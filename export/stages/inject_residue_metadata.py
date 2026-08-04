# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
Inject external residue salient config into a restored ModelOpt model.

This stage is intended for workflows such as GPTQ+Res:
    1. load base model
    2. restore ModelOpt checkpoint saved after base quant calibration
    3. inject external residue salient indices
    4. extend weights on the live, uncompressed model
    5. export

The input file is compatible with exported ``residue_kernel_metadata.json`` but
may also come from another project, as long as it provides per-layer
``salient_indices`` under ``layers``.

Input:
    - ctx.refs["model"]: Restored, uncompressed ModelOpt model
    - params["residue_metadata_path"]: Path to external residue salient config JSON
    - params["residue_ratio"] (optional): Global ratio for compatibility checks only

Output:
    - ctx.refs["model"]: Same model, updated in place
    - ctx.artifacts["injected_residue_metadata"]: Metadata path consumed
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from modelopt.torch.utils.logging import print_rank_0

from .base import Stage, StageContext


class InjectResidueMetadataStage(Stage):
    """Inject residue metadata and trigger weight extension."""

    def __init__(self, name: str = "inject_residue_metadata"):
        super().__init__(name)

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        if ctx.get_ref("model") is None:
            raise ValueError(
                "InjectResidueMetadataStage requires:\n"
                "  - ctx.refs['model']\n"
                "Hint: run modelopt_restore_checkpoint first."
            )

        residue_metadata_path = params.get("residue_metadata_path")
        if not residue_metadata_path:
            raise ValueError(
                "InjectResidueMetadataStage requires:\n"
                "  - params['residue_metadata_path']"
            )

    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        from quant.modelopt_integration import (
            create_and_replace_residue_linear_on_the_fly,
            inject_residue_salient_map,
        )

        model = ctx.get_ref("model")
        metadata_path = ctx.resolve_path(Path(params["residue_metadata_path"]))
        requested_ratio = params.get("residue_ratio")
        residue_ratio = float(requested_ratio) if requested_ratio is not None else None
        verbose = params.get("verbose", False)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Residue metadata not found: {metadata_path}")

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        layers = metadata.get("layers")
        if not isinstance(layers, dict) or not layers:
            raise ValueError(
                f"Invalid residue metadata: expected non-empty 'layers' dict in {metadata_path}"
            )

        global_meta = metadata.get("global", {})
        metadata_ratio = global_meta.get("residue_ratio")
        if (
            residue_ratio is not None
            and metadata_ratio is not None
            and abs(float(metadata_ratio) - residue_ratio) > 1e-9
        ):
            raise ValueError(
                f"Residue ratio mismatch: params['residue_ratio']={residue_ratio} "
                f"but metadata declares {metadata_ratio} in {metadata_path}"
            )
        declared_ratio = residue_ratio if residue_ratio is not None else (
            float(metadata_ratio) if metadata_ratio is not None else None
        )
        helper_ratio = declared_ratio if declared_ratio is not None else 0.25

        create_and_replace_residue_linear_on_the_fly(model)
        modules = dict(model.named_modules())
        block_size = int(global_meta.get("block_size", params.get("calib_block_size", params.get("block_size", 8))))
        salient_map: Dict[str, torch.Tensor] = {}
        skipped_non_salient_layers = []

        print_rank_0(f"Injecting residue metadata from: {metadata_path}")
        if declared_ratio is None:
            print_rank_0("  Residue ratio: external/per-layer indices (no global ratio declared)")
        else:
            print_rank_0(f"  Residue ratio: {declared_ratio}")
        print_rank_0(f"  Block size: {block_size}")
        print_rank_0(f"  Metadata layers: {len(layers)}")

        for layer_name, layer_meta in layers.items():
            raw_indices = layer_meta.get("salient_indices")
            if raw_indices is None:
                skipped_non_salient_layers.append(layer_name)
                if verbose:
                    print_rank_0(
                        f"  Skipping metadata-only layer without salient_indices: {layer_name}"
                    )
                continue

            if layer_name not in modules:
                raise KeyError(f"Layer from residue metadata not found in model: {layer_name}")

            module = modules[layer_name]
            if not hasattr(module, "input_quantizer") or not hasattr(module, "weight_quantizer"):
                raise TypeError(
                    f"Layer '{layer_name}' is not a quantized linear module "
                    f"(missing input_quantizer/weight_quantizer)"
                )

            salient_indices = torch.as_tensor(raw_indices, dtype=torch.int64)
            if salient_indices.ndim != 1:
                raise ValueError(
                    f"Layer '{layer_name}' salient_indices must be 1D, got ndim={salient_indices.ndim}"
                )
            salient_indices = salient_indices.contiguous()
            if salient_indices.numel() == 0:
                raise ValueError(f"Layer '{layer_name}' has empty salient_indices")

            declared_num_salient = layer_meta.get("num_salient")
            if declared_num_salient is not None and int(declared_num_salient) != salient_indices.numel():
                raise ValueError(
                    f"Layer '{layer_name}' num_salient mismatch: "
                    f"declared={declared_num_salient}, actual={salient_indices.numel()}"
                )

            current_in_features = getattr(
                module,
                "_original_in_features",
                getattr(module, "in_features", None),
            )
            if current_in_features is None:
                raise ValueError(
                    f"Layer '{layer_name}' has no in_features/_original_in_features for validation"
                )

            min_idx = int(salient_indices.min().item())
            max_idx = int(salient_indices.max().item())
            if min_idx < 0 or max_idx >= int(current_in_features):
                raise ValueError(
                    f"Layer '{layer_name}' salient_indices out of bounds: "
                    f"min={min_idx}, max={max_idx}, in_features={current_in_features}"
                )

            # ratio-1.0 layers are not K-extended (mext_r1 runtime mode), so
            # the extended-K block-alignment invariant does not apply to them.
            if int(salient_indices.numel()) != int(current_in_features):
                self._validate_weight_block_alignment(
                    module=module,
                    current_in_features=int(current_in_features),
                    num_salient=int(salient_indices.numel()),
                    layer_name=layer_name,
                )
            salient_map[layer_name] = salient_indices

        summary = inject_residue_salient_map(
            model,
            salient_map,
            block_size=block_size,
            residue_ratio=helper_ratio,
            debug=verbose,
            replace_modules=False,
            preserve_existing_amax=True,
        )

        extended_count = int(summary["newly_extended_layers"])
        already_extended_count = int(summary["already_extended_layers"])
        mext_r1_count = int(summary.get("mext_r1_layers", 0))

        ctx.set_artifact("injected_residue_metadata", metadata_path)
        ctx.set_meta("residue_metadata_path", str(metadata_path))
        ctx.set_meta("residue_ratio", declared_ratio)
        ctx.set_meta("injected_residue_layers", extended_count + already_extended_count + mext_r1_count)
        ctx.set_meta("skipped_non_salient_metadata_layers", skipped_non_salient_layers)
        ctx.metrics["injected_residue_layers"] = (
            extended_count + already_extended_count + mext_r1_count
        )
        ctx.metrics["newly_extended_residue_layers"] = extended_count
        ctx.metrics["already_extended_residue_layers"] = already_extended_count
        ctx.metrics["mext_r1_residue_layers"] = mext_r1_count
        ctx.metrics["skipped_non_salient_metadata_layers"] = len(skipped_non_salient_layers)

        print_rank_0(
            f"✓ Injected residue metadata for "
            f"{extended_count + already_extended_count + mext_r1_count} layers "
            f"({extended_count} newly extended, {already_extended_count} already "
            f"extended, {mext_r1_count} mext_r1/ratio-1.0 without weight extension)"
        )
        if skipped_non_salient_layers:
            print_rank_0(
                f"  Skipped {len(skipped_non_salient_layers)} metadata-only layers without salient indices"
            )

        return ctx

    @staticmethod
    def _validate_weight_block_alignment(
        module,
        current_in_features: int,
        num_salient: int,
        layer_name: str,
    ):
        block_sizes = getattr(getattr(module, "weight_quantizer", None), "_block_sizes", None)
        quant_block_size = None
        if isinstance(block_sizes, dict):
            quant_block_size = block_sizes.get(-1)

        if not isinstance(quant_block_size, int) or quant_block_size <= 1:
            return

        extended_in_features = current_in_features + num_salient
        if extended_in_features % quant_block_size != 0:
            raise ValueError(
                f"Layer '{layer_name}' would extend from {current_in_features} to "
                f"{extended_in_features}, which is not divisible by quant block size "
                f"{quant_block_size}. Use block-aligned salient metadata before export."
            )
