# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
ExportHFStage - Export model to HuggingFace format.

This stage exports a model (quantized or compressed) to HuggingFace format.

Input:
    - ctx.refs["model"], ctx.refs["tokenizer"]

Output:
    - ctx.artifacts["export_dir"]: Path to exported model directory
"""
from pathlib import Path
from typing import Dict, Any

from modelopt.torch.utils.logging import print_rank_0

from .base import Stage, StageContext


def _calibration_record(provenance) -> dict | None:
    """DatasetProvenance -> the dict the pod inventory reads, or None.

    Key names match what the inventory probe prints (dataset / num_samples /
    seq_len / seed); the rest ride along for full reproducibility. None in,
    None out: the ModelOpt-dataloader path records no provenance, and writing
    a half-empty record would make 'unknown' look like 'known'.
    """
    if provenance is None:
        return None
    import dataclasses

    d = (dataclasses.asdict(provenance)
         if dataclasses.is_dataclass(provenance) else dict(provenance))
    d["dataset"] = d.pop("dataset_ref", None)
    return d


class ExportHFStage(Stage):
    """
    Export model to HuggingFace format.

    This stage saves a model and tokenizer. Residue vLLM exports use standard
    HF config plus residue_kernel_metadata.json; legacy custom HF model support
    is archived and no longer generated here.
    """

    def __init__(self, name: str = "export"):
        """Initialize export stage."""
        super().__init__(name)

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        """Validate required inputs."""
        if ctx.get_ref("model") is None or ctx.get_ref("tokenizer") is None:
            raise ValueError(
                "ExportHFStage requires:\n"
                "  - ctx.refs['model']\n"
                "  - ctx.refs['tokenizer']"
            )

        # Must have export_dir
        export_dir = ctx.get_artifact("export_dir") or params.get("export_dir")
        if not export_dir:
            raise ValueError(
                "ExportHFStage requires export directory in:\n"
                "  - ctx.artifacts['export_dir'], or\n"
                "  - params['export_dir']"
            )

    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        """
        Export model to HuggingFace format.

        Args:
            ctx: Stage context with model and tokenizer in refs
            params: Parameters including:
                - export_dir: Directory to export to
                - safe_serialization: Use safetensors (default: True)
                - use_custom_config: Deprecated legacy option, ignored
                - export_target: Target runtime ("huggingface" or "vllm")
                - save_kernel_metadata: Save kernel metadata for custom deployment (default: False)
                - residue_ratio: Residue ratio (required if save_kernel_metadata=True)
                - verbose: (optional) Enable debug logging

        Returns:
            Updated context with export_dir in artifacts
        """
        verbose = params.get("verbose", False)
        safe_serialization = params.get("safe_serialization", True)
        use_custom_config = params.get("use_custom_config", False)
        export_target = params.get("export_target", "huggingface")
        save_kernel_metadata = params.get("save_kernel_metadata", False)
        residue_ratio = params.get("residue_ratio")

        if use_custom_config:
            print_rank_0(
                "  Note: use_custom_config is archived and ignored; "
                "export uses standard HF config plus residue metadata."
            )
            use_custom_config = False

        # Get export directory (resolve relative to case_dir)
        export_dir = ctx.get_artifact("export_dir") or params.get("export_dir")
        export_dir = ctx.resolve_path(Path(export_dir))

        # Get model and tokenizer
        model = ctx.get_ref("model")
        tokenizer = ctx.get_ref("tokenizer")
        runtime_asset_source = (
            params.get("runtime_asset_source")
            or ctx.get_artifact("model_path")
            or params.get("source_checkpoint")
        )

        # Export model
        print_rank_0(f"Exporting model to: {export_dir}")
        print_rank_0(f"  Export target: {export_target}")
        export_path = self._export_model(
            model,
            tokenizer,
            export_dir,
            safe_serialization,
            use_custom_config,
            export_target,
            save_kernel_metadata,
            residue_ratio,
            int(params.get("calib_block_size", params.get("block_size", 8))),
            verbose,
            calibration=_calibration_record(ctx.get_meta("dataset_provenance")),
            max_shard_size=params.get("max_shard_size", "10GB"),
            source_checkpoint=params.get("source_checkpoint"),
            processor=ctx.get_ref("processor"),
            runtime_asset_source=runtime_asset_source,
        )

        # Update context
        ctx.set_artifact("export_dir", export_path)

        return ctx

    def _export_model(
        self,
        model,
        tokenizer,
        export_dir: Path,
        safe_serialization: bool = True,
        use_custom_config: bool = False,
        export_target: str = "huggingface",
        save_kernel_metadata: bool = False,
        residue_ratio: float = None,
        kernel_metadata_block_size: int = 8,
        verbose: bool = False,
        calibration: dict | None = None,
        max_shard_size: str = "10GB",
        source_checkpoint=None,
        processor=None,
        runtime_asset_source: str | Path | None = None,
    ) -> Path:
        """Export native ModelOpt weights and common residue metadata."""
        if model is None:
            raise RuntimeError(
                "Model is None! Cannot export. "
                "This indicates a bug in a previous stage."
            )

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        if verbose:
            print_rank_0(f"  Saving model (safe_serialization={safe_serialization})...")

        # ALWAYS use ModelOpt export to correctly save FP4 packed weights
        # export_hf_checkpoint() calls to_quantized_weight() to properly pack FP16 → FP4
        # Standard save_pretrained() doesn't understand QTensor packing
        import torch

        def _resolve_export_dtype(m: torch.nn.Module) -> torch.dtype:
            # Prefer HF config dtype when present.
            candidate = getattr(m.config, "torch_dtype", None)
            if isinstance(candidate, str):
                candidate = getattr(torch, candidate, None)
            if isinstance(candidate, torch.dtype) and candidate.is_floating_point:
                return candidate

            # Some models expose a top-level dtype.
            candidate = getattr(m, "dtype", None)
            if isinstance(candidate, torch.dtype) and candidate.is_floating_point:
                return candidate

            # Fall back to any floating parameter/buffer (avoid uint8 fp4 packed weights).
            for p in m.parameters():
                if isinstance(p.dtype, torch.dtype) and p.dtype.is_floating_point:
                    return p.dtype
            for b in m.buffers():
                if isinstance(b.dtype, torch.dtype) and b.dtype.is_floating_point:
                    return b.dtype

            # Last resort: BF16 is the safest default for TRTLLM FP4 GEMM.
            return torch.bfloat16

        model_dtype = _resolve_export_dtype(model)
        if verbose:
            print_rank_0(f"  Export dtype: {model_dtype}")

        from quant.modelopt_integration.export import export_residue_checkpoint

        export_residue_checkpoint(
            model=model, export_dir=export_path, dtype=model_dtype,
            residue_ratio=residue_ratio if residue_ratio is not None else 0.125,
            block_size=kernel_metadata_block_size, calibration=calibration,
            max_shard_size=max_shard_size, source_checkpoint=source_checkpoint,
            processor=processor, tokenizer=tokenizer,
            save_kernel_metadata=save_kernel_metadata,
        )

        if verbose:
            print_rank_0(f"  ✓ Exported FP4 packed weights (nibble-packed uint8)")

        if runtime_asset_source is not None:
            from export.hf_runtime_assets import copy_hf_runtime_assets

            runtime_assets = copy_hf_runtime_assets(
                runtime_asset_source, export_path
            )
            if verbose and runtime_assets:
                print_rank_0("  ✓ Runtime assets: " + ", ".join(runtime_assets))

        if verbose:
            print_rank_0(f"  ✓ Model saved to {export_path}")

        return export_path
