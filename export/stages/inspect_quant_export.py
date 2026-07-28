# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Lightweight quant/export sanity checks for smoke suites."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from modelopt.torch.quantization.utils import is_quantized_linear
from modelopt.torch.utils.logging import print_rank_0

from .base import Stage, StageContext


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return value


def _values_equal(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
        if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
            return False
        return lhs.shape == rhs.shape and torch.equal(lhs.cpu(), rhs.cpu())
    return lhs == rhs


class SnapshotQuantizerStateStage(Stage):
    """Snapshot quantizer state before external residue injection."""

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        if ctx.get_ref("model") is None:
            raise ValueError("SnapshotQuantizerStateStage requires ctx.refs['model']")

    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        model = ctx.get_ref("model")
        snapshot: Dict[str, Dict[str, Any]] = {}

        for name, module in model.named_modules():
            if not is_quantized_linear(module):
                continue
            if not hasattr(module, "input_quantizer") or not hasattr(module, "weight_quantizer"):
                continue
            if not getattr(module.weight_quantizer, "is_enabled", False):
                continue

            snapshot[name] = {
                "weight_amax": _clone_value(getattr(module.weight_quantizer, "amax", None)),
                "input_amax": _clone_value(getattr(module.input_quantizer, "amax", None)),
                "input_axis": getattr(module.input_quantizer, "axis", None),
                "input_enabled": getattr(module.input_quantizer, "is_enabled", None),
                "weight_enabled": getattr(module.weight_quantizer, "is_enabled", None),
            }

        ctx.set_ref("quantizer_state_snapshot", snapshot)
        ctx.metrics["quantizer_snapshot_layers"] = len(snapshot)
        print_rank_0(f"Snapshot quantizer state for {len(snapshot)} layers")
        return ctx


class CheckQuantizerStateStage(Stage):
    """Check that external residue injection preserved quantizer scale state."""

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        if ctx.get_ref("model") is None:
            raise ValueError("CheckQuantizerStateStage requires ctx.refs['model']")
        if ctx.get_ref("quantizer_state_snapshot") is None:
            raise ValueError("CheckQuantizerStateStage requires quantizer_state_snapshot")

    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        model = ctx.get_ref("model")
        modules = dict(model.named_modules())
        snapshot: Dict[str, Dict[str, Any]] = ctx.get_ref("quantizer_state_snapshot")
        mismatches = []

        for name, expected in snapshot.items():
            if name not in modules:
                mismatches.append(f"{name}: module missing after injection")
                continue

            module = modules[name]
            current = {
                "weight_amax": _clone_value(getattr(module.weight_quantizer, "amax", None)),
                "input_amax": _clone_value(getattr(module.input_quantizer, "amax", None)),
                "input_axis": getattr(module.input_quantizer, "axis", None),
                "input_enabled": getattr(module.input_quantizer, "is_enabled", None),
                "weight_enabled": getattr(module.weight_quantizer, "is_enabled", None),
            }
            for key, expected_value in expected.items():
                if not _values_equal(expected_value, current[key]):
                    mismatches.append(f"{name}.{key}")

        ctx.metrics["quantizer_state_checked_layers"] = len(snapshot)
        ctx.metrics["quantizer_state_mismatches"] = len(mismatches)
        if mismatches:
            preview = ", ".join(mismatches[:10])
            raise AssertionError(
                f"Quantizer state changed after residue injection: {preview}"
            )

        print_rank_0(f"Quantizer state preserved for {len(snapshot)} layers")
        return ctx


class InspectQuantExportStage(Stage):
    """Inspect exported safetensors/index metadata for scale tensors."""

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        export_dir = ctx.get_artifact("export_dir") or params.get("export_dir")
        if not export_dir:
            raise ValueError("InspectQuantExportStage requires export_dir")

    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        export_dir = ctx.resolve_path(Path(ctx.get_artifact("export_dir") or params["export_dir"]))
        keys = self._load_export_keys(export_dir)

        expected_counts = params.get("expected_export_suffix_counts", {})
        actual_counts = {}
        for suffix, expected in expected_counts.items():
            normalized_suffix = suffix if str(suffix).startswith(".") else f".{suffix}"
            actual = sum(key.endswith(normalized_suffix) for key in keys)
            actual_counts[str(suffix)] = actual
            if int(actual) != int(expected):
                raise AssertionError(
                    f"{export_dir}: expected {expected} tensors ending with "
                    f"{normalized_suffix}, got {actual}"
                )

        if params.get("forbid_salient_indices_in_safetensors", False):
            salient_keys = [key for key in keys if "_salient_indices" in key]
            if salient_keys:
                raise AssertionError(
                    f"{export_dir}: exported safetensors still contain salient indices: "
                    f"{salient_keys[:5]}"
                )

        finite_counts = {}
        if params.get("require_finite_export_tensors", False):
            suffixes = params.get(
                "finite_export_tensor_suffixes",
                ["input_scale", "weight_scale", "weight_scale_2"],
            )
            finite_counts = self._check_finite_tensors(export_dir, suffixes)

        min_metadata_layers = params.get("expected_residue_metadata_min_layers")
        min_extended_dims = params.get("expected_extended_linear_dims_min")
        if min_metadata_layers is not None or min_extended_dims is not None:
            metadata_path = export_dir / "residue_kernel_metadata.json"
            if not metadata_path.exists():
                raise AssertionError(f"{export_dir}: missing residue_kernel_metadata.json")
            metadata = json.loads(metadata_path.read_text())
            layers = metadata.get("layers", {})
            extended_dims = metadata.get("extended_linear_dims", {})
            if min_metadata_layers is not None and len(layers) < int(min_metadata_layers):
                raise AssertionError(
                    f"{metadata_path}: expected at least {min_metadata_layers} metadata layers, "
                    f"got {len(layers)}"
                )
            if min_extended_dims is not None and len(extended_dims) < int(min_extended_dims):
                raise AssertionError(
                    f"{metadata_path}: expected at least {min_extended_dims} extended dims, "
                    f"got {len(extended_dims)}"
                )

        ctx.metrics["export_tensor_keys"] = len(keys)
        ctx.metrics["export_suffix_counts"] = actual_counts
        if finite_counts:
            ctx.metrics["finite_export_tensor_counts"] = finite_counts
        print_rank_0(f"Inspect export OK: {export_dir}")
        return ctx

    @staticmethod
    def _load_export_keys(export_dir: Path) -> list[str]:
        index_path = export_dir / "model.safetensors.index.json"
        if index_path.exists():
            data = json.loads(index_path.read_text())
            return sorted(data.get("weight_map", {}).keys())

        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to inspect unsharded exports without an index"
            ) from exc

        keys: list[str] = []
        for shard in sorted(export_dir.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                keys.extend(handle.keys())
        if not keys:
            raise FileNotFoundError(f"No safetensors found in {export_dir}")
        return sorted(keys)

    @staticmethod
    def _check_finite_tensors(export_dir: Path, suffixes: list[str]) -> dict[str, int]:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("safetensors is required to inspect exported tensor values") from exc

        normalized_suffixes = [
            suffix if str(suffix).startswith(".") else f".{suffix}"
            for suffix in suffixes
        ]
        counts = {suffix.lstrip("."): 0 for suffix in normalized_suffixes}
        nonfinite = []

        for shard in sorted(export_dir.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    matched_suffix = next(
                        (suffix for suffix in normalized_suffixes if key.endswith(suffix)),
                        None,
                    )
                    if matched_suffix is None:
                        continue

                    counts[matched_suffix.lstrip(".")] += 1
                    tensor = handle.get_tensor(key)
                    values = tensor.float()
                    bad = int((~torch.isfinite(values)).sum().item())
                    if bad:
                        nonfinite.append(
                            f"{key}: {bad} non-finite values, shape={tuple(tensor.shape)}, "
                            f"dtype={tensor.dtype}"
                        )

        if nonfinite:
            preview = "; ".join(nonfinite[:10])
            raise AssertionError(f"{export_dir}: non-finite exported tensors: {preview}")

        return counts
