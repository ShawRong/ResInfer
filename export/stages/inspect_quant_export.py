# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Lightweight quant/export sanity checks for smoke suites."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

# ModelOpt is imported LAZILY, and that is load-bearing rather than tidy.
# This module is what a pod-side task probes to answer "does this checkout
# carry the tensor-opening export guard?" -- and it probes from the RUNTIME
# venv, which deliberately has no ModelOpt (that lives in env-export). With a
# module-level import the probe died on ModuleNotFoundError and the task
# reported "stale export guard, sync the pod", so a dependency problem read as
# a version problem and no amount of syncing could fix it. The width check
# itself needs only json + safetensors; only _collect_quantized_linears needs
# ModelOpt, and it asks for it when it runs.


def print_rank_0(*args, **kwargs):
    """ModelOpt's rank-aware print when available, plain print otherwise."""
    try:
        from modelopt.torch.utils.logging import print_rank_0 as _p
    except ImportError:
        print(*args, **kwargs)
    else:
        _p(*args, **kwargs)

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
            from modelopt.torch.quantization.utils import is_quantized_linear

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
            config = json.loads((export_dir / "config.json").read_text())
            quant_config = config.get("quantization_config", {})
            is_mxfp4 = (quant_config.get("quant_method") == "modelopt" and
                        str(quant_config.get("quant_algo", "")).upper() == "MXFP4")
            metadata_path = export_dir / "residue_kernel_metadata.json"
            if not metadata_path.exists():
                raise AssertionError(f"{export_dir}: missing {metadata_path.name}")
            metadata = json.loads(metadata_path.read_text())
            layers = metadata.get("layers", {})
            extended_dims = metadata.get("extended_linear_dims", {})
            if min_metadata_layers is not None and len(layers) < int(min_metadata_layers):
                raise AssertionError(
                    f"{metadata_path}: expected at least {min_metadata_layers} metadata layers, "
                    f"got {len(layers)}"
                )
            # mext_r1 (ratio-1.0) layers carry residue without weight
            # extension, so they count as residue evidence alongside
            # extended_linear_dims entries.
            mext_r1_layers = sum(
                1 for v in layers.values()
                if isinstance(v, dict) and v.get("runtime_mode", v.get("mode")) == "mext_r1"
            )
            if (min_extended_dims is not None
                    and len(extended_dims) + mext_r1_layers < int(min_extended_dims)):
                raise AssertionError(
                    f"{metadata_path}: expected at least {min_extended_dims} extended dims "
                    f"(or mext_r1 layers), got {len(extended_dims)} extended + "
                    f"{mext_r1_layers} mext_r1"
                )

            # Every check above reads the METADATA -- which is exactly the
            # artefact that can be right while the export is wrong. A driver
            # whose chain never widened the weights still writes
            # runtime_mode=extended_k with a full num_salient, passes all of
            # the above, records PASS, and hands over a checkpoint that dies
            # at load with a device-side assert when the loader indexes
            # columns that were never written. Verifying a claim against the
            # same claim is not verification; open the tensors.
            if is_mxfp4:
                self._assert_mxfp4_tensors(export_dir, layers)
            else:
                self._assert_extended_widths(export_dir, layers)

        ctx.metrics["export_tensor_keys"] = len(keys)
        ctx.metrics["export_suffix_counts"] = actual_counts
        if finite_counts:
            ctx.metrics["finite_export_tensor_counts"] = finite_counts
        print_rank_0(f"Inspect export OK: {export_dir}")
        return ctx

    @classmethod
    def _assert_mxfp4_tensors(cls, export_dir: Path, layers: dict):
        from safetensors import safe_open
        from quant.kernel_metadata import residue_spec, padded_k
        config = json.loads((export_dir / "config.json").read_text())
        quant = config.get("quantization_config", {})
        if quant.get("quant_method") != "modelopt" or str(quant.get("quant_algo", "")).upper() != "MXFP4":
            raise AssertionError("expected native ModelOpt MXFP4 config")
        expected = {}
        for shard in sorted(export_dir.glob("*.safetensors")):
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    if not key.endswith(".weight_scale"):
                        continue
                    name = key[:-len(".weight_scale")]
                    base = cls._expected_in_features(name, config.get("text_config", config))
                    if base is None:
                        raise AssertionError(f"unknown MXFP4 source input dimension: {name}")
                    n = f.get_slice(key).get_shape()[0]
                    spec = residue_spec(layers.get(name), base, n)
                    kp = padded_k(base, len(spec.get("indices", [])))
                    for suffix, divisor in (("weight", 2), ("weight_scale", 32)):
                        expected[f"{name}.{suffix}"] = [n, kp // divisor]
        if not expected or any(name + ".weight" not in expected for name in layers):
            raise AssertionError("missing MXFP4 tensors for residue metadata")
        seen = {}
        for shard in sorted(export_dir.glob("*.safetensors")):
            with safe_open(shard, framework="pt") as f:
                for name in f.keys():
                    if name in seen:
                        raise AssertionError(f"duplicate MXFP4 tensor: {name}")
                    seen[name] = shard.name
                    if name in expected:
                        tensor = f.get_tensor(name)
                        if tensor.dtype != torch.uint8 or list(tensor.shape) != expected[name]:
                            raise AssertionError(f"MXFP4 tensor shape/dtype mismatch: {name}")
                        if name.endswith(".weight_scale") and (tensor == 255).any().item():
                            raise AssertionError(f"NaN E8M0 scale in {name}")
        index_path = export_dir / "model.safetensors.index.json"
        index = (json.loads(index_path.read_text())["weight_map"]
                 if index_path.exists() else seen)
        if seen != index or set(expected) - seen.keys():
            raise AssertionError("MXFP4 shard index or tensor inventory mismatch")

    @staticmethod
    def _expected_in_features(name: str, cfg: dict) -> int | None:
        """Logical input width of a projection, from the model config.

        The independent source this check needs. `num_salient` and the stored
        width alone cannot decide anything: 1280 packed columns is a perfectly
        legal base width AND a perfectly legal extended one, so a comparison
        between the two numbers has no fixed point. The config is the only
        thing here that was not written by the export being audited.
        """
        h = cfg.get("hidden_size")
        if h is None:
            return None
        inter = cfg.get("intermediate_size")
        if name.endswith(("q_proj", "k_proj", "v_proj",
                          "gate_proj", "up_proj", "qkv_proj", "gate_up_proj")):
            return h
        if name.endswith("o_proj"):
            heads = cfg.get("num_attention_heads")
            if not heads:
                return None
            return heads * cfg.get("head_dim", h // heads)
        if name.endswith("down_proj"):
            return inter
        if name.endswith(("in_proj_qkv", "in_proj_z", "in_proj_qkvz")):
            return h
        if name.endswith("linear_attn.out_proj"):
            return cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
        return None       # unrecognised layer: not this check's business

    @classmethod
    def _assert_extended_widths(cls, export_dir: Path, layers: dict) -> int:
        """Every extended_k layer must be as wide as its metadata claims.

        FP4 is nibble-packed, so the stored width is half the logical one:

            packed == (in_features + num_salient) // 2

        A chain that declared the extension but never widened the tensors is
        short by exactly num_salient // 2, and nothing else in this stage can
        see it -- every other check reads the metadata, which is the artefact
        that is right while the weights are wrong.

        mext_r1 layers are skipped deliberately: their residue lives in the
        activation's M dimension and base width is CORRECT for them. Asserting
        extension there would reject good exports (a real b80 export has three
        extended_k layers and the rest mext_r1).
        """
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to verify extended weight widths"
            ) from exc

        cfg_path = export_dir / "config.json"
        if not cfg_path.exists():
            return 0
        cfg = json.loads(cfg_path.read_text())
        cfg = cfg.get("text_config", cfg)

        widths: dict[str, int] = {}
        for shard in sorted(export_dir.glob("*.safetensors")):
            with safe_open(shard, framework="pt") as f:
                for k in f.keys():
                    if k.endswith(".weight"):
                        shape = f.get_slice(k).get_shape()
                        if len(shape) == 2:
                            widths[k[: -len(".weight")]] = int(shape[1])

        bad, checked = [], 0
        for name, meta in layers.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("runtime_mode") != "extended_k":
                continue
            num_salient = int(meta.get("num_salient", 0))
            if num_salient <= 0:
                continue
            got = widths.get(name)
            in_features = cls._expected_in_features(name, cfg)
            if got is None or in_features is None:
                continue
            expected = (int(in_features) + num_salient) // 2
            checked += 1
            if got != expected:
                bad.append(f"{name}: num_salient={num_salient} "
                           f"in_features={in_features} expected packed="
                           f"{expected} got={got}")
        if bad:
            raise AssertionError(
                f"{export_dir}: {len(bad)} of {checked} extended_k layers are "
                f"NOT widened -- the metadata declares an extension the "
                f"weights do not carry, and such a checkpoint dies at load "
                f"with a device-side assert. First: " + bad[0])
        return checked

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
            # Name the resolution, not just the miss. `export_dir` arrives
            # through ctx.resolve_path, which makes a RELATIVE path relative
            # to case_dir -- correct for a stage, and surprising to a caller
            # who passed `--out foo` from inside `foo` and now reads
            # "No safetensors found in foo/foo". The behaviour is not the
            # defect; a message that says "missing file" when the truth is
            # "different directory than you meant" is.
            raise FileNotFoundError(
                f"No safetensors found in {export_dir} "
                f"(exists={export_dir.exists()}). Relative export_dir values "
                f"are resolved against the stage's case_dir, so a relative "
                f"--out can land one level deeper than intended; pass an "
                f"absolute path if this is not the directory you meant.")
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
