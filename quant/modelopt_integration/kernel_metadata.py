"""Metadata helpers for vLLM residue kernel export.

This module is the active home for residue export metadata. The old custom
HuggingFace model implementation lives under archive/hf_custom.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def is_mext_r1_layer(num_salient: int, in_features: int) -> bool:
    """Does this layer's residue live in the M dimension rather than in K?

    True on exactly one condition: EVERY input channel is salient, i.e.
    residue ratio 1.0. That is the whole rule -- `mext_r1` is not a mode
    competing with the ratio options, it IS ratio 1.0, stored so the weight
    is kept once instead of twice:

        extended_k at ratio 1.0:  [x_q | r_q] @ [W | W]^T = x_q@W^T + r_q@W^T
        mext_r1:                  rows (x_q, r_q) @ W^T, summed in the epilogue

    Both compute the same thing; only mext_r1 avoids duplicating W.

    This predicate is the ONLY place the equivalence is decided. It used to be
    re-derived at four sites with four different expressions
    (`salient_indices.numel() == in_features`, `== module.weight.shape[1]`,
    `== k_base`, `num_salient == int(in_features)`), which is how a reader can
    meet the rule repeatedly without ever meeting a statement of it.

    Note this is a PROPERTY OF THE LAYER, decided at export. It is not the
    runtime execution path: a mext_r1 layer can still be served by a non-fold
    kernel depending on M. Do not infer one from the other -- read
    `runtime_mode` for the former and the dispatcher's own state for the
    latter. Conflating them (via the overloaded `num_salient == 0` sentinel)
    has already produced one silent-wrong-numerics bug.
    """
    return int(num_salient) == int(in_features)


def _debug(msg: str) -> None:
    if os.getenv("RESIDUE_DEBUG", "0") == "1":
        print(msg)


def collect_extended_linear_dims(model) -> dict[str, int]:
    """Collect Linear layers whose physical input dimension was extended."""
    extended_dims: dict[str, int] = {}

    for name, module in model.named_modules():
        if hasattr(module, "_original_in_features"):
            original_in = int(module._original_in_features)
            actual_in = int(module.in_features)
            if actual_in != original_in:
                extended_dims[name] = actual_in
                _debug(f"[Collect] Extended: {name} {original_in} -> {actual_in}")
        elif isinstance(module, torch.nn.Linear):
            actual_in = int(module.weight.shape[1])
            declared_in = int(module.in_features)
            if actual_in != declared_in:
                extended_dims[name] = actual_in
                _debug(f"[Collect] Extended (by shape): {name} {declared_in} -> {actual_in}")

    _debug(f"[Collect] Found {len(extended_dims)} extended Linear layers")
    return extended_dims


def collect_salient_indices(model) -> dict[str, list[int]]:
    """Collect input-quantizer salient indices for JSON serialization."""
    indices: dict[str, list[int]] = {}

    for name, module in model.named_modules():
        input_quantizer = getattr(module, "input_quantizer", None)
        salient = getattr(input_quantizer, "_salient_indices", None)
        if salient is not None:
            indices[name] = salient.cpu().tolist()
            _debug(f"[Collect] Salient indices: {name} ({len(indices[name])} channels)")

    _debug(f"[Collect] Found salient indices for {len(indices)} layers")
    return indices


def collect_kernel_metadata(
    model,
    residue_ratio: float = 0.25,
    block_size: int = 8,
) -> dict[str, Any]:
    """Collect metadata consumed by residue vLLM linear methods."""
    metadata: dict[str, Any] = {
        "layers": {},
        "global": {
            "residue_ratio": residue_ratio,
            "block_size": block_size,
        },
    }

    for name, module in model.named_modules():
        input_quantizer = getattr(module, "input_quantizer", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)
        if input_quantizer is None or weight_quantizer is None:
            continue

        layer_data: dict[str, Any] = {}
        salient = getattr(input_quantizer, "_salient_indices", None)
        if salient is not None:
            layer_data["salient_indices"] = salient.cpu().tolist()
            layer_data["num_salient"] = len(layer_data["salient_indices"])
            # ratio-1.0 layers are never K-extended: they carry
            # runtime_mode=mext_r1 (set at injection time) and are served by
            # the M-extension fold path with the weight at original K. All
            # other residue layers stay on the extended_k contract (which is
            # also the reader default for legacy checkpoints).
            runtime_mode = getattr(module, "_residue_runtime_mode", None)
            if runtime_mode is None and not hasattr(module, "_original_in_features"):
                in_features = getattr(module, "in_features", None)
                if in_features is not None and is_mext_r1_layer(
                    layer_data["num_salient"], in_features
                ):
                    runtime_mode = "mext_r1"
            layer_data["runtime_mode"] = runtime_mode or "extended_k"

        # Keep per-layer metadata intentionally minimal. Historical debug
        # fields such as input_amax/weight_amax/original_in_features/
        # extended_in_features/out_features had ambiguous meanings after
        # ModelOpt export mutates weights into FP4 packed form. Runtime paths
        # should derive dimensions from tensors or top-level extended_linear_dims
        # instead of trusting stale per-layer debug fields.

        if layer_data:
            metadata["layers"][name] = layer_data

    return metadata


def _infer_mask_in_features(module, salient_indices: list[int]) -> int | None:
    """Infer the logical input dimension K for optional channel mask export."""
    if hasattr(module, "_original_in_features"):
        return int(module._original_in_features)

    if hasattr(module, "in_features"):
        return int(module.in_features)

    if hasattr(module, "weight"):
        weight = module.weight
        if hasattr(weight, "shape") and len(weight.shape) >= 2:
            # Last-resort dense fallback. Avoid using this for packed FP4
            # metadata; this helper is only for optional channel_masks.pt.
            inferred = int(weight.shape[1])
            if salient_indices and max(salient_indices) < inferred:
                return inferred

    return None


def save_kernel_metadata(
    model,
    export_dir: str | Path,
    residue_ratio: float = 0.25,
    block_size: int = 8,
    save_channel_masks: bool = True,
    calibration: dict | None = None,
) -> dict[str, Path]:
    """Save vLLM residue kernel metadata and optional bool channel masks.

    `calibration` records what the quantizers were calibrated ON (dataset,
    num_samples, seq_len, seed, ...). Two checkpoints that differ in nothing
    but this are different models, and without the record the difference is
    invisible: a results table comparing them cannot separate the algorithm
    from the calibration, which already forced one conclusion to be
    downgraded. Absent stays absent -- the pod inventory prints NOT RECORDED
    for those rather than letting the gap read as fine.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    saved_files: dict[str, Path] = {}

    metadata = collect_kernel_metadata(
        model, residue_ratio, block_size=block_size)
    metadata["extended_linear_dims"] = collect_extended_linear_dims(model)
    if calibration:
        metadata["calibration"] = calibration
    metadata_path = export_dir / "residue_kernel_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    saved_files["metadata_json"] = metadata_path

    if save_channel_masks:
        masks: dict[str, torch.Tensor] = {}
        modules = dict(model.named_modules())
        for name, layer_data in metadata["layers"].items():
            if "salient_indices" not in layer_data:
                continue
            module = modules.get(name)
            if module is None:
                continue
            num_channels = _infer_mask_in_features(module, layer_data["salient_indices"])
            if num_channels is None:
                continue
            indices = torch.tensor(layer_data["salient_indices"], dtype=torch.long)
            if indices.numel() == 0 or int(indices.max().item()) >= int(num_channels):
                continue
            mask = torch.zeros(int(num_channels), dtype=torch.bool)
            mask[indices] = True
            masks[name] = mask

        if masks:
            masks_path = export_dir / "channel_masks.pt"
            torch.save(masks, masks_path)
            saved_files["channel_masks"] = masks_path

    return saved_files


__all__ = [
    "collect_extended_linear_dims",
    "collect_salient_indices",
    "collect_kernel_metadata",
    "save_kernel_metadata",
]
