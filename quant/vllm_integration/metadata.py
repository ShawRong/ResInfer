"""Shared residue sidecar loading and layer-name lookup for NVFP4 and MXFP4.

No tensor/backend imports at module load. Metadata is cached on the owning
quantization config, so separate model loads do not share mutable state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from quant.kernel_metadata import FUSED_PROJECTIONS


def read_residue_metadata_json(metadata_path: str | Path) -> dict:
    """Read one sidecar; propagate file and JSON errors without caching."""
    with open(metadata_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Residue metadata must be an object: {metadata_path}")
    return data


def load_residue_metadata(model_dir: str | Path) -> dict:
    """Load a model's optional sidecar; plain checkpoints may omit it."""
    try:
        return read_residue_metadata_json(Path(model_dir) / "residue_kernel_metadata.json")
    except FileNotFoundError:
        return {}


def get_residue_metadata(quant_config, *, require_local=False) -> dict:
    """Resolve the model directory and read once per quantization config."""
    data = getattr(quant_config, "_residue_metadata", None)
    if data is None:
        model_dir = os.environ.get("RESIDUE_VLLM_MODEL_DIR")
        if not model_dir:
            from vllm.config import get_current_vllm_config_or_none

            cfg = get_current_vllm_config_or_none()
            model_dir = cfg.model_config.model if cfg and cfg.model_config else None
        if not model_dir or not Path(model_dir).is_dir():
            # Plain NVFP4 can use a Hub model through the upstream loader.
            # MXFP4 and explicit sidecar directories retain their local contract.
            if not require_local and not os.environ.get("RESIDUE_VLLM_MODEL_DIR"):
                return {}  # Do not cache an unresolved or non-local model path.
            raise ValueError("Residue metadata requires a local model directory; "
                             "set RESIDUE_VLLM_MODEL_DIR to the checkpoint directory")
        data = load_residue_metadata(model_dir)
        quant_config._residue_metadata = data
    return data


def _layer_aliases(layer_name: str) -> list[str]:
    # Qwen3.5's checkpoint and runtime nest the text trunk differently.
    names = [layer_name]
    for source, target in (("language_model.model.", "model.language_model."),
                           ("model.language_model.", "language_model.model.")):
        if layer_name.startswith(source):
            names.append(target + layer_name[len(source):])
            break
    return names


def get_fused_hf_candidates(layer_name: str) -> list[list[str]]:
    """HF constituent groups for each supported runtime prefix spelling."""
    groups = []
    for candidate in _layer_aliases(layer_name):
        parent, separator, leaf = candidate.rpartition(".")
        if leaf in FUSED_PROJECTIONS:
            groups.append([parent + separator + name for name in FUSED_PROJECTIONS[leaf]])
    return groups


def get_residue_layer_candidates(layer_name: str) -> list[str]:
    """Whole-layer names first, followed by their checkpoint constituents."""
    names = _layer_aliases(layer_name)
    for group in get_fused_hf_candidates(layer_name):
        names.extend(group)
    return list(dict.fromkeys(names))


def get_extended_dim(quant_config, layer_name: str) -> int | None:
    """Look up checkpoint input width, retaining legacy short-name matching."""
    dims = get_residue_metadata(quant_config).get("extended_linear_dims", {})
    for candidate in get_residue_layer_candidates(layer_name):
        if candidate in dims:
            return dims[candidate]
    for key, dim in dims.items():
        if key.endswith(layer_name) or layer_name.endswith(key):
            return dim
    return None


def get_layer_metadata(quant_config, layer_name: str) -> dict | None:
    """Look up a layer record, including aliases and fused constituents."""
    layers = get_residue_metadata(quant_config).get("layers", {})
    for candidate in get_residue_layer_candidates(layer_name):
        meta = layers.get(candidate)
        if meta:
            return meta
    return None
