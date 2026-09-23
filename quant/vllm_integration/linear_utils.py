"""Utility helpers for residue vLLM linear methods."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn

_RESIDUE_METADATA_CACHE: dict[str, dict] = {}


class _InputQuantizerState(nn.Module):
    """Placeholder module to receive input_quantizer.* checkpoint tensors."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_salient_indices", torch.empty(0, dtype=torch.int64))


def get_residue_layer_candidates(layer_name: str) -> list[str]:
    """Return metadata lookup candidates for possibly merged vLLM layers."""

    candidates = [layer_name]
    # Qwen3.5's outer HF checkpoint and vLLM runtime use opposite nesting
    # orders for the dense text trunk. Metadata is keyed by the HF spelling.
    if layer_name.startswith("language_model.model."):
        candidates.append(
            "model.language_model." + layer_name[len("language_model.model.") :]
        )
    elif layer_name.startswith("model.language_model."):
        candidates.append(
            "language_model.model." + layer_name[len("model.language_model.") :]
        )
    vllm_to_hf_suffixes = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        # Qwen3.5 stores the GDN projections separately. vLLM maps the
        # already-fused qkv tensor to shards 0..2 and z to shard 3 of one
        # in_proj_qkvz module, so its shared input dimension and residue
        # metadata must be recovered from the checkpoint-side names.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    for candidate in list(candidates):
        for vllm_suffix, hf_suffixes in vllm_to_hf_suffixes.items():
            if candidate.endswith(vllm_suffix):
                base_name = candidate[: -len(vllm_suffix)]
                candidates.extend(base_name + hf_suffix for hf_suffix in hf_suffixes)

    deduped: list[str] = []
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def load_residue_metadata_json(metadata_path: str | Path) -> dict:
    """Load residue_kernel_metadata.json once per process."""

    cache_key = str(metadata_path)
    cached = _RESIDUE_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        with open(metadata_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"[residue] Warning: Failed to load metadata: {e}")
        data = {}

    _RESIDUE_METADATA_CACHE[cache_key] = data
    return data


def get_extended_dims_dict(quant_config) -> dict | None:
    """Get extended_linear_dims from quant config or model-dir metadata."""

    if hasattr(quant_config, "_extended_linear_dims"):
        if quant_config._extended_linear_dims is None:
            if hasattr(quant_config, "_load_metadata"):
                quant_config._load_metadata()
        if quant_config._extended_linear_dims:
            return quant_config._extended_linear_dims

    model_dir = os.environ.get("RESIDUE_VLLM_MODEL_DIR")
    if model_dir:
        metadata_path = Path(model_dir) / "residue_kernel_metadata.json"
        data = load_residue_metadata_json(metadata_path)
        if data:
            return data.get("extended_linear_dims", {})

    return None


def get_extended_dim(quant_config, layer_name: str) -> int | None:
    """Look up an extended input dimension for a layer from metadata."""

    dims = get_extended_dims_dict(quant_config)
    if not dims:
        return None

    if layer_name in dims:
        return dims[layer_name]

    for candidate in get_residue_layer_candidates(layer_name):
        if candidate in dims:
            return dims[candidate]

    for key, dim in dims.items():
        if key.endswith(layer_name) or layer_name.endswith(key):
            return dim

    return None


def get_layer_metadata(quant_config, layer_name: str) -> dict | None:
    """Get residue metadata for a layer, including merged-layer fallbacks."""

    if hasattr(quant_config, "get_layer_metadata"):
        for candidate in get_residue_layer_candidates(layer_name):
            meta = quant_config.get_layer_metadata(candidate)
            if meta:
                return meta
        return None

    model_dir = os.environ.get("RESIDUE_VLLM_MODEL_DIR")
    if not model_dir:
        return None

    metadata_path = Path(model_dir) / "residue_kernel_metadata.json"
    data = load_residue_metadata_json(metadata_path)
    layers = data.get("layers", {}) if data else {}

    for candidate in get_residue_layer_candidates(layer_name):
        meta = layers.get(candidate)
        if meta:
            return meta

    return None


def get_fused_hf_candidates(layer_name: str) -> list[list[str]]:
    """HF constituent groups for each supported runtime prefix spelling."""
    suffixes = {
        "qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "gate_up_proj": ("gate_proj", "up_proj"),
        "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
        "in_proj_ba": ("in_proj_b", "in_proj_a"),
    }
    result = []
    for candidate in get_residue_layer_candidates(layer_name):
        parent, _, leaf = candidate.rpartition(".")
        if leaf in suffixes:
            result.append([(parent + "." if parent else "") + name for name in suffixes[leaf]])
    return result
