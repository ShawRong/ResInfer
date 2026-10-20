# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Validate selective NVFP4 weight extension against the model config."""

import json
from pathlib import Path


def _expected_in_features(name: str, cfg: dict) -> int | None:
    """Logical projection input width, independently read from model config."""
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


def validate_nvfp4_widths(export_dir: Path, layers: dict) -> int:
    """Check NVFP4 residue width against model config; ratio1 keeps base weights."""
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
        in_features = _expected_in_features(name, cfg)
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
