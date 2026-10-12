# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python shape resolution for the SM100 MExt paper sweep.

The physical GEMM shape is ``A[M,K] @ W[N,K].T``.  A shape may have several
model/projection/TP aliases; time it once, then retain every alias for analysis.
"""

from __future__ import annotations

from pathlib import Path

from .json_config import load_model_config


def _text_config(config: dict) -> dict:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("text_config must be a mapping")
    return text


def projection_shapes(config: dict, projections: list[str]) -> dict[str, tuple[int, int]]:
    text = _text_config(config)
    model_type = str(text.get("model_type", config.get("model_type", "")))
    if model_type != "qwen3" and not model_type.startswith("qwen3_5"):
        raise ValueError(f"unsupported model_type {model_type!r}; inspect the checkpoint first")
    h = int(text["hidden_size"])
    q_heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    qk_dim = int(text.get("qk_head_dim", text.get("head_dim", 0)))
    v_dim = int(text.get("v_head_dim", text.get("head_dim", 0)))
    if min(h, q_heads, kv_heads, qk_dim, v_dim) <= 0:
        raise ValueError("attention dimensions must be positive")
    q_out = q_heads * qk_dim
    k_out = kv_heads * qk_dim
    v_out = kv_heads * v_dim
    shapes = {"qkv": (q_out + k_out + v_out, h), "o": (h, q_heads * v_dim)}
    if "gate_up" in projections or "down" in projections:
        inter = int(text["intermediate_size"])
        if inter <= 0:
            raise ValueError("intermediate_size must be positive")
        shapes.update({"gate_up": (2 * inter, h), "down": (h, inter)})
    return {name: shapes[name] for name in projections}


def resolve(suite: dict, overrides: dict[str, Path],
            selected_models: list[str] | None = None,
            tp_overrides: dict[str, list[int]] | None = None) -> tuple[list[dict], dict]:
    model_names = selected_models or list(suite["models"])
    tp_overrides = tp_overrides or {}
    unknown = (set(model_names) - set(suite["models"])) | (set(tp_overrides) - set(model_names))
    if unknown or set(overrides) - set(suite["models"]):
        raise ValueError(f"unknown models: {sorted(unknown | (set(overrides) - set(suite['models'])))}")
    by_shape: dict[tuple[int, int], dict] = {}
    sources = {}
    for model in model_names:
        model_spec = suite["models"][model]
        if model in tp_overrides and not model_spec.get("tp_required"):
            raise ValueError(f"{model} TP is fixed by the suite")
        tp_values = tp_overrides.get(model, model_spec.get("tp_values"))
        if not tp_values:
            raise ValueError(f"{model} needs --model-tp {model}=<deployment TP>")
        if any(type(tp) is not int or tp < 1 for tp in tp_values) or len(set(tp_values)) != len(tp_values):
            raise ValueError(f"{model} TP values must be positive and unique")
        config, source = load_model_config(model, model_spec, overrides.get(model))
        source_hash = source["sha256"]
        text = _text_config(config)
        sources[model] = {**source,
                          "model_type": text.get("model_type", config.get("model_type")),
                          "tp_values": tp_values}
        for projection, (n_full, k_full) in projection_shapes(
            config, model_spec["projections"]
        ).items():
            for tp in tp_values:
                if projection in {"qkv", "gate_up"}:
                    if n_full % tp:
                        raise ValueError(f"{model}/{projection} N={n_full} cannot split TP={tp}")
                    n, k = n_full // tp, k_full
                else:
                    if k_full % tp:
                        raise ValueError(f"{model}/{projection} K={k_full} cannot split TP={tp}")
                    n, k = n_full, k_full // tp
                if n % 32 or k % 64:
                    raise ValueError(f"{model}/{projection}/tp{tp} yields unaligned N={n},K={k}")
                item = by_shape.setdefault((n, k), {"n": n, "k": k, "aliases": []})
                item["aliases"].append({"model": model, "projection": projection,
                                        "tp": tp, "config_sha256": source_hash})
    shapes = sorted(by_shape.values(), key=lambda item: (item["n"], item["k"]))
    for item in shapes:
        item["shape_id"] = f"n{item['n']}_k{item['k']}"
        item["aliases"].sort(key=lambda a: (a["model"], a["projection"], a["tp"]))
    return shapes, sources
