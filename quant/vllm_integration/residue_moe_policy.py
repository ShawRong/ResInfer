# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Per-MoE-layer residue policy: the serving half of the paired design.'

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .linear_utils import load_residue_metadata_json

MOE_IMPL_OFF = "off"
MOE_IMPL_MEXT_R1 = "mext_r1"
MOE_IMPL_EXTENDED_K = "extended_k"  # schema-reserved; no serving path yet

_KNOWN_IMPLS = {MOE_IMPL_OFF, MOE_IMPL_MEXT_R1, MOE_IMPL_EXTENDED_K}

# impl -> the ratio it IS (kernel_metadata.is_mext_r1_layer: mext_r1 is not
# a mode competing with the ratios, it IS ratio 1.0).
_IMPL_RATIO = {MOE_IMPL_OFF: 0.0, MOE_IMPL_MEXT_R1: 1.0}


@dataclass(frozen=True)
class MoELayerPolicy:
    impl: str
    ratio: float


#: No-metadata default == today's behavior (all layers on, full residue).
DEFAULT_MOE_POLICY = MoELayerPolicy(MOE_IMPL_MEXT_R1, 1.0)


def load_moe_policy_section(model_dir: str | None = None) -> dict | None:
    """Return the metadata's ``moe`` section, or None (= no policy).

    Reads through linear_utils' per-process JSON cache, so per-layer method
    instances share one parse.
    """
    if model_dir is None:
        model_dir = os.environ.get("RESIDUE_VLLM_MODEL_DIR")
    if not model_dir:
        return None
    data = load_residue_metadata_json(
        Path(model_dir) / "residue_kernel_metadata.json")
    section = data.get("moe") if data else None
    if section is None:
        return None
    granularity = section.get("granularity", "layer")
    if granularity != "layer":
        raise ValueError(
            f"MoE policy granularity {granularity!r} is not servable: the "
            "grouped path serves one layer per call over a shared activation "
            "buffer (per-expert policies need a permute-based path).")
    return section


def _match_entry(layers: dict, layer_name: str) -> tuple[str, dict] | None:
    """Match a vLLM layer name against export-side module paths.

    Export keys the HF MoE module ("model.layers.0.mlp"); vLLM's
    RoutedExperts registers a child of it ("model.layers.0.mlp.experts").
    Exact match wins; otherwise accept a unique dotted prefix/suffix
    relation in either direction. Ambiguity raises — a policy that two keys
    both claim is a broken policy, not a tie to break silently.
    """
    if layer_name in layers:
        return layer_name, layers[layer_name]
    hits = [
        (key, entry)
        for key, entry in layers.items()
        if layer_name.startswith(key + ".") or key.startswith(layer_name + ".")
    ]
    if not hits:
        return None
    if len(hits) > 1:
        keys = sorted(k for k, _ in hits)
        raise ValueError(
            f"MoE policy is ambiguous for layer {layer_name!r}: "
            f"matched {keys}.")
    return hits[0]


def resolve_moe_layer_policy(
    section: dict | None,
    layer_name: str,
) -> MoELayerPolicy:
    """Resolve one layer's policy. See module docstring for the rules."""
    if section is None:
        return DEFAULT_MOE_POLICY

    if not layer_name:
        raise ValueError(
            "A MoE policy is present but this layer has no layer_name to "
            "resolve it with; refusing to guess.")

    layers = section.get("layers", {})
    hit = _match_entry(layers, layer_name)
    if hit is None:
        raise KeyError(
            f"MoE policy names {len(layers)} layers but not {layer_name!r}. "
            "A partial policy is ambiguous; every MoE layer must have an "
            "entry (impl=mext_r1 to keep it on, impl=off to turn it off).")
    key, entry = hit

    impl = entry.get("impl")
    if impl not in _KNOWN_IMPLS:
        raise ValueError(
            f"MoE policy {key!r} has unknown impl {impl!r}; "
            f"known: {sorted(_KNOWN_IMPLS)}.")
    if impl == MOE_IMPL_EXTENDED_K:
        raise NotImplementedError(
            f"MoE policy {key!r} demands impl=extended_k, which this server "
            "cannot serve yet (PLAN_moe_per_layer_ratio.md W1-W3). Refusing "
            "to fall back — serving stock while the export was built for "
            "K-extension is the silent mismatch this contract exists to "
            "prevent.")

    ratio = entry.get("ratio")
    expected = _IMPL_RATIO[impl]
    if ratio is not None and abs(float(ratio) - expected) > 1e-6:
        raise ValueError(
            f"MoE policy {key!r}: impl={impl!r} IS ratio {expected}, but the "
            f"entry says ratio={ratio}. The file is inconsistent; fix the "
            "writer instead of trusting either field.")

    return MoELayerPolicy(impl, expected)


__all__ = [
    "MOE_IMPL_OFF",
    "MOE_IMPL_MEXT_R1",
    "MOE_IMPL_EXTENDED_K",
    "MoELayerPolicy",
    "DEFAULT_MOE_POLICY",
    "load_moe_policy_section",
    "resolve_moe_layer_policy",
]
