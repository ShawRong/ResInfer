# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Make residue salience compatible with runtime-fused dense linears."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .algo_helper import get_salient_indices


_FUSED_FAMILIES = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("mlp.gate_proj", "mlp.up_proj"),
    # Qwen3.5 GDN names. vLLM materializes these as in_proj_qkvz and
    # in_proj_ba respectively.
    ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
    ("linear_attn.in_proj_b", "linear_attn.in_proj_a"),
)


def _split_family(name: str) -> tuple[str, tuple[str, ...], str] | None:
    for family in _FUSED_FAMILIES:
        for suffix in family:
            marker = "." + suffix
            if name.endswith(marker):
                return name[: -len(marker)], family, suffix
    return None


def align_runtime_fused_salience(
    model: torch.nn.Module,
    *,
    ratio: float,
    block_size: int,
    salience_metric: str,
) -> dict[str, Any]:
    """Give every member of a runtime-fused Linear one group-optimal top-k.

    For ``nvfp4_error_weighted`` each per-projection score estimates output
    error energy.  Squared scores therefore add across output projections;
    ranking by ``sqrt(sum(score**2))`` minimizes the fused projection's joint
    estimate.  For activation-only salience the member scores should be the
    same input statistic, and their arithmetic mean is the stable equivalent.
    """

    modules = dict(model.named_modules())
    grouped: dict[tuple[str, tuple[str, ...]], dict[str, str]] = defaultdict(dict)
    for name, module in modules.items():
        if not isinstance(getattr(module, "_residue_salience_score", None), torch.Tensor):
            continue
        parsed = _split_family(name)
        if parsed is None:
            continue
        prefix, family, suffix = parsed
        grouped[(prefix, family)][suffix] = name

    aligned_groups = 0
    aligned_layers = 0
    details: list[dict[str, Any]] = []
    ratio_key = f"{float(ratio):g}"
    for (prefix, family), suffix_to_name in sorted(grouped.items()):
        if not all(suffix in suffix_to_name for suffix in family):
            continue
        names = [suffix_to_name[suffix] for suffix in family]
        scores = [modules[name]._residue_salience_score for name in names]
        shapes = {tuple(score.shape) for score in scores}
        if len(shapes) != 1:
            raise AssertionError(
                f"Fused salience score shapes differ for {names}: {sorted(shapes)}"
            )
        stacked = torch.stack(scores).float()
        if salience_metric == "nvfp4_error_weighted":
            group_score = stacked.square().sum(dim=0).sqrt()
        else:
            group_score = stacked.mean(dim=0)
        indices_cpu = get_salient_indices(group_score, block_size, ratio).cpu()

        for name in names:
            module = modules[name]
            indices = indices_cpu.to(module.weight.device)
            module.input_quantizer._salient_indices = indices
            sidecar = getattr(module, "_residue_salient_indices_by_ratio", None)
            if sidecar is None:
                sidecar = {}
                module._residue_salient_indices_by_ratio = sidecar
            sidecar[ratio_key] = indices_cpu.clone().to(torch.int64).contiguous()
        aligned_groups += 1
        aligned_layers += len(names)
        details.append(
            {
                "runtime_prefix": prefix,
                "members": names,
                "num_salient": int(indices_cpu.numel()),
            }
        )

    scored_layers = 0
    for module in modules.values():
        if hasattr(module, "_residue_salience_score"):
            scored_layers += 1
            delattr(module, "_residue_salience_score")

    return {
        "scored_layers": scored_layers,
        "aligned_groups": aligned_groups,
        "aligned_layers": aligned_layers,
        "details": details,
    }
