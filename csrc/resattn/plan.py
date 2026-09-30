# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Static launch policy and compiler variants for ResAttn decode.'
import os


def _reduction_policy(value=None):
    value = os.environ.get("RESIDUE_ATTN_REDUCTION", "legacy") if value is None else value
    if value not in ("legacy", "auto", "deterministic"):
        raise ValueError(f"unknown ResAttn reduction policy: {value}")
    return value


def select_decode_plan(batch, num_kv_heads, max_kv_len, sm_count, *,
                       output_nvfp4=False, grouped_head_tile=16,
                       residue_fold=False, reduction_policy=None):
    """Select the static split/reducer, then apply the caller's reduction policy."""
    policy = _reduction_policy(reduction_policy)
    fused = grouped_head_tile == 16 or (grouped_head_tile == 8 and residue_fold)
    cga = sm_count == 148 and residue_fold and grouped_head_tile in (8, 16)
    splits, reducer = _select_decode_plan(
        batch, num_kv_heads, max_kv_len, sm_count, output_nvfp4,
        fused=fused, cga=cga, fused_split_cap=16 if residue_fold else 32)
    if policy == "deterministic" and reducer not in ("none", "kernel"):
        reducer = "kernel"
    return splits, reducer


def static_compile_plans(*, output_nvfp4=False, grouped_head_tile=16,
                         residue_fold=False):
    """Represent every static compiler variant, not a sampled input grid.

    Batch, KV length and split count are runtime arguments. Only reducer and
    scheduler specialize compilation: fused split4/8 and split16/32 use two
    schedulers. A few extra variants are intentional to cover policy changes.
    """
    plans = [(1, "none"), (2, "kernel")]
    if output_nvfp4 and (grouped_head_tile == 16 or
                         grouped_head_tile == 8 and residue_fold):
        plans.extend(((4, "gmem_fused"), (16, "gmem_fused")))
    elif not output_nvfp4 and residue_fold and grouped_head_tile in (8, 16):
        plans.append((4, "cga_sharded"))
    return tuple(plans)


def _power_of_two(value):
    return 1 << (value.bit_length() - 1)


def _select_decode_plan(batch, heads, length, sms, packed, *, fused, cga, fused_split_cap):
    'Expose parallel work, then bound reduction cost.'
    grid = max(1, batch * heads)
    tiles = max(1, (length + 255) // 256)
    budget = _power_of_two(max(1, sms // grid))
    splits = min(budget, _power_of_two((tiles + 1) // 2), 64)
    if splits == 1:
        return 1, "none"
    if packed and fused:
        # More than 16 fused splits need at least eight KV tiles per CTA.
        cap = min(fused_split_cap, max(16, _power_of_two(max(1, tiles // 8))))
        splits = min(budget, max(4, _power_of_two(max(1, tiles // 4))), cap)
        return splits, "gmem_fused" if splits >= 4 else "kernel"
    if packed or not cga:
        return splits, "kernel"
    cga = min(16, _power_of_two(max(2, sms // (2 * grid))), _power_of_two(tiles))
    if cga >= 2 and (length <= 2048 * cga or splits == 2):
        return cga, "cga_sharded"
    return splits, "kernel"
