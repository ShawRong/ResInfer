# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Static launch policy and compiler variants for ResAttn decode.'
import os

QMUL4_KERNEL_REDUCTION_MIN_SEQ = 49152
ATOMIC_REPRODUCIBLE_SPLITS = (1, 2)


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
    if sm_count == 148 and residue_fold and grouped_head_tile in (8, 16):
        splits, reducer = _select_residue_decode_plan(
            batch, num_kv_heads, max_kv_len, sm_count, output_nvfp4)
    else:
        splits, reducer = _select_decode_plan(
            batch, num_kv_heads, max_kv_len, sm_count, output_nvfp4=output_nvfp4,
            grouped_head_tile=grouped_head_tile, residue_fold=residue_fold)
    if (policy == "deterministic" and reducer not in ("none", "kernel")
            or policy == "auto" and reducer == "atomic"
            and splits not in ATOMIC_REPRODUCIBLE_SPLITS):
        reducer = "kernel"
    return splits, reducer


def static_compile_plans(*, output_nvfp4=False, grouped_head_tile=16,
                         residue_fold=False):
    """Represent every static compiler variant, not a sampled input grid.

    Batch, KV length and split count are runtime arguments. Only reducer and
    scheduler specialize compilation: fused split4/8 and split16/32 use two
    schedulers. A few extra variants are intentional to cover policy changes.
    """
    plans = [(1, "none")]
    if output_nvfp4 and grouped_head_tile == 8 and not residue_fold:
        return tuple(plans)
    plans.append((2, "kernel"))
    if output_nvfp4:
        plans.extend(((4, "gmem_fused"), (16, "gmem_fused")))
    else:
        plans.append((2, "atomic"))
        if residue_fold and grouped_head_tile in (8, 16):
            plans.append((4, "cga_sharded"))
    return tuple(plans)


def _power_of_two(value):
    return 1 << (value.bit_length() - 1)


def _select_residue_decode_plan(batch, heads, length, sms, packed):
    'B200 residue policy: expose parallel work, then bound reduction cost.'
    grid = max(1, batch * heads)
    tiles = max(1, (length + 255) // 256)
    budget = _power_of_two(max(1, sms // grid))
    splits = min(budget, _power_of_two((tiles + 1) // 2), 64)
    if splits == 1:
        return 1, "none"
    if packed:
        # Four splits can help even at 1K; keep the packed scheduler's split16 cap.
        splits = min(budget, max(4, _power_of_two(max(1, tiles // 4))), 16)
        return splits, "gmem_fused" if splits >= 4 else "kernel"
    cga = min(16, _power_of_two(max(2, sms // (2 * grid))), _power_of_two(tiles))
    if cga >= 2 and (length <= 2048 * cga or splits == 2):
        return cga, "cga_sharded"
    return splits, "kernel"


def _select_decode_plan(batch, num_kv_heads, max_kv_len, sm_count, *,
                        output_nvfp4=False, grouped_head_tile=16,
                        residue_fold=False):
    'Return (splits, reducer); caller supplies the physical padded Q tile.'
    grid = max(1, batch * num_kv_heads)
    tiles = max(1, (max_kv_len + 255) // 256)
    wave = max(1, sm_count // grid)

    if output_nvfp4 and grouped_head_tile == 8:
        if not residue_fold:
            return 1, "none"
        splits = min(wave, tiles // 4, 16)
        if splits == 2 and max_kv_len >= 8192:
            return 2, "kernel"
        if splits < 4:
            return 1, "none"
        return _power_of_two(splits), "gmem_fused"

    if output_nvfp4 and grouped_head_tile == 16:
        if grid == 1:
            if tiles <= 4:
                return 1, "none"
            if tiles <= 8:
                return 4, "gmem_fused"
            if tiles <= 16:
                return 8, "gmem_fused"
            if max_kv_len < QMUL4_KERNEL_REDUCTION_MIN_SEQ:
                return 16, "gmem_fused"
            return 32, "gmem_fused"
        if tiles <= 4:
            return 1, "none"
        splits = _power_of_two(min(wave, max(4, tiles // 4), 16))
        return (1, "none") if splits < 4 else (splits, "gmem_fused")

    if grid == 1 and max_kv_len >= QMUL4_KERNEL_REDUCTION_MIN_SEQ:
        return 32, "kernel"
    if tiles <= 4 or (tiles <= 16 and grid <= 32):
        return 1, "none"
    splits = _power_of_two(min(wave, max(1, tiles // 2), 16))
    return (1, "none") if splits == 1 else (splits, "atomic")
