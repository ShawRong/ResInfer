# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Legal tuning candidates and fallback policy for ResAttn decode.'
import os

QMUL4_KERNEL_REDUCTION_MIN_SEQ = 49152
ATOMIC_REPRODUCIBLE_SPLITS = (1, 2)


def _split_reduction(splits):
    mode = os.environ.get("RESIDUE_ATTN_REDUCTION", "legacy")
    if mode == "deterministic":
        return "kernel"
    if mode == "auto":
        return "atomic" if splits in ATOMIC_REPRODUCIBLE_SPLITS else "kernel"
    return "atomic"


def _power_of_two(value):
    return 1 << (value.bit_length() - 1)


def select_decode_plan(batch, num_kv_heads, max_kv_len, sm_count, *,
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
    return (1, "none") if splits == 1 else (splits, _split_reduction(splits))


def decode_plan_candidates(max_kv_len, *, output_nvfp4=False,
                           grouped_head_tile=16, residue_fold=False,
                           reduction_policy=None):
    """Bounded split/reducer search space for the production Q8/Q16 path.

    These are capability constraints, not a speed ranking. Each split must
    have at least one 256-token tile at the supplied maximum length; shorter
    requests in the batch may have empty splits, handled by the kernel.
    Cluster atomics are limited to portable cluster sizes (up to eight CTAs).
    Experimental CGA reducers and split64 are outside this tuning scope.
    """
    if max_kv_len < 1 or grouped_head_tile not in (8, 16):
        raise ValueError("decode tuning requires positive length and Q8/Q16")
    policy = (os.environ.get("RESIDUE_ATTN_REDUCTION", "legacy")
              if reduction_policy is None else reduction_policy)
    if policy not in ("legacy", "auto", "deterministic"):
        raise ValueError(f"unknown ResAttn reduction policy: {policy}")
    plans = [(1, "none")]
    if output_nvfp4 and grouped_head_tile == 8 and not residue_fold:
        return tuple(plans)
    tiles = (max_kv_len + 255) // 256
    for splits in (2, 4, 8, 16, 32):
        if splits > tiles:
            break
        plans.append((splits, "kernel"))
        if output_nvfp4:
            if splits >= 4 and policy != "deterministic":
                plans.append((splits, "gmem_fused"))
        elif (splits <= 8 and policy != "deterministic"
              and (policy != "auto" or splits in ATOMIC_REPRODUCIBLE_SPLITS)):
            plans.append((splits, "atomic"))
    return tuple(plans)
