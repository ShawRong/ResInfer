# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Deterministic launch heuristics for the conventional-V decode kernel.'
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
