# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Caller-side glue for the residue graft. OURS, not vendored.'

from __future__ import annotations

import torch


def double_tables_for_residue(
    token_id_mapping: torch.Tensor,   # (permuted_tokens,) int32, token space
    tile_idx_to_mn_limit: torch.Tensor,   # (num_tiles,) int32, token rows
    seq_len: int,
    topk: int,
    row_pair: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Token-space tables (built at tile_tokens_dim = 64) -> physical-space.

    row_pair=False: activation is block-concatenated (base rows [0:seq),
    residue rows [seq:2*seq)) -- odd entries add topk*seq_len so the
    kernel's //topk lands on token+seq (the probe's layout).

    row_pair=True: activation rows interleave (base at 2t, residue at
    2t+1) -- our CUDA quantiser's LAYOUT_ROW_PAIR output. Odd/even entries
    encode (2*token+parity)*topk + choice, so //topk yields 2t / 2t+1
    directly (and %topk still yields the right choice for GEMM2's meta).

    Returns (gemm1_mapping, gemm2_mapping, mn_limit_physical). All device
    ops, capture-safe (no host sync, shapes static in the doubled space).
    """
    doubled = token_id_mapping.repeat_interleave(2)
    if row_pair:
        c = token_id_mapping % topk
        even = 2 * token_id_mapping - c        # (2t)*topk + c
        gemm1 = torch.stack([even, even + topk], dim=1).reshape(-1)
    else:
        gemm1 = doubled.clone()
        gemm1[1::2] += topk * seq_len          # odd rows: //topk -> token+seq
    return gemm1.contiguous(), doubled.contiguous(), tile_idx_to_mn_limit * 2


def sf_unswizzle_indices(rows: int, cols: int,
                         device) -> torch.Tensor:
    """Precomputed gather indices: swizzled 128x4-atom SF bytes -> linear.

    Our CUDA quantiser emits the fold-GEMM's swizzled SF block; the gather
    GEMM1 wants a PLAIN (rows, cols/16) row-major SFA. The swizzle is a
    fixed permutation for a fixed shape, so: precompute once per shape,
    one `gather` per step (capture-safe), zero kernel changes anywhere.

    `rows` = physical activation rows (2*seq_len), `cols` = K.
    """
    m = torch.arange(rows, device=device).view(-1, 1)
    ks = torch.arange(cols // 16, device=device).view(1, -1)
    n_kt = (cols + 63) // 64
    return ((m // 128) * (n_kt * 512) + (ks // 4) * 512
            + (m % 32) * 16 + ((m % 128) // 32) * 4 + (ks % 4)).reshape(-1)
