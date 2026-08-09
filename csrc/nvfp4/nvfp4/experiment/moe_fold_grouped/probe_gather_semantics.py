# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Probe 1: what does `token_id_mapping` actually address?

The docstring says the value is `token_idx * topk + k_idx` and that it is
"used by LDGSTS to gather from A tensor" -- but A has only `seq_len` rows, so
the kernel must decode it somehow. Which decode it uses decides whether we can
point at an arbitrary row of a 2M-row (base, residue) activation, or only at
original tokens.

Rather than keep reading a @cute.jit body, ask the kernel.

Method: give every row of A a distinct constant, make B a single all-ones
column, run with gating off. Then out[j] is proportional to the sum of the row
that row j gathered, so the output reads back the gather index directly.

    python probe_gather_semantics.py
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    try:
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4,
            create_gather_gemm_tensors,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"flashinfer gather GEMM unavailable: {exc}")
        return 2

    torch.manual_seed(0)
    dev = "cuda"

    # Smallest shape that still satisfies the kernel's tile/alignment rules.
    seq_len = 8          # 8 "tokens"
    topk = 1             # keep the encoding as simple as possible first
    num_experts = 2
    k = 128              # hidden
    inter = 128          # intermediate (N of gemm1, per gate/up half)
    mma_tiler_m = 128

    group_m_list = [4, 4]  # 4 rows per expert

    token_map, tile_map, mn_limit, num_tiles, valid_m, aligned_m = (
        create_gather_gemm_tensors(
            seq_len=seq_len, topk=topk, group_m_list=group_m_list,
            mma_tiler_m=mma_tiler_m,
        )
    )
    print(f"  seq_len={seq_len} topk={topk} experts={num_experts} "
          f"k={k} inter={inter}")
    print(f"  valid_m={valid_m}  aligned_m={aligned_m}  num_tiles={num_tiles.tolist()}")
    print(f"  token_id_mapping (helper's own) = {token_map.tolist()}")
    print(f"  tile_idx_to_expert_idx          = {tile_map.tolist()}")
    print(f"  tile_idx_to_mn_limit            = {mn_limit.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
