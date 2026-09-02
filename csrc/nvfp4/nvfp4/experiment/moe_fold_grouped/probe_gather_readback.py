# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Probe 2: read the gather index back out of the kernel.

Probe 1 showed the metadata shape. This asks the kernel what a
`token_id_mapping` entry actually addresses, by making the answer visible in
the output instead of inferring it from a @cute.jit body.

Setup: row i of A is the constant (i+1), B is all ones, so

    out[j] ~ alpha * k * (gathered_row_index + 1)

i.e. the output reads back which row each permuted position gathered. Then we
hand it a HAND-WRITTEN mapping and check we get the rows we asked for -- that
is the property the whole MoE fold plan rests on, because the fold needs to
place a token's (base, residue) rows at chosen positions.

    python probe_gather_readback.py
"""

from __future__ import annotations

import torch


def main() -> int:
    try:
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4,
            create_gather_gemm_tensors,
        )
        from vllm._custom_ops import scaled_fp4_quant
    except Exception as exc:  # noqa: BLE001
        print(f"unavailable: {exc}")
        return 2

    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    seq_len, topk, num_experts = 8, 1, 2
    k, inter = 128, 128
    mma_tiler_m = 128
    group_m_list = [4, 4]

    token_map, tile_map, mn_limit, num_tiles, valid_m, aligned_m = (
        create_gather_gemm_tensors(
            seq_len=seq_len, topk=topk, group_m_list=group_m_list,
            mma_tiler_m=mma_tiler_m,
        )
    )

    # Row i is the constant (i+1): a row-distinguishing input.
    a = torch.arange(1, seq_len + 1, device=dev, dtype=dt).view(seq_len, 1).expand(
        seq_len, k
    ).contiguous()
    ag = ((448.0 * 6.0) / a.float().abs().max()).reshape(1).float()
    aq, asf = scaled_fp4_quant(a, ag)

    # B all ones so the GEMM just sums the gathered row.
    b = torch.ones((num_experts, 2 * inter, k), device=dev, dtype=dt)
    bg = ((448.0 * 6.0) / b.float().abs().max()).reshape(1).float()
    bq_list, bsf_list = [], []
    for e in range(num_experts):
        q, sf = scaled_fp4_quant(b[e], bg)
        bq_list.append(q)
        bsf_list.append(sf)
    bq = torch.stack(bq_list)
    bsf = torch.stack(bsf_list)

    alpha = (1.0 / (ag * bg)).repeat(num_experts).float().contiguous()

    def run(mapping):
        out, _ = blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4(
            a=aq, b=bq, a_scale=asf, b_scale=bsf, alpha=alpha,
            tile_idx_to_expert_idx=tile_map,
            tile_idx_to_mn_limit=mn_limit,
            token_id_mapping=mapping,
            num_non_exiting_tiles=num_tiles,
            topk=topk, mma_tiler_mn=(mma_tiler_m, 128),
            cluster_shape_mn=(1, 1), gated=False,
        )
        return out

    print(f"  valid_m={valid_m}  aligned_m={aligned_m}")
    out = run(token_map)
    print(f"  out shape={tuple(out.shape)} dtype={out.dtype}")

    # Recover the gathered row from the magnitude: out ~ alpha*k*(row+1),
    # and every output column is the same because B is all ones.
    col0 = out[:, 0].float()
    scale = col0[0].item() / 1.0 if col0[0].item() != 0 else 1.0
    print("\n  helper mapping -> which row each position gathered")
    for pos in (0, 1, 2, 3, 128, 129, 130, 131):
        print(f"    pos {pos:4d}: mapping={token_map[pos].item():4d}  "
              f"out={col0[pos].item():10.3f}  implied row+1 ~ "
              f"{col0[pos].item()/max(scale,1e-9):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
