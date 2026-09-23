"""Verify the pinned FlashInfer TRTLLM NVFP4 decode call on synthetic KV bytes.

This checks the real stock kernel ABI before constructing a performance A/B.
Select one idle physical GPU with CUDA_VISIBLE_DEVICES before running.
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from vllm import envs
from vllm._custom_ops import scaled_fp8_quant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gqa", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=257)
    args = parser.parse_args()
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    batch, kv_heads, dim = args.batch, 8, 128
    heads = kv_heads * args.gqa
    width = math.ceil(args.seq / 128)
    pages = batch * width
    shape = (pages, kv_heads, 128, dim // 2)
    k_data = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    v_data = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    sf = torch.full((pages, kv_heads, 128, dim // 16), .125,
                    dtype=torch.float8_e4m3fn, device=device)
    table = torch.arange(pages, dtype=torch.int32, device=device).reshape(
        batch, width)
    seq_lens = torch.full((batch,), args.seq, dtype=torch.int32, device=device)
    q = torch.randn(batch, heads, dim, dtype=torch.bfloat16, device=device)
    q_fp8, _ = scaled_fp8_quant(q.reshape(batch, heads * dim),
                                scale=torch.ones(1, device=device))
    q_fp8 = q_fp8.reshape_as(q).contiguous()
    out = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=device)
    workspace = torch.zeros(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
                            dtype=torch.uint8, device=device)
    trtllm_batch_decode_with_kv_cache(
        query=q_fp8, kv_cache=(k_data, v_data),
        workspace_buffer=workspace, block_tables=table, seq_lens=seq_lens,
        max_seq_len=args.seq, bmm1_scale=dim ** -.5, bmm2_scale=1.,
        kv_cache_sf=(sf, sf), out=out, backend="trtllm-gen",
        kv_layout="HND", q_len_per_req=1,
    )
    torch.cuda.synchronize()
    values = out.float()
    print(json.dumps({
        "status": "STOCK_NVFP4_DECODE_OK",
        "gqa": args.gqa, "batch": batch, "seq": args.seq,
        "dtype": str(out.dtype), "finite": bool(torch.isfinite(values).all()),
        "mean_abs": float(values.abs().mean()),
    }), flush=True)


if __name__ == "__main__":
    main()
