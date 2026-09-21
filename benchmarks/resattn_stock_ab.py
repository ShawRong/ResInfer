"""Reproducible stock TRT-LLM versus ResAttn decode serving-chain probe.

Both arms read identical HND NVFP4 KV bytes and start with the same BF16 Q.
Both end with vLLM-layout packed NVFP4 data and block scales. This is a
synthetic decode-chain experiment, not an engine throughput result.

Select an idle GPU with CUDA_VISIBLE_DEVICES and provide a checksum-verified
FLASHINFER_CUBIN_DIR before running. Results include raw interleaved samples
and a stock/stock A/A control. No base-only ResAttn arm is used.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import statistics

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from nvfp4.cute_attn.append import quantize_paged_trtllm_format
from nvfp4.cute_attn import host
from nvfp4.cute_attn.kv_layout import NvFp4PagedKVPool
from vllm import envs
from vllm._custom_ops import scaled_fp8_quant


def unpack_nvfp4(data: torch.Tensor, scales: torch.Tensor, heads: int,
                 dim: int) -> torch.Tensor:
    """Decode the vLLM/CUTLASS 128x4 scale layout for validation only."""
    batch = data.shape[0]
    packed = data.reshape(batch, heads, dim // 2).to(torch.int32)
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(
        batch, heads, dim)
    lut = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device=data.device)
    values = lut[nibbles & 7] * torch.where((nibbles & 8) == 0, 1., -1.)
    blocks = heads * dim // 16
    k_idx = torch.arange(blocks, device=data.device)
    rows = torch.arange(batch, device=data.device)
    sf_offsets = (
        (rows[:, None] // 128 * math.ceil(blocks / 4)
         + k_idx[None, :] // 4) * 512
        + rows[:, None] % 32 * 16
        + (rows[:, None] // 32) % 4 * 4
        + k_idx[None, :] % 4
    )
    sf = scales.reshape(-1)[sf_offsets].float().reshape(batch, heads, dim // 16)
    return (values.reshape(batch, heads, dim // 16, 16) * sf[..., None]).reshape(
        batch, heads, dim)


def capture(call, warmups: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def sample(graph: torch.cuda.CUDAGraph, iterations: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def run(args: argparse.Namespace) -> dict:
    if args.batch < 1 or args.seq < 1 or args.iterations < 1 or args.repeats < 2:
        raise ValueError("batch, seq, iterations and repeats must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    batch, kv_heads, dim = args.batch, 8, 128
    heads = kv_heads * args.gqa
    width = math.ceil(args.seq / 128)
    pages = batch * width
    block_tables = torch.arange(pages, dtype=torch.int32, device=device).reshape(
        batch, width)
    kv_indices = block_tables.reshape(-1)
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * width
    seq_lens = torch.full((batch,), args.seq, dtype=torch.int32, device=device)
    k = torch.randn(batch, args.seq, kv_heads, dim,
                    dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    k_data, k_sf, v_data, v_sf, k_deq, v_deq = (
        quantize_paged_trtllm_format(
            k, v, seq_lens, kv_indices, kv_indptr, pages))
    del k, v, k_deq, v_deq
    q = torch.randn(batch, heads, dim, dtype=torch.bfloat16, device=device)
    pool = NvFp4PagedKVPool(
        num_kv_heads=kv_heads, head_dim=dim, k_data=k_data, v_data=v_data,
        k_sf=k_sf, v_sf=v_sf,
        v_staging=torch.empty(0, dtype=torch.bfloat16, device=device),
        page_size=128, layout_profile="trtllm", k_global_scale=1.,
        v_global_scale=1.)
    output_scale_inv = torch.ones(1, dtype=torch.float32, device=device)
    q_scale = torch.ones(1, dtype=torch.float32, device=device)
    auto_plan = host.plan_qmul4_decode_nvfp4(
        batch, heads, kv_heads, args.seq, device, residue_fold=True)
    plan = auto_plan if args.plan == "auto" else (
        int(args.plan.split("/")[0]), args.plan.split("/")[1])
    physical_gqa = 8 if args.gqa == 4 else 16
    work = None
    if plan[1] == "gmem_fused":
        work = host.allocate_qmul4_gmem_reduction_workspace(
            plan[0], batch, kv_heads * physical_gqa, dim, device)
    elif plan[1] == "kernel":
        work = host.allocate_qmul4_reduction_workspace(
            plan[0], batch, kv_heads * physical_gqa, dim, device)
    elif plan != (1, "none"):
        raise RuntimeError(f"unsupported direct output plan: {plan}")

    stock_outputs = [host.allocate_nvfp4_attention_output(
        batch, heads, dim, device) for _ in range(2)]
    res_output = host.allocate_nvfp4_attention_output(batch, heads, dim, device)
    fp8_q = [torch.empty_like(q, dtype=torch.float8_e4m3fn) for _ in range(2)]
    fp8_o = [torch.empty_like(q, dtype=torch.float8_e4m3fn) for _ in range(2)]
    bf16_o = [torch.empty_like(q) for _ in range(2)]
    stock_work = [torch.zeros(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
                              dtype=torch.uint8, device=device) for _ in range(2)]

    def stock(arm: int) -> None:
        scaled_fp8_quant(q.reshape(batch, heads * dim), scale=q_scale,
                         output=fp8_q[arm].reshape(batch, heads * dim),
                         group_shape=(-1, -1))
        trtllm_batch_decode_with_kv_cache(
            query=fp8_q[arm], kv_cache=(k_data, v_data),
            workspace_buffer=stock_work[arm], block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=args.seq, bmm1_scale=dim ** -.5,
            bmm2_scale=1., kv_cache_sf=(k_sf, v_sf), out=fp8_o[arm],
            backend="trtllm-gen", kv_layout="HND", q_len_per_req=1)
        bf16_o[arm].copy_(fp8_o[arm])
        data, scales = stock_outputs[arm]
        torch.ops._C.scaled_fp4_quant.out(
            bf16_o[arm].reshape(batch, heads * dim), output_scale_inv, True,
            output=data, output_scale=scales.view(torch.int32))

    def resattn() -> None:
        host.run_paged_fp4_decode_qmul4_tmem_nvfp4(
            q, pool, v_data, v_sf, kv_indices, kv_indptr, seq_lens,
            dim ** -.5, output_scale_inv, out=res_output[0],
            output_block_scale=res_output[1], max_seq_len=args.seq,
            k_sf_mode="linear", value_scale=1., reduction_workspace=work,
            plan=plan, residue_fold=True)

    calls = {"stock0": lambda: stock(0), "resattn": resattn,
             "stock1": lambda: stock(1)}
    for call in calls.values():
        call()
    torch.cuda.synchronize()
    stock_value = unpack_nvfp4(*stock_outputs[0], heads, dim)
    res_value = unpack_nvfp4(*res_output, heads, dim)
    cos = torch.nn.functional.cosine_similarity(
        stock_value.float(), res_value.float(), dim=-1)
    min_cos = float(cos.min())
    median_cos = float(cos.median())
    rms_error = float((stock_value.float() - res_value.float()).square().mean().sqrt())
    stock_rms = float(stock_value.float().square().mean().sqrt())
    if not bool(torch.isfinite(res_value).all()) or min_cos < args.min_cosine:
        raise RuntimeError(
            f"cross-arm correctness failed: min_cosine={min_cos} "
            f"median_cosine={median_cos} rms_error={rms_error} "
            f"stock_rms={stock_rms}")
    if not (torch.equal(stock_outputs[0][0], stock_outputs[1][0]) and
            torch.equal(stock_outputs[0][1], stock_outputs[1][1])):
        raise RuntimeError("stock A/A outputs differ")
    graphs = {name: capture(call, args.warmups) for name, call in calls.items()}
    raw = {name: [] for name in calls}
    forward = tuple(calls)
    for repeat in range(args.repeats):
        order = forward if repeat % 2 == 0 else tuple(reversed(forward))
        for name in order:
            raw[name].append(sample(graphs[name], args.iterations))
    medians = {name: statistics.median(values) for name, values in raw.items()}
    return {
        "status": "SYNTHETIC_CHAIN_OK", "gqa": args.gqa,
        "batch": batch, "seq": args.seq, "kv_heads": kv_heads, "dim": dim,
        "seed": args.seed, "plan": list(plan), "auto_plan": list(auto_plan),
        "min_cross_cosine": min_cos,
        "median_cross_cosine": median_cos, "rms_error": rms_error,
        "stock_rms": stock_rms,
        "iterations": args.iterations, "repeats": args.repeats,
        "raw_us": raw, "median_us": medians,
        "stock_over_resattn": medians["stock0"] / medians["resattn"],
        "stock_aa_ratio": medians["stock0"] / medians["stock1"],
        "scope": "BF16 Q to packed NVFP4 attention output, synthetic KV",
        "runtime": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "vllm": importlib.metadata.version("vllm"),
            "flashinfer_python": importlib.metadata.version("flashinfer-python"),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "flashinfer_cubin_dir": os.getenv("FLASHINFER_CUBIN_DIR"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gqa", type=int, choices=(4, 6), required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--min-cosine", type=float, default=.97)
    ap.add_argument("--plan", choices=("auto", "1/none", "4/gmem_fused",
                                         "8/gmem_fused", "16/gmem_fused"),
                    default="auto")
    args = ap.parse_args()
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
