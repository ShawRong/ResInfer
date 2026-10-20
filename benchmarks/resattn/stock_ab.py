"""Reproducible stock TRT-LLM versus ResAttn decode serving-chain probe.

Both arms read identical HND NVFP4 KV bytes and start with the same BF16 Q.
The output contract is selected by --output-contract: BF16 matches the
current real-model E2E path; NVFP4 measures the prospective fused path.
This is a synthetic decode-chain experiment, not an engine throughput result.

Select an idle GPU with CUDA_VISIBLE_DEVICES and provide a checksum-verified
FLASHINFER_CUBIN_DIR before running. Results include raw interleaved samples
and a stock/stock A/A control.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
from pathlib import Path

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from benchmarks.resattn.chain_reference import make_kv, attention_reference
from nvfp4.cute_attn import host as production_host
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


def capture(call, warmups: int, repetitions: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            call()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def sample(graph: torch.cuda.CUDAGraph, iterations: int, repetitions: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (iterations * repetitions)


def reference_ok(metrics, min_cosine, max_relative_rms):
    """Both arms must preserve direction and magnitude against the reference."""
    return (all(math.isfinite(value) for value in metrics.values())
            and metrics["min_cosine"] >= min_cosine
            and metrics["relative_rms"] <= max_relative_rms)


def run(args: argparse.Namespace) -> dict:
    host = production_host
    value_scale = getattr(args, "value_scale", 1.)
    if min(args.batch, args.seq, args.iterations, args.graph_repetitions) < 1:
        raise ValueError("batch, seq and iterations must be positive")
    if args.repeats < 2 or args.warmups < 0:
        raise ValueError("repeats must be at least 2 and warmups nonnegative")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    batch, kv_heads, dim = args.batch, args.kv_heads, getattr(args, "head_dim", 128)
    heads = kv_heads * args.gqa
    width = math.ceil(args.seq / 128)
    pages = batch * width
    # TRT-LLM derives page-table row stride from its width; both arms use
    # contiguous rows. Strided ResAttn tables are tested separately.
    block_tables = torch.empty((batch, width), dtype=torch.int32, device=device)
    block_tables.copy_(torch.randperm(pages, device=device).reshape(batch, width))
    seq_lens = torch.full((batch,), args.seq, dtype=torch.int32, device=device)
    k_data, k_sf, v_data, v_sf = make_kv(pages, kv_heads, device, dim)
    if getattr(args, "vllm_kv_layout", False):
        from vllm.utils.torch_utils import nvfp4_split_data_scale
        cache = torch.empty((pages, 2 * kv_heads, 128, dim // 2 + dim // 16),
                            dtype=torch.uint8, device=device)
        k_side, v_side = cache.split(kv_heads, dim=1)
        kd, ks = nvfp4_split_data_scale(k_side)
        vd, vs = nvfp4_split_data_scale(v_side)
        for dst, src in zip((kd, ks, vd, vs), (k_data, k_sf, v_data, v_sf)):
            dst.copy_(src)
        k_data, k_sf, v_data, v_sf = kd, ks, vd, vs
    q = torch.randn(batch, heads, dim, dtype=torch.bfloat16, device=device)
    pool = NvFp4PagedKVPool(
        num_kv_heads=kv_heads, head_dim=dim, k_data=k_data, v_data=v_data,
        k_sf=k_sf, v_sf=v_sf,
        page_size=128, k_global_scale=1.,
        v_global_scale=value_scale)
    output_scale_inv = torch.full((1,), args.output_scale_inv, dtype=torch.float32, device=device)
    q_scale = torch.ones(1, dtype=torch.float32, device=device)
    nvfp4_output = args.output_contract == "nvfp4"
    physical_gqa = host._residue_padded_group_size(heads, kv_heads)
    plan = (
        host.plan_qmul4_decode_nvfp4(
            batch, heads, kv_heads, args.seq, device, residue_fold=True)
        if nvfp4_output else
        host.plan_qmul4_decode(
            batch, kv_heads, args.seq, device,
            grouped_head_tile=physical_gqa, residue_fold=True,
            **({"head_dim": dim} if dim != 128 else {}))
    )
    def workspace(chosen):
        if chosen[1] == "gmem_fused":
            return host.allocate_qmul4_gmem_reduction_workspace(
                chosen[0], batch, kv_heads * physical_gqa, dim, device)
        if chosen[1] == "kernel":
            return host.allocate_qmul4_reduction_workspace(
                chosen[0], batch, kv_heads * physical_gqa, dim, device)
        if chosen == (1, "none") or (not nvfp4_output and chosen[1] in ("atomic", "cga_sharded")):
            return None
        raise RuntimeError(f"unsupported direct output plan: {chosen}")
    work = workspace(plan)

    stock_output = (host.allocate_nvfp4_attention_output(
        batch, heads, dim, device) if nvfp4_output else None)
    res_output = (host.allocate_nvfp4_attention_output(batch, heads, dim, device)
                  if nvfp4_output else torch.empty_like(q))
    fp8_q = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    fp8_o = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    bf16_o = torch.empty_like(q)
    stock_work = torch.zeros(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
                             dtype=torch.uint8, device=device)

    def stock() -> None:
        scaled_fp8_quant(q.reshape(batch, heads * dim), scale=q_scale,
                         output=fp8_q.reshape(batch, heads * dim),
                         group_shape=(-1, -1))
        trtllm_batch_decode_with_kv_cache(
            query=fp8_q, kv_cache=(k_data, v_data),
            workspace_buffer=stock_work, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=args.seq, bmm1_scale=dim ** -.5,
            bmm2_scale=value_scale, kv_cache_sf=(k_sf, v_sf), out=fp8_o,
            backend="trtllm-gen", kv_layout="HND", q_len_per_req=1)
        bf16_o.copy_(fp8_o)
        if nvfp4_output:
            data, scales = stock_output
            torch.ops._C.scaled_fp4_quant.out(
                bf16_o.reshape(batch, heads * dim), output_scale_inv, True,
                output=data, output_scale=scales.view(torch.int32))

    def resattn() -> None:
        if nvfp4_output:
            host.run_paged_fp4_decode_qmul4_tmem_nvfp4(
                q, pool, v_data, v_sf, block_tables, seq_lens,
                dim ** -.5, output_scale_inv, out=res_output[0],
                output_block_scale=res_output[1], max_seq_len=args.seq,
                k_sf_mode="linear", value_scale=value_scale, reduction_workspace=work,
                plan=plan, residue_fold=True)
        else:
            host.run_paged_fp4_decode_qmul4_tmem(
                q, pool, v_data, v_sf, block_tables, seq_lens,
                dim ** -.5, out=res_output, max_seq_len=args.seq,
                k_sf_mode="linear", value_scale=value_scale, reduction_workspace=work,
                plan=plan, residue_fold=True, direct_output=True)

    stock()
    stock_snapshot = ((stock_output[0].clone(), stock_output[1].clone())
                      if nvfp4_output else bf16_o.clone())
    resattn()
    stock()
    torch.cuda.synchronize()
    stock_value = (unpack_nvfp4(*stock_output, heads, dim) / args.output_scale_inv
                   if nvfp4_output else bf16_o.float())
    res_value = (unpack_nvfp4(*res_output, heads, dim) / args.output_scale_inv
                 if nvfp4_output else res_output.float())
    reference = attention_reference(q, k_data, k_sf, v_data, v_sf,
                                    block_tables, seq_lens) * value_scale
    reference_rms = float(reference.square().mean().sqrt())
    diagnostics = {}
    for name, value in (("stock", stock_value), ("resattn", res_value)):
        cosine = torch.nn.functional.cosine_similarity(value, reference, dim=-1)
        rms_error = float((value - reference).square().mean().sqrt())
        metrics = dict(min_cosine=float(cosine.min()),
                       median_cosine=float(cosine.median()),
                       relative_rms=rms_error / reference_rms
                       if reference_rms > 0 else float("inf"))
        diagnostics[name] = metrics
        if not bool(torch.isfinite(value).all()) or not reference_ok(
                metrics, args.reference_min_cosine, args.reference_max_relative_rms):
            raise RuntimeError(f"{name} failed independent reference: {metrics}")
    if ((nvfp4_output and not (
            torch.equal(stock_snapshot[0], stock_output[0]) and
            torch.equal(stock_snapshot[1], stock_output[1]))) or
            (not nvfp4_output and not torch.equal(stock_snapshot, bf16_o))):
        raise RuntimeError("stock A/A outputs differ")
    stock_graph = capture(stock, args.warmups, args.graph_repetitions)
    resattn_graph = capture(resattn, args.warmups, args.graph_repetitions)
    # A/A reuses the same graph and allocations, exposing timing drift.
    graphs = {"stock0": stock_graph, "resattn": resattn_graph, "stock1": stock_graph}
    raw = {name: [] for name in graphs}
    forward = tuple(graphs)
    for repeat in range(args.repeats):
        order = forward if repeat % 2 == 0 else tuple(reversed(forward))
        for name in order:
            raw[name].append(sample(graphs[name], args.iterations, args.graph_repetitions))
    medians = {name: statistics.median(values) for name, values in raw.items()}
    return {
        "status": ("SYNTHETIC_CHAIN_OK" if
                   abs(medians["stock0"] / medians["stock1"] - 1) <= .02 else "NOISY_AA"),
        "gqa": args.gqa, "value_scale": value_scale,
        "cold_l2_cache": False, "residue_fold": True,
        "vllm_kv_layout": getattr(args, "vllm_kv_layout", False),
        "block_table_stride": list(block_tables.stride()),
        "kv_strides": {"data": list(k_data.stride()), "scale": list(k_sf.stride())},
        "kv_generation": "chunked Gaussian BF16, independently quantized K/V, permuted physical pages",
        "output_contract": args.output_contract, "output_scale_inv": args.output_scale_inv,
        "batch": batch, "seq": args.seq, "kv_heads": kv_heads, "dim": dim,
        "seed": args.seed, "plan": list(plan),
        "reference_diagnostic": diagnostics,
        "reference_rms": reference_rms,
        "correctness_gate": dict(min_cosine=args.reference_min_cosine,
                                 max_relative_rms=args.reference_max_relative_rms),
        "iterations": args.iterations, "repeats": args.repeats,
        "graph_repetitions": args.graph_repetitions,
        "raw_us": raw, "median_us": medians,
        "stock_over_resattn": medians["stock0"] / medians["resattn"],
        "stock_aa_ratio": medians["stock0"] / medians["stock1"],
        "scope": f"BF16 Q to {args.output_contract} attention output, synthetic KV",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "source_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), Path(__file__).with_name("chain_reference.py"),
                                    Path(host.__file__), Path(host.__file__).with_name("kernel_paged_decode.py")]},
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
    ap.add_argument("--vllm-kv-layout", action="store_true",
                    help="Use vLLM's combined K/V data-scale allocation and page strides")
    ap.add_argument("--gqa", type=int, choices=(4, 6, 8, 16, 32), required=True)
    ap.add_argument("--output-contract", choices=("bf16", "nvfp4"),
                    default="bf16")
    ap.add_argument("--kv-heads", type=int, choices=(1, 2, 4, 8), default=8)
    ap.add_argument("--head-dim", type=int, choices=(128, 256), default=128)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--output-scale-inv", type=float, default=1.)
    ap.add_argument("--value-scale", type=float, default=1.,
                    help="Shared V global scale; use 256 for long Gaussian-KV "
                         "comparisons to avoid tiny FP8 intermediate values")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--graph-repetitions", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--reference-min-cosine", type=float, default=.98,
                    help="minimum per-head cosine for each arm against independent reference")
    ap.add_argument("--reference-max-relative-rms", type=float,
                    default=.15)
    args = ap.parse_args()
    if (args.value_scale <= 0 or not math.isfinite(args.value_scale)
            or args.output_scale_inv <= 0 or not math.isfinite(args.output_scale_inv)
            or not 0 < args.reference_min_cosine <= 1 or
            not math.isfinite(args.reference_max_relative_rms) or
            args.reference_max_relative_rms <= 0):
        ap.error("scales must be positive and finite; thresholds positive; "
                 "cosines must be at most 1")
    if args.head_dim == 256 and args.output_contract == "nvfp4":
        ap.error("D256 supports BF16 output only")
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
