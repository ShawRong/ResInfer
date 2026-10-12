#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 ratio sweep: stock, full-residue Fold and selective K-extension.

Run `python -m benchmarks.kernel.kext_paper_sweep --help` for options.
Measures online activation quantization plus GEMM, using FlashInfer CUDA Graph
cold-L2 timing. Selective ratios are 0.125/0.25/0.5; full residue is a separate
Fold arm. This is a kernel pipeline benchmark, not full-model throughput.
All arms tune and warm before measurement. Results include correctness,
latency samples, selected tactics and GPU/runtime provenance.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks import manifest as spec_mod
from benchmarks.kernel.timing import (check_output, fi_autotuned, iters_for_cold_l2,
                                l2_accounting, timed_detail, warm_equally)
from benchmarks._runtime import assert_gpu_exclusive, gpu_state, provenance


# ----------------------------------------------------------------- shapes
# (N, K) for x[M,K] @ W[K,N] -> y[M,N], i.e. the vLLM *fused* projections.
# Read off the public configs; fused sizes are what actually gets launched.
SHAPES: dict[str, dict[str, tuple[int, int]]] = {
    "qwen3_4b": {                       # hidden 2560, inter 9728, 36 layers
        "qkv": (6144, 2560), "o": (2560, 4096),
        "gate_up": (19456, 2560), "down": (2560, 9728),
    },
    "qwen3_8b": {                       # hidden 4096, inter 12288, 36 layers
        "qkv": (6144, 4096), "o": (4096, 4096),
        "gate_up": (24576, 4096), "down": (4096, 12288),
    },
    "qwen3_32b": {                      # hidden 5120, inter 25600, 64 layers
        "qkv": (10240, 5120), "o": (5120, 8192),
        "gate_up": (51200, 5120), "down": (5120, 25600),
    },
}




SUPPORTED_RATIOS = (0.125, 0.25, 0.5)

# ------------------------------------------------------------------ setup
def global_scale(t: torch.Tensor) -> torch.Tensor:
    """vLLM/CUTLASS FP4 convention: 448*6/amax, passed into the quantiser."""
    return (448.0 * 6.0 / t.abs().max().float()).reshape(())


def make_weight(n: int, k: int, device: torch.device, dtype: torch.dtype):
    """Quantise a random weight the way vLLM does, and return both forms.

    Both the packed FP4 weight and the bf16 tensor it came from: the latter is
    the reference the correctness check compares against, so it must be the
    exact tensor that was quantised, not a fresh draw.
    """
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale)

    w_bf16 = torch.randn(n, k, device=device, dtype=dtype) / (k ** 0.5)
    w_gs = global_scale(w_bf16)
    w_fp4, w_sf = scaled_fp4_quant(w_bf16, w_gs, is_sf_swizzled_layout=False)
    return w_bf16, w_fp4, swizzle_blockscale(w_sf), w_gs


class FakeLayer(torch.nn.Module):
    """The attribute surface `NvFp4LinearKernel.apply_weights` reads.

    Deliberately minimal and deliberately NOT a reimplementation: vLLM's own
    kernel object does the work, this only supplies what it looks up.
    """

    def __init__(self, weight, weight_scale, alpha, input_gs, out_size,
                 weight_gs=None):
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self.alpha = alpha
        self.input_global_scale_inv = input_gs
        self.output_size_per_partition = out_size
        self.weights_padding_cols = 0
        # Unused by the fast kernels, which fold both global scales into
        # `alpha`. Kept because vLLM's emulation kernel reads it, so this
        # layer can be handed to that path unchanged if a reference is ever
        # wanted.
        self.weight_global_scale = weight_gs


def salient_indices(k: int, ratio: float, device: torch.device) -> torch.Tensor:
    """The per-8-channel salient pattern the export format uses.

    Selection is structurally confined to each 8-channel block
    (`residue_per_8`), so a benchmark that picked globally would be measuring
    a layout the exporter cannot produce. Which channels within the block are
    picked does not affect cost, so a deterministic prefix is used.
    """
    if k % 8:
        raise ValueError(f"k must be a multiple of 8, got {k}")
    per_block = int(round(ratio * 8))
    if per_block == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    base = torch.arange(0, k, 8, dtype=torch.long, device=device)
    off = torch.arange(per_block, dtype=torch.long, device=device)
    return (base[:, None] + off[None, :]).reshape(-1).contiguous()


def residue_arm(x, weight, scale, base_scale, scale_inv, alpha, mask,
                k, n, num_salient, ratio1):
    """Use the production route, exposing its active weight/scales to rotation."""
    from quant.vllm_integration.residue_linear_op import _linear_policy
    from quant.flashinfer_integration.nvfp4_linear_adapter import _route

    policy = _linear_policy(k, num_salient, 0, n, True, False, ratio1)
    route = _route(x.shape[0], policy)
    if route in ('sm10x_fold', 'sm120_fold', 'ratio1_prefill'):
        weight, scale = weight[:, :k // 2], base_scale

    def forward(a, w, sf):
        return torch.ops.residue.nvfp4_linear(
            a, w, scale_inv, sf, sf, mask, alpha,
            k, num_salient, 0, n, True, False, ratio1)

    return forward, (x, weight, scale), route, policy


def selected_fold_tactic(route, policy):
    """Read the selected Fold configuration; never infer it from log messages."""
    if route == 'sm10x_fold':
        from quant.flashinfer_integration import sm100_fold_candidates as candidates
        enabled = policy.sm10x_tuner
    elif route == 'sm120_fold':
        from quant.flashinfer_integration import sm120_fold_candidates as candidates
        enabled = policy.sm120_tuner
    else:
        return {}
    return {'fold_tactic_source': candidates.last_tactic_source() if enabled else 'offline_table',
            'fold_tactic_tile': candidates.last_tactic_tile()}


def make_extended_weights(w_ref, k, ratios):
    import nvfp4
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

    extended = {}
    for ratio in ratios:
        idx = salient_indices(k, ratio, w_ref.device)
        w_ext = torch.cat([w_ref, w_ref.index_select(-1, idx)], dim=-1)
        gs = global_scale(w_ext)
        packed, scales = scaled_fp4_quant(w_ext, gs, is_sf_swizzled_layout=False)
        extended[ratio] = (packed, swizzle_blockscale(scales),
            swizzle_blockscale(scales[:, :k // 16].contiguous()), gs,
            nvfp4.indices_to_channel_masks(idx, k), idx.numel())
    return extended


def measure_arms(arms, reference, args):
    device, m = reference.device, reference.shape[0]
    niw = max(iters_for_cold_l2(inputs, device) for _, _, _, inputs, _, _ in arms)
    stock_us = None
    for family, ratio, fn, inputs, route, policy in arms:
        if args.autotune:
            fi_autotuned(fn, inputs, m_bucket=m)
        error = check_output(fn(*inputs), reference)
        warm_equally(fn, inputs)
        measured = timed_detail(fn, inputs, iters_within_graph=niw,
                                repeats=args.repeats)
        if family == 'stock':
            stock_us = measured['us']
        yield {'kind': 'measure', 'family': family, 'ratio': ratio,
              'route': route,
              'autotune_requested': args.autotune, 'relmax_bf16': error,
              'vs_stock': measured['us'] / stock_us,
              **(selected_fold_tactic(route, policy) if policy else {}),
              **l2_accounting(inputs, device, niw), **measured}


def run(args) -> None:
    from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
    from quant.vllm_integration import residue_linear_op  # registers the op

    if not 0 <= args.gpu < torch.cuda.device_count():
        raise ValueError('--gpu must index the devices visible through CUDA_VISIBLE_DEVICES')
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    stock_kernel = init_nvfp4_linear_kernel()
    stock_name = type(stock_kernel).__name__
    major, minor = torch.cuda.get_device_capability(device)
    sm = f'sm{major}{minor}'
    doc = args.manifest_doc
    grids = {family: spec_mod.m_grid_for(doc, family, sm)
             for family in ('mext', 'kext')}
    for family in grids:
        if args.family not in ('both', family):
            grids[family] = []
        if args.m_max is not None:
            grids[family] = [m for m in grids[family] if m <= args.m_max]
    m_grid = sorted(set(grids['mext']) | set(grids['kext']))
    if not m_grid:
        raise ValueError(f'no M points for {sm}')
    ratios = args.ratios
    projections = doc.get('kernel', {}).get('projections', ('qkv', 'o', 'gate_up', 'down'))
    run_id = args.run_id or time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A failed partial run must not be silently appended to another run.
    with out.open('x') as stream:
        def emit(row):
            row.update(run_id=run_id, rep=args.rep,
                       ts=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            stream.write(json.dumps(row) + '\n')
            stream.flush()

        neighbors = assert_gpu_exclusive(args.gpu, exclusive=not args.allow_shared_gpu)
        emit({**provenance(args.gpu, run_id), **spec_mod.stamp(doc),
              'stock_kernel_class': stock_name, 'gpu_neighbors': neighbors,
              'm_grid': m_grid, 'm_grid_mext': grids['mext'], 'm_grid_kext': grids['kext'],
              'ratios': ratios, 'autotune_requested': args.autotune,
              'manifest_overrides': args.manifest_overrides})
        for model in args.models:
            for proj in projections:
                n, k = SHAPES[model][proj]
                w_ref, w_base, sf_base, wgs = make_weight(n, k, device, torch.bfloat16)
                layer = FakeLayer(w_base, sf_base, None, None, n, weight_gs=wgs)
                extended = make_extended_weights(w_ref, k, ratios if grids['kext'] else ())
                for m in m_grid:
                    assert_gpu_exclusive(args.gpu, exclusive=not args.allow_shared_gpu)
                    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
                    reference = x @ w_ref.T
                    xgs = global_scale(x)
                    layer.alpha = (1 / (xgs * wgs)).reshape(())
                    layer.input_global_scale_inv = xgs

                    def stock(a, w, sf):
                        layer.weight, layer.weight_scale = w, sf
                        return stock_kernel.apply_weights(layer, a)

                    arms = [('stock', 0., stock, (x, w_base, sf_base), 'stock', None)]
                    if m in grids['mext']:
                        mask = torch.empty(1, dtype=torch.uint8, device=device)
                        fn, inputs, route, policy = residue_arm(
                            x, w_base, sf_base, sf_base, xgs, layer.alpha, mask,
                            k, n, 0, True)
                        arms.append(('mext', 1., fn, inputs, route, policy))
                    if m in grids['kext']:
                        for ratio, (w, sf, bsf, gs, mask, salient) in extended.items():
                            fn, inputs, route, policy = residue_arm(
                                x, w, sf, bsf, xgs, (1 / (xgs * gs)).reshape(()),
                                mask, k, n, salient, False)
                            arms.append(('kext', ratio, fn, inputs, route, policy))
                    for row in measure_arms(arms, reference, args):
                        emit(dict(row, model=model, proj=proj, n=n, k=k, m=m,
                                  stock_kernel_class=stock_name))
                    layer.weight, layer.weight_scale = w_base, sf_base
                print(f'[bench] done {model}/{proj}', flush=True)
        emit({'kind': 'run_summary', 'status': 'complete',
              'gpu_neighbors_end': assert_gpu_exclusive(args.gpu, exclusive=not args.allow_shared_gpu),
              'gpu_end': gpu_state(args.gpu)})
    print(f'[bench] wrote {out}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=None,
                   help="benchmark spec YAML (default: the one shipped in "
                        "benchmarks/suites/, default speed_dense.yaml). Values it sets become the "
                        "defaults; an explicit flag still wins, and the "
                        "override is recorded in provenance.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--models",
                   help=f"comma-separated: {', '.join(SHAPES)}")
    p.add_argument("--family", default="both", choices=["mext", "kext", "both"])
    p.add_argument("--ratios")
    p.add_argument("--m-max", type=int, default=None,
                   help="optional upper bound on each family's manifest M grid")
    p.add_argument("--repeats", type=int, help="timing repeats per point")
    p.add_argument("--rep", type=int, default=0, help="whole-process repetition index")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default=None)
    p.add_argument("--allow-shared-gpu", action="store_true",
                   help="time even when another process holds the GPU. Does "
                        "NOT clean the numbers -- it annotates every row with "
                        "the neighbouring PIDs instead of refusing.")
    p.add_argument("--no-autotune", dest="autotune", action="store_false",
                   help="skip FlashInfer profiling for all arms; existing cache "
                        "entries may still be used")
    p.set_defaults(autotune=True)
    p.add_argument("--out", default="residue_kernel_bench.jsonl")
    a = p.parse_args()

    a.manifest_doc = spec_mod.load(a.manifest)
    kernel = a.manifest_doc["kernel"]
    a.manifest_overrides = [name for name in ("models", "ratios", "repeats")
                            if getattr(a, name) is not None]
    a.models = ([m.strip() for m in a.models.split(",") if m.strip()]
                if a.models is not None else kernel["models"])
    a.ratios = ([float(r) for r in a.ratios.split(",") if r.strip()]
                if a.ratios is not None else
                kernel.get("ratios", spec_mod.ratios_for(a.manifest_doc, "kext")))
    a.repeats = a.repeats if a.repeats is not None else kernel.get("repeats", 5)
    if not a.models or (a.family != "mext" and not a.ratios):
        p.error("models and selective ratios must not be empty")
    for r in a.ratios:
        if not any(abs(r - s) < 1e-9 for s in SUPPORTED_RATIOS):
            raise SystemExit(f"unsupported ratio {r}; expected one of "
                             f"{SUPPORTED_RATIOS} (1.0 is the mext family)")
    for m in a.models:
        if m not in SHAPES:
            raise SystemExit(f"unknown model {m}; expected one of {list(SHAPES)}")
    if a.repeats < 1 or (a.m_max is not None and a.m_max < 1):
        p.error("--repeats and --m-max must be positive")
    run(a)


if __name__ == "__main__":
    main()
