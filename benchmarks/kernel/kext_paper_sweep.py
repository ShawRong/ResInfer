#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'NVFP4 ratio sweep: stock, full-residue Fold and selective K-extension.'
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections.abc import Callable
from typing import NamedTuple
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks import manifest as spec_mod
from benchmarks.kernel.json_config import load_models, parse_config_overrides, parse_tp_overrides
from benchmarks.kernel.shapes import resolve
from benchmarks.kernel.inputs import (global_scale, make_weight,
                                      make_extended_weights)
from benchmarks.kernel.timing import fi_autotuned, timed_detail, warm_equally
from benchmarks._runtime import assert_gpu_exclusive, gpu_state, provenance


# Ratio 1.0 reuses the base weight and the runtime ratio1_prefill route.
SUPPORTED_RATIOS = (0.125, 0.25, 0.5, 1.0)


class BenchmarkArm(NamedTuple):
    """One callable and its explicit rotating inputs, plus result labels."""
    family: str
    ratio: float
    forward: Callable
    inputs: tuple
    route: str
    policy: object = None

def residue_arm(x, weight, scale, base_scale, scale_inv, alpha, mask,
                *, k, n, num_salient, ratio1):
    """Use the production route, exposing its active weight/scales to rotation."""
    import torch
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


def measure_arms(arms, args, *, order_offset=0):
    """Rotate measurement order; compute ratios after the stock result is known."""
    m = arms[0].inputs[0].shape[0]
    shift = order_offset % len(arms)
    ordered_arms = arms[shift:] + arms[:shift]
    rows = []
    for arm in ordered_arms:
        with fi_autotuned(arm.forward, arm.inputs, m_bucket=m):
            warm_equally(arm.forward, arm.inputs)
            measured = timed_detail(arm.forward, arm.inputs,
                                    repeats=args.repeats)
            tactic = selected_fold_tactic(arm.route, arm.policy) if arm.policy else {}
        rows.append({
            'kind': 'measure', 'family': arm.family, 'ratio': arm.ratio,
            'route': arm.route, 'autotune_requested': True,
            'ratio1_prefill_path': arm.policy.prefill if arm.route == 'ratio1_prefill' else None,
            **tactic, **measured,
        })
    stock_us = next(row['us'] for row in rows if row['family'] == 'stock')
    for row in rows:
        yield dict(row, vs_stock=row['us'] / stock_us)


def measure_projection(n, k, stock_kernel, *, device, grids, args):
    """Prepare one projection's weights once, then measure its requested M grid."""
    import torch

    w_ref, w_base, sf_base, wgs = make_weight(n, k, device, torch.bfloat16)
    # Attribute interface consumed by vLLM's apply_weights, including emulation.
    layer = SimpleNamespace(
        weight=w_base, weight_scale=sf_base, alpha=None,
        input_global_scale_inv=None, output_size_per_partition=n,
        weights_padding_cols=0, weight_global_scale=wgs)
    ratios = args.ratios if grids['kext'] else ()
    extended = make_extended_weights(w_ref, k, [r for r in ratios if r < 1.0])
    if 1.0 in ratios:
        extended[1.0] = (w_base, sf_base, sf_base, wgs,
                         torch.empty(1, dtype=torch.uint8, device=device), 0)
    m_grid = sorted(set(grids['mext']) | set(grids['kext']))
    for m_index, m in enumerate(m_grid):
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        xgs = global_scale(x)
        layer.alpha = (1 / (xgs * wgs)).reshape(())
        layer.input_global_scale_inv = xgs

        def stock(a, w, sf):
            layer.weight, layer.weight_scale = w, sf
            return stock_kernel.apply_weights(layer, a)

        arms = [BenchmarkArm('stock', 0., stock, (x, w_base, sf_base), 'stock')]
        if m in grids['mext']:
            mask = torch.empty(1, dtype=torch.uint8, device=device)
            fn, inputs, route, policy = residue_arm(
                x, w_base, sf_base, sf_base, xgs, layer.alpha, mask,
                k=k, n=n, num_salient=0, ratio1=True)
            arms.append(BenchmarkArm('mext', 1., fn, inputs, route, policy))
        if m in grids['kext']:
            for ratio, (w, sf, bsf, gs, mask, salient) in extended.items():
                fn, inputs, route, policy = residue_arm(
                    x, w, sf, bsf, xgs, (1 / (xgs * gs)).reshape(()),
                    mask, k=k, n=n, num_salient=salient, ratio1=ratio == 1.0)
                arms.append(BenchmarkArm('kext', ratio, fn, inputs, route, policy))
        for row in measure_arms(arms, args, order_offset=args.rep + m_index):
            yield dict(row, n=n, k=k, m=m)
        layer.weight, layer.weight_scale = w_base, sf_base


def run(args) -> None:
    'Implementation details.'
    import torch
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
              'ratios': ratios, 'autotune_requested': True,
              'manifest_overrides': args.manifest_overrides,
              'shapes': args.shapes, 'config_sources': args.config_sources})
        for shape in args.shapes:
            for row in measure_projection(
                shape['n'], shape['k'], stock_kernel, device=device, grids=grids, args=args,
            ):
                # Retain per-projection rows, sharing one physical measurement.
                for alias in shape['aliases']:
                    emit(dict(row, model=alias['model'], proj=alias['projection'],
                              tp=alias['tp'], shape_id=shape['shape_id'],
                              config_sha256=alias['config_sha256'],
                              stock_kernel_class=stock_name))
            print(f"[bench] done {shape['shape_id']}", flush=True)
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
    p.add_argument("--models", help="comma-separated model names from linear_models.json")
    p.add_argument("--model-config", action="append", default=[], metavar="MODEL=PATH")
    p.add_argument("--model-tp", action="append", default=[], metavar="MODEL=TP[,TP]")
    p.add_argument("--dry-run", action="store_true", help="print resolved shapes without GPU work")
    p.add_argument("--family", default="both", choices=["mext", "kext", "both"])
    p.add_argument("--ratios")
    p.add_argument("--m-max", type=int, default=None,
                   help="optional upper bound on each family's manifest M grid")
    p.add_argument("--repeats", type=int, help="timing repeats per point")
    p.add_argument("--rep", type=int, default=0, help="whole-process repetition index")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default=None)
    p.add_argument("--allow-shared-gpu", action="store_true",
                   help="allow exploratory timing on a shared GPU; record "
                        "other PIDs at run entry/exit")
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
        p.error("models and ratios must not be empty")
    for r in a.ratios:
        if not any(abs(r - s) < 1e-9 for s in SUPPORTED_RATIOS):
            raise SystemExit(f"unsupported ratio {r}; expected one of "
                             f"{SUPPORTED_RATIOS}")
    models = load_models(a.models)
    if kernel.get("projections"):
        for spec in models.values():
            spec["projections"] = kernel["projections"]
    a.shapes, a.config_sources = resolve(
        {"models": models}, parse_config_overrides(a.model_config), a.models,
        parse_tp_overrides(a.model_tp))
    if a.repeats < 1 or (a.m_max is not None and a.m_max < 1):
        p.error("--repeats and --m-max must be positive")
    if a.dry_run:
        print(json.dumps({"shapes": a.shapes, "config_sources": a.config_sources}, indent=2))
        return
    # Paper scores use production autotuning, independent of diagnostic shell overrides.
    os.environ["RESIDUE_RATIO1_PREFILL_PATH"] = "auto"
    run(a)


if __name__ == "__main__":
    main()
