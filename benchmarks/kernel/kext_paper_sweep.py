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

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks import manifest as spec_mod
from benchmarks.kernel.json_config import load_models, parse_config_overrides, parse_tp_overrides
from benchmarks.kernel.shapes import resolve
from benchmarks.kernel.inputs import SELECTIVE_RATIOS, prepare_mext_weights, make_extended_weights
from benchmarks.kernel.linear import (KEXT_BASELINES, STATIC_SELECTION,
                                      make_kext_arms, selected_fold_tactic)
from benchmarks.kernel.timing import fi_autotuned, timed_detail, warm_equally
from benchmarks._runtime import assert_gpu_exclusive, gpu_state, provenance


# Ratio 1.0 reuses the base weight and the runtime ratio1_prefill route.
SUPPORTED_RATIOS = (*SELECTIVE_RATIOS, 1.0)


def measure_arms(arms, args, *, order_offset=0):
    """Rotate measurement order; compute ratios after the stock result is known."""
    m = arms[0].inputs[0].shape[0]
    shift = order_offset % len(arms)
    ordered_arms = arms[shift:] + arms[:shift]
    rows = []
    for arm in ordered_arms:
        selection = STATIC_SELECTION.get(arm.arm, "flashinfer_autotune")
        with fi_autotuned(arm.forward, arm.inputs, m_bucket=m,
                          enabled=selection == "flashinfer_autotune"):
            warm_equally(arm.forward, arm.inputs)
            measured = timed_detail(arm.forward, arm.inputs,
                                    repeats=args.repeats)
            tactic = selected_fold_tactic(arm.route, arm.policy) if arm.policy else {}
        rows.append({
            'kind': 'measure', 'family': arm.family, 'ratio': arm.ratio,
            'arm': arm.arm or arm.family, 'selection': selection,
            'route': arm.route, 'autotune_requested': selection == 'flashinfer_autotune',
            'ratio1_prefill_path': arm.policy.prefill if arm.route == 'ratio1_prefill' else None,
            **tactic, **measured,
        })
    stock_us = next(row['us'] for row in rows if row['family'] == 'stock')
    for row in rows:
        yield dict(row, vs_stock=row['us'] / stock_us)


def measure_projection(n, k, *, device, grids, args):
    """Prepare one projection and compare all ratios against the shared baselines."""
    import torch

    weights = prepare_mext_weights(n, k, device, KEXT_BASELINES)
    ratios = args.ratios if grids['kext'] else ()
    extended = make_extended_weights(weights['reference'], k, [r for r in ratios if r < 1])
    for m_index, m in enumerate(sorted(set(grids['mext']) | set(grids['kext']))):
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        arms = make_kext_arms(x, weights, extended, ratios if m in grids['kext'] else ())
        if m in grids['mext']:
            full = make_kext_arms(x, weights, extended, [1.0])[-1]
            arms.append(full._replace(family='mext', arm='mext_r1'))
        for row in measure_arms(arms, args, order_offset=args.rep + m_index):
            yield dict(row, n=n, k=k, m=m)


def run(args) -> None:
    'Implementation details.'
    import torch
    from quant.vllm_integration import residue_linear_op  # registers the op

    if not 0 <= args.gpu < torch.cuda.device_count():
        raise ValueError('--gpu must index the devices visible through CUDA_VISIBLE_DEVICES')
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    stock_name = KEXT_BASELINES[0]
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
              'stock_arm': stock_name, 'gpu_neighbors': neighbors,
              'm_grid': m_grid, 'm_grid_mext': grids['mext'], 'm_grid_kext': grids['kext'],
              'ratios': ratios, 'baseline_arms': KEXT_BASELINES,
              'mask_source': 'synthetic_per8_prefix', 'selective_route': 'k_ext',
              'manifest_overrides': args.manifest_overrides,
              'shapes': args.shapes, 'config_sources': args.config_sources})
        for shape in args.shapes:
            for row in measure_projection(
                shape['n'], shape['k'], device=device, grids=grids, args=args,
            ):
                # Retain per-projection rows, sharing one physical measurement.
                for alias in shape['aliases']:
                    emit(dict(row, model=alias['model'], proj=alias['projection'],
                              tp=alias['tp'], shape_id=shape['shape_id'],
                              config_sha256=alias['config_sha256'],
                              stock_arm=stock_name))
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
        if r not in SUPPORTED_RATIOS:
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
