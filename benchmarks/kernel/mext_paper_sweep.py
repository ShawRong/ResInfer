#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""One-command, five-process SM100 MExt paper kernel sweep.

Run ``python -m benchmarks.kernel.mext_paper_sweep --outdir RUN --gpu 3
--model-tp custom_model=8``. Use ``--model-config`` to override saved dimensions.  ``--dry-run`` resolves and
prints the physical shapes without importing CUDA packages or writing files.

Flow: main -> _driver -> fresh _worker per shape/repetition -> _measure_shape.
linear.py defines backend calls; _measure_arm owns one measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import uuid

from .timing import TIMING_METHOD
from .inputs import prepare_mext_weights
from .linear import ARM_FORMAT, STATIC_SELECTION, W4A8_CONFIG, make_runners, selected_tactic
from .mext_paper_aggregate import aggregate, write_outputs
from .json_config import load_suite, parse_config_overrides, parse_tp_overrides
from .shapes import resolve

SUITE = pathlib.Path(__file__).resolve().parents[1] / "suites" / "mext_paper_sm100.json"
REQUIRED_ENV = {
    "RESIDUE_FOLD_MEXT": "1",
    "RESIDUE_SM10X_FOLD_TUNER": "1",
    "RESIDUE_MEXT_FOLD_MAX_M": "512",
    "NVFP4_MEXT_R1_PDL": "1",
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
}

def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=pathlib.Path, default=SUITE)
    parser.add_argument("--outdir", type=pathlib.Path)
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU index")
    parser.add_argument("--model-config", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--model-tp", action="append", default=[], metavar="MODEL=TP[,TP]",
                        help="required deployment TP for models marked tp_required in suite")
    parser.add_argument("--source-commit", help="exact source commit for archive-based GPU runs")
    parser.add_argument("--models", help="comma-separated subset")
    parser.add_argument("--shape-id", help="one resolved physical shape")
    parser.add_argument("--m-values", help="comma-separated subset")
    parser.add_argument("--arms", help="comma-separated backend subset")
    parser.add_argument("--reps", type=int, help="independent process repetitions")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resolved", type=pathlib.Path, help=argparse.SUPPRESS)
    parser.add_argument("--rep", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shard-stem", type=pathlib.Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.dry_run and args.outdir is None:
        parser.error("--outdir is required unless --dry-run")
    return args


def _prepare(args: argparse.Namespace) -> dict:
    """Resolve the requested subset and freeze its source/configuration identity."""
    suite = load_suite(args.suite)
    models = args.models.split(",") if args.models else None
    overrides = parse_config_overrides(args.model_config)
    tp_overrides = parse_tp_overrides(args.model_tp)
    shapes, sources = resolve(suite, overrides, models, tp_overrides)
    if args.shape_id:
        shapes = [shape for shape in shapes if shape["shape_id"] == args.shape_id]
        if not shapes:
            raise ValueError(f"unknown --shape-id {args.shape_id!r}")
    m_values = [int(x) for x in args.m_values.split(",")] if args.m_values else suite["m_values"]
    if not m_values or len(set(m_values)) != len(m_values) or any(m not in suite["m_values"] for m in m_values):
        raise ValueError("--m-values must be a nonempty unique subset of suite M values")
    reps = args.reps if args.reps is not None else suite["process_reps"]
    if reps < 1:
        raise ValueError("--reps must be positive")
    arms = args.arms.split(",") if args.arms else suite["arms"]
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(suite["arms"]):
        raise ValueError("--arms must be a nonempty unique subset of suite arms")
    if set(arms) - ARM_FORMAT.keys():
        raise ValueError(f"unknown backend: {set(arms) - ARM_FORMAT.keys()}")
    source_commit = args.source_commit
    source_root = pathlib.Path(__file__).resolve().parents[2]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root,
                         capture_output=True, text=True, check=False)
    if git.returncode == 0:
        live_commit = git.stdout.strip()
        if source_commit and source_commit != live_commit:
            raise ValueError("--source-commit does not match the source checkout")
        source_commit = live_commit
    if not args.dry_run and not source_commit:
        raise ValueError("archive-based GPU runs require --source-commit")
    subset = bool(args.models or args.shape_id or set(m_values) != set(suite["m_values"])
                  or set(arms) != set(suite["arms"]) or reps != suite["process_reps"])
    return {"scope": "subset" if subset else "full_suite", "suite_arms": suite["arms"],
            "schema": 1, "mxfp4_fold_protocol": "shared_buckets_pdl_early_v2", "timing_method": TIMING_METHOD, "run_id": uuid.uuid4().hex, "suite": suite["name"],
            "suite_path": str(args.suite.resolve()),
            "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
            "source_commit": source_commit,
            "config_sources": sources, "shapes": shapes, "m_values": m_values,
            "arms": arms, "process_reps": reps,
            "timing_repeats": suite["timing_repeats"], "required_env": REQUIRED_ENV}


def _driver(args: argparse.Namespace) -> int:
    """Run each shape/repetition in a fresh process, then aggregate its outputs."""
    resolved = _prepare(args)
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0
    env = os.environ.copy()
    for key, value in REQUIRED_ENV.items():
        if key in env and env[key] != value:
            raise ValueError(f"{key}={env[key]!r} conflicts with suite {value!r}")
        env[key] = value
    visible = env.get("CUDA_VISIBLE_DEVICES", str(args.gpu))
    if visible != str(args.gpu):
        from benchmarks._runtime import _nvsmi
        if not visible.startswith("GPU-") or _nvsmi(args.gpu, "gpu=uuid", required=True) != [visible]:
            raise ValueError("CUDA_VISIBLE_DEVICES conflicts with --gpu")
    env["CUDA_VISIBLE_DEVICES"] = visible
    output = args.outdir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "resolved.json"
    frozen.write_text(json.dumps(resolved, indent=2) + "\n")
    files = []
    # A fresh process per shape bounds GPU memory and autotuner state.
    # Every invocation owns a new directory: no old results are reused.
    for rep in range(resolved["process_reps"]):
        directory = output / f"rep{rep}"
        directory.mkdir()
        combined = output / f"rep{rep}.jsonl"
        files.append(combined)
        with combined.open("x") as stream:
            for shape in resolved["shapes"]:
                stem = directory / shape["shape_id"]
                command = [sys.executable, "-m", "benchmarks.kernel.mext_paper_sweep",
                           "--worker", "--resolved", str(frozen), "--outdir", str(output),
                           "--rep", str(rep), "--shape-id", shape["shape_id"],
                           "--shard-stem", str(stem)]
                with stem.with_suffix(".log").open("x") as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=pathlib.Path(__file__).resolve().parents[2], check=True)
                stream.write(stem.with_suffix(".jsonl").read_text())
    summary = aggregate(files, resolved["shapes"], resolved["m_values"],
                        resolved["arms"],
                        run_id=resolved["run_id"])
    summary.update(scope=resolved["scope"], suite=resolved["suite"],
                   suite_arms=resolved["suite_arms"])
    write_outputs(summary, output)
    print(f"Complete ({resolved['scope']}): {output / 'summary.json'}", flush=True)
    return 0


def _measure_arm(arm, runner, base, *, repeats):
    """Tune, warm and time one backend. Errors propagate to the worker."""
    from .timing import fi_autotuned, timed_detail, warm_equally

    fn, inputs = runner
    m = base["m"]
    fmt_w, fmt_a, provider = ARM_FORMAT[arm]
    selection = STATIC_SELECTION.get(arm, "flashinfer_autotune")
    row = dict(base,
               arm=arm, weight_format=fmt_w, activation_format=fmt_a,
               provider=provider, selection=selection,
               physical_m=max(4, m) if arm == "fi_mxfp4_mxfp8_w4a8" else m)
    if arm == "fi_mxfp4_mxfp8_w4a8":
        row["kernel_config"] = dict(W4A8_CONFIG)
    with fi_autotuned(fn, inputs, m_bucket=m, enabled=selection == "flashinfer_autotune"):
        warm_equally(fn, inputs, calls=20)
        row.update(timed_detail(fn, inputs, repeats=repeats))
        row.update(selected_tactic(arm))
    if arm == "resinfer_mxfp4_mext_r1_tuned" and row.get("tactic") is not None:
        family, mode, tile, _cluster, _ab = row["tactic"]
        pad = 1
        if family[0] == "tile8":
            pad = max(8, tile[0] // 2) if mode == "direct_mext" else 8
        row["physical_m"] = 2 * ((m + pad - 1) // pad) * pad
    return dict(row, status="passed")


def _measure_shape(shape, resolved, rep, emit):
    """Reuse weights across M points and rotate backend order between points."""
    import torch

    device = torch.device("cuda:0")
    n, k = shape["n"], shape["k"]
    weights = prepare_mext_weights(n, k, device, resolved["arms"])
    for m_index, m in enumerate(resolved["m_values"]):
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        runners = make_runners(x, weights, resolved["arms"])
        active = [a for a in resolved["arms"] if a in runners]
        base = {"schema": 1, "kind": "measure", "run_id": resolved["run_id"],
                "rep": rep, "shape_id": shape["shape_id"],
                "n": n, "k": k, "m": m, "aliases": shape["aliases"]}
        # Interleave arm order between M points and process reps.
        shift = (rep + m_index) % len(active)
        active = active[shift:] + active[:shift]
        for arm in active:
            row = _measure_arm(
                arm, runners[arm], base, repeats=resolved["timing_repeats"],
            )
            emit(row)


def _worker(args: argparse.Namespace) -> int:
    """Initialize one GPU process and write its provenance and measurement rows."""
    import torch
    import nvfp4
    from benchmarks._runtime import assert_gpu_exclusive, provenance
    nvfp4.ensure_loaded()
    from quant.vllm_integration import residue_linear_op  # registers the op
    if torch.cuda.get_device_capability(0) != (10, 0):
        raise RuntimeError("paper suite requires SM100 (B200/Blackwell)")
    assert_gpu_exclusive(0)
    resolved = json.loads(args.resolved.read_text())
    shape = next(s for s in resolved["shapes"] if s["shape_id"] == args.shape_id)
    torch.manual_seed(20260921 + args.rep)
    stem = args.shard_stem
    prov = provenance(0, resolved["run_id"])
    from quant.flashinfer_integration import sm100_fold_candidates
    prov["candidate_hash"] = sm100_fold_candidates.candidates_hash()
    if "resinfer_mxfp4_mext_r1_tuned" in resolved["arms"]:
        from quant.flashinfer_integration import mxfp4_linear_adapter
        prov["mxfp4_candidate_hash"] = mxfp4_linear_adapter.candidates_hash()
    if any(arm.startswith("trtllm_") for arm in resolved["arms"]):
        from importlib.metadata import version
        prov["trtllm_version"] = version("tensorrt-llm")
    prov.update(rep=args.rep, shape_id=shape["shape_id"],
                source_commit=resolved["source_commit"], required_env=resolved["required_env"],
                mxfp4_fold_protocol=resolved["mxfp4_fold_protocol"])
    stem.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    with stem.with_suffix(".jsonl").open("x") as stream:
        def emit(row):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        _measure_shape(shape, resolved, args.rep, emit)
    assert_gpu_exclusive(0)
    return 0


def main() -> int:
    args = _arguments()
    return _worker(args) if args.worker else _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
