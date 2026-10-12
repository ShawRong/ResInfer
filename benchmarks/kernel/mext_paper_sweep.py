#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""One-command, five-process SM100 MExt paper kernel sweep.

Run ``python -m benchmarks.kernel.mext_paper_sweep --outdir RUN --gpu 3
--model-config custom_model=/path/to/config.json``.  ``--dry-run`` resolves and
prints the physical shapes without importing CUDA packages or writing files.
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
from .mext_paper_aggregate import aggregate, write_outputs
from .json_config import load_suite, parse_config_overrides, parse_tp_overrides
from .mext_paper_shapes import resolve

SUITE = pathlib.Path(__file__).resolve().parents[1] / "suites" / "mext_paper_sm100.json"
REQUIRED_ENV = {
    "RESIDUE_FOLD_MEXT": "1",
    "RESIDUE_SM10X_FOLD_TUNER": "1",
    "RESIDUE_MEXT_FOLD_MAX_M": "512",
    "NVFP4_MEXT_R1_PDL": "1",
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
}
ARM_FORMAT = {
    "mext_r1": ("NVFP4", "NVFP4+residue", "residue Fold"),
    "fi_nvfp4_cutlass": ("NVFP4", "NVFP4", "FlashInfer CUTLASS"),
    "fi_nvfp4_cutedsl": ("NVFP4", "NVFP4", "FlashInfer CuTeDSL"),
    "resinfer_mxfp4_mext_r1": ("MXFP4", "MXFP4+residue", "ResInfer Fold"),
    "resinfer_mxfp4_mext_r1_tuned": ("MXFP4", "MXFP4+residue", "ResInfer Fold tuned"),
    "fi_mxfp4_mxfp4_stock": ("MXFP4", "MXFP4", "FlashInfer CuTeDSL"),
    "fi_mxfp4_mxfp4_cutedsl_quant": ("MXFP4", "MXFP4", "FlashInfer CuTeDSL quant+GEMM"),
    "fi_mxfp4_mxfp8_w4a8": ("MXFP4", "MXFP8", "FlashInfer grouped"),
    "trtllm_nvfp4_fp8_w4a8": ("NVFP4", "FP8", "TensorRT-LLM"),
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
    required_arms = [arm for arm in suite.get("required_arms", suite["arms"])
                     if arm in arms]
    if len(set(required_arms)) != len(required_arms) or set(required_arms) - set(suite["arms"]):
        raise ValueError("required_arms must be a unique subset of suite arms")
    return {"schema": 1, "timing_method": TIMING_METHOD, "run_id": uuid.uuid4().hex, "suite": suite["name"],
            "suite_path": str(args.suite.resolve()),
            "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
            "source_commit": source_commit,
            "config_sources": sources, "shapes": shapes, "m_values": m_values,
            "arms": arms, "required_arms": required_arms, "process_reps": reps,
            "timing_repeats": suite["timing_repeats"], "required_env": REQUIRED_ENV}


def _driver(args: argparse.Namespace) -> int:
    resolved = _prepare(args)
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0
    env = os.environ.copy()
    for key, value in REQUIRED_ENV.items():
        if key in env and env[key] != value:
            raise ValueError(f"{key}={env[key]!r} conflicts with suite {value!r}")
        env[key] = value
    if env.get("CUDA_VISIBLE_DEVICES", str(args.gpu)) != str(args.gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES conflicts with --gpu")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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
                        resolved["arms"], resolved["required_arms"],
                        run_id=resolved["run_id"])
    write_outputs(summary, output)
    print(f"Complete: {output / 'summary.json'}", flush=True)
    return 0


def _prepare_weights(n, k, device, arms):
    import torch
    from flashinfer import SfLayout, mxfp4_quantize, nvfp4_quantize
    from .mext_trtllm_adapter import prepare_weight as prepare_trtllm_weight, require_dense_ops

    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16) / (k ** 0.5)
    wgs = (448.0 * 6.0 / weight.float().abs().max()).reshape(())
    w_nv, ws_nv = nvfp4_quantize(weight, wgs, sfLayout=SfLayout.layout_128x4)
    ws_mext = ws_nv.view(torch.float8_e4m3fn)
    # FlashInfer quantizers already return SM100-interleaved scale
    # factors. Re-interleaving them silently corrupts W4A8 output.
    w_mx, ws_mx = mxfp4_quantize(weight)
    ws_mx_flat = ws_mx.flatten().contiguous()
    w_mx_batch = w_mx.unsqueeze(0)
    ws_mx_batch = ws_mx.unsqueeze(0)
    if "trtllm_nvfp4_fp8_w4a8" in arms:
        trtllm_fp4_utils = require_dense_ops()
        w_trt, ws_trt, wgs_trt = prepare_trtllm_weight(weight, trtllm_fp4_utils)
    return {"reference": weight, "nvfp4": (w_nv, ws_nv, ws_mext, wgs),
            "mxfp4": (w_mx, ws_mx, ws_mx_flat, w_mx_batch, ws_mx_batch),
            "trtllm": (w_trt, ws_trt, wgs_trt)
            if "trtllm_nvfp4_fp8_w4a8" in arms else None}


def _make_runners(x, weights, arms):
    import torch
    from flashinfer import SfLayout, mm_fp4, mxfp4_quantize, mxfp8_quantize, nvfp4_quantize
    from flashinfer.gemm import group_gemm_mxfp8_mxfp4_nt_groupwise
    from nvfp4.mxfp4 import fold_gemm as mxfp4_fold_gemm, quantize_mext as mxfp4_quantize_mext
    from .mext_trtllm_adapter import linear as trtllm_w4a8_linear

    if "resinfer_mxfp4_mext_r1_tuned" in arms:
        from quant.flashinfer_integration import mxfp4_linear_adapter
    m, k = x.shape
    n, device = weights["reference"].shape[0], x.device
    w_nv, ws_nv, ws_mext, wgs = weights["nvfp4"]
    w_mx, ws_mx, ws_mx_flat, w_mx_batch, ws_mx_batch = weights["mxfp4"]
    mask = torch.empty(1, device=device, dtype=torch.uint8)
    xgs = (448.0 * 6.0 / x.float().abs().max()).reshape(())
    alpha = (1.0 / (xgs * wgs)).reshape(1).float()
    if weights["trtllm"] is not None:
        w_trt, ws_trt, wgs_trt = weights["trtllm"]
        # TRT's static FP8 op takes the dequantization scale
        # (amax / 448); the FP4 weight producer takes its inverse.
        x_scale_trt = (x.float().abs().max() / 448.0).reshape(1).float()
        alpha_trt = (x_scale_trt / wgs_trt).reshape(1).float()
    outputs = {arm: torch.empty(m, n, device=device, dtype=torch.bfloat16)
               for arm in arms if arm != "mext_r1"}
    padded_m = max(4, m)
    indptr = torch.tensor([0, padded_m], device=device, dtype=torch.int32)
    mx_output = torch.empty(padded_m, n, device=device, dtype=torch.bfloat16)

    def mext(a, w, ws):
        return torch.ops.residue.nvfp4_linear(
            a, w, xgs, ws, ws, mask, alpha,
            k, 0, 0, n, True, False, True)

    def fi_cutlass(a, w, ws):
        aq, asc = nvfp4_quantize(a, xgs, sfLayout=SfLayout.layout_128x4)
        return mm_fp4(aq, w.T, asc, ws.T, alpha, torch.bfloat16,
                      outputs["fi_nvfp4_cutlass"], backend="cutlass",
                      use_8x4_sf_layout=False)

    def fi_cutedsl(a, w, ws):
        aq, asc = nvfp4_quantize(a, xgs, sfLayout=SfLayout.layout_128x4)
        return mm_fp4(aq, w.T, asc, ws.T, alpha, torch.bfloat16,
                      outputs["fi_nvfp4_cutedsl"], backend="cute-dsl",
                      use_8x4_sf_layout=False)

    def fi_w4a8(a, w, ws):
        if a.shape[0] < 4:
            padded = torch.zeros(4, a.shape[1], device=a.device, dtype=a.dtype)
            padded[:a.shape[0]].copy_(a)
            a = padded
        aq, asf = mxfp8_quantize(a)
        return group_gemm_mxfp8_mxfp4_nt_groupwise(
            aq, w, asf, ws, indptr, out=mx_output,
            out_dtype=torch.bfloat16)[:m]

    def res_mxfp4_mext(a, w, ws):
        aq, asc = mxfp4_quantize_mext(a)
        return mxfp4_fold_gemm(aq, w, asc, ws,
                               out_dtype=torch.bfloat16)

    def res_mxfp4_mext_tuned(a, w, ws):
        return mxfp4_linear_adapter.tuned_mxfp4_fold(
            a, w, ws, pdl_mode="early",
            out_dtype=torch.bfloat16)

    def fi_mxfp4_stock(a, w, ws):
        aq, asc = mxfp4_quantize(a)
        return mm_fp4(
            aq, w.T, asc, ws.T, None, torch.bfloat16,
            outputs["fi_mxfp4_mxfp4_stock"], block_size=32,
            backend="cute-dsl", use_nvfp4=False)

    def fi_mxfp4_cutedsl_quant(a, w, ws):
        aq, asc = mxfp4_quantize(a, backend="cute-dsl")
        return mm_fp4(
            aq, w.T, asc, ws.T, None, torch.bfloat16,
            outputs["fi_mxfp4_mxfp4_cutedsl_quant"], block_size=32,
            backend="cute-dsl", use_nvfp4=False)

    def trtllm_w4a8(a, w, ws, input_scale, alpha):
        return trtllm_w4a8_linear(a, w, ws, input_scale, alpha)

    runners = {
        "mext_r1": (mext, (x, w_nv, ws_mext)),
        "fi_nvfp4_cutlass": (fi_cutlass, (x, w_nv, ws_nv)),
        "fi_nvfp4_cutedsl": (fi_cutedsl, (x, w_nv, ws_nv)),
        "resinfer_mxfp4_mext_r1": (res_mxfp4_mext, (x, w_mx, ws_mx_flat)),
        "resinfer_mxfp4_mext_r1_tuned": (
            res_mxfp4_mext_tuned, (x, w_mx, ws_mx_flat)),
        "fi_mxfp4_mxfp4_stock": (fi_mxfp4_stock, (x, w_mx, ws_mx)),
        "fi_mxfp4_mxfp4_cutedsl_quant": (
            fi_mxfp4_cutedsl_quant, (x, w_mx, ws_mx)),
        "fi_mxfp4_mxfp8_w4a8": (fi_w4a8, (x, w_mx_batch, ws_mx_batch)),
    }
    if weights["trtllm"] is not None:
        runners["trtllm_nvfp4_fp8_w4a8"] = (
            trtllm_w4a8, (x, w_trt, ws_trt, x_scale_trt, alpha_trt))
    return runners


def _measure_shape(shape, resolved, rep, emit):
    import torch
    from .timing import (check_output, fi_autotuned, iters_for_cold_l2,
                         l2_accounting, timed_detail, warm_equally)
    from benchmarks._runtime import assert_gpu_exclusive
    from quant.flashinfer_integration import sm100_fold_candidates

    if "resinfer_mxfp4_mext_r1_tuned" in resolved["arms"]:
        from quant.flashinfer_integration import mxfp4_linear_adapter
    device = torch.device("cuda:0")
    n, k = shape["n"], shape["k"]
    weights = _prepare_weights(n, k, device, resolved["arms"])
    for m in resolved["m_values"]:
        assert_gpu_exclusive(0)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        reference = x @ weights["reference"].T
        runners = _make_runners(x, weights, resolved["arms"])
        padded_m = max(4, m)
        active = [a for a in resolved["arms"] if a in runners]
        niw = max(iters_for_cold_l2(runners[a][1], device) for a in active)
        base = {"schema": 1, "kind": "measure", "run_id": resolved["run_id"],
                "rep": rep, "shape_id": shape["shape_id"],
                "n": n, "k": k, "m": m, "aliases": shape["aliases"],
                "niw": niw}
        # Interleave arm order between M points and process reps.
        shift = (rep + resolved["m_values"].index(m)) % len(active)
        active = active[shift:] + active[:shift]
        for arm in active:
            fn, fn_args = runners[arm]
            fmt_w, fmt_a, provider = ARM_FORMAT[arm]
            row = dict(base, **l2_accounting(fn_args, device, niw), arm=arm, weight_format=fmt_w,
                       activation_format=fmt_a, provider=provider,
                       physical_m=padded_m if arm == "fi_mxfp4_mxfp8_w4a8" else m)
            try:
                fi_autotuned(fn, fn_args, m_bucket=m)
                warm_equally(fn, fn_args, calls=20)
                actual = fn(*fn_args)
                torch.cuda.synchronize()
                row["relmax_bf16"] = check_output(actual, reference)
                if arm == "mext_r1":
                    row["tactic_source"] = sm100_fold_candidates.last_tactic_source()
                    row["tactic"] = sm100_fold_candidates.last_tactic_tile()
                    if (row["tactic_source"] not in {"autotuned", "autotuned_fallback"}
                        or row["tactic"] is None):
                        raise RuntimeError("MExt Fold did not serve a profiled tactic")
                if arm == "resinfer_mxfp4_mext_r1_tuned":
                    row["tactic_source"] = mxfp4_linear_adapter.last_tactic_source()
                    row["tactic"] = mxfp4_linear_adapter.last_tactic_tile()
                    if (row["tactic_source"] not in {"autotuned", "autotuned_fallback"}
                        or row["tactic"] is None):
                        raise RuntimeError("MXFP4 Fold did not serve a profiled tactic")
                    family, mode, tile, _cluster, _ab = row["tactic"]
                    pad = (max(8, tile[0] // 2) if mode == "direct_mext"
                           else 8) if family[0] == "tile8" else 1
                    row["physical_m"] = 2 * ((m + pad - 1) // pad) * pad
                measured = timed_detail(fn, fn_args,
                                        iters_within_graph=niw,
                                        repeats=resolved["timing_repeats"])
                row.update(measured)
                row["status"] = "passed"
            except Exception as exc:
                row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            emit(row)
            if row["status"] != "passed":
                raise RuntimeError(f"{shape['shape_id']} M={m} {arm}: {row['reason']}")
        assert_gpu_exclusive(0)


def _worker(args: argparse.Namespace) -> int:
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
    if "trtllm_nvfp4_fp8_w4a8" in resolved["arms"]:
        from importlib.metadata import version
        prov["trtllm_version"] = version("tensorrt-llm")
    prov.update(rep=args.rep, shape_id=shape["shape_id"],
                source_commit=resolved["source_commit"], required_env=resolved["required_env"])
    stem.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    with stem.with_suffix(".jsonl").open("x") as stream:
        def emit(row):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        _measure_shape(shape, resolved, args.rep, emit)
    return 0


def main() -> int:
    args = _arguments()
    return _worker(args) if args.worker else _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
