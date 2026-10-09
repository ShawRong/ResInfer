#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""One-command, five-process SM100 MExt paper kernel sweep.

Run ``python -m benchmarks.mext_paper_sweep --outdir RUN --gpu 3
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

from .mext_paper_aggregate import aggregate, write_outputs
from .mext_paper_shapes import load_suite, parse_config_overrides, parse_tp_overrides, resolve

SUITE = pathlib.Path(__file__).with_name("suites") / "mext_paper_sm100.json"
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
    parser.add_argument("--models", help="comma-separated subset; non-canonical")
    parser.add_argument("--shape-id", help="one resolved physical shape; non-canonical")
    parser.add_argument("--m-values", help="comma-separated subset; non-canonical")
    parser.add_argument("--arms", help="comma-separated backend subset; non-canonical")
    parser.add_argument("--reps", type=int, help="override process reps; non-canonical")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="resume only shards matching the frozen manifest")
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
    canonical = not (args.models or args.shape_id or args.m_values or args.arms
                     or args.reps is not None)
    source_commit = args.source_commit
    source_root = pathlib.Path(__file__).resolve().parents[1]
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
    return {"schema": 1, "run_id": uuid.uuid4().hex, "suite": suite["name"],
            "suite_path": str(args.suite.resolve()),
            "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
            "source_commit": source_commit, "canonical_matrix": canonical,
            "config_sources": sources, "shapes": shapes, "m_values": m_values,
            "arms": arms, "required_arms": required_arms, "process_reps": reps,
            "timing_repeats": suite["timing_repeats"], "required_env": REQUIRED_ENV}


def _driver(args: argparse.Namespace) -> int:
    requested = _prepare(args)
    output = args.outdir.resolve() if args.outdir else None
    if args.resume:
        if output is None:
            raise ValueError("--resume requires --outdir")
        resolved = json.loads((output / "resolved.json").read_text())
        if {k: v for k, v in resolved.items() if k != "run_id"} != {
            k: v for k, v in requested.items() if k != "run_id"
        }:
            raise ValueError("--resume arguments/source/config differ from frozen manifest")
    else:
        resolved = requested
    print(json.dumps({"suite": resolved["suite"], "unique_shapes": len(resolved["shapes"]),
                      "aliases": sum(len(s["aliases"]) for s in resolved["shapes"]),
                      "m_values": resolved["m_values"], "arms": resolved["arms"],
                      "process_reps": resolved["process_reps"],
                      "config_sources": resolved["config_sources"]}, indent=2), flush=True)
    if args.dry_run:
        for shape in resolved["shapes"]:
            print(json.dumps(shape), flush=True)
        return 0
    env = os.environ.copy()
    for key, value in REQUIRED_ENV.items():
        if key in env and env[key] != value:
            raise ValueError(f"{key}={env[key]!r} conflicts with paper suite {value!r}")
        env[key] = value
    if "CUDA_VISIBLE_DEVICES" in env and env["CUDA_VISIBLE_DEVICES"] != str(args.gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES conflicts with --gpu")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    frozen = output / "resolved.json"
    if not args.resume:
        frozen.write_text(json.dumps(resolved, indent=2) + "\n")
    shards_dir = output / "shards"
    shards_dir.mkdir(exist_ok=args.resume)
    rep_files = []
    all_provenance = []
    for rep in range(resolved["process_reps"]):
        rep_file = output / f"rep{rep}.jsonl"
        rep_files.append(rep_file)
        selected_shards = []
        for shape in resolved["shapes"]:
            shape_dir = shards_dir / f"rep{rep}" / shape["shape_id"]
            shape_dir.mkdir(parents=True, exist_ok=True)
            attempts = sorted({int(p.name.split(".")[0].removeprefix("attempt"))
                               for p in shape_dir.glob("attempt*.*")})
            selected = None
            for attempt in attempts:
                stem = shape_dir / f"attempt{attempt}"
                try:
                    _validate_shard(stem, resolved, rep, shape)
                except (ValueError, OSError):
                    continue
                selected = stem
                break
            if selected is None:
                stem = shape_dir / f"attempt{max(attempts, default=-1) + 1}"
                command = [sys.executable, "-m", "benchmarks.mext_paper_sweep",
                           "--worker", "--resolved", str(frozen), "--outdir", str(output),
                           "--rep", str(rep), "--shape-id", shape["shape_id"],
                           "--shard-stem", str(stem)]
                with pathlib.Path(f"{stem}.log").open("x") as log:
                    result = subprocess.run(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT,
                        cwd=pathlib.Path(__file__).resolve().parents[1], check=False)
                if result.returncode:
                    (output / "status.json").write_text(json.dumps(
                        {"status": "incomplete", "failed_rep": rep,
                         "shape_id": shape["shape_id"], "attempt": stem.name,
                         "exit_code": result.returncode}, indent=2) + "\n")
                    print(f"{stem} failed (exit={result.returncode}); see {stem}.log",
                          file=sys.stderr)
                    return result.returncode
                _validate_shard(stem, resolved, rep, shape)
                selected = stem
            selected_shards.append(selected)
            all_provenance.append(json.loads(pathlib.Path(f"{selected}.provenance.json").read_text()))
        tmp = output / f"rep{rep}.jsonl.tmp"
        with tmp.open("w") as stream:
            for stem in selected_shards:
                stream.write(pathlib.Path(f"{stem}.jsonl").read_text())
        tmp.replace(rep_file)
        (output / f"rep{rep}.provenance.json").write_text(json.dumps(
            {"kind": "shard_manifest", "rep": rep,
             "shards": [str(stem) for stem in selected_shards]}, indent=2) + "\n")
    fingerprints = {_runtime_fingerprint(prov) for prov in all_provenance}
    if len(fingerprints) != 1:
        reason = "shards have mixed Runtime/binary/GPU fingerprints"
        (output / "status.json").write_text(json.dumps(
            {"status": "incomplete", "reason": reason}, indent=2) + "\n")
        raise ValueError(reason)
    try:
        summary = aggregate(rep_files, resolved["shapes"], resolved["m_values"],
                            resolved["arms"], resolved["required_arms"])
    except ValueError as exc:
        (output / "status.json").write_text(json.dumps(
            {"status": "incomplete", "reason": str(exc)}, indent=2) + "\n")
        raise
    write_outputs(summary, output)
    (output / "status.json").write_text(json.dumps(
        {"status": "complete", "run_id": resolved["run_id"],
         "canonical_matrix": resolved["canonical_matrix"]}, indent=2) + "\n")
    print(f"Complete: {output / 'summary.json'}", flush=True)
    return 0


def _runtime_fingerprint(prov: dict) -> str:
    return json.dumps({
        "source_commit": prov.get("source_commit"),
        "suite_sha256": prov.get("suite_sha256"),
        "loaded_nvfp4_so_sha256": prov.get("loaded_nvfp4_so_sha256"),
        "trtllm_version": prov.get("trtllm_version"),
        "candidate_hash": prov.get("candidate_hash"),
        "mxfp4_candidate_hash": prov.get("mxfp4_candidate_hash"),
        "mxfp4_fold_pdl_mode": prov.get("mxfp4_fold_pdl_mode"),
        "torch": prov.get("torch"), "flashinfer": prov.get("flashinfer"),
        "cuda": prov.get("cuda"), "driver_version": prov.get("driver_version"),
        "gpu_uuid": prov.get("gpu", {}).get("uuid"),
    }, sort_keys=True)


def _validate_shard(stem: pathlib.Path, resolved: dict, rep: int,
                    shape: dict) -> None:
    rows_path = pathlib.Path(f"{stem}.jsonl")
    provenance_path = pathlib.Path(f"{stem}.provenance.json")
    if not rows_path.exists() or not provenance_path.exists():
        raise ValueError(f"incomplete shard {stem}")
    prov = json.loads(provenance_path.read_text())
    if (prov.get("run_id") != resolved["run_id"] or prov.get("rep") != rep or
        prov.get("source_commit") != resolved["source_commit"] or
        prov.get("suite_sha256") != resolved["suite_sha256"] or
        not prov.get("loaded_nvfp4_so_sha256")):
        raise ValueError(f"shard provenance mismatch: {stem}")
    expected = {(m, arm) for m in resolved["m_values"] for arm in resolved["arms"]}
    try:
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed shard {stem}") from exc
    keys = {(row.get("m"), row.get("arm")) for row in rows}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError(f"missing or duplicate shard cells: {stem}")
    for row in rows:
        if (row.get("run_id") != resolved["run_id"] or row.get("rep") != rep or
            row.get("shape_id") != shape["shape_id"] or
            row.get("n") != shape["n"] or row.get("k") != shape["k"] or
            row.get("aliases") != shape["aliases"] or
            row.get("cold_l2_effective") is not True or
            row.get("status") != "passed" or
            not isinstance(row.get("us"), (int, float)) or row["us"] <= 0):
            raise ValueError(f"invalid shard cell: {stem}")


def _worker(args: argparse.Namespace) -> int:
    # Imports deliberately delayed: --dry-run and aggregation work on CPU-only hosts.
    import hashlib
    import importlib.util
    import importlib.metadata

    import torch
    from flashinfer import (SfLayout, mm_fp4, mxfp4_quantize, mxfp8_quantize,
                            nvfp4_quantize)
    from flashinfer.gemm import group_gemm_mxfp8_mxfp4_nt_groupwise

    # The legacy provenance module prepends its own source-tree csrc to
    # sys.path. Load the explicitly selected nvfp4 build first, so that
    # bookkeeping cannot shadow the compiled extension used for measurement.
    import nvfp4
    nvfp4.ensure_loaded()
    from nvfp4.mxfp4 import fold_gemm as mxfp4_fold_gemm
    from nvfp4.mxfp4 import quantize_mext as mxfp4_quantize_mext

    from .mext_trtllm_adapter import linear as trtllm_w4a8_linear
    from .mext_trtllm_adapter import prepare_weight as prepare_trtllm_weight
    from .mext_trtllm_adapter import require_dense_ops
    resolved = json.loads(args.resolved.read_text())
    mxfp4_linear_adapter = None
    if "resinfer_mxfp4_mext_r1_tuned" in resolved["arms"]:
        from quant.flashinfer_integration import mxfp4_linear_adapter
    trtllm_fp4_utils = (require_dense_ops()
                       if "trtllm_nvfp4_fp8_w4a8" in resolved["arms"] else None)

    from benchmarks._common import (assert_args_consumed, fi_autotuned,
                                    iters_for_cold_l2, l2_accounting,
                                    timed_detail, warm_equally)
    from benchmarks.bench_residue_nvfp4 import (assert_gpu_exclusive,
                                                 physical_index, provenance)
    from quant.vllm_integration import residue_linear_op  # noqa: F401
    from quant.flashinfer_integration import sm100_fold_candidates

    if torch.cuda.get_device_capability(0) != (10, 0):
        raise RuntimeError("paper suite requires SM100 (B200/Blackwell)")
    for key, value in resolved["required_env"].items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"missing suite environment: {key}={value}")
    if len({c[0] for c in sm100_fold_candidates._candidates()}) != 3:
        raise RuntimeError("A4/A6 are not in the live FlashInfer candidate set")
    torch.manual_seed(20260921 + args.rep)
    device = torch.device("cuda:0")
    neighbors = assert_gpu_exclusive(0)
    prov = provenance(0, resolved["run_id"])
    prov.update({"rep": args.rep, "suite": resolved["suite"],
                 "source_commit": resolved["source_commit"],
                 "suite_sha256": resolved["suite_sha256"],
                 "pdl": os.environ["NVFP4_MEXT_R1_PDL"],
                 "fold_max_m": os.environ["RESIDUE_MEXT_FOLD_MAX_M"],
                 "fold_mext": os.environ["RESIDUE_FOLD_MEXT"],
                 "candidate_hash": sm100_fold_candidates.candidates_hash()})
    if "resinfer_mxfp4_mext_r1_tuned" in resolved["arms"]:
        prov["mxfp4_candidate_hash"] = mxfp4_linear_adapter.candidates_hash()
        prov["mxfp4_fold_pdl_mode"] = "early"
    if trtllm_fp4_utils is not None:
        prov["trtllm_version"] = importlib.metadata.version("tensorrt-llm")
    extension = importlib.util.find_spec("nvfp4._C")
    if extension is None or extension.origin is None:
        raise RuntimeError("nvfp4._C is not installed; build the patched extension first")
    with open(extension.origin, "rb") as binary:
        prov["loaded_nvfp4_so_sha256"] = hashlib.file_digest(binary, "sha256").hexdigest()
    prov["loaded_nvfp4_so"] = extension.origin
    driver = subprocess.run(["nvidia-smi", "-i", str(physical_index(0)),
                             "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=False)
    prov["driver_version"] = driver.stdout.strip() if driver.returncode == 0 else None
    stem = args.shard_stem or args.outdir / f"rep{args.rep}"
    pathlib.Path(f"{stem}.provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    output = pathlib.Path(f"{stem}.jsonl")
    shapes = [shape for shape in resolved["shapes"]
              if args.shape_id is None or shape["shape_id"] == args.shape_id]
    if len(shapes) != 1:
        raise RuntimeError("worker requires exactly one resolved physical shape")
    with output.open("x") as stream:
        def emit(row: dict) -> None:
            stream.write(json.dumps(row) + "\n")
            stream.flush()

        for shape in shapes:
            n, k = shape["n"], shape["k"]
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
            if trtllm_fp4_utils is not None:
                w_trt, ws_trt, wgs_trt = prepare_trtllm_weight(weight, trtllm_fp4_utils)
            mask = torch.empty(1, device=device, dtype=torch.uint8)
            for m in resolved["m_values"]:
                assert_gpu_exclusive(0)
                x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
                xgs = (448.0 * 6.0 / x.float().abs().max()).reshape(())
                alpha = (1.0 / (xgs * wgs)).reshape(1).float()
                if trtllm_fp4_utils is not None:
                    # TRT's static FP8 op takes the dequantization scale
                    # (amax / 448); the FP4 weight producer takes its inverse.
                    x_scale_trt = (x.float().abs().max() / 448.0).reshape(1).float()
                    alpha_trt = (x_scale_trt / wgs_trt).reshape(1).float()
                reference = x @ weight.T
                outputs = {arm: torch.empty(m, n, device=device, dtype=torch.bfloat16)
                           for arm in resolved["arms"] if arm != "mext_r1"}
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
                if trtllm_fp4_utils is not None:
                    runners["trtllm_nvfp4_fp8_w4a8"] = (
                        trtllm_w4a8, (x, w_trt, ws_trt, x_scale_trt, alpha_trt))
                niw = iters_for_cold_l2(runners["fi_nvfp4_cutlass"][1], device)
                l2 = l2_accounting(runners["fi_nvfp4_cutlass"][1], device, niw)
                if l2.get("cold_l2_effective") is not True:
                    raise RuntimeError(f"cold-L2 ineffective for {shape['shape_id']} M={m}: {l2}")
                base = {"schema": 1, "kind": "measure", "run_id": resolved["run_id"],
                        "rep": args.rep, "shape_id": shape["shape_id"],
                        "n": n, "k": k, "m": m, "aliases": shape["aliases"],
                        "niw": niw, **l2}
                # Interleave arm order between M points and process reps.
                active = [a for a in resolved["arms"] if a in runners]
                shift = (args.rep + resolved["m_values"].index(m)) % len(active)
                active = active[shift:] + active[:shift]
                for arm in active:
                    fn, fn_args = runners[arm]
                    fmt_w, fmt_a, provider = ARM_FORMAT[arm]
                    row = dict(base, arm=arm, weight_format=fmt_w,
                               activation_format=fmt_a, provider=provider,
                               physical_m=padded_m if arm == "fi_mxfp4_mxfp8_w4a8" else m)
                    try:
                        assert_args_consumed(fn, fn_args, label=arm)
                        fi_autotuned(fn, fn_args, m_bucket=m)
                        warm_equally(fn, fn_args, calls=20)
                        actual = fn(*fn_args)
                        torch.cuda.synchronize()
                        if actual.shape != reference.shape or not bool(torch.isfinite(actual).all()):
                            raise RuntimeError("wrong output shape or non-finite values")
                        rel = float((actual.float() - reference.float()).abs().max()) / (
                            float(reference.float().abs().max()) + 1e-9)
                        if rel > 0.25:
                            raise RuntimeError(f"BF16 sanity gate failed: relative max={rel:.4f}")
                        row["relmax_bf16"] = rel
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
    return 0


def main() -> int:
    args = _arguments()
    return _worker(args) if args.worker else _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
