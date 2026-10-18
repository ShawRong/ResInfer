#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Serving speed sweep using vllm serve and vllm bench serve.

Uses a manifest-defined fixed-length workload and concurrency grid, rotates
checkpoint arms across rounds, and requires a fresh output directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import time
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks import manifest as spec_mod
from benchmarks._runtime import assert_gpu_exclusive, runtime_provenance
from benchmarks.serving import engagement, stop_server, check_port_available, wait_ready
from benchmarks.checkpoints import validate_kind, validate_context

REPO = pathlib.Path(__file__).resolve().parents[2]


def _vllm_cli() -> str:
    """The `vllm` executable, resolved from the environment, not the repo."""
    beside = pathlib.Path(sys.executable).parent / "vllm"
    if beside.exists():
        return str(beside)
    found = shutil.which("vllm")
    if found:
        return found
    raise SystemExit(
        "no `vllm` executable found next to "
        f"{sys.executable} or on PATH. Run this script with the python of "
        "the environment vLLM is installed in.")

def provenance(gpu: int, neighbors: list) -> dict:
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,clocks.sm,temperature.gpu",
         "--format=csv,noheader", "-i", str(gpu)],
        capture_output=True, text=True)
    return {**runtime_provenance(), "gpu_index": gpu, "gpu": q.stdout.strip(),
            "gpu_neighbors": neighbors, "vllm_cli": _vllm_cli()}


CONCURRENCIES = (1, 4, 16, 64)
INPUT_LEN = 1024
OUTPUT_LEN = 256


def _num_prompts(conc: int) -> int:
    # Multiple request waves at every concurrency, including large batches.
    return max(32, 8 * conc)


def assert_gpu_free(gpu: int, need_gib: float = 20.0,
                    exclusive: bool = True) -> list:
    """Refuse to measure unless the GPU has both the room AND no neighbours."""
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True, check=True)
    used, total = (int(x) for x in q.stdout.strip().split(", "))
    free_gib = (total - used) / 1024
    if free_gib < need_gib:
        raise SystemExit(
            f"gpu{gpu} has only {free_gib:.1f} GiB free; need ~{need_gib}. "
            "Check for orphaned vLLM EngineCore (kill -9 -$pgid, not the pid).")

    return assert_gpu_exclusive(gpu, exclusive=exclusive, physical=True)


_MEM_PATTERNS = (
    ("weights_gib", r"Model loading took ([0-9.]+) GiB", float),
    ("kv_cache_gib", r"Available KV cache memory: ([0-9.]+) GiB", float),
    ("kv_cache_tokens", r"GPU KV cache size: ([0-9,]+) tokens",
     lambda v: int(v.replace(",", ""))),
)


def gpu_state(gpu: int) -> dict:
    """Record clocks, temperature and throttling for each arm."""
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,"
         "power.draw,clocks_event_reasons.active",
         "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True)
    row = (q.stdout or "").strip().splitlines()
    if not row:
        return {}
    v = [x.strip() for x in row[0].split(",")]
    keys = ("clock_sm_mhz", "clock_max_sm_mhz", "temp_c", "power_w",
            "throttle_reasons")
    return {f"{k}_at_arm": val for k, val in zip(keys, v)}



def parse_memory(log_text: str) -> dict:
    """Weight footprint and KV-cache headroom, read off vLLM's own startup log."""
    out = {}
    for key, pattern, conv in _MEM_PATTERNS:
        m = re.search(pattern, log_text)
        out[key] = conv(m.group(1)) if m else None
    return out


def _gpu_used_mib(gpu: int) -> int:
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu)], capture_output=True, text=True)
    try:
        return int(q.stdout.strip())
    except ValueError:
        return -1


def _shutdown(proc: subprocess.Popen, gpu: int, settle_s: int = 90) -> None:
    """Stop the server AND its EngineCore, then wait for the GPU to come back."""
    stop_server(proc)

    t0 = time.time()
    while time.time() - t0 < settle_s:
        if _gpu_used_mib(gpu) < 2000:
            return
        time.sleep(3)
    print(f"[warn] gpu{gpu} still holds {_gpu_used_mib(gpu)} MiB after "
          f"{settle_s}s. Look for an orphaned EngineCore:\n"
          f"  nvidia-smi -i {gpu} --query-compute-apps=pid --format=csv\n"
          f"  kill -9 -$(ps -o pgid= -p <pid> | tr -d ' ')", flush=True)


def _bench(model: str, port: int, conc: int, out_json: str, env: dict) -> dict:
    if pathlib.Path(out_json).exists():
        raise FileExistsError(f"Refusing to reuse benchmark result: {out_json}")
    cmd = [
        sys.executable, _vllm_cli(), "bench", "serve",
        "--backend", "vllm", "--model", model,
        "--host", "127.0.0.1", "--port", str(port),
        "--dataset-name", "random",
        "--random-input-len", str(INPUT_LEN),
        "--random-output-len", str(OUTPUT_LEN),
        "--random-range-ratio", "0.0",
        "--ignore-eos", "--request-rate", "inf",
        "--max-concurrency", str(conc),
        "--num-prompts", str(_num_prompts(conc)),
        "--save-result", "--save-detailed", "--result-filename", out_json,
    ]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=1800)
        tail = ((p.stdout or "") + (p.stderr or ""))[-2000:]
        if p.returncode:
            return {"error": f"bench exited {p.returncode}", "tail": tail}
    except subprocess.TimeoutExpired:
        return {"error": "bench timed out"}
    try:
        with open(out_json) as f:
            result = json.load(f)
        if result.get("completed") != _num_prompts(conc) or result.get("failed", 0):
            raise ValueError("not all requests completed")
        expected = _num_prompts(conc)
        for phase, length in (("input", INPUT_LEN), ("output", OUTPUT_LEN)):
            lengths = result.get(f"{phase}_lens")
            if (not isinstance(lengths, list) or len(lengths) != expected
                    or any(type(n) is not int or n != length for n in lengths)
                    or result.get(f"total_{phase}_tokens") != expected * length):
                raise ValueError(f"actual {phase} lengths differ from the fixed workload")
        for key in _KEEP:
            value = result[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid metric: {key}")
        if result["output_throughput"] <= 0:
            raise ValueError("no output throughput")
        return result
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        return {"error": f"invalid result: {exc}", "tail": tail}


_KEEP = ("mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
         "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
         "mean_itl_ms", "p99_itl_ms",
         "output_throughput", "total_token_throughput",
         "request_throughput", "completed")


def run_arm(tag: str, ratio: float, quant: str, model: str, gpu: int,
            port: int, outdir: pathlib.Path,
            ready_timeout: int = 2400,
            neighbors: list | None = None,
            hygiene_env: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("RESIDUE_", "RESINFER_DEV_"))}
    # "<unset>" explicitly removes an inherited environment variable.
    for k, v in (hygiene_env or {}).items():
        if str(v) == "<unset>":
            env.pop(k, None)
        else:
            env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), str(REPO / "csrc")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    # Do not let a previous arm's format gates leak into this server.
    for key in ("RESIDUE_NVFP4", "RESIDUE_MXFP4", "RESIDUE_PATCH_MODE",
                "RESIDUE_ENABLE_DECODE_MEXT", "RESIDUE_SHARE_AUTOTUNE_CACHE",
                "RESIDUE_VLLM_MODEL_DIR"):
        env.pop(key, None)
    if quant == "residue_nvfp4":
        env["RESIDUE_NVFP4"] = "1"
        env["RESIDUE_ENABLE_DECODE_MEXT"] = "1"
    elif quant == "modelopt_mxfp4":
        env["RESIDUE_MXFP4"] = "1"
    if quant in ("residue_nvfp4", "modelopt_mxfp4"):
        env["RESIDUE_VLLM_MODEL_DIR"] = model

    rec = {"tag": tag, "ratio": ratio, "quant": quant, "gpu": gpu,
           "model": model, "runs": {},
           "residue_env": {k: v for k, v in sorted(env.items())
                           if k.startswith("RESIDUE_")}}

    validate_context(model, INPUT_LEN + OUTPUT_LEN)
    log_path = outdir / f"serve_{tag}.log"
    cmd = [
        sys.executable, _vllm_cli(), "serve", model,
        # Omitted entirely for an unquantised reference arm -- vLLM then reads
        # the checkpoint's own config, which is what a bf16 ceiling needs.
        *(["--quantization", "modelopt_fp4" if quant == "residue_nvfp4" else quant]
          if quant else []),
        "--dtype", "bfloat16",
        "--max-model-len", str(INPUT_LEN + OUTPUT_LEN),
        "--generation-config", "vllm",
        "--gpu-memory-utilization", "0.85",
        "--port", str(port),
        "--trust-remote-code",
    ]
    print(f"[{tag}] gpu{gpu} port{port} launching ({quant}, budget={ratio}), "
          f"conc={CONCURRENCIES}", flush=True)
    check_port_available(port)
    with open(log_path, "wb") as logf:
        
        
        
        
        
        
        proc = subprocess.Popen(cmd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            ready_s = wait_ready(port, ready_timeout, proc, model)
            if ready_s is None:
                rec["error"] = (f"serve not ready in {ready_timeout}s "
                                f"(check the log)")
                return rec
            rec["ready_s"] = round(ready_s, 1)
            early = log_path.read_text(errors="ignore")
            eng = engagement(early, proc.pid, quant=quant,
                             multiprocessing=env.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0")
            rec.update(eng)
            if eng.get("engagement_error"):
                rec["error"] = eng["engagement_error"]
                return rec
            print(f"[{tag}] ready in {ready_s:.0f}s", flush=True)
            for conc in CONCURRENCIES:
                rj = outdir / f"{tag}_c{conc}.json"
                r = _bench(model, port, conc, str(rj), env)
                rec["runs"][str(conc)] = (
                    {k: r.get(k) for k in _KEEP} if "error" not in r else r)
                if "error" in r:
                    rec["error"] = f"c={conc}: {r['error']}"
                    return rec
                got = rec["runs"][str(conc)]
                print(f"[{tag}] c={conc:<3} "
                      f"ttft={got.get('mean_ttft_ms')} "
                      f"tpot={got.get('mean_tpot_ms')} "
                      f"out_tps={got.get('output_throughput')}", flush=True)
        finally:
            _shutdown(proc, gpu)

    log = log_path.read_text(errors="ignore")
    if "CUDA Error:" in log:
        rec["error"] = "CUDA driver error in server log"
    rec["gpu_neighbors"] = neighbors
    rec.update(parse_memory(log))
    return rec


def validate_workload(e: dict) -> None:
    workload = e.get("workload", {})
    expected = {"dataset": "random", "request_rate": "inf", "ignore_eos": True}
    for key, value in expected.items():
        actual = workload.get(key, value)
        if key == "request_rate":
            actual = str(actual).lower()
        if actual != value or (key == "ignore_eos" and actual is not True):
            raise ValueError(f"Unsupported workload {key}={actual!r}; expected {value!r}")
    for key, default in (("input_len", INPUT_LEN), ("output_len", OUTPUT_LEN)):
        value = workload.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    conc = e.get("concurrencies", CONCURRENCIES)
    if not conc or any(type(n) is not int or n <= 0 for n in conc):
        raise ValueError("concurrencies must contain positive integers")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--manifest", default=None,
                    help="benchmark spec YAML (default: the one shipped in "
                         "benchmarks/suites/, default speed_dense.yaml). It fixes the workload, the "
                         "concurrency grid and WHICH arms must be present; "
                         "--models-root supplies where each one lives.")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--allow-shared-gpu", action="store_true",
                    help="run even when another process holds the GPU. This "
                         "does NOT make the timings clean -- every row is "
                         "annotated with the neighbouring PIDs so a reader can "
                         "judge. A 1.4 GiB neighbour passes any memory check "
                         "and still takes SMs.")
    ap.add_argument("--ready-timeout", type=int, default=2400,
                    help='Seconds to wait for /health. Prewarm the tactic cache or increase this timeout when cold autotuning is enabled.')
    ap.add_argument("--models-root", default=None,
                    help="directory holding the exports. Each arm's `path` in "
                         "the manifest is resolved against it, which is what "
                         "keeps the manifest free of machine-specific paths.")
    a = ap.parse_args()

    doc = spec_mod.load(a.manifest)
    global CONCURRENCIES, INPUT_LEN, OUTPUT_LEN
    e = doc.get("e2e", {})
    CONCURRENCIES = tuple(e.get("concurrencies", CONCURRENCIES))
    wl = e.get("workload", {})
    validate_workload(e)
    INPUT_LEN = wl.get("input_len", INPUT_LEN)
    OUTPUT_LEN = wl.get("output_len", OUTPUT_LEN)
    if a.ready_timeout == 2400 and e.get("ready_timeout_s"):
        a.ready_timeout = int(e["ready_timeout_s"])

    if not a.models_root:
        raise SystemExit(
            f"--models-root is required. Manifest {doc['name']} expects these "
            f"directories under it:\n" +
            "\n".join(f"    {x['id']:<8} {x['path']}" for x in doc["arms"]))
    resolved = spec_mod.resolve_arms(doc, a.models_root)
    arms = [(r["id"], float(r.get("bits", 0.0)),
             validate_kind(r["abs_path"], r["kind"]), r["abs_path"]) for r in resolved]
    print(f"[manifest] {doc['name']}  arms={len(arms)}  "
          f"conc={list(CONCURRENCIES)}  isl={INPUT_LEN} osl={OUTPUT_LEN}",
          flush=True)
    outdir = pathlib.Path(a.outdir or
                          f"runs/residue_e2e_"
                          f"{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=False)
    neighbors = assert_gpu_free(a.gpu, exclusive=not a.allow_shared_gpu)

    rounds = int(e.get("rounds", 1) or 1)
    rotate = bool(e.get("rotate_arms", False))
    with open(outdir / "spec_stamp.json", "w") as fh:
        json.dump({**spec_mod.stamp(doc), **provenance(a.gpu, neighbors),
                   "concurrencies": list(CONCURRENCIES),
                   "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
                   "ready_timeout_s": a.ready_timeout}, fh, indent=2)
    summary = outdir / "summary.jsonl"
    for rnd in range(rounds):
        # Rotate to distribute temperature and other time-dependent drift.
        order = arms[rnd % len(arms):] + arms[:rnd % len(arms)] if rotate else arms
        rdir = outdir / f"round{rnd}" if rounds > 1 else outdir
        rdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[round {rnd + 1}/{rounds}] order="
              f"{[t for t, *_ in order]}", flush=True)
        for slot, (tag, ratio, quant, model) in enumerate(order):
            neighbors = assert_gpu_free(a.gpu, exclusive=not a.allow_shared_gpu)
            before = gpu_state(a.gpu)
            rec = run_arm(tag, ratio, quant, model, a.gpu, a.port, rdir,
                          ready_timeout=a.ready_timeout,
                          neighbors=neighbors,
                          hygiene_env=(doc.get("hygiene", {}) or {}).get("env"))
            # Before AND after: one sample cannot show drift, and drift is
            # what an unlocked multi-hour sweep has to be read against.
            rec["gpu_before"] = before
            rec["gpu_after"] = gpu_state(a.gpu)
            # `round` and `arm_slot` are what make the rotation analysable:
            # without the slot you cannot tell a real arm effect from a
            # position effect, which is the whole reason for rotating.
            rec["round"] = rnd
            rec["arm_slot"] = slot
            with open(summary, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if "error" in rec:
                raise RuntimeError(f"[{tag}] {rec['error']}")

    print(f"\ndone -> {summary}  ({rounds} round(s))", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
