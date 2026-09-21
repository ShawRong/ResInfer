'Resumable GQA4/GQA6 stock FlashInfer versus ResAttn decode grid.'
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_BATCHES = (1, 4, 16, 32, 64, 128)
DEFAULT_SEQS = (1024, 4096, 8192, 16384, 32768)
DEFAULT_GQAS = (4, 6)
DEFAULT_CONTRACTS = ("bf16", "nvfp4")


def gpu_snapshot() -> dict:
    command = ["nvidia-smi", "--query-gpu=index,uuid,clocks.sm,temperature.gpu,"
               "memory.used,memory.total", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=20)
    return {"returncode": result.returncode,
            "output": result.stdout.strip() if result.returncode == 0 else
            result.stderr.strip()[:1000]}


def run_cell(probe: Path, *, gqa: int, batch: int, seq: int,
             contract: str, timeout: int, iterations: int,
             repeats: int, reference_gate_arm: str,
             reference_min_cosine: float,
             reference_min_median_cosine: float,
             reference_max_relative_rms: float) -> dict:
    command = [sys.executable, str(probe), "--gqa", str(gqa), "--batch",
               str(batch), "--seq", str(seq), "--output-contract", contract,
               "--reference-all-heads", "--iterations", str(iterations),
               "--repeats", str(repeats), "--reference-gate-arm",
               reference_gate_arm, "--reference-min-cosine",
               str(reference_min_cosine),
               "--reference-min-median-cosine",
               str(reference_min_median_cosine),
               "--reference-max-relative-rms",
               str(reference_max_relative_rms)]
    started = datetime.now(timezone.utc).isoformat()
    before = gpu_snapshot()
    t0 = time.monotonic()
    try:
        process = subprocess.run(command, text=True, capture_output=True,
                                 timeout=timeout)
        stdout, stderr, code = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(
            exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(
            exc.stderr, bytes) else (exc.stderr or "")
        code = 124
    parsed = None
    for line in reversed(stdout.splitlines()):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    status = ("ok" if code == 0 and isinstance(parsed, dict) and
              parsed.get("status") == "SYNTHETIC_CHAIN_OK" else
              "correctness_failed" if isinstance(parsed, dict) and
              parsed.get("status") == "CORRECTNESS_FAILED" else
              "timeout" if code == 124 else "failed")
    return {"status": status, "gqa": gqa, "batch": batch, "seq": seq,
            "output_contract": contract, "started_utc": started,
            "elapsed_seconds": round(time.monotonic() - t0, 3),
            "command": command, "exit_code": code, "gpu_before": before,
            "gpu_after": gpu_snapshot(), "result": parsed,
            "stdout_tail": stdout[-3000:] if status != "ok" else None,
            "stderr_tail": stderr[-6000:] if status != "ok" else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=Path(__file__).with_name(
        "resattn_stock_ab.py"))
    ap.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    ap.add_argument("--seqs", type=int, nargs="+", default=DEFAULT_SEQS)
    ap.add_argument("--gqas", type=int, nargs="+", default=DEFAULT_GQAS)
    ap.add_argument("--contracts", nargs="+", choices=DEFAULT_CONTRACTS,
                    default=DEFAULT_CONTRACTS)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--reference-gate-arm", choices=("both", "resattn"),
                    default="both")
    ap.add_argument("--reference-min-cosine", type=float, default=.98)
    ap.add_argument("--reference-min-median-cosine", type=float, default=.98)
    ap.add_argument("--reference-max-relative-rms", type=float, default=.15)
    ap.add_argument("--retry-from", type=Path,
                    help="run only failed or stock A/A drift >2%% cells from this directory")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="0 runs all missing cells; positive limits this invocation")
    args = ap.parse_args()
    if (not os.getenv("CUDA_VISIBLE_DEVICES") or
            not os.getenv("FLASHINFER_CUBIN_DIR") or
            os.getenv("FLASHINFER_NO_DOWNLOAD") != "1"):
        ap.error("set CUDA_VISIBLE_DEVICES, FLASHINFER_CUBIN_DIR, "
                 "FLASHINFER_NO_DOWNLOAD=1 explicitly")
    if any(x < 1 for x in (*args.batches, *args.seqs, args.timeout,
                           args.iterations)) or args.repeats < 2 or any(
            x not in DEFAULT_GQAS for x in args.gqas):
        ap.error("batch, seq and timeout must be positive; GQA must be 4 or 6")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    probe = args.probe.resolve(strict=True)
    probe_sha256 = hashlib.sha256(probe.read_bytes()).hexdigest()
    grid = [(g, b, s, c) for g in args.gqas for b in args.batches
            for s in args.seqs for c in args.contracts]
    if args.retry_from:
        grid = []
        for path in sorted(args.retry_from.glob("*.json")):
            prior = json.loads(path.read_text())
            aa = (prior.get("result") or {}).get("stock_aa_ratio")
            if prior["status"] != "ok" or aa is None or abs(aa - 1) > .02:
                grid.append((prior["gqa"], prior["batch"], prior["seq"],
                             prior["output_contract"]))
    launched = 0
    for gqa, batch, seq, contract in grid:
        name = f"gqa{gqa}-b{batch}-s{seq}-{contract}.json"
        target = args.result_dir / name
        if target.exists():
            continue
        if args.max_cells and launched >= args.max_cells:
            break
        record = run_cell(probe, gqa=gqa, batch=batch, seq=seq,
                          contract=contract, timeout=args.timeout,
                          iterations=args.iterations, repeats=args.repeats,
                          reference_gate_arm=args.reference_gate_arm,
                          reference_min_cosine=args.reference_min_cosine,
                          reference_min_median_cosine=
                          args.reference_min_median_cosine,
                          reference_max_relative_rms=
                          args.reference_max_relative_rms)
        record["probe_sha256"] = probe_sha256
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary.rename(target)
        launched += 1
        print(json.dumps({"cell": name, "status": record["status"],
                          "elapsed_seconds": record["elapsed_seconds"],
                          "ratio": (record["result"] or {}).get(
                              "stock_over_resattn")}), flush=True)


if __name__ == "__main__":
    main()
