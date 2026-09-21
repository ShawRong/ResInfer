"""Resumable CUDA Graph BF16-output real-model ResAttn E2E grid.

Four server rounds run stock, ResAttn, ResAttn, stock. Every round visits the
same input-length/concurrency cells, reversing cell order on alternate rounds.
Only processes started by this driver are signalled during cleanup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from resattn_e2e_plan import build_plan


def gpu_snapshot(index: str) -> dict:
    command = ["nvidia-smi", "-i", index,
               "--query-gpu=uuid,clocks.sm,temperature.gpu,memory.used,memory.total",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=20)
    return {"returncode": result.returncode,
            "output": result.stdout.strip() if result.returncode == 0 else
            result.stderr.strip()[:1000]}


def wait_gpu_idle(index: str, timeout: int = 90) -> None:
    command = ["nvidia-smi", "-i", index, "--query-compute-apps=pid",
               "--format=csv,noheader,nounits"]
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = subprocess.run(command, text=True, capture_output=True,
                                timeout=20)
        if result.returncode != 0:
            raise RuntimeError(f"GPU ownership query failed: {result.stderr[:500]}")
        if not result.stdout.strip():
            return
        time.sleep(3)
    raise RuntimeError(f"GPU {index} still has compute processes: "
                       f"{result.stdout.strip()}")


def require_port_free() -> None:
    try:
        connection = socket.create_connection(("127.0.0.1", 8123), timeout=2)
    except OSError:
        return
    connection.close()
    raise RuntimeError("port 8123 is occupied; refusing to benchmark another server")


def ready(process: subprocess.Popen, timeout: int) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup: {process.returncode}")
        try:
            with urlopen("http://127.0.0.1:8123/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError("server /health did not become ready")


def stop(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=15)


def single_result(directory: Path) -> dict:
    files = list(directory.glob("*.json"))
    if len(files) != 1:
        raise RuntimeError(f"expected one vLLM result JSON in {directory}: {files}")
    result = json.loads(files[0].read_text())
    if result.get("failed") or result.get("completed") != result.get("num_prompts"):
        raise RuntimeError(f"incomplete serving run: {result.get('completed')}/"
                           f"{result.get('num_prompts')}, failed={result.get('failed')}")
    return {"result_file": str(files[0]),
            "completed": result["completed"], "failed": result["failed"],
            "total_input_tokens": result["total_input_tokens"],
            "total_output_tokens": result["total_output_tokens"],
            "observed_input_lens": sorted(set(result["input_lens"])),
            "observed_output_lens": sorted(set(result["output_lens"])),
            **{key: result[key] for key in (
                "mean_tpot_ms", "mean_itl_ms", "mean_ttft_ms",
                "mean_e2el_ms", "output_throughput")}}


def iteration_trace(log_text: str, input_len: int, output_len: int) -> dict:
    pattern = re.compile(r"Iteration\(\d+\): \d+ context requests, "
                         r"\d+ context tokens, (\d+) generation requests, "
                         r"\d+ generation tokens")
    histogram: dict[int, int] = {}
    for match in pattern.finditer(log_text):
        batch = int(match.group(1))
        if batch:
            histogram[batch] = histogram.get(batch, 0) + 1
    return {"generation_batch_histogram": histogram,
            "generation_iterations": sum(histogram.values()),
            "kv_length_bounds": ([input_len + 1,
                                  input_len + output_len - 1]
                                 if output_len > 1 else None),
            "exact_per_request_kv_lengths_logged": False}


def derived_decode_kv_lengths(result_file: str) -> dict[int, int]:
    """Lengths implied by completed outputs; not a per-step scheduler trace."""
    result = json.loads(Path(result_file).read_text())
    histogram: dict[int, int] = {}
    for input_len, output_len in zip(result["input_lens"], result["output_lens"]):
        # First output token comes from prefill; subsequent tokens decode.
        for kv_len in range(input_len + 1, input_len + output_len):
            histogram[kv_len] = histogram.get(kv_len, 0) + 1
    return histogram


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--cubin-dir", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--input-lens", type=int, nargs="+",
                    default=(1024, 8192, 32768))
    ap.add_argument("--concurrencies", type=int, nargs="+", default=(1, 4, 16))
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--bench-timeout", type=int, default=1800)
    ap.add_argument("--kv-cache-bytes", type=int)
    ap.add_argument("--rounds", type=int, nargs="+", default=(1, 2, 3, 4))
    ap.add_argument("--trace-scheduler", action="store_true",
                    help="separate diagnostic run; iteration logging perturbs latency")
    args = ap.parse_args()
    if not args.cubin_dir.is_dir() or not args.model.is_dir():
        ap.error("model and verified cubin directory must exist")
    if any(x < 1 for x in (*args.input_lens, *args.concurrencies,
                            args.output_len)):
        ap.error("lengths and concurrency must be positive")
    config_bytes = (args.model / "config.json").read_bytes()
    config = json.loads(config_bytes)
    model_limit = config.get("text_config", config)["max_position_embeddings"]
    max_len = max(args.input_lens) + args.output_len
    if max_len > model_limit:
        ap.error(f"requested context {max_len} exceeds model limit {model_limit}")
    root = args.result_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cells = [(length, concurrency) for length in args.input_lens
             for concurrency in args.concurrencies]
    rounds = ((1, "stock"), (2, "resattn"),
              (3, "resattn"), (4, "stock"))
    for number, arm in rounds:
        if number not in args.rounds:
            continue
        round_dir = root / f"round{number}-{arm}"
        round_dir.mkdir(exist_ok=True)
        planned = build_plan(
            str(args.model), config, python=args.python, gpu=args.gpu,
            input_len=max(args.input_lens), output_len=args.output_len,
            concurrency=max(args.concurrencies), max_model_len=max_len,
            offline_cubin_dir=str(args.cubin_dir),
            kv_cache_bytes=args.kv_cache_bytes)
        spec = planned["arms"][arm]
        server_command = list(spec["server_argv"])
        if args.trace_scheduler:
            server_command.append("--enable-logging-iteration-details")
        wait_gpu_idle(args.gpu)
        require_port_free()
        env = os.environ.copy()
        for key in spec["env_unset"]:
            env.pop(key, None)
        env.update(spec["env_set"])
        env["RESIDUE_ATTN_REDUCTION"] = "auto" if arm == "resattn" else ""
        if arm == "stock":
            env.pop("RESIDUE_ATTN_REDUCTION")
        server_log = round_dir / "server.log"
        with server_log.open("a") as output:
            output.write(f"\nSTART {datetime.now(timezone.utc).isoformat()}\n")
            output.flush()
            process = subprocess.Popen(server_command, env=env,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                ready(process, args.startup_timeout)
                order = cells if number % 2 else list(reversed(cells))
                for length, concurrency in order:
                    cell = round_dir / f"input{length}-c{concurrency}"
                    summary_file = cell / "summary.json"
                    if summary_file.exists():
                        continue
                    cell.mkdir(exist_ok=True)
                    result_dir = cell / ("vllm-results-" + datetime.now(
                        timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
                    result_dir.mkdir(exist_ok=True)
                    cell_plan = build_plan(
                        str(args.model), config, python=args.python,
                        gpu=args.gpu, input_len=length,
                        output_len=args.output_len, concurrency=concurrency,
                        max_model_len=max_len,
                        offline_cubin_dir=str(args.cubin_dir),
                        kv_cache_bytes=args.kv_cache_bytes)
                    command = cell_plan["benchmark_argv"] + [
                        "--result-dir", str(result_dir), "--num-prompts",
                        str(max(32, concurrency * 4))]
                    started = datetime.now(timezone.utc).isoformat()
                    gpu_before = gpu_snapshot(args.gpu)
                    log_offset = server_log.stat().st_size
                    with (cell / "bench.log").open("w") as bench_log:
                        try:
                            bench = subprocess.run(command, env=env, stdout=bench_log,
                                                   stderr=subprocess.STDOUT,
                                                   timeout=args.bench_timeout)
                            status = "ok" if bench.returncode == 0 else "failed"
                            exit_code = bench.returncode
                        except subprocess.TimeoutExpired:
                            status, exit_code = "timeout", 124
                    summary = {"status": status, "arm": arm, "round": number,
                               "input_len": length, "concurrency": concurrency,
                               "output_len": args.output_len, "started_utc": started,
                               "model_config_sha256": hashlib.sha256(
                                   config_bytes).hexdigest(),
                               "server_command": server_command,
                               "bench_command": command,
                               "exit_code": exit_code,
                               "trace_only": args.trace_scheduler,
                               "gpu_before": gpu_before,
                               "gpu_after": gpu_snapshot(args.gpu)}
                    if args.trace_scheduler:
                        time.sleep(1)
                        with server_log.open("rb") as log_file:
                            log_file.seek(log_offset)
                            summary["scheduler_trace"] = iteration_trace(
                                log_file.read().decode(errors="replace"),
                                length, args.output_len)
                        if (status == "ok" and not summary["scheduler_trace"]
                                ["generation_iterations"]):
                            summary["status"] = "trace_missing"
                    if status == "ok":
                        try:
                            summary.update(single_result(result_dir))
                            if args.trace_scheduler:
                                summary["scheduler_trace"][
                                    "derived_decode_kv_length_histogram"] = (
                                    derived_decode_kv_lengths(
                                        summary["result_file"]))
                        except Exception as exc:
                            summary.update(status="invalid_result", error=str(exc))
                    temporary = summary_file.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(summary, indent=2,
                                                    sort_keys=True) + "\n")
                    temporary.rename(summary_file)
                    print(json.dumps({"round": number, "arm": arm,
                                      "input_len": length,
                                      "concurrency": concurrency,
                                      "status": summary["status"],
                                      "mean_tpot_ms": summary.get(
                                          "mean_tpot_ms")}), flush=True)
            finally:
                stop(process)
        server_text = server_log.read_text(errors="replace")
        evidence = {
            "arm": arm, "round": number,
            "stock_trtllm_gen_selected": "decode_backend=trtllm-gen" in server_text,
            "custom_backend_registered": "CUSTOM" in server_text,
            "cuda_graph_mentioned": "CUDA graph" in server_text or
                                    "CUDA Graph" in server_text,
            "server_log": str(server_log),
        }
        (round_dir / "server-evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
