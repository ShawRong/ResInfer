# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-server AIME/GPQA driver with isolated per-arm caches and checked results.

The external lighteval 0.13 environment owns tasks and metrics. Its scheduled
items are recorded before generation and checked against the saved details.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from benchmarks.engine_config import accuracy_engine_config
from benchmarks.serving import engagement, stop_server, check_port_available, wait_ready
from benchmarks.quality.common.protocol import load_manifest, resolve_arms, SAMPLING_FIELDS
from benchmarks.quality.common.runtime import configure_env
SERVING_PYTHON = pathlib.Path(sys.executable)
_SERVERS: list[subprocess.Popen] = []


def teardown() -> None:
    for proc in _SERVERS:
        stop_server(proc)
    _SERVERS.clear()


def on_signal(signum, _frame) -> None:
    print(f"[lighteval_driver] got signal {signum}, tearing down", flush=True)
    teardown()
    raise SystemExit(143)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def strip_proxies(env: dict) -> None:
    # lighteval -> litellm -> httpx routes through any configured proxy, even
    # for localhost when the proxy is SOCKS (NO_PROXY does not suppress SOCKS).
    # Symptom when not stripped: every request fails, all scores come back 0/N.
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1"
    for k in ("http_proxy", "https_proxy", "all_proxy", "ftp_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
              "socks_proxy", "SOCKS_PROXY"):
        env.pop(k, None)


def write_json(path: pathlib.Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def build_server_env(arm: dict, a) -> dict:
    env = os.environ.copy()
    strip_proxies(env)
    # Start from a clean residue slate: an inherited RESIDUE_* would make the
    # served arm differ from what this file constructs AND records.
    for k in [k for k in env if k.startswith("RESIDUE_")]:
        env.pop(k)
    # Drop the obsolete backend override from inherited environments.
    if env.pop("VLLM_NVFP4_GEMM_BACKEND", None) is not None:
        print("[lighteval_driver] dropped inherited VLLM_NVFP4_GEMM_BACKEND "
              "(dead knob; distorts baselines)", flush=True)

    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    if arm["kind"] == "residue" and arm.get("mext", "on") != "on":
        raise ValueError("Runtime ablations belong in .dev; use a stock NVFP4 checkpoint")
    configure_env(env, model=arm["abs_path"], kind=arm["kind"], repo=REPO,
                  mext_max_m=getattr(a, "mext_max_m", 64))
    return env


def build_server_cmd(arm: dict, man: dict, a, port: int) -> list[str]:
    template_kwargs = {"enable_thinking": man["enable_thinking"]}
    if "reasoning_effort" in man:
        template_kwargs["reasoning_effort"] = man["reasoning_effort"]
    cmd = [str(SERVING_PYTHON), "-m", "benchmarks.quality.common.vllm_server",
           "serve", arm["abs_path"],
           "--port", str(port),
           "--dtype", a.dtype,
           "--generation-config", "vllm",
           "--compilation-config", json.dumps(accuracy_engine_config({}, {})["compilation_config"]),
           "--default-chat-template-kwargs", json.dumps(template_kwargs),
           "--seed", str(man["seed"]),
           "--max-model-len", str(man["max_model_len"]),
           "--max-num-batched-tokens", str(a.max_num_batched_tokens),
           "--max-num-seqs", str(a.max_num_seqs),
           "--gpu-memory-utilization", str(a.gpu_mem_util),
           "--no-enable-flashinfer-autotune"]
    if arm["kind"] in ("residue", "modelopt_fp4"):
        cmd += ["--quantization", "modelopt_fp4"]
    elif arm["kind"] == "modelopt_mxfp4":
        cmd += ["--quantization", "modelopt_mxfp4"]
    return cmd


def launch_server(cmd: list[str], env: dict, case_out: pathlib.Path,
                  port: int, timeout_s: int, model: str) -> subprocess.Popen:
    check_port_available(port)
    with (case_out / "server.log").open("wb") as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    _SERVERS.append(proc)
    print(f"[lighteval_driver] server pid={proc.pid} port={port}", flush=True)
    if wait_ready(port, timeout_s, proc, model) is None:
        raise RuntimeError(f"server not ready after {timeout_s}s; see {case_out / 'server.log'}")
    return proc


def mext_resolved_gate(server_log: pathlib.Path, arm: dict,
                       server_pid: int | None = None) -> str | None:
    text = server_log.read_text(errors="ignore") if server_log.is_file() else ""
    quant = "modelopt_mxfp4" if arm["kind"] == "modelopt_mxfp4" else "residue_nvfp4"
    result = engagement(text, server_pid, multiprocessing=False, quant=quant)
    problem = result.get("engagement_error")
    return f"arm {arm['id']}: {problem}; see {server_log}" if problem else None


def sampling_parameters(man: dict) -> dict:
    """One explicit protocol for scorer requests and provenance."""
    return {key: man[key] for key in SAMPLING_FIELDS}


def write_lighteval_config(arm: dict, man: dict, a, port: int,
                           case_out: pathlib.Path,
                           cache_dir: pathlib.Path) -> pathlib.Path:
    import yaml

    # model_name='openai/<path>': litellm sends everything after 'openai/' as
    # the OpenAI 'model' field, which vLLM matches against its model_id.
    model_parameters = {
        "model_name": f"openai/{arm['abs_path']}",
        "provider": "openai",
        "base_url": f"http://localhost:{port}/v1",
        "api_key": "dummy",  # vLLM ignores it
        "concurrent_requests": a.concurrent_requests,
        "timeout": a.request_timeout,
        "max_model_length": int(man["max_model_len"]),
        "generation_parameters": sampling_parameters(man),
    }

    # Arms sharing a checkpoint still need separate generation caches.
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_parameters["cache_dir"] = str(cache_dir)
    cfg_path = case_out / "lighteval_model_args.yaml"
    cfg_path.write_text(yaml.safe_dump({"model_parameters": model_parameters},
                                       sort_keys=False))
    return cfg_path


def reject_legacy_lcb(tasks: str) -> None:
    if any(
        part.strip().lower().split(":")[0] in ("lcb", "livecodebench")
        for task in tasks.split(",") for part in task.split("|")
    ):
        raise ValueError(
            "LCB moved to python -m benchmarks.quality.lcb run; "
            "use an explicit official release_v6 manifest. "
            "See benchmarks/quality/lighteval/README.md."
        )


def run_lighteval(arm: dict, man: dict, a, cfg_path: pathlib.Path,
                  case_out: pathlib.Path) -> dict:
    scorer_bin = pathlib.Path(a.lighteval_env).expanduser().resolve() / "bin"
    scorer_cli, scorer_python = scorer_bin / "lighteval", scorer_bin / "python"
    if not scorer_cli.exists():
        raise SystemExit(
            f"lighteval CLI not found: {scorer_cli}; pass --lighteval-env with an installed scorer environment.")
    raw = case_out / "lighteval_raw"
    raw.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks or man["_tasks"]
    reject_legacy_lcb(tasks)
    cmd = [str(scorer_python), str(scorer_cli), "endpoint", "litellm", str(cfg_path), tasks,
           "--output-dir", str(raw)]
    if a.custom_tasks:
        cmd += ["--custom-tasks", str(pathlib.Path(a.custom_tasks).resolve())]
    if a.max_samples is not None:
        cmd += ["--max-samples", str(a.max_samples)]
    cmd.append("--save-details")

    env = os.environ.copy()
    strip_proxies(env)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "True",
    })
    if a.nltk_data:
        env["NLTK_DATA"] = str(pathlib.Path(a.nltk_data).expanduser().resolve())
    if a.data_root:
        env["RESINFER_EVAL_DATA_ROOT"] = str(
            pathlib.Path(a.data_root).expanduser().resolve())
    env["RESINFER_LIGHTEVAL_SAMPLING_COMPAT"] = "1"
    compat = REPO / "benchmarks" / "quality" / "lighteval" / "compat"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(compat), str(REPO)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    # Shared HF cache so pre-fetched datasets are visible to the scorer venv.
    env.setdefault("HF_HOME", str(pathlib.Path.home() / ".cache" / "huggingface"))
    preflight_lighteval(env, case_out, scorer_python)
    write_json(case_out / "lighteval_invocation.json",
               {"cmd": cmd, "config": str(cfg_path), "tasks": tasks})
    validator = [str(scorer_python), "-m", "benchmarks.quality.lighteval.results",
                 "--tasks", tasks]
    if a.custom_tasks:
        validator += ["--custom-tasks", str(pathlib.Path(a.custom_tasks).resolve())]
    expected = case_out / "expected_items.json"
    prepare = [*validator, "prepare", "--expected", str(expected)]
    if a.max_samples is not None:
        prepare += ["--max-samples", str(a.max_samples)]
    with (case_out / "result_validation.log").open("wb") as log:
        subprocess.run(prepare, env=env, cwd=str(REPO), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print(f"[lighteval_driver] scorer: {' '.join(cmd)}", flush=True)
    with (case_out / "lighteval.log").open("wb") as f:
        proc = subprocess.run(cmd, cwd=str(case_out), env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"lighteval exited {proc.returncode}; see "
                           f"{case_out / 'lighteval.log'}")
    checked = case_out / "validated_results.json"
    validate_cmd = [*validator, "validate", "--expected", str(expected),
                    "--raw", str(raw), "--out", str(checked)]
    with (case_out / "result_validation.log").open("ab") as log:
        subprocess.run(validate_cmd, env=env,
                       cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(checked.read_text())


def preflight_lighteval(env: dict, case_out: pathlib.Path, scorer_python: pathlib.Path) -> None:
    """Fail before task loading if the offline scorer assets are incomplete."""
    if not scorer_python.exists():
        raise RuntimeError(f"lighteval interpreter not found: {scorer_python}")
    if not env.get("NLTK_DATA"):
        raise RuntimeError(
            "Offline lighteval requires local NLTK resources; pass --nltk-data")
    script = r'''import importlib.util, pathlib
import nltk, xxhash
spec = importlib.util.find_spec("lighteval")
assert spec and spec.origin, "lighteval is not installed"
task_dir = pathlib.Path(spec.origin).resolve().parent / "tasks" / "tasks"
assert (task_dir / "tinyBenchmarks.pkl").is_file(), "tinyBenchmarks.pkl is missing"
for resource in (
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "taggers/averaged_perceptron_tagger_eng",
):
    nltk.data.find(resource)
xxhash.xxh64("resinfer-text-compat").hexdigest()
print("lighteval offline preflight passed")
'''
    result = subprocess.run(
        [str(scorer_python), "-c", script], cwd=str(REPO), env=env,
        capture_output=True, text=True,
    )
    log_path = case_out / "lighteval_preflight.log"
    log_path.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"lighteval offline preflight exited {result.returncode}; see {log_path}")


def git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                             capture_output=True, text=True)
        return out.stdout.strip() or None
    except OSError:
        return None


def run_case(arm: dict, man: dict, a, out_root: pathlib.Path) -> tuple[str, dict]:
    """One arm end to end. Returns (status, results)."""
    case_out = out_root / arm["id"]
    if case_out.exists() and any(case_out.iterdir()):
        raise ValueError(f"Nonempty experiment directory: {case_out}; choose a new --out")
    case_out.mkdir(parents=True, exist_ok=True)
    with (case_out / "run.claim").open("x") as stream:
        stream.write(arm["id"])
    t0 = time.time()
    port = a.port or free_port()

    env = build_server_env(arm, a)
    cmd = build_server_cmd(arm, man, a, port)
    cache_dir = case_out / "lighteval_cache"

    status, results, error, resolved_line = "fail", {}, None, None
    server_log = case_out / "server.log"
    try:
        proc = launch_server(cmd, env, case_out, port, a.ready_timeout_sec, arm["abs_path"])
        if arm["kind"] in ("residue", "modelopt_mxfp4"):
            problem = mext_resolved_gate(server_log, arm, proc.pid)
            if problem:
                status, error = "fail_mext", problem
                raise RuntimeError(problem)
        if server_log.is_file():
            hits = [ln for ln in server_log.read_text(errors="ignore").splitlines()
                    if "mode=" in ln and "decode_mext=" in ln]
            resolved_line = hits[-1].strip() if hits else None
        cfg_path = write_lighteval_config(arm, man, a, port, case_out, cache_dir)
        results = run_lighteval(arm, man, a, cfg_path, case_out)
        problem = engagement(server_log.read_text(errors="ignore"), quant=None).get("engagement_error")
        if problem:
            raise RuntimeError(problem)
        status = "pass"
    except Exception as exc:  # recorded per case; the suite keeps going
        error = error or repr(exc)
        print(f"[lighteval_driver] arm {arm['id']} FAILED: {error}", flush=True)
    finally:
        teardown()

    wall = time.time() - t0
    write_json(case_out / "provenance.json", {
        "arm": arm["id"], "model": arm["abs_path"], "kind": arm["kind"],
        "mext": arm.get("mext"), "manifest": man["_path"],
        "git_head": git_head(),
        "resolved_mode": resolved_line,
        "runtime_env": {k: v for k, v in sorted(env.items())
                        if k.startswith(("RESIDUE_", "VLLM_")) or k == "CUDA_VISIBLE_DEVICES"},
        "cache_dir": str(cache_dir),
        "wall_time_sec": round(wall, 1),
        "server_cmd": cmd,
        "sampling": sampling_parameters(man),
        "enable_thinking": man["enable_thinking"],
        "model_type": man["model_type"],
        "protocol_source": man["protocol_source"],
        "sampling_transport": "lighteval-0.13-openai-extra-body-v1",
        "engine_knobs": {
            "enforce_eager": False,
            "compilation_config": accuracy_engine_config({}, {})["compilation_config"],
            "max_num_batched_tokens": a.max_num_batched_tokens,
            "max_num_seqs": a.max_num_seqs,
            "max_model_len": man["max_model_len"],
            "gpu_memory_utilization": a.gpu_mem_util,
            "enable_flashinfer_autotune": False,
            "seed": man["seed"],
        },
    })
    write_json(case_out / "bench_result.json", {
        "status": status, "arm": arm["id"], "model": arm["abs_path"],
        "kind": arm["kind"], "mext": arm.get("mext"),
        "tasks": a.tasks or man["_tasks"], "error": error,
        "results_file": "validated_results.json" if results else None,
        "wall_time_sec": round(wall, 1),
    })
    for task, metrics in results.items():
        nums = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        print(f"[lighteval_driver]   {arm['id']} | {task}: {nums}", flush=True)
    return status, results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Single-server AIME/GPQA driver; see benchmarks/quality/lighteval/README.md.")
    ap.add_argument("--lighteval-env", required=True,
                    help="external scorer virtual environment (outside .dev)")
    ap.add_argument("--manifest", required=True,
                    help="AIME/GPQA manifest, e.g. benchmarks/suites/aime_4b.yaml")
    ap.add_argument("--models-root", required=True,
                    help="dir the manifest's relative arm paths resolve under")
    ap.add_argument("--out", required=True, help="run output directory")
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm ids; default: every arm")
    ap.add_argument("--gpu", type=int, default=None,
                    help="CUDA_VISIBLE_DEVICES for the server (single GPU)")
    ap.add_argument("--port", type=int, default=0, help="0 = fresh free port per arm")
    ap.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    ap.add_argument("--tasks", default=None,
                    help="override the manifest task spec (e.g. the avg8 shim "
                         "'custom|aime24_avg8|0,custom|aime25_avg8|0', or the "
                         "aime stability_tasks set)")
    ap.add_argument("--custom-tasks", default=None,
                    help="lighteval --custom-tasks module, e.g. "
                         "benchmarks/quality/lighteval/aime.py for avg@8")
    ap.add_argument("--data-root", default=None,
                    help="root containing offline AIME/GPQA datasets for "
                         "benchmarks/quality/lighteval/tasks.py")
    ap.add_argument("--nltk-data", default=os.environ.get("NLTK_DATA"),
                    help="pre-staged NLTK data directory for offline scoring")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="debug only: score the first N samples")
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048,
                    help="vLLM scheduler token budget per iteration")
    ap.add_argument("--max-num-seqs", type=int, default=64,
                    help="maximum concurrent sequences in vLLM")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--concurrent-requests", type=int, default=64)
    ap.add_argument("--request-timeout", type=float, default=6000.0)
    ap.add_argument("--ready-timeout-sec", type=int, default=1800)
    ap.add_argument("--mext-max-m", type=int, default=64)
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    man = load_manifest(a.manifest)
    reject_legacy_lcb(a.tasks or man["_tasks"])
    arms = resolve_arms(man, a.models_root, a.arms)
    out_root = pathlib.Path(a.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    print(f"[lighteval_driver] manifest={man['name']} arms="
          f"{','.join(x['id'] for x in arms)} tasks={a.tasks or man['_tasks']}",
          flush=True)
    statuses: dict[str, str] = {}
    for arm in arms:
        print(f"[lighteval_driver] === arm {arm['id']} ({arm['kind']}"
              f"{', mext ' + str(arm['mext']) if 'mext' in arm else ''}) ===",
              flush=True)
        status, _ = run_case(arm, man, a, out_root)
        statuses[arm["id"]] = status

    write_json(out_root / "summary.json",
               {"manifest": man["_path"], "statuses": statuses,
                "tasks": a.tasks or man["_tasks"]})
    print("[lighteval_driver] --- summary ---", flush=True)
    for arm_id, status in statuses.items():
        print(f"[lighteval_driver]   {arm_id:<16} {status}", flush=True)
    failed = [k for k, v in statuses.items() if v != "pass"]
    if failed:
        print(f"[lighteval_driver] FAILED arm(s): {', '.join(failed)} -- see "
              f"bench_result.json in each case dir", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
