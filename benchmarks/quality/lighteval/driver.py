# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-server AIME/GPQA driver with isolated per-arm caches and checked results.

The external lighteval 0.13 environment owns tasks and metrics. Its scheduled
items are recorded before generation and checked against the saved details.
Throughput logs are diagnostic only; they do not prove cache use or generation.
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
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from benchmarks.engine_config import accuracy_engine_config
from benchmarks.quality.common.protocol import load_manifest, resolve_arms, SAMPLING_FIELDS
VLLM_BIN = REPO / ".venv" / "bin" / "vllm"
SERVING_PYTHON = REPO / ".venv" / "bin" / "python"
_SERVERS: list[subprocess.Popen] = []


# ----------------------------------------------------------- process plumbing
def teardown() -> None:
    for proc in _SERVERS:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        proc.wait()
    _SERVERS.clear()


def on_signal(signum, _frame) -> None:
    print(f"[lighteval_driver] got signal {signum}, tearing down", flush=True)
    teardown()
    raise SystemExit(143)


def health_ok(port: int) -> bool:
    # ProxyHandler({}) so a shell-level SOCKS/HTTP proxy cannot swallow the
    # localhost health check.
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(f"http://localhost:{port}/health", timeout=5).status == 200
    except Exception:
        return False


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


# ------------------------------------------------------------- server side
def build_server_env(arm: dict, a, case_out: pathlib.Path) -> dict:
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
    env["VLLM_NO_USAGE_STATS"] = "1"
    # Match runtime.install_env: use the same process mode for every arm.
    # The patch requires in-process EngineCore on the tested runtime.
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Keep compile-cache reuse disabled for accuracy comparisons.
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    # FlashInfer JIT needs the driver library when the stub directory lacks libcuda.
    env["LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + env.get("LIBRARY_PATH", "")
    # The repository sitecustomize registers the selected runtime integration.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), str(REPO / "csrc")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    if arm["kind"] == "residue":
        env["RESIDUE_NVFP4"] = "1"
        env["RESIDUE_VLLM_MODEL_DIR"] = arm["abs_path"]
        # Keep the extended-K loader enabled during compute-time ablations.
        env["RESIDUE_ENABLE_DECODE_MEXT"] = "1"
        env["RESIDUE_FORCE_DECODE_MEXT"] = "0"
        env["RESIDUE_DECODE_MEXT_MAX_M"] = str(a.mext_max_m)
        if str(arm.get("mext", "on")) == "off":
            # MIN_M=0 disables residue for every M while preserving weight loading.
            env["RESIDUE_PREFILL_RESIDUE_OFF"] = "1"
            env["RESIDUE_PREFILL_RESIDUE_OFF_MIN_M"] = "0"
        # Cubins push modelopt_fp4 onto trtllm-gen paths that bypass or break
        # the patched linear method (vllm-integration.md rule 3).
        for var in ("FLASHINFER_CUBIN_DIR", "VLLM_HAS_FLASHINFER_CUBIN"):
            if env.pop(var, None) is not None:
                print(f"[lighteval_driver] dropped {var}: incompatible with a "
                      f"residue arm", flush=True)
    return env


def build_server_cmd(arm: dict, man: dict, a, port: int) -> list[str]:
    if not VLLM_BIN.exists():
        raise SystemExit(f"vLLM entrypoint not found: {VLLM_BIN} -- the server "
                         f"runs from the MAIN venv; is .venv bootstrapped?")
    template_kwargs = {"enable_thinking": man["enable_thinking"]}
    if "reasoning_effort" in man:
        template_kwargs["reasoning_effort"] = man["reasoning_effort"]
    # Migrated venv console scripts can retain a stale absolute shebang.
    # Run the script with the selected serving interpreter explicitly.
    cmd = [str(SERVING_PYTHON), str(VLLM_BIN), "serve", arm["abs_path"],
           "--port", str(port),
           "--dtype", a.dtype,
           "--generation-config", "vllm",
           "--compilation-config", json.dumps(accuracy_engine_config({}, {})["compilation_config"]),
           "--default-chat-template-kwargs", json.dumps(template_kwargs),
           # enforce_eager stays False (knob 1: non-eager is the standard).
           "--seed", str(man["seed"]),                          # knob 4
           "--max-model-len", str(man["max_model_len"]),        # knob 6
           # Match batching across comparison arms.
           "--max-num-batched-tokens", str(a.max_num_batched_tokens),
           "--max-num-seqs", str(a.max_num_seqs),               # knob 10
           "--gpu-memory-utilization", str(a.gpu_mem_util),     # knob 7
           "--no-enable-flashinfer-autotune"]                   # knob 3
    if arm["kind"] in ("residue", "modelopt_fp4"):
        cmd += ["--quantization", "modelopt_fp4"]
    return cmd


def launch_server(cmd: list[str], env: dict, case_out: pathlib.Path,
                  port: int, timeout_s: int) -> None:
    with (case_out / "server.log").open("wb") as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    _SERVERS.append(proc)
    print(f"[lighteval_driver] server pid={proc.pid} port={port}", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            raise RuntimeError(
                f"server exited with code {returncode} before /health became "
                f"ready -- see {case_out / 'server.log'}")
        if health_ok(port):
            return
        time.sleep(2)
    raise RuntimeError(f"server not healthy after {timeout_s}s -- see "
                       f"{case_out / 'server.log'}")


def preflight_server(arm: dict, env: dict, case_out: pathlib.Path) -> dict:
    """Verify the residue package, extension ABI, and vLLM patch before serve."""
    if arm["kind"] != "residue":
        return {}
    if not SERVING_PYTHON.exists():
        raise RuntimeError(f"serving interpreter not found: {SERVING_PYTHON}")

    script = r'''import hashlib, importlib.util, json, pathlib, sys
import nvfp4, quant, torch
from nvfp4.extension import ensure_loaded
from quant.vllm_integration.patch import is_patched
assert ensure_loaded(True), "nvfp4 extension did not load"
assert is_patched(), "residue vLLM patch is not installed"
spec = importlib.util.find_spec("nvfp4._C")
origin = pathlib.Path(spec.origin).resolve() if spec and spec.origin else None
assert origin and origin.is_file(), "nvfp4._C origin is unavailable"
payload = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "extension": str(origin),
    "extension_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
}
print("RESINFER_PREFLIGHT_JSON=" + json.dumps(payload, sort_keys=True))
'''
    proc = subprocess.run(
        [str(SERVING_PYTHON), "-c", script], cwd=str(REPO), env=env,
        capture_output=True, text=True,
    )
    log_path = case_out / "server_preflight.log"
    log_path.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"residue server preflight exited {proc.returncode}; see {log_path}")
    marker = "RESINFER_PREFLIGHT_JSON="
    lines = [line for line in proc.stdout.splitlines() if line.startswith(marker)]
    if not lines:
        raise RuntimeError(f"server preflight produced no ABI record; see {log_path}")
    payload = json.loads(lines[-1][len(marker):])
    extension = pathlib.Path(payload["extension"])
    try:
        extension.relative_to(REPO / "csrc" / "nvfp4")
    except ValueError as exc:
        raise RuntimeError(
            f"nvfp4 extension is not from this checkout: {extension}") from exc
    payload["source_commit"] = git_head()
    return payload


# ---------------------------------------------------------------- the gates
def mext_resolved_gate(server_log: pathlib.Path, arm: dict) -> str | None:
    """Delta 4: confirm the RESOLVED mode in the server log after the server
    is up and BEFORE the first prompt. Returns an error string naming
    expected vs found, or None when confirmed."""
    text = server_log.read_text(errors="ignore") if server_log.is_file() else ""
    resolved = [ln for ln in text.splitlines()
                if "[residue:env]" in ln and "mode=" in ln
                and "WARNING" not in ln and "FATAL" not in ln]
    if not resolved:
        return (f"arm {arm['id']}: no [residue:env] resolved-mode line in "
                f"{server_log}. The residue plugin never ran in the engine "
                f"process -- the arm would serve UNPATCHED and measure the "
                f"plain floor under a method arm's name "
                f"(implementation constraints rule 1).")
    line = resolved[-1].strip()
    mext_on = str(arm.get("mext", "on")) == "on"
    if mext_on:
        # Primary signal is decode_mext=on; a mode naming mext (mext_r1) is
        # also accepted as a resolved-mode confirmation.
        if "decode_mext=on" not in line and "mode=mext" not in line:
            return (f"arm {arm['id']}: expected decode_mext=on (or a mode "
                    f"naming mext) in the resolved line; found: {line}")
        if "prefill_residue=OFF" in line:
            return (f"arm {arm['id']}: expected prefill_residue=on for a "
                    f"method arm, but the compute-time kill is engaged; "
                    f"found: {line}")
    else:
        # The mext-off floor arm relies on the compute-time kill; without it
        # the arm silently serves residue under the floor's name -- the same
        # hazard, inverted.
        if "prefill_residue=OFF" not in line:
            return (f"arm {arm['id']}: expected prefill_residue=OFF (the "
                    f"compute-time residue kill that makes this the plain "
                    f"floor); found: {line}")
    return None


def generation_provenance(server_log: pathlib.Path, server_env: dict) -> dict:
    """Record observations; periodic logs cannot prove cache replay or completion."""
    served = 0
    if server_log.is_file():
        served = server_log.read_text(errors="ignore").count(
            "Avg generation throughput")
    env_fp = {k: v for k, v in sorted(server_env.items())
              if k.startswith(("RESIDUE_", "VLLM_")) or k == "CUDA_VISIBLE_DEVICES"}
    return {"generated": None, "cache_replay": None,
            "decode_throughput_lines": served, "runtime_env": env_fp}


# ------------------------------------------------------------- scorer side



def sampling_parameters(man: dict) -> dict:
    """One explicit protocol for scorer requests and provenance."""
    return {key: man[key] for key in SAMPLING_FIELDS}


def write_lighteval_config(arm: dict, man: dict, a, port: int,
                           case_out: pathlib.Path,
                           cache_dir: pathlib.Path | None) -> pathlib.Path:
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

    if cache_dir is not None:
        # Delta 2: run-scoped, per case. Per CASE and not per run because this
        # manifest's floor arm shares a model path with a method arm -- with
        # one shared dir the second of the two would replay the first
        # (lighteval-cache.md rule 1: per-arm is REQUIRED when arms share a
        # model path).
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


# ------------------------------------------------------------------ one case
def git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                             capture_output=True, text=True)
        return out.stdout.strip() or None
    except OSError:
        return None


def run_case(arm: dict, man: dict, a, out_root: pathlib.Path) -> tuple[str, dict]:
    """One arm end to end. Returns (status, results)."""
    if a.global_cache:
        raise ValueError("Shared generation caches are unsupported; use a fresh per-arm cache")
    case_out = out_root / arm["id"]
    if case_out.exists() and any(case_out.iterdir()):
        raise ValueError(f"Nonempty experiment directory: {case_out}; choose a new --out")
    case_out.mkdir(parents=True, exist_ok=True)
    with (case_out / "run.claim").open("x") as stream:
        stream.write(arm["id"])
    t0 = time.time()
    port = a.port or free_port()

    env = build_server_env(arm, a, case_out)
    cmd = build_server_cmd(arm, man, a, port)
    cache_dir = case_out / "lighteval_cache"

    status, results, error, resolved_line = "fail", {}, None, None
    server_preflight = {}
    server_log = case_out / "server.log"
    try:
        server_preflight = preflight_server(arm, env, case_out)
        launch_server(cmd, env, case_out, port, a.ready_timeout_sec)
        # Delta 4: the pre-gate runs after health and before the first prompt.
        if arm["kind"] == "residue":
            problem = mext_resolved_gate(server_log, arm)
            if problem:
                status, error = "fail_mext", problem
                raise RuntimeError(problem)
        if server_log.is_file():
            hits = [ln for ln in server_log.read_text(errors="ignore").splitlines()
                    if "[residue:env]" in ln and "mode=" in ln]
            resolved_line = hits[-1].strip() if hits else None
        cfg_path = write_lighteval_config(arm, man, a, port, case_out, cache_dir)
        results = run_lighteval(arm, man, a, cfg_path, case_out)
        status = "pass"
    except Exception as exc:  # recorded per case; the suite keeps going
        error = error or repr(exc)
        print(f"[lighteval_driver] arm {arm['id']} FAILED: {error}", flush=True)
    finally:
        teardown()

    prov = generation_provenance(server_log, env)
    wall = time.time() - t0
    write_json(case_out / "provenance.json", {
        "arm": arm["id"], "model": arm["abs_path"], "kind": arm["kind"],
        "mext": arm.get("mext"), "manifest": man["_path"],
        "git_head": git_head(),
        "server_preflight": server_preflight,
        "resolved_mode": resolved_line,
        "runtime_env": prov["runtime_env"],
        "cache_dir": str(cache_dir) if cache_dir else "GLOBAL (opted out)",
        "decode_throughput_lines": prov["decode_throughput_lines"],
        "generated": prov["generated"], "cache_replay": prov["cache_replay"],
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
    write_json(case_out / "lighteval_results.json",
               {"results": results, "provenance": prov})
    write_json(case_out / "bench_result.json", {
        "status": status, "arm": arm["id"], "model": arm["abs_path"],
        "kind": arm["kind"], "mext": arm.get("mext"),
        "tasks": a.tasks or man["_tasks"], "error": error, "results": results,
        "wall_time_sec": round(wall, 1),
    })
    for task, metrics in results.items():
        nums = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        print(f"[lighteval_driver]   {arm['id']} | {task}: {nums}", flush=True)
    return status, results


# ----------------------------------------------------------------------- main
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
    ap.add_argument("--dtype", default="bfloat16")
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
    ap.add_argument("--global-cache", action="store_true",
                    help="OPT OUT of the run-scoped lighteval cache (hazard!)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="debug only: score the first N samples")
    ap.add_argument("--save-details", action="store_true")
    # Determinism knobs -- pinned defaults; identical across every arm of a
    # table by construction, since one driver invocation runs the whole table.
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048,
                    help="knob 2, pinned (historical short-prompt value)")
    ap.add_argument("--max-num-seqs", type=int, default=64,
                    help="knob 10, pinned (the paper LCB harness value)")
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

    if a.global_cache:
        raise ValueError("--global-cache is unsupported; use fresh per-arm outputs")
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
