#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end serving sweep across a residue budget ladder.

    python benchmarks/bench_residue_e2e.py --gpu 0 \
        --manifest benchmarks/suites/speed_dense.yaml \
        --models-root <dir holding the exports> --outdir results_speed

ARCHITECTURE
  Nothing here is architecture-specific: it launches `vllm serve` and drives
  it with `vllm bench serve`. The same script produces the sm120 (RTX 5090)
  and sm100 (B200) tables -- what differs between them is where the process
  runs, which is a launcher concern, not a measurement one. Do not fork it
  per GPU; run it inside whatever container that GPU lives in.

  The residue kernels themselves DO differ by architecture (the fold is a
  CuTeDSL kernel on sm100/sm103 and a different one on sm120, with different
  M thresholds), which is exactly why the same harness has to produce both
  tables: any difference in the numbers should come from the kernels, not
  from two harnesses that drifted apart.

WHAT IT ANSWERS
  How do the e2e serving metrics (TTFT / TPOT / ITL / throughput) move with
  effective-bit budget, and does the ordering between budgets change with
  load? One fixed workload (1024 in / 256 out, ignore-eos, request-rate inf)
  swept over concurrency {1, 4, 16, 64} for every arm:

      whatever `--arm` names, with a plain-NVFP4 arm as the no-residue floor

  Expectations going in (falsifiable):
    - TTFT rises with budget (K-extension is a prefill-time cost) and the
      gap WIDENS with concurrency (prefill GEMMs saturate the machine).
    - TPOT is ~flat across budgets at low concurrency (decode fold is
      ratio-blind) but may spread at high concurrency, where decode batches
      ride above fold_max_m and the K-ext GEMM serves them.
    - b80 is the outlier to watch: every layer is mext_r1, so its cost
      mechanism is doubled token ROWS rather than a widened K, and its weight
      is stored once at base width. Expect its memory footprint to match the
      plain arm's despite carrying the largest budget in the table.

SERVER HYGIENE
  - arms run SERIALLY on one GPU; concurrent servers starve each other during
    autotune and the readiness timeout fires.
  - assert_gpu_free first: orphaned EngineCores squat 28 GiB and turn into
    "not ready in 2400s" 40 minutes later.
  - loud manipulation checks: a residue arm whose plugin failed serves plain
    NVFP4 with a residue label; refuse the arm instead of measuring it.

AUTOTUNE
  RESIDUE_SHARE_AUTOTUNE_CACHE=1 for residue arms + an offline prewarm
  beforehand (benchmarks/prewarm_autotune_cache.py) means arms start warm; the
  per-arm ready_s in the summary is the live measurement of that machinery.
  The plain arm cannot share (no residue plugin -> no patch) and tunes its
  own 4 shapes once.

  NOT an eval. mnbt stays at vLLM's default per eval-determinism.md's scope
  note: benchmarks run each arm at its own best, only evals pin the knob.

ARMS
Named by the manifest, located by `--models-root`: each arm's `path` is
relative to that root, so the manifest holds no machine-specific path and is
the spec you commit. Every arm is checked to exist BEFORE any GPU time is
spent, because a table missing a column still looks like a complete table.

The quantisation mode is DETECTED from each export (a residue one carries
`residue_kernel_metadata.json`) rather than declared. Serving a checkpoint
under the wrong arm label is the most expensive mistake available here: it
produces a complete and plausible table of the wrong thing.

MEMORY
Half of what residue costs is memory, not time -- K-extension multiplies weight
bytes by (1+r), and on a fixed GPU that comes out of the KV cache. Every arm
records `weights_gib`, `kv_cache_gib` and `kv_cache_tokens`, all read off
vLLM's own startup log.

Resumable: an (arm, concurrency) whose result JSON already parses is skipped,
so a crashed sweep continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import signal
import subprocess
import time
import shutil
import sys
import urllib.request

try:                          # `python -m benchmarks.bench_residue_e2e`
    from benchmarks import manifest as spec_mod
except ImportError:           # `python benchmarks/bench_residue_e2e.py`
    # Both invocations must work. Under `-m` the package parent is on
    # sys.path, so a bare `import manifest` is not found; run as a plain file
    # the script's own directory is sys.path[0], so the package name is not
    # found. Neither form can be dropped: the README advertises the module
    # form and the suites are driven as files.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import manifest as spec_mod

REPO = pathlib.Path(__file__).resolve().parents[1]


def _vllm_cli() -> str:
    """The `vllm` executable, resolved from the environment, not the repo.

    This used to be `REPO/.venv/bin/vllm`, which works only when the venv
    happens to live inside the checkout. It does not for anyone reproducing
    from a normal install, and the failure is a FileNotFoundError deep inside
    the first server launch -- after the sweep has already claimed a GPU.

    Preference order is the interpreter's own bin directory first, so running
    the script with a venv's python uses that venv's vllm even when a
    different one is earlier on PATH.
    """
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
    """Everything needed, months later, to decide whether two tables compare.

    The kernel benchmark in this directory carried this from the start; this
    one did not, and the omission is the same failure it exists to prevent.
    A summary.jsonl without it is not reproducible: the arm names and numbers
    are there, but which commit, which vLLM, which GPU and which RESIDUE_*
    produced them lives only in whoever ran it.

    Env keys are written as "<unset>" rather than dropped, so "not set" and
    "not recorded" stay distinguishable -- for RESIDUE_TENSOR_DUMP, "not set"
    is exactly the claim being made.
    """
    import hashlib
    import platform

    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:                                       # noqa: BLE001
            return None

    def _git(*a):
        try:
            r = subprocess.run(["git", "-C", str(REPO), *a],
                               capture_output=True, text=True, timeout=20)
            return r.stdout.strip() or None
        except Exception:                                       # noqa: BLE001
            return None

    so = next(iter(sorted((REPO / "csrc" / "nvfp4").glob("_C*.so"))),
              None)
    md5 = None
    if so:
        h = hashlib.md5()
        with open(so, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        md5 = h.hexdigest()

    import torch
    tracked = ("RESIDUE_NVFP4", "RESIDUE_PATCH_MODE",
               "RESIDUE_ENABLE_DECODE_MEXT", "RESIDUE_DECODE_MEXT_MAX_M",
               "RESIDUE_RATIO1_PREFILL_PATH", "RESIDUE_MEXT_PREFILL_PATH", "RESIDUE_SHARE_AUTOTUNE_CACHE",
               "RESIDUE_TENSOR_DUMP", "RESIDUE_ATTN",
               "VLLM_DISABLE_COMPILE_CACHE", "VLLM_ENABLE_V1_MULTIPROCESSING",
               "VLLM_PLUGINS", "FLASHINFER_CUBIN_DIR", "CPATH", "CUDA_HOME")
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,clocks.sm,temperature.gpu",
         "--format=csv,noheader", "-i", str(gpu)],
        capture_output=True, text=True)
    return {
        "kind": "provenance",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(REPO),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_subject": _git("log", "-1", "--format=%s"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "ext_so": str(so) if so else None,
        "ext_so_md5": md5,
        "ext_so_mtime": (time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(so.stat().st_mtime))
                         if so else None),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _v("torch"), "vllm": _v("vllm"),
        "flashinfer": _v("flashinfer"), "cuda": torch.version.cuda,
        "gpu_index": gpu, "gpu": q.stdout.strip(),
        "gpu_neighbors": neighbors,
        "vllm_cli": _vllm_cli(),
        "env": {k: os.environ.get(k, "<unset>") for k in tracked},
    }


def _detect_quant(path: str) -> str | None:
    """Read the quantisation mode off the checkpoint instead of being told it.

    Three outcomes, because there are three kinds of arm and conflating any
    two of them breaks the run:

        residue_nvfp4   the export carries `residue_kernel_metadata.json`
        modelopt_fp4    `config.json` carries a `quantization_config`
        None            neither -- an unquantised reference, which must be
                        served with NO --quantization flag at all

    That last case is not hypothetical. Passing `modelopt_fp4` to a plain bf16
    checkpoint dies at startup with "Cannot find the config file for
    modelopt_fp4", which reads like a missing file rather than like the arm
    being mislabelled. Found by running it, not by reading it.

    Detecting rather than letting the manifest declare it is deliberate:
    serving a checkpoint under the wrong arm label is the most expensive
    mistake available here, because it produces a complete and plausible table
    of the wrong thing.
    """
    d = pathlib.Path(path)
    if (d / "residue_kernel_metadata.json").exists():
        return "residue_nvfp4"
    try:
        cfg = json.load(open(d / "config.json"))
    except Exception:                                           # noqa: BLE001
        return None
    return "modelopt_fp4" if cfg.get("quantization_config") else None

CONCURRENCIES = (1, 4, 16, 64)
INPUT_LEN = 1024
OUTPUT_LEN = 256


def _num_prompts(conc: int) -> int:
    # Enough requests that steady state dominates ramp-up, bounded so the
    # sweep finishes: 32 at conc=1, 8x concurrency above that, cap 256.
    return min(256, max(32, 8 * conc))


def assert_gpu_free(gpu: int, need_gib: float = 20.0,
                    exclusive: bool = True) -> None:
    'Refuse to measure unless the GPU has both the room AND no neighbours.'
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True)
    used, total = (int(x) for x in q.stdout.strip().split(", "))
    free_gib = (total - used) / 1024
    if free_gib < need_gib:
        raise SystemExit(
            f"gpu{gpu} has only {free_gib:.1f} GiB free; need ~{need_gib}. "
            "Check for orphaned vLLM EngineCore (kill -9 -$pgid, not the pid).")

    a = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True)
    mine = {os.getpid()}
    others = []
    for line in a.stdout.strip().splitlines():
        if not line.strip():
            continue
        pid_s, mem_s = (p.strip() for p in line.split(",")[:2])
        if pid_s.isdigit() and int(pid_s) not in mine:
            others.append((int(pid_s), mem_s))
    if others and exclusive:
        # Name the neighbour while it still exists (see _common.pid_identity):
        # a refusal that says only "pid 12345" cannot answer "whose job?"
        # ten seconds later.
        def _ident(q):
            try:
                from benchmarks._common import pid_identity
            except ImportError:
                from _common import pid_identity
            d = pid_identity(q)
            extra = ""
            if d.get("cmdline"):
                extra += f"\n      cmd  {d['cmdline']}"
            if d.get("cwd"):
                extra += f"\n      cwd  {d['cwd']}"
            if d.get("note"):
                extra += f"\n      ({d['note']})"
            return extra

        listing = "\n".join(f"    pid {q}  {m} MiB{_ident(q)}"
                            for q, m in others)
        raise SystemExit(
            f"gpu{gpu} is NOT exclusive -- {len(others)} other compute "
            f"process(es):\n{listing}\n"
            f"  Free memory is not the test: a 1.4 GiB neighbour passes every "
            f"memory threshold\n  and still takes SMs. Move to an empty GPU, "
            f"or re-run with --allow-shared-gpu, which does not make the\n"
            f"  numbers trustworthy -- it records the neighbours on every row "
            f"so the reader can\n  decide. nvidia-smi's memory column is the "
            f"handiest one and the most misleading.")
    if others:
        print(f"[warn] gpu{gpu} shared with {len(others)} process(es): "
              f"{[q for q, _ in others]} -- recorded on every row; these "
              f"timings are annotated, not clean.", flush=True)
    return [{"pid": q, "used_mib": m} for q, m in others]


def _wait_ready(port: int, timeout_s: int) -> float | None:
    """Seconds until /health returns 200, or None on timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    return None


_MEM_PATTERNS = (
    ("weights_gib", r"Model loading took ([0-9.]+) GiB", float),
    ("kv_cache_gib", r"Available KV cache memory: ([0-9.]+) GiB", float),
    ("kv_cache_tokens", r"GPU KV cache size: ([0-9,]+) tokens",
     lambda v: int(v.replace(",", ""))),
)


def gpu_state(gpu: int) -> dict:
    """Clock and temperature AT THIS MOMENT, for stamping onto one arm's row.

    `provenance()` samples once, before the sweep. An eight-round sweep runs
    for hours, and the manifest's own comment promises that "clocks are
    recorded per round and thermal drift becomes a column" -- which a single
    sample at t=0 cannot deliver. A claim in a manifest that the code does not
    implement is the defect this directory keeps paying for; here it is one
    call per arm.
    """
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


def engagement(log: str) -> dict:
    'Did the residue patch reach the process that runs forward?'
    applied = re.findall(r"\[residue:plugin\] applied in pid=(\d+)", log)
    in_engine = bool(re.search(
        r"\(EngineCore pid=(\d+)\)\s*\[residue:plugin\] applied", log))
    env_lines = re.findall(r"\[residue:env\] (mode=[^\n]*)", log)
    resolved = env_lines[-1].strip() if env_lines else None
    mext_on = bool(resolved and "decode_mext=on" in resolved)

    out = {"residue_plugin_pids": sorted(set(applied)),
           "residue_plugin_in_engine": in_engine,
           "residue_env_resolved": resolved,
           "engagement_error": None}
    if "[residue:plugin] FAILED" in log:
        out["engagement_error"] = ("residue plugin FAILED -- this arm would "
                                   "serve plain NVFP4 under a residue label")
    elif not applied:
        out["engagement_error"] = ("no '[residue:plugin] applied' in the serve "
                                   "log -- the plugin never ran in any process")
    elif not in_engine:
        out["engagement_error"] = (
            "the plugin applied only outside EngineCore. The launcher's log "
            "looks healthy and the process that runs forward is unpatched; "
            "set VLLM_ENABLE_V1_MULTIPROCESSING=0 or fix plugin coverage")
    elif resolved is None:
        out["engagement_error"] = ("no '[residue:env]' line -- the env "
                                   "contract never resolved, so what the arm "
                                   "actually serves is unrecorded")
    elif not mext_on:
        out["engagement_error"] = (
            f"decode_mext is not on ({resolved}) -- a residue export served "
            f"this way takes the legacy K-ext branch and measures the plain "
            f"floor under a method arm's name")
    return out


def parse_decode_batch(log_text: str) -> dict:
    """The decode batch sizes actually reached, from vLLM's own progress lines.

    This decides what the high-concurrency columns MEAN. The residue fold is
    taken per layer, and for a non-mext_r1 (K-extended) layer only while
    M <= the arch's fold band -- 128 on sm10x, 64 on sm120. Above it that layer
    falls to the legacy K-extension path instead.

    So a sweep whose decode batch crosses the band is measuring two different
    kernel paths in one column, and nothing in the throughput number says which.
    Recording the batch makes the crossing visible instead of inferred.

    Every kind of layer crosses the band; what differs is where it lands. A
    K-extended layer falls back to the legacy K-ext GEMM. A mext_r1 layer goes
    to the wrapped K-loop (or two-GEMM) prefill path -- it does NOT keep
    folding, and the residue math is applied either way. What 4083dee changed
    is that the band stopped being a NUMERICS boundary: before it, a mext_r1
    layer above the band fell through to the plain-NVFP4 bypass and silently
    computed no residue at all, so the same served model did different maths at
    different load. It is now a performance crossover and nothing more.

    The consequence for this table: two arms can diverge at high concurrency
    because their layers land on different post-band kernels, which has nothing
    to do with their budgets. Hence this column.
    """
    import re
    vals = [int(m) for m in re.findall(r"Running: (\d+) reqs", log_text)]
    vals = [v for v in vals if v > 0]
    if not vals:
        return {"decode_batch_max": None, "decode_batch_median": None,
                "decode_batch_samples": 0}
    vals.sort()
    return {"decode_batch_max": vals[-1],
            "decode_batch_median": vals[len(vals) // 2],
            "decode_batch_samples": len(vals)}


def parse_memory(log_text: str) -> dict:
    "Weight footprint and KV-cache headroom, read off vLLM's own startup log."
    import re
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
    """Stop the server AND its EngineCore, then wait for the GPU to come back.

    Signalling the process GROUP is the whole point: vLLM V1's EngineCore is a
    spawned child, and killing only the parent leaves it reparented to init
    with ~28 GiB still allocated. Observed exactly that here -- the parent's
    group was 2811640 while the surviving EngineCore sat in 2811661.

    Waiting for the memory to actually drop matters as much as the kill: the
    next arm starts with assert_gpu_free, and CUDA teardown is not instant, so
    returning the moment the process dies races the next launch.
    """
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            proc.wait(timeout=30 if sig is signal.SIGINT else 15)
            break
        except subprocess.TimeoutExpired:
            continue

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
    cmd = [
        _vllm_cli(), "bench", "serve",
        "--backend", "vllm", "--model", model,
        "--host", "127.0.0.1", "--port", str(port),
        "--dataset-name", "random",
        "--random-input-len", str(INPUT_LEN),
        "--random-output-len", str(OUTPUT_LEN),
        "--random-range-ratio", "0.0",
        "--ignore-eos", "--request-rate", "inf",
        "--max-concurrency", str(conc),
        "--num-prompts", str(_num_prompts(conc)),
        "--save-result", "--result-filename", out_json,
    ]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=1800)
        tail = (p.stdout or "")[-500:]
    except subprocess.TimeoutExpired:
        return {"error": "bench timed out"}
    try:
        return json.load(open(out_json))
    except Exception:
        return {"error": "no result json", "tail": tail}


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
    env = dict(os.environ)
    # The manifest's `hygiene.env` block, applied. It used to be read by
    # nothing at all: the yaml declared VLLM_DISABLE_COMPILE_CACHE=1 and no
    # code consulted it, so the knob was decoration on a file that is
    # committed as the spec itself. "<unset>" means remove the variable, which
    # is how the spec says "this must not be set" as opposed to "leave it".
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
    if quant == "residue_nvfp4":
        env["RESIDUE_NVFP4"] = "1"
        env["RESIDUE_ENABLE_DECODE_MEXT"] = "1"
        env["RESIDUE_SHARE_AUTOTUNE_CACHE"] = "1"
        # RESIDUE_PATCH_MODE is left unset, and this changes NOTHING about
        # what runs -- it is worth saying so, because the variable's name
        # suggests otherwise and at least one comment elsewhere in the tree
        # still describes its pre-merge behaviour.
        #
        # V2 and MExtR1 are one class now (`_choose_linear_method_cls`: "the
        # class returned is always V2"), so the variable resolves to a log
        # label and selects no code path. What actually decides is per LAYER:
        # `runtime_mode` in residue_kernel_metadata.json, defaulting to
        # extended_k where a layer has no entry. Decode takes the fold because
        # a mext_r1 layer enables it unconditionally and everything else
        # enables it under RESIDUE_ENABLE_DECODE_MEXT=1, which is set above.
        env.pop("RESIDUE_PATCH_MODE", None)
        
        
        
        
        
        
        env["RESIDUE_VLLM_MODEL_DIR"] = model
    else:
        for k in ("RESIDUE_NVFP4", "RESIDUE_PATCH_MODE",
                  "RESIDUE_ENABLE_DECODE_MEXT",
                  "RESIDUE_SHARE_AUTOTUNE_CACHE",
                  "RESIDUE_VLLM_MODEL_DIR"):
            env.pop(k, None)

    rec = {"tag": tag, "ratio": ratio, "quant": quant, "gpu": gpu,
           "model": model, "runs": {}}

    # Resume: if every concurrency already has a parsed result, skip the arm
    # without starting a server.
    pending = []
    for conc in CONCURRENCIES:
        rj = outdir / f"{tag}_c{conc}.json"
        try:
            rec["runs"][str(conc)] = {
                k: json.load(open(rj)).get(k) for k in _KEEP}
        except Exception:
            pending.append(conc)
    if not pending:
        rec["resumed"] = True
        return rec

    log_path = outdir / f"serve_{tag}.log"
    cmd = [
        _vllm_cli(), "serve", model,
        # Omitted entirely for an unquantised reference arm -- vLLM then reads
        # the checkpoint's own config, which is what a bf16 ceiling needs.
        *(["--quantization", quant] if quant else []),
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--port", str(port),
        "--trust-remote-code",
    ]
    print(f"[{tag}] gpu{gpu} port{port} launching ({quant}, budget={ratio}), "
          f"pending conc={pending}", flush=True)
    with open(log_path, "wb") as logf:
        
        
        
        
        
        
        proc = subprocess.Popen(cmd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            ready_s = _wait_ready(port, ready_timeout)
            if ready_s is None:
                rec["error"] = (f"serve not ready in {ready_timeout}s "
                                f"(check the log)")
                return rec
            rec["ready_s"] = round(ready_s, 1)
            early = log_path.read_text(errors="ignore")
            eng = engagement(early)
            rec.update(eng)
            if quant == "residue_nvfp4" and eng.get("engagement_error"):
                rec["error"] = eng["engagement_error"]
                return rec
            print(f"[{tag}] ready in {ready_s:.0f}s", flush=True)
            for conc in pending:
                rj = outdir / f"{tag}_c{conc}.json"
                r = _bench(model, port, conc, str(rj), env)
                rec["runs"][str(conc)] = (
                    {k: r.get(k) for k in _KEEP} if "error" not in r else r)
                got = rec["runs"][str(conc)]
                print(f"[{tag}] c={conc:<3} "
                      f"ttft={got.get('mean_ttft_ms')} "
                      f"tpot={got.get('mean_tpot_ms')} "
                      f"out_tps={got.get('output_throughput')}", flush=True)
        finally:
            _shutdown(proc, gpu)

    log = log_path.read_text(errors="ignore")
    rec["gpu_neighbors"] = neighbors
    rec.update(parse_memory(log))
    rec.update(parse_decode_batch(log))
    rec["residue_patch_applied"] = "[residue" in log
    rec["fold_fired"] = "FIRED" in log
    rec["shared_cache"] = "SHARED across models" in log
    return rec


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
             _detect_quant(r["abs_path"]), r["abs_path"]) for r in resolved]
    print(f"[manifest] {doc['name']}  arms={len(arms)}  "
          f"conc={list(CONCURRENCIES)}  isl={INPUT_LEN} osl={OUTPUT_LEN}",
          flush=True)
    outdir = pathlib.Path(a.outdir or
                          f"runs/residue_e2e_"
                          f"{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)
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
        # Round k starts at arm k. Anything that drifts over the run -- GPU
        # temperature, page cache, a neighbour waking up -- is otherwise
        # collinear with arm order and lands entirely on whichever arm goes
        # last. Rotation spreads it across all of them instead.
        order = arms[rnd % len(arms):] + arms[:rnd % len(arms)] if rotate else arms
        rdir = outdir / f"round{rnd}" if rounds > 1 else outdir
        rdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[round {rnd + 1}/{rounds}] order="
              f"{[t for t, *_ in order]}", flush=True)
        for slot, (tag, ratio, quant, model) in enumerate(order):
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
                print(f"[{tag}] ERROR: {rec['error']}", flush=True)

    print(f"\ndone -> {summary}  ({rounds} round(s))", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
