"""GPU ownership and runtime provenance shared by Linear benchmarks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import pathlib
import platform
import subprocess
import sys
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _md5(path: pathlib.Path) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                           capture_output=True, text=True, timeout=20)
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except Exception:                                           # noqa: BLE001
        return None


def _version(mod: str) -> str | None:
    loaded = getattr(sys.modules.get(mod), "__version__", None)
    if loaded is not None:
        return loaded
    try:
        return importlib.metadata.version({"flashinfer": "flashinfer-python"}.get(mod, mod))
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvsmi(index: int | str, fields: str, *, required: bool = False) -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{fields}", "--format=csv,noheader,nounits",
             "-i", str(index)], capture_output=True, text=True, timeout=20,
            check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        if required:
            raise RuntimeError(f"Cannot verify GPU {index} ownership") from exc
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def physical_index(logical: int) -> int | str:
    """Translate CUDA-visible ordinals; nvidia-smi also accepts GPU UUIDs."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return logical
    devices = [v.strip() for v in visible.split(",")]
    if logical < 0 or logical >= len(devices):
        raise ValueError(f"Device {logical} is outside CUDA_VISIBLE_DEVICES={visible!r}")
    selected = devices[logical]
    if selected.isdigit():
        return int(selected)
    if selected.startswith(("GPU-", "MIG-")):
        return selected
    raise ValueError(f"Invalid CUDA_VISIBLE_DEVICES entry: {selected!r}")


def gpu_state(index: int) -> dict:
    """Best-effort GPU telemetry; missing counters remain absent."""
    index = physical_index(index)
    out: dict = {"index": index}
    rows = _nvsmi(index, "gpu=name,uuid,clocks.sm,clocks.mem,clocks.max.sm,"
                         "temperature.gpu,"
                         "power.draw,memory.used,memory.total")
    if rows:
        keys = ("name", "uuid", "clock_sm_mhz", "clock_mem_mhz",
                "clock_max_sm_mhz", "temp_c",
                "power_w", "mem_used_mib", "mem_total_mib")
        out.update(dict(zip(keys, [v.strip() for v in rows[0].split(",")])))
    # Driver 580+ renamed clocks_throttle_reasons -> clocks_event_reasons;
    # ask for the new name and fall back, rather than losing the field on
    # whichever driver the reproducer has.
    for field in ("clocks_event_reasons.active",
                  "clocks_throttle_reasons.active"):
        rows = _nvsmi(index, f"gpu={field}")
        if rows and not rows[0].lower().startswith(("not supported", "n/a")):
            out["throttle_reasons"] = rows[0].strip()
            break
    apps = _nvsmi(index, "compute-apps=pid,used_memory")
    out["neighbors"] = sum(row.split(",")[0].strip() != str(os.getpid()) for row in apps)
    return out


TRACKED_ENV = (
    "RESIDUE_NVFP4", "RESIDUE_PATCH_MODE", "RESIDUE_ENABLE_DECODE_MEXT",
    "RESIDUE_FORCE_DECODE_MEXT", "RESIDUE_DECODE_MEXT_MAX_M",
    "RESIDUE_RATIO1_PREFILL_PATH", "RESIDUE_MEXT_R1_LAYOUT",
    "RESIDUE_PREFILL_RESIDUE_OFF", "RESIDUE_TENSOR_DUMP", "RESIDUE_SWAP",
    "VLLM_DISABLE_COMPILE_CACHE", "VLLM_BATCH_INVARIANT", "VLLM_PLUGINS",
    "FLASHINFER_CUBIN_DIR", "CUDA_VISIBLE_DEVICES",
)


def assert_gpu_exclusive(gpu: int, exclusive: bool = True) -> list[dict]:
    """Reject other compute processes; failed ownership queries are errors."""
    rows = _nvsmi(physical_index(gpu), "compute-apps=pid,used_memory", required=True)
    me = os.getpid()
    others = []
    for line in rows:
        parts = [x.strip() for x in line.split(",")]
        if not parts[0].isdigit():
            raise RuntimeError(f"Unrecognized GPU process row: {line!r}")
        if int(parts[0]) != me:
            # Identity NOW: the neighbour may exit before anyone asks whose
            # job it was, and then the pid alone answers nothing.
            other = {"used_mib": parts[1] if len(parts) > 1 else None}
            other.update(pid_identity(int(parts[0])))
            others.append(other)
    if others and exclusive:
        listing = "\n".join(
            f"    pid {o['pid']}  {o['used_mib']} MiB"
            + (f"\n      cmd  {o['cmdline']}" if o.get("cmdline") else "")
            + (f"\n      cwd  {o['cwd']}" if o.get("cwd") else "")
            + (f"\n      ({o['note']})" if o.get("note") else "")
            for o in others)
        raise SystemExit(
            f"gpu{gpu} is NOT exclusive -- {len(others)} other compute "
            f"process(es):\n{listing}\n"
            f"  This benchmark measures time, so a neighbour invalidates it. "
            f"Free memory is not\n  the test: 1.4 GiB clears any threshold and "
            f"still takes SMs. Use an empty GPU,\n  or --allow-shared-gpu, "
            f"which annotates every row instead of cleaning anything.")
    if others:
        print(f"[bench] WARNING: sharing gpu{gpu} with "
              f"{[o['pid'] for o in others]} -- every row is annotated; these "
              f"timings are not clean.", flush=True)
    return others


def provenance(gpu: int, run_id: str) -> dict:
    import torch

    spec = importlib.util.find_spec("nvfp4._C")
    so = pathlib.Path(spec.origin) if spec and spec.origin else None
    major, minor = torch.cuda.get_device_capability(gpu)
    return {
        "kind": "provenance",
        "run_id": run_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(REPO_ROOT),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_subject": _git("log", "-1", "--format=%s"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "ext_so": str(so) if so else None,
        "ext_so_mtime": (time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(so.stat().st_mtime))
                         if so else None),
        "ext_so_md5": _md5(so) if so else None,
        "sm": f"sm{major}{minor}",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _version("torch"),
        "vllm": _version("vllm"),
        "flashinfer": _version("flashinfer"),
        "cuda": torch.version.cuda,
        "gpu": gpu_state(gpu),
        "env": {k: os.environ.get(k, "<unset>") for k in TRACKED_ENV},
    }




def pid_identity(pid: int) -> dict:
    """Capture available /proc identity now; missing fields mean the PID exited."""
    import os
    import pathlib

    out: dict = {"pid": int(pid)}
    base = pathlib.Path(f"/proc/{int(pid)}")
    try:
        raw = (base / "cmdline").read_bytes()
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        if cmd:
            out["cmdline"] = cmd[:300]
    except OSError:
        pass
    try:
        # os.readlink, not Path.resolve(): resolve() is non-strict by default
        # and happily returns "/proc/<dead pid>/cwd" for a process that no
        # longer exists -- a plausible-looking string standing in for a
        # missing fact, which is the one thing this helper must not produce.
        out["cwd"] = os.readlink(str(base / "cwd"))
    except OSError:
        pass
    try:
        # Field 22 of /proc/pid/stat is starttime in clock ticks since boot.
        # comm may contain spaces/parens, so split after the last ')'.
        stat = (base / "stat").read_text()
        fields = stat[stat.rfind(")") + 1:].split()
        out["start_ticks"] = int(fields[19])
    except (OSError, IndexError, ValueError):
        pass
    if len(out) == 1:
        out["note"] = "process already exited -- identity unrecoverable"
    return out
