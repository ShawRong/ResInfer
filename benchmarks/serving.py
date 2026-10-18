"""Server diagnostics and cleanup shared by benchmark entrypoints."""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request


def check_port_available(port: int) -> None:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def wait_ready(port: int, timeout_s: int, proc, model: str) -> float | None:
    """Wait for a live server advertising the requested model, without proxies."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with code {proc.returncode} before readiness")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                healthy = response.status == 200
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                models = json.load(response)
        except (OSError, ValueError):
            time.sleep(3)
            continue
        if healthy:
            if model not in {item["id"] for item in models.get("data", [])}:
                raise RuntimeError("Ready endpoint advertises a different model")
            if proc.poll() is not None:
                raise RuntimeError("Server exited during readiness check")
            return time.monotonic() - started
        time.sleep(3)
    return None


def engagement(log: str, server_pid=None, *, multiprocessing=True,
               quant="residue_nvfp4") -> dict:
    """Require format-specific diagnostics from the process owning the model."""
    if "CUDA Error:" in log:
        return {"engagement_error": "CUDA driver error in server log"}
    if quant not in ("residue_nvfp4", "modelopt_mxfp4"):
        return {}
    engine_pids = set(re.findall(r"\(EngineCore(?:_[^ )]+)? pid=(\d+)\)", log))
    owners = engine_pids or ({str(server_pid)} if not multiprocessing else set())
    patched, enabled = set(), set()
    for line in log.splitlines():
        pids = re.findall(r"\bpid=(\d+)", line)
        if not pids:
            continue
        pid = pids[-1]
        if quant == "modelopt_mxfp4":
            if "[residue:plugin] applied" in line:
                patched.add(pid)
            if "MXFP4 weights ready: ResidueMxfp4LinearMethod mode=" in line:
                enabled.add(pid)
        else:
            if "ModelOpt NVFP4 -> ResidueNvFp4LinearMethodV2" in line:
                patched.add(pid)
            if "mode=" in line and "decode_mext=True" in line:
                enabled.add(pid)
    engaged = owners & patched & enabled
    expected = ("MXFP4 plugin and prepared Linear weights" if quant == "modelopt_mxfp4"
                else "residue patch and decode_mext=True")
    error = None if engaged else f"No model-process confirmation of {expected}"
    if "[residue:sitecustomize] patch FAILED" in log or "[residue:plugin] FAILED" in log:
        error = "Residue initialization failed"
    return {"residue_engine_pids": sorted(engaged), "engagement_error": error}


def stop_server(proc, timeout_s: float = 6) -> None:
    """Stop a server started with start_new_session, including orphaned workers."""
    # The group ID survives its leader; getpgid(proc.pid) does not.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            proc.poll()
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            continue
        break
    proc.wait(timeout=timeout_s)
