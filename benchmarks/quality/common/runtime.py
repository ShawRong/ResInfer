"""Shared ResInfer environment setup and vLLM configuration for quality runs."""
from __future__ import annotations

import json
import os
import pathlib
import sys

from benchmarks.checkpoints import validate_kind
from benchmarks.engine_config import accuracy_engine_config, load_engine_config


def install_env(a):
    """Every RESIDUE_* the workers need, set BEFORE vllm is imported."""
    # Keep Hugging Face repo IDs unchanged; resolve only existing local paths.
    if pathlib.Path(a.model).exists():
        a.model = str(pathlib.Path(a.model).resolve())
        config_path = pathlib.Path(a.model) / "config.json"
        if config_path.is_file():
            checkpoint_config = json.loads(config_path.read_text())
            checkpoint_quant = checkpoint_config.get("quantization_config") or {}
            if checkpoint_quant.get("quant_method") == "residue_mxfp4":
                raise ValueError("Legacy residue_mxfp4 checkpoints require the isolated "
                                 "historical adapter; this runner supports native ModelOpt "
                                 "MXFP4 only. Do not relabel the checkpoint config.")
    quantization = getattr(a, "quantization", None)
    if a.bf16 and quantization:
        raise ValueError("--bf16 cannot be combined with --quantization")
    mxfp4 = quantization == "modelopt_mxfp4"
    kind = ("bf16" if a.bf16 else "modelopt_mxfp4" if mxfp4
            else "modelopt_fp4" if a.plain else "residue")
    if pathlib.Path(a.model).is_dir():
        validate_kind(a.model, kind)
    if (any(getattr(a, key, False) for key in ("prefill_residue_off", "residue_off"))
            or getattr(a, "prefill_only", None) is not None):
        raise ValueError("Runtime ablations belong in .dev; use a stock checkpoint for the baseline")
    if a.bf16:
        a.plain = True
    repo = a.repo or os.environ.get("RESINFER_REPO", "")
    configure_env(os.environ, model=a.model, kind=kind, repo=repo,
                  mext_max_m=getattr(a, "mext_max_m", 64))
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    print("[kit] execution contract: " + " ".join(
        f"{key}={os.environ.get(key, '<unset>')}" for key in (
            "VLLM_ENABLE_V1_MULTIPROCESSING", "RESIDUE_NVFP4", "RESIDUE_MXFP4",
            "RESIDUE_ENABLE_DECODE_MEXT", "VLLM_DISABLE_COMPILE_CACHE")), flush=True)


def configure_env(env, *, model, kind, repo, mext_max_m=64):
    """Set the same arm environment for offline engines and HTTP servers."""
    env["RESIDUE_NVFP4"] = "1" if kind == "residue" else "0"
    env["RESIDUE_MXFP4"] = "1" if kind == "modelopt_mxfp4" else "0"
    env["RESIDUE_ATTN"] = "0"
    if kind in ("residue", "modelopt_mxfp4"):
        env["RESIDUE_VLLM_MODEL_DIR"] = model
    if kind == "residue":
        env["RESIDUE_ENABLE_DECODE_MEXT"] = "1"
        env["RESIDUE_FORCE_DECODE_MEXT"] = "0"
        env["RESIDUE_DECODE_MEXT_MAX_M"] = str(mext_max_m)
        for var in ("FLASHINFER_CUBIN_DIR", "VLLM_HAS_FLASHINFER_CUBIN"):
            if env.pop(var, None) is not None:
                print(f"[kit] dropped {var}: incompatible with a residue arm")
    # The patch requires in-process EngineCore on the tested runtime, for every arm.
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if env.get("RESIDUE_EVALKIT_ALLOW_CACHE", "0") == "1":
        print("[kit] compile-cache override enabled; not an isolated accuracy run", flush=True)
    else:
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    if repo:
        env["PYTHONPATH"] = os.pathsep.join(p for p in (
            str(repo), env.get("PYTHONPATH", "")) if p)
    # FlashInfer needs libcuda; empty entries make GCC search cwd for its specs file.
    env["LIBRARY_PATH"] = os.pathsep.join(p for p in (
        "/usr/lib/x86_64-linux-gnu", *env.get("LIBRARY_PATH", "").split(os.pathsep)) if p)


def quality_quantization(a):
    if a.bf16:
        return None
    return getattr(a, "quantization", None) or "modelopt_fp4"


def initialize_runtime():
    """Use the repository plugin even when its wheel entry point is not installed."""
    from runtime.vllm_plugin.residue_vllm_plugin import initialize

    initialize()


def pinned_engine_kwargs(a, max_model_len: int) -> dict:
    """Apply shared engine JSON over the runner's defaults."""
    short = getattr(a, "cmd", None) == "task" or getattr(a, "_venue", "") == "task"
    base = dict(
        enforce_eager=bool(getattr(a, "enforce_eager", False)),
        max_num_batched_tokens=(getattr(a, "max_num_batched_tokens", None)
                                or (2048 if short else max_model_len)),
        kernel_config={"enable_flashinfer_autotune": bool(
            getattr(a, "flashinfer_autotune", False))},
        seed=a.seed, max_model_len=max_model_len,
        gpu_memory_utilization=a.gpu_mem_util, max_num_seqs=a.max_num_seqs,
    )
    return accuracy_engine_config(base, load_engine_config(getattr(a, "engine_config", None)))


def write_effective_engine_config(llm, a):
    """Record vLLM's resolved engine settings after initialization."""
    config = llm.llm_engine.vllm_config
    # Do not asdict/deepcopy VllmConfig: compilation state contains live layers.
    fields = {
        "model_config": ("model", "dtype", "quantization", "max_model_len", "seed", "enforce_eager"),
        "cache_config": ("cache_dtype", "block_size", "gpu_memory_utilization",
                         "kv_cache_memory_bytes", "enable_prefix_caching", "calculate_kv_scales"),
        "parallel_config": ("tensor_parallel_size", "pipeline_parallel_size"),
        "scheduler_config": ("max_num_seqs", "max_num_batched_tokens",
                             "enable_chunked_prefill", "async_scheduling"),
        "compilation_config": ("mode", "cudagraph_mode", "cudagraph_capture_sizes",
                               "max_cudagraph_capture_size", "inductor_compile_config", "pass_config"),
        "kernel_config": ("enable_flashinfer_autotune",),
        "attention_config": ("backend",),
    }
    effective = {}
    for name, keys in fields.items():
        section = getattr(config, name, None)
        if section is None:
            continue
        effective[name] = {key: getattr(section, key) for key in keys if hasattr(section, key)}
    path = pathlib.Path(a.out) / f"{a.arm}__engine.json"
    path.write_text(json.dumps(effective, indent=2, default=str) + "\n")


def write_provenance(a, out_dir, engine_kwargs: dict, extra: dict | None = None):
    """Save the model, runtime, environment and calibration identity for one arm."""
    import platform
    rec = {"arm": a.arm, "model": a.model, "cmd": sys.argv[1:],
           "knobs": engine_kwargs, "quantization": quality_quantization(a),
           "env": {k: os.environ.get(k) for k in sorted(
               [e for e in os.environ if e.startswith(("RESIDUE_", "VLLM_"))])},
           "python": platform.python_version(), "platform": platform.platform()}
    # Record the export recipe alongside the runtime configuration.
    rec["calibration"] = None
    try:
        cp = pathlib.Path(a.model) / "calibration_provenance.json"
        if cp.exists():
            rec["calibration"] = json.loads(cp.read_text())
    except Exception as exc:  # noqa: BLE001
        rec["calibration"] = {"error": f"{type(exc).__name__}: {exc}"}
    for mod in ("vllm", "torch", "flashinfer", "transformers"):
        try:
            rec.setdefault("versions", {})[mod] = __import__(mod).__version__
        except Exception:
            rec.setdefault("versions", {})[mod] = None
    try:
        import subprocess
        rec["git_head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=a.repo or ".", capture_output=True,
            text=True).stdout.strip() or None
    except Exception:
        rec["git_head"] = None
    try:
        import torch
        rec["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        rec["gpu"] = None
    if extra:
        rec.update(extra)
    p = pathlib.Path(out_dir) / f"{a.arm}__provenance.json"
    with p.open("x") as stream:
        stream.write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(f"[kit] provenance -> {p}", flush=True)


def add_runtime_arguments(parser):
    """Add shared model and engine options to an evaluation command."""
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default="", help="repository root for runtime imports")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-mem-util", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mext-max-m", type=int, default=64)
    for flag in ("bf16", "plain"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--flashinfer-autotune", action="store_true",
                        help="enable FlashInfer autotuning (default: disabled)")
    parser.add_argument("--engine-config", help="JSON vLLM kwargs; overrides legacy engine flags")
    parser.add_argument("--quantization", choices=("modelopt_fp4", "modelopt_mxfp4"),
                        help="loader; default modelopt_fp4; MXFP4 needs the installed vLLM plugin")
