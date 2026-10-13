"""Shared ResInfer environment setup and vLLM configuration for quality runs."""
from __future__ import annotations

import json
import os
import pathlib
import sys

from benchmarks.engine_config import accuracy_engine_config, load_engine_config, merge_engine_config


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
    if mxfp4 and any(getattr(a, key, False) for key in (
            "prefill_residue_off", "residue_off", "decode_mext_off", "prefill_only")):
        raise ValueError("MXFP4 phase ablations have not been validated by this runner")
    os.environ["RESIDUE_MXFP4"] = "1" if mxfp4 else "0"
    if mxfp4:
        os.environ["RESIDUE_NVFP4"] = "0"
        os.environ["RESIDUE_VLLM_MODEL_DIR"] = a.model
    if a.bf16:
        a.plain = True
    if a.plain:
        os.environ["RESIDUE_NVFP4"] = "0"
    # Use the same process mode for every arm. The patch requires in-process
    # EngineCore on the tested vLLM runtime.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if not a.plain and not mxfp4:
        os.environ["RESIDUE_NVFP4"] = "1"
        os.environ["RESIDUE_VLLM_MODEL_DIR"] = a.model
        # Keep the extended-K loader enabled during ablations. Disable residue
        # at compute time with the PREFILL_RESIDUE_OFF bounds below.
        os.environ["RESIDUE_ENABLE_DECODE_MEXT"] = "1"
        os.environ["RESIDUE_FORCE_DECODE_MEXT"] = "0"
        os.environ["RESIDUE_DECODE_MEXT_MAX_M"] = str(getattr(a, "mext_max_m", 64))
        # Prevent inherited GEMM backend overrides from bypassing the residue patch.
        for var in ("FLASHINFER_CUBIN_DIR", "VLLM_HAS_FLASHINFER_CUBIN"):
            if os.environ.pop(var, None) is not None:
                print(f"[kit] dropped {var}: incompatible with a residue arm")
    if getattr(a, "decode_mext_off", False):
        raise SystemExit(
            "--decode-mext-off is not implementable today. RESIDUE_ENABLE_DECODE_MEXT=0\n"
            "breaks weight loading (KeyError 'weight_scale_base', vLLM 0.26) and\n"
            "RESIDUE_DECODE_MEXT_MAX_M=0 silently keeps residue on. See install_env().")
    if getattr(a, "prefill_residue_off", False) or getattr(a, "residue_off", False):
        if a.plain:
            raise SystemExit("--prefill-residue-off is meaningless with --plain/--bf16")
        os.environ["RESIDUE_PREFILL_RESIDUE_OFF"] = "1"
        if getattr(a, "residue_off", False):
            # threshold 0 -> every M is "> 0" -> residue killed everywhere
            os.environ["RESIDUE_PREFILL_RESIDUE_OFF_MIN_M"] = "0"
    # MAX_M disables residue in the decode band, preserving prefill residue.
    if getattr(a, "prefill_only", None) is not None:
        if a.plain:
            raise SystemExit("--prefill-only is meaningless with --plain/--bf16")
        os.environ["RESIDUE_PREFILL_RESIDUE_OFF"] = "1"
        os.environ["RESIDUE_PREFILL_RESIDUE_OFF_MAX_M"] = str(a.prefill_only)
        if os.environ.get("RESIDUE_PREFILL_RESIDUE_OFF_MIN_M") is not None:
            # Reject overlapping ablations that would disable residue in both bands.
            raise SystemExit(
                "--prefill-only cannot be combined with --residue-off: both "
                "bounds set kills BOTH bands and yields the plain cell.")
    repo = a.repo or os.environ.get("RESINFER_REPO", "")
    if repo:
        parts = [repo, str(pathlib.Path(repo) / "csrc")]
        os.environ["PYTHONPATH"] = ":".join(
            [p for p in parts + [os.environ.get("PYTHONPATH", "")] if p])
        # Update this process as well as the worker import path.
        for p in reversed(parts):
            if p not in sys.path:
                sys.path.insert(0, p)
    # Disable compile-cache reuse across accuracy arms by default.
    if os.environ.get("RESIDUE_EVALKIT_ALLOW_CACHE", "0") == "1":
        # Report the explicit cache override in the run log.
        print("[kit] " + "!" * 62, flush=True)
        print("[kit] !! RESIDUE_EVALKIT_ALLOW_CACHE=1 -- torch.compile cache is ON.")
        print("[kit] !! NUMBERS FROM THIS RUN ARE NOT PUBLISHABLE. An arm's score")
        print("[kit] !! depends on what ran before it in the queue.")
        print("[kit] " + "!" * 62, flush=True)
    else:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    # Log effective environment settings for comparison across arms.
    print("[kit] execution contract: " + " ".join(
        f"{k}={os.environ.get(k, '<unset>')}" for k in (
            "VLLM_ENABLE_V1_MULTIPROCESSING", "RESIDUE_NVFP4",
            "RESIDUE_ENABLE_DECODE_MEXT", "RESIDUE_PREFILL_RESIDUE_OFF",
            "RESIDUE_PREFILL_RESIDUE_OFF_MIN_M",
            "RESIDUE_PREFILL_RESIDUE_OFF_MAX_M",
            "VLLM_DISABLE_COMPILE_CACHE")), flush=True)
    # flashinfer JIT links -lcuda from a stub dir that may not contain it
    os.environ["LIBRARY_PATH"] = ("/usr/lib/x86_64-linux-gnu:"
                                  + os.environ.get("LIBRARY_PATH", ""))


def quality_quantization(a):
    if a.bf16:
        return None
    return getattr(a, "quantization", None) or "modelopt_fp4"


def pinned_engine_kwargs(a, max_model_len: int) -> dict:
    """Legacy defaults < shared JSON < explicit compilation diagnostics."""
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
    config = accuracy_engine_config(base, load_engine_config(getattr(a, "engine_config", None)))
    compilation = {}
    if getattr(a, "cudagraph_mode", ""):
        compilation["cudagraph_mode"] = a.cudagraph_mode
    if getattr(a, "compile_mode", ""):
        compilation["mode"] = a.compile_mode
    if getattr(a, "inductor_config", ""):
        value = json.loads(a.inductor_config)
        if not isinstance(value, dict):
            raise ValueError("--inductor-config must be a JSON object")
        compilation["inductor_compile_config"] = value
    return merge_engine_config(config, {"compilation_config": compilation})


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
    for flag in ("bf16", "plain", "prefill-residue-off", "residue-off"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--decode-mext-off", action="store_true",
                        help="unsupported: disabling the extended-K loader prevents weight loading")
    parser.add_argument("--flashinfer-autotune", action="store_true",
                        help="enable FlashInfer autotuning (default: disabled)")
    parser.add_argument("--engine-config", help="JSON vLLM kwargs; overrides legacy engine flags")
    parser.add_argument("--quantization", choices=("modelopt_fp4", "modelopt_mxfp4"),
                        help="loader; default modelopt_fp4; MXFP4 needs the installed vLLM plugin")
