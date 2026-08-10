"""Patch vLLM's ModelOpt to use Residue linear methods.

This module provides a monkey-patch that replaces vLLM's standard
ModelOptNvFp4LinearMethod with our Residue NVFP4 linear method.

This is a HACK / workaround approach:
- We're monkey-patching vLLM's internal class attribute
- This bypasses vLLM's normal quantization registration mechanism
- The patch must be applied BEFORE model loading

Why we need it:
- vLLM's auto-detection aggressively overrides user quantization settings
- hf_quant_config.json with quant_method="modelopt" triggers this override
- There's no official way to extend ModelOpt quantization in vLLM

Usage:
    from quant.vllm_integration.patch import patch_vllm_modelopt
    patch_vllm_modelopt(model_dir="./exported")  # Pass model dir!

    llm = LLM(model="./exported", quantization="modelopt_fp4")
"""

import os

# Environment variable for passing model directory across process boundaries
# vLLM uses spawn multiprocessing, so class variables don't survive
RESIDUE_MODEL_DIR_ENV = "RESIDUE_VLLM_MODEL_DIR"
RESIDUE_PATCH_MODE_ENV = "RESIDUE_PATCH_MODE"

_patched = False


def _choose_linear_method_cls(version: str = "auto"):
    """Resolve the residue LinearMethod class + tag.

    There used to be two distinct subclasses (V2 + MExtR1). MExt has since been
    folded into V2.PWAL — both modes now resolve to the same class. We still
    accept the `mext_r1` / `mext` aliases (legacy suites and scripts pass them
    via `RESIDUE_PATCH_MODE`) and return a `"mext_r1"` tag so `is_patched_mext_r1`
    can still distinguish what mode the user asked for, but the class returned
    is always V2.
    """
    if version == "auto":
        patch_mode = os.environ.get(RESIDUE_PATCH_MODE_ENV, "legacy")
        patch_mode = patch_mode.strip().lower().replace("-", "_")
        if patch_mode in {"mext_r1", "mext"}:
            version = "mext_r1"
        elif patch_mode in {"", "legacy", "default"}:
            version = os.environ.get("RESIDUE_VLLM_PATCH_VERSION", "v2")
        else:
            raise ValueError(
                f"Unknown {RESIDUE_PATCH_MODE_ENV}={patch_mode!r}. "
                "Expected one of: legacy, mext_r1."
            )

    if version not in {"mext_r1", "mext", "auto", "legacy", "default", "v2"}:
        raise ValueError(f"Unknown residue vLLM patch version: {version}")

    from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel  # noqa: F401
    from .linear_method_v2 import ResidueNvFp4LinearMethodV2

    tag = "mext_r1" if version in {"mext_r1", "mext"} else "v2"
    return ResidueNvFp4LinearMethodV2, tag


def _install_shared_autotune_cache() -> None:
    "Let models with the same GEMM shapes reuse each other's tuned tactics."
    if os.environ.get("RESIDUE_SHARE_AUTOTUNE_CACHE", "0") != "1":
        return

    
    
    
    
    
    
    fac = None
    try:
        from vllm.model_executor.warmup import flashinfer_autotune_cache as fac
    except ImportError:
        pass
    try:
        from vllm.model_executor.warmup import kernel_warmup as kw
    except Exception as e:                                # noqa: BLE001
        print(f"[residue] shared autotune cache unavailable: {e}", flush=True)
        return

    holders = []                    # (module, attribute) pairs to rebind
    if fac is not None and hasattr(fac, "resolve_flashinfer_autotune_file"):
        holders.append((fac, "resolve_flashinfer_autotune_file"))
        # kernel_warmup imported the symbol by name at module load, so
        # rebinding the source module alone would leave the caller pointing
        # at the original.
        if hasattr(kw, "resolve_flashinfer_autotune_file"):
            holders.append((kw, "resolve_flashinfer_autotune_file"))
    elif hasattr(kw, "_resolve_flashinfer_autotune_file"):
        holders.append((kw, "_resolve_flashinfer_autotune_file"))
    if not holders:
        print("[residue] shared autotune cache unavailable: no known "
              "resolver in this vLLM's warmup module", flush=True)
        return

    anchor = holders[0][0]
    if getattr(anchor, "_residue_shared_cache_installed", False):
        return
    _orig_resolve = getattr(*holders[0])

    def _resolve_shared(runner):
        # Reuse the original to get the fully-formed path (it also creates the
        # directory and honours VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR), then swap
        # the model-hash component for a fixed one. Deriving the root by hand
        # would duplicate vLLM's version/arch layout and drift from it.
        try:
            per_model = _orig_resolve(runner)
            shared_dir = per_model.parent.parent / "shared"
            shared_dir.mkdir(parents=True, exist_ok=True)
            return shared_dir / per_model.name
        except Exception as e:                            # noqa: BLE001
            print(f"[residue] shared autotune cache path failed, using "
                  f"per-model: {type(e).__name__}: {e}", flush=True)
            return _orig_resolve(runner)

    for mod, attr in holders:
        setattr(mod, attr, _resolve_shared)
    anchor._residue_shared_cache_installed = True
    print("[residue] FlashInfer autotune cache SHARED across models "
          "(RESIDUE_SHARE_AUTOTUNE_CACHE=1). Entries are keyed by "
          "(op, runner, shape), so only genuinely new shapes are profiled. "
          "Do not start two servers concurrently against it -- FlashInfer "
          "documents concurrent writes as best-effort.", flush=True)


def _install_sm120_cutedsl_fold_warmup() -> None:
    'Pre-compile the SM120 CuTeDSL fold BEFORE cudagraph capture.'
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:                                # noqa: BLE001
        print(f"[residue] sm120 fold warmup hook unavailable: {e}", flush=True)
        return
    if getattr(GPUModelRunner, "_residue_sm120_warmup_installed", False):
        return
    _orig_capture = GPUModelRunner.capture_model

    def _capture_with_warmup(self, *args, **kwargs):
        # Belt and braces only: the real pre-compile happens in
        # register_sm120_fold_shape at PWAL, because this hook is NOT reliably
        # reached (vLLM also captures from profile_cudagraph_memory). Anything
        # PWAL already compiled is an lru_cache hit here, so this costs nothing
        # when it does run and covers a shape PWAL somehow missed when it does.
        try:
            from .residue_linear_op import (_FOLD_MAX_M,
                                            warmup_sm120_cutedsl_fold)

            sizes = getattr(self.vllm_config.compilation_config,
                            "cudagraph_capture_sizes", None) or []
            fold_sizes = sorted({int(m) for m in sizes if int(m) <= _FOLD_MAX_M})
            # Capture sizes are the graph shapes; the fold also runs eagerly at
            # other M below the threshold, so cover the powers of two too.
            fold_sizes = sorted(set(fold_sizes) | {
                m for m in (1, 2, 4, 8, 16, 32, 64, 128) if m <= _FOLD_MAX_M})
            if fold_sizes:
                warmup_sm120_cutedsl_fold(fold_sizes)
        except Exception as e:                            # noqa: BLE001
            print(f"[residue] sm120 fold warmup skipped: "
                  f"{type(e).__name__}: {str(e)[:90]}", flush=True)

        # sm10x: shape-independent, so it needs no registration -- one call
        # covers the whole space. Nothing did this before, which is why
        # B200/B300 compiled the fold on first use. Cheap (a handful of
        # kernels at ~1.3 s) and it removes the entire cold-start category.
        try:
            import torch

            major, minor = torch.cuda.get_device_capability()
            if major != 12:
                from nvfp4.fold import warmup as _fold_warmup

                n = _fold_warmup(int(major), int(minor))
                if n:
                    print(f"[residue] sm10x fold pre-compiled: {n} kernels",
                          flush=True)
        except Exception as e:                            # noqa: BLE001
            print(f"[residue] sm10x fold warmup skipped: "
                  f"{type(e).__name__}: {str(e)[:90]}", flush=True)

        # OBSERVATION covers both arches. A clean capture prints nothing; if
        # either warmup above under-covered, this names the keys it missed.
        # Guard ONLY the import. Putting _orig_capture inside the try would
        # re-run capture if vLLM itself raised ImportError somewhere in it.
        try:
            from nvfp4.fold import observe_compiles

            ctx = observe_compiles("cudagraph capture")
        except Exception:                                 # noqa: BLE001
            import contextlib

            ctx = contextlib.nullcontext()
        with ctx:
            return _orig_capture(self, *args, **kwargs)

    GPUModelRunner.capture_model = _capture_with_warmup
    GPUModelRunner._residue_sm120_warmup_installed = True


def _maybe_dump_fold_tuning() -> None:
    """Write the CuTeDSL fold tuner's decisions to RESIDUE_FOLD_TUNE_DUMP.

    Purely observational: the FlashInfer layer's e2e result is already a file
    (the autotune cache JSON), and this makes the python-tuner layer
    comparable the same way, so scripts/audit_autotune_coverage.py can diff a
    kernel-level tune against what the server actually ended up serving with.
    """
    path = os.environ.get("RESIDUE_FOLD_TUNE_DUMP")
    if not path:
        return
    try:
        import json

        from nvfp4.cute_fold.tactics import dump_state

        with open(path, "w") as f:
            json.dump(dump_state(), f, indent=1, sort_keys=True)
        print(f"[residue] fold tuning state dumped to {path}", flush=True)
    except Exception as e:                                # noqa: BLE001
        print(f"[residue] fold tuning dump failed: "
              f"{type(e).__name__}: {str(e)[:70]}", flush=True)







# The fold band's bucket ladder (fold_max_m caps at 128 on sm10x).
_FOLD_DECODE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


def _fold_tune_sizes(runner, warmup_tok: int) -> list[int]:
    'Token counts to run inside the fold tuning window.'
    sizes = {warmup_tok}
    try:
        import torch

        from .residue_linear_op import _SM120_FOLD_SHAPES

        if torch.cuda.get_device_capability()[0] == 10 and _SM120_FOLD_SHAPES:
            mnbt = int(getattr(runner.scheduler_config,
                               "max_num_batched_tokens", 0) or 0)

            def _cap(bucket: int) -> int:
                # A dummy run above max_num_batched_tokens is not legal;
                # min() keeps the forward legal while _m_bucket still files
                # the measurement under the bucket serving will look up.
                return min(bucket, mnbt) if mnbt else bucket

            sizes |= {_cap(b) for b in _FOLD_DECODE_BUCKETS}
    except Exception as e:                                # noqa: BLE001
        print(f"[residue] fold bucket probe failed "
              f"({type(e).__name__}: {str(e)[:70]}); tuning the given decode "
              "size only", flush=True)
    return sorted(sizes)


def _install_fold_decode_autotune_warmup() -> None:
    """Make the decode-only fold participate in FlashInfer autotuning -- and
    ride vLLM's cache + rank-0 broadcast so it is correct under TP too.

    vLLM's ``flashinfer_autotune`` runs ONE prefill-sized dummy forward inside
    ``autotune(tune_mode=True, cache=<file>)`` on rank 0, then broadcasts that
    cache file to every rank. The fold is decode-only (it engages at
    ``M <= fold_max_m``), so that prefill forward never exercises it and its
    ``choose_one`` always returns the fallback tactic (-1) -- the fold is never
    tuned.

    We do NOT tune the fold *after* vLLM's warmup: that would miss the leader's
    cache file and its broadcast, so each rank would tune its own shard and
    could pick a different tactic (a real TP hazard). Instead we inject the
    decode forwards INTO vLLM's warmup: while ``flashinfer_autotune`` runs, we
    temporarily wrap ``runner._dummy_run`` so its profiling call ALSO runs
    decode-sized forwards. Those execute inside the SAME ``autotune(cache=)``
    context, so the fold's tactic -- keyed by op name ``mext_fold_fp4_gemm`` --
    lands in the same JSON cache the leader saves and broadcasts. Every rank
    then loads one identical cache: TP-consistent, and persisted for fast
    restarts. Gated on ``is_tuning_mode`` so only the tuning (leader) rank pays
    the extra profiling. Best-effort: any failure leaves vLLM's warmup untouched.
    """
    try:
        import vllm.model_executor.warmup.kernel_warmup as kw
        from flashinfer.autotuner import AutoTuner
    except Exception as e:  # pragma: no cover - vLLM/flashinfer layout drift
        print(f"[residue] fold autotune warmup unavailable: {e}", flush=True)
        return

    try:
        from nvfp4.fold import tuning_mode as _fold_tuning_mode
    except Exception:
        # No CuTeDSL fold in this build; the FlashInfer half still applies.
        import contextlib as _cl

        def _fold_tuning_mode():
            return _cl.nullcontext()

    if getattr(kw, "_residue_fold_warmup_installed", False):
        return
    _orig_flashinfer_autotune = kw.flashinfer_autotune

    # ONE decode-sized forward covers the FlashInfer-registered ops entirely:
    # choose_one tunes EVERY bucket in the op's gen_tuning_buckets in one call
    # (_generate_optimization_profiles emits one profile per bucket), so the
    # sm120 C++ fold op is fully tuned by it. The CuTeDSL sm10x tuner is
    # BUCKET-LOCAL instead -- run_fold_autotuned caches per (shape, m-bucket)
    # and only measures the bucket it was called at. A mext_r1 layer takes the
    # fold at EVERY M (4083dee), so on sm10x its prefill buckets need their own
    # tuning forwards or they silently serve the static fallback table.
    _FOLD_MAX_M = 128

    def _flashinfer_autotune_with_fold(runner) -> None:
        try:
            max_seqs = int(getattr(runner.scheduler_config, "max_num_seqs", 128))
            warmup_tok = max(1, min(_FOLD_MAX_M, max_seqs))
        except Exception:
            warmup_tok = _FOLD_MAX_M
        tune_sizes = _fold_tune_sizes(runner, warmup_tok)

        _orig_dummy_run = runner._dummy_run
        _state = {"done": False}

        def _dummy_run_with_decode(*args, **kwargs):
            out = _orig_dummy_run(*args, **kwargs)
            # After vLLM's prefill profiling forward, run the tuning forwards
            # -- but only while actually tuning (leader rank), so non-leaders do
            # no extra work and only the leader's cache is broadcast.
            if (not _state["done"] and kwargs.get("is_profile")
                    and AutoTuner.get().is_tuning_mode):
                _state["done"] = True
                # The same forwards prime BOTH tuners. The C++ sm120 fold is
                # tuned by FlashInfer via this autotune(cache=) context (tactic
                # keyed on mext_fold_fp4_gemm; all buckets in the first call).
                # The CuTeDSL sm10x fold keeps its own python cache instead, so
                # it needs a window in which run_fold() is allowed to measure --
                # it must never do that while serving, where a cudagraph
                # capture is live. Per-size guard: losing one bucket degrades
                # that bucket to the static table, which must not stop startup.
                with _fold_tuning_mode():
                    for m in tune_sizes:
                        try:
                            _orig_dummy_run(num_tokens=m, skip_eplb=True,
                                            is_profile=True)
                        except Exception as e:            # noqa: BLE001
                            print(f"[residue] fold tuning forward m={m} "
                                  f"failed: {type(e).__name__}: "
                                  f"{str(e)[:70]}", flush=True)
            return out

        runner._dummy_run = _dummy_run_with_decode
        try:
            _orig_flashinfer_autotune(runner)
        finally:
            runner._dummy_run = _orig_dummy_run
        tuned_cutedsl = None
        try:
            from nvfp4.cute_fold.tactics import _AUTOTUNE_CACHE
            tuned_cutedsl = len(_AUTOTUNE_CACHE)
        except Exception:
            pass
        print(f"[residue] fold autotune folded into flashinfer warmup "
              f"(m_tok={tune_sizes}, tuned={_state['done']}, "
              f"cutedsl_cached={tuned_cutedsl})", flush=True)
        _maybe_dump_fold_tuning()



    kw.flashinfer_autotune = _flashinfer_autotune_with_fold
    kw._residue_fold_warmup_installed = True
    print("[residue] Installed fold decode autotune warmup", flush=True)


def patch_vllm_modelopt(model_dir: str | None = None, version: str = "auto") -> bool:
    """Replace ModelOptNvFp4Config.LinearMethodCls with a residue method.

    Must be called BEFORE vLLM creates model. Safe to call multiple times.

    Args:
        model_dir: Path to the exported model directory. REQUIRED for residue
            quantization to work. This is passed via environment variable
            to survive vLLM's spawn multiprocessing.

    Returns:
        True if patch was applied, False if already patched.
    """
    global _patched

    # Always set/update model_dir even if already patched
    # (user might call with different model)
    if model_dir is not None:
        os.environ[RESIDUE_MODEL_DIR_ENV] = str(model_dir)
        print(f"[residue] Set {RESIDUE_MODEL_DIR_ENV}={model_dir}")

    # The env contract, enforced where every launcher converges. Prints one
    # resolved-config line (so logs always say what actually ran) and refuses
    # known-fatal combinations at startup instead of hours later -- each rule
    # in env_contract.py names the real incident it prevents.
    # RESIDUE_ENV_CHECK=0 downgrades refusals to warnings.
    from .env_contract import enforce as _enforce_env_contract
    _enforce_env_contract(model_dir=os.environ.get(RESIDUE_MODEL_DIR_ENV))

    if _patched:
        return False

    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        # RESIDUE_LINEAR_OFF=1: leave every LinearBase layer on stock vLLM
        # (auto kernel selection included) and install ONLY the MoE hook +
        # persistent-cache patch. For MoE-focused runs on checkpoints whose
        # linear layers carry no residue, the linear takeover buys nothing
        # and costs the kernel-choice gap (mext pwal only speaks the
        # cutlass weight format, so it force-pins the linear backend).
        if os.environ.get("RESIDUE_LINEAR_OFF", "0") == "1":
            try:
                from . import patch_flashinfer_persistent_cache  # noqa: F401
            except Exception as _e:  # pragma: no cover - best-effort
                print(f"[residue] persistent-cache patch import failed: "
                      f"{_e}", flush=True)
            from .residue_moe import patch_moe_method, patch_moe_observer

            if not patch_moe_method():
                patch_moe_observer()
            _patched = True
            print("[residue] LINEAR OFF -- stock linear layers, MoE hook "
                  "only", flush=True)
            return True

        target_cls, resolved_version = _choose_linear_method_cls(version)

        # Store original for potential restoration
        if not hasattr(ModelOptNvFp4Config, "_original_LinearMethodCls"):
            ModelOptNvFp4Config._original_LinearMethodCls = (
                ModelOptNvFp4Config.LinearMethodCls
            )

        # Apply patch
        ModelOptNvFp4Config.LinearMethodCls = target_cls
        ModelOptNvFp4Config._residue_patch_version = resolved_version

        # vLLM >= 0.23 also sets `self.LinearMethodCls` inside
        # ModelOptNvFp4Config.__init__ based on quant_algo, which overrides the
        # class attribute above for each new config instance. Patch __init__ so
        # serialized NVFP4 checkpoints still instantiate the residue method.
        if not hasattr(ModelOptNvFp4Config, "_residue_original_init"):
            ModelOptNvFp4Config._residue_original_init = ModelOptNvFp4Config.__init__

            def _residue_init(self, *args, **kwargs):
                self._residue_original_init(*args, **kwargs)
                qm = getattr(self, "quant_method", None)
                if qm == "NVFP4":
                    self.LinearMethodCls = target_cls
                if os.environ.get("RESIDUE_DEBUG_CREATE_WEIGHTS", "0") == "1":
                    print(
                        f"[residue:config_init] quant_method={qm!r} "
                        f"LinearMethodCls={getattr(self, 'LinearMethodCls', None)}",
                        flush=True,
                    )

            ModelOptNvFp4Config.__init__ = _residue_init

        # Newer vLLM calls LinearMethodCls(self) from get_quant_method() and
        # does not pass the layer prefix into the method constructor. The
        # prefix is still needed for residue metadata lookup, especially for
        # vLLM-fused layers such as qkv_proj and gate_up_proj whose checkpoint
        # keys are split as q_proj/k_proj/v_proj and gate_proj/up_proj.
        if not hasattr(ModelOptNvFp4Config, "_residue_original_get_quant_method"):
            ModelOptNvFp4Config._residue_original_get_quant_method = (
                ModelOptNvFp4Config.get_quant_method
            )

            def _residue_get_quant_method(self, layer, prefix: str):
                method = self._residue_original_get_quant_method(layer, prefix)
                if isinstance(method, target_cls):
                    method.layer_prefix = prefix
                if os.environ.get("RESIDUE_DEBUG_CREATE_WEIGHTS", "0") == "1":
                    print(
                        f"[residue:get_quant_method] prefix={prefix!r} "
                        f"method={type(method).__name__} "
                        f"cfg_cls={getattr(self, 'LinearMethodCls', None)}",
                        flush=True,
                    )
                return method

            ModelOptNvFp4Config.get_quant_method = _residue_get_quant_method

        # CRITICAL: Add our LinearMethod to WEIGHT_LOADER_V2_SUPPORTED
        # Without this, vLLM uses the old weight_loader which doesn't correctly
        # handle PerTensorScaleParameter (weight_scale_2) for merged layers
        # like gate_up_proj. The old loader walks into an else branch that
        # asserts param_data.shape == loaded_weight.shape, but:
        #   - param_data.shape = [2] (for gate_up_proj with 2 partitions)
        #   - loaded_weight.shape = [] (scalar from checkpoint)
        # The new weight_loader_v2 correctly calls param.load_merged_column_weight()
        # which handles scalar-to-array mapping via _load_into_shard_id().
        try:
            from vllm.model_executor.layers import linear

            if target_cls.__name__ not in linear.WEIGHT_LOADER_V2_SUPPORTED:
                linear.WEIGHT_LOADER_V2_SUPPORTED.append(target_cls.__name__)
                print(f"[residue] Added {target_cls.__name__} to WEIGHT_LOADER_V2_SUPPORTED")
        except (ImportError, AttributeError) as e:
            print(f"[residue] Warning: Could not patch WEIGHT_LOADER_V2_SUPPORTED: {e}")

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        _install_fold_decode_autotune_warmup()

        # Same reasoning, already applied here: extended_k checkpoints take the
        # fold too (via --enable_decode_mext) and need the same pre-compile.
        _install_sm120_cutedsl_fold_warmup()

        # Must run BEFORE kernel_warmup resolves the cache path. Self-gated on
        # RESIDUE_SHARE_AUTOTUNE_CACHE=1 and a no-op otherwise.
        _install_shared_autotune_cache()

        
        
        
        
        
        
        if os.environ.get("RESIDUE_STACK_DUMP_SIGNAL") == "1":
            import faulthandler
            import signal as _signal

            faulthandler.register(_signal.SIGUSR1, all_threads=True,
                                  chain=False)
            print(f"[residue] SIGUSR1 stack dumps armed in pid={os.getpid()}",
                  flush=True)

        # Re-enable vLLM's FlashInfer PERSISTENT autotune cache so the tuned
        # tactics (fold + fp4_gemm) are written once and reloaded on later boots
        # instead of re-tuned every start. Safe now that flashinfer #3367 fixed
        # the use_8x4_sf_layout key collision; the patch self-gates and refuses
        # on an unfixed flashinfer. Not restricted to mext_r1 -- plain NVFP4
        # serving benefits too.
        try:
            from . import patch_flashinfer_persistent_cache  # noqa: F401  auto-installs on import
        except Exception as _e:  # pragma: no cover - best-effort
            print(f"[residue] persistent-cache patch import failed: {_e}", flush=True)

        # LinearMethodCls only covers LinearBase. `RoutedExperts` layers go
        # to FusedMoEMethodCls. RESIDUE_ENABLE_MOE=1 installs the REAL
        # residue expert method (ResidueNvFp4Experts on the fold chain);
        # RESIDUE_MOE_OBSERVE=1 the numerics-neutral coverage observer.
        # Method wins when both are set.
        try:
            from .residue_moe import patch_moe_method, patch_moe_observer

            if not patch_moe_method():
                patch_moe_observer()
        except Exception as _e:  # pragma: no cover - best-effort
            print(f"[residue] MoE hook install failed: {_e}", flush=True)

        _patched = True
        print(
            "[residue] Patched ModelOptNvFp4Config.LinearMethodCls "
            f"-> {target_cls.__name__} ({resolved_version})"
        )
        return True

    except ImportError as e:
        print(f"[residue] Warning: Could not patch vLLM: {e}")
        return False


def unpatch_vllm_modelopt() -> bool:
    """Restore original ModelOptNvFp4LinearMethod.

    Returns:
        True if restored, False if not patched.
    """
    global _patched
    if not _patched:
        return False

    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        if hasattr(ModelOptNvFp4Config, "_original_LinearMethodCls"):
            ModelOptNvFp4Config.LinearMethodCls = (
                ModelOptNvFp4Config._original_LinearMethodCls
            )
            _patched = False
            print("[residue] Restored original ModelOptNvFp4LinearMethod")
            return True
    except ImportError:
        pass

    return False


def patch_vllm_modelopt_v2(model_dir: str | None = None) -> bool:
    return patch_vllm_modelopt(model_dir=model_dir, version="v2")


def patch_vllm_modelopt_mext_r1(model_dir: str | None = None) -> bool:
    return patch_vllm_modelopt(model_dir=model_dir, version="mext_r1")


def is_patched() -> bool:
    """Check if the patch's DECLARED configuration is actually installed.

    NOTE: This checks actual class attributes, not the module-level _patched
    variable. This is important because in spawn multiprocessing, _patched
    resets to False in worker processes, but the class attribute check tells
    us the actual state.

    What "installed" means depends on the configuration:
      * RESIDUE_LINEAR_OFF=1 -- the linear side is DELIBERATELY stock, so
        checking LinearMethodCls would report failure on a correct install
        (found the hard way: the polhalf e2e's plugin refused to serve while
        everything it asked for was in place). There the installed state is
        the MoE hook (when RESIDUE_ENABLE_MOE=1) and nothing else.
      * otherwise -- the linear method class is the residue method.
    """
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        if os.environ.get("RESIDUE_LINEAR_OFF", "0") == "1":
            if os.environ.get("RESIDUE_ENABLE_MOE", "0") == "1":
                moe_cls = getattr(
                    ModelOptNvFp4Config, "FusedMoEMethodCls", None)
                return getattr(moe_cls, "__name__", "") == (
                    "ModelOptNvFp4FusedMoEResidue")
            # LINEAR_OFF with no MoE hook requested: an intentional no-op
            # install (env contract warns about it); nothing to verify.
            return True

        from .linear_method_v2 import ResidueNvFp4LinearMethodV2

        return ModelOptNvFp4Config.LinearMethodCls is ResidueNvFp4LinearMethodV2
    except ImportError:
        return False


def is_patched_mext_r1() -> bool:
    """Whether the patch was installed in mext_r1 mode.

    Since MExt was folded into V2, both modes resolve to the same class — this
    check now reads the patch *tag* (stored at patch time) rather than checking
    class identity, so callers can still see what mode the user requested.
    """
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        return getattr(ModelOptNvFp4Config, "_residue_patch_version", None) == "mext_r1"
    except ImportError:
        return False
