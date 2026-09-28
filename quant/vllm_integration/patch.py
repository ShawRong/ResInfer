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
        patch_mode = os.environ.get(RESIDUE_PATCH_MODE_ENV, "v2")
        patch_mode = patch_mode.strip().lower().replace("-", "_")
        if patch_mode in {"", "v2", "mext_r1"}:
            version = patch_mode or "v2"
        else:
            
            
            
            
            
            
            
            
            raise ValueError(
                f"Unknown {RESIDUE_PATCH_MODE_ENV}={patch_mode!r}. Valid: "
                "'v2' (default -- metadata-driven; metadata-less layers "
                "stay plain) or 'mext_r1' (additionally name-force "
                "metadata-less projections onto the runtime M-extension "
                "path)."
            )

    if version not in {"v2", "mext_r1"}:
        raise ValueError(
            f"Unknown residue vLLM patch version: {version!r}. "
            "Valid: 'v2', 'mext_r1'."
        )

    from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel  # noqa: F401
    from .linear_method_v2 import ResidueNvFp4LinearMethodV2

    tag = "mext_r1" if version == "mext_r1" else "v2"
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
    """
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

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
