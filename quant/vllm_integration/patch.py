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

    if getattr(kw, "_residue_fold_warmup_installed", False):
        return
    _orig_flashinfer_autotune = kw.flashinfer_autotune

    # A SINGLE decode-sized forward is enough. FlashInfer's choose_one tunes
    # EVERY bucket in the op's gen_tuning_buckets (for the fold that is
    # RESIDUE_MEXT_AUTOTUNE_M2_BUCKETS = 16,32,64,128,256) in one call --
    # _generate_optimization_profiles emits one profile per bucket and choose_one
    # loops over all of them. So one forward at any decode M covers the whole
    # bucket range; extra forwards would just re-hit the cache. Cap the M at
    # fold_max_m (128) and at max_num_seqs so the fold path is actually taken.
    _FOLD_MAX_M = 128

    def _flashinfer_autotune_with_fold(runner) -> None:
        try:
            max_seqs = int(getattr(runner.scheduler_config, "max_num_seqs", 128))
            warmup_tok = max(1, min(_FOLD_MAX_M, max_seqs))
        except Exception:
            warmup_tok = _FOLD_MAX_M

        _orig_dummy_run = runner._dummy_run
        _state = {"done": False}

        def _dummy_run_with_decode(*args, **kwargs):
            out = _orig_dummy_run(*args, **kwargs)
            # After vLLM's prefill profiling forward, run ONE decode-sized forward
            # -- but only while actually tuning (leader rank), so non-leaders do
            # no extra work and only the leader's cache is broadcast.
            if (not _state["done"] and kwargs.get("is_profile")
                    and AutoTuner.get().is_tuning_mode):
                _state["done"] = True
                _orig_dummy_run(num_tokens=warmup_tok, skip_eplb=True,
                                is_profile=True)
            return out

        runner._dummy_run = _dummy_run_with_decode
        try:
            _orig_flashinfer_autotune(runner)
        finally:
            runner._dummy_run = _orig_dummy_run
        print(f"[residue] fold decode autotune folded into flashinfer warmup "
              f"(m_tok={warmup_tok}, tuned={_state['done']})", flush=True)

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

        # The decode-only fold is invisible to vLLM's prefill-sized autotune
        # warmup; add a decode-sized warmup so its tactic actually gets tuned.
        if resolved_version == "mext_r1":
            _install_fold_decode_autotune_warmup()

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
    """Check if patch is active.

    NOTE: This checks the actual class attribute, not the module-level _patched variable.
    This is important because in spawn multiprocessing, _patched resets to False in
    worker processes, but the class attribute check tells us the actual state.
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
