# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'FlashInfer-autotuned dispatch for the mext_r1 prefill GEMM on sm120.'

from __future__ import annotations

from typing import Callable

import os

import torch





KLOOP_BUCKETS = (128, 256, 512, 1024, 2048)
# sm120 kloop tiles (3-tuples, per-(shape, tile) compiles -> whitelist).
KLOOP_TILES = ((128, 128, 128),)
# sm100 kloop (tile, cluster) configs. Symbolic-shape compiles: ONE compile
# per config covers every shape and m -- no buckets, no padding, whitelist
# keyed on the config alone. (256,128) is a 2-SM MMA and needs cluster (2,1).
KLOOP_SM100_CFGS = (((128, 128), (1, 1)), ((256, 128), (2, 1)))

_CUSTOM_OP = "residue_mext_prefill_fp4_gemm"

# sm120: (n_w, k_logical, tile) whose kloop kernel exists in the lru cache.
# sm100: ("sm100", tile, cluster).
_PRECOMPILED: set = set()


def _bucket_of(m: int) -> int:
    for b in KLOOP_BUCKETS:
        if m <= b:
            return b
    return KLOOP_BUCKETS[-1]


def _map_to_tuning_buckets(x: int) -> int:
    # Module-level named function: TuningConfig is hashed into the cache key,
    # so this must be ONE stable object, not a fresh lambda per call.
    return _bucket_of(int(x))


_TUNING_CONFIG = None


def _tuning_config():
    global _TUNING_CONFIG
    if _TUNING_CONFIG is None:
        from flashinfer.autotuner import DynamicTensorSpec, TuningConfig

        _TUNING_CONFIG = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    # tuples, not lists: the spec is hashed into the cache key
                    # (its docstring says "list", its __hash__ says otherwise)
                    input_idx=(0,),
                    dim_idx=(0,),
                    gen_tuning_buckets=KLOOP_BUCKETS,
                    map_to_tuning_buckets=_map_to_tuning_buckets,
                ),
            ),
        )
    return _TUNING_CONFIG


def _pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    if x.shape[0] == rows:
        return x
    xp = torch.zeros(rows, x.shape[1], dtype=x.dtype, device=x.device)
    xp[: x.shape[0]] = x
    return xp


def _is_sm10x() -> bool:
    return torch.cuda.get_device_capability()[0] == 10


def _run_kloop(inputs, tactic_idx) -> torch.Tensor:
    import nvfp4

    x, w, gs_inv, wsb, alpha = inputs
    m = x.shape[0]
    if _is_sm10x():
        # Symbolic-shape kernel: no padding, no buckets, D is [m, n_w].
        from nvfp4.cute_fold.host import kext_kloop_gemm_sm100

        tile, cluster = KLOOP_SM100_CFGS[tactic_idx]
        f, s = nvfp4.scaled_fp4_quant_mext_r1(
            x, gs_inv, layout_mode="concat_k")
        return kext_kloop_gemm_sm100(
            w, f, wsb, s, alpha, x.dtype, mma_tiler_mn=tile,
            cluster_shape_mn=cluster)
    from nvfp4.cute_fold_sm120.host import kext_kloop_gemm_sm120

    # Pad to the tile-N boundary only (fast path: no TMA OOB handling on the
    # B/SFB stream). NOT a compile bucket -- the kernel takes both row counts
    # at runtime, and the store is predicated on the real m.
    xp = _pad_rows(x, (m + 127) // 128 * 128)
    f, s = nvfp4.scaled_fp4_quant_mext_r1(xp, gs_inv, layout_mode="concat_k")
    return kext_kloop_gemm_sm120(
        w, f, wsb, s, alpha, x.dtype, tile_shape_mnk=KLOOP_TILES[tactic_idx],
        m_tok_real=m)


def precompile_kloop_sm100(out_dtype=torch.bfloat16) -> int:
    """Compile the (few) symbolic sm100 kloop kernels; whitelist per config."""
    from nvfp4.cute_fold.host import _compile_fold_gemm

    # A real allocation, not torch.cuda.init(): init() alone leaves no CURRENT
    # context, and cutlass HardwareInfo then dies with
    # CUDA_ERROR_INVALID_CONTEXT. PWAL callers always have one; offline
    # tooling (prewarm scripts, probes) is exactly who hits this.
    torch.zeros(1, device="cuda")
    done = 0
    for tile, cluster in KLOOP_SM100_CFGS:
        try:
            _compile_fold_gemm(tile, cluster, out_dtype,
                               mode="kloop_tma", kernel_arch="sm100")
            _PRECOMPILED.add(("sm100", tile, cluster))
            done += 1
        except Exception as e:                            # noqa: BLE001
            print(f"[residue] sm100 kloop precompile failed tile={tile} "
                  f"cluster={cluster}: {type(e).__name__}: {str(e)[:70]}",
                  flush=True)
    return done


def precompile_kloop_sm120(n_w: int, k: int,
                           out_dtype=torch.bfloat16) -> int:
    'Compile the kloop kernels for one mext_r1 shape -- one per tile.'
    from nvfp4.cute_fold_sm120.host import compile_fold_gemm_sm120

    # cute.compile needs a live CURRENT context; PWAL always has one, but be
    # self-sufficient for offline/tooling callers that compile first. Must be
    # a real allocation -- torch.cuda.init() alone leaves no current context
    # and cutlass HardwareInfo dies with CUDA_ERROR_INVALID_CONTEXT.
    torch.zeros(1, device="cuda")

    done = 0
    for tile in KLOOP_TILES:
        try:
            compile_fold_gemm_sm120(
                n_w, k, tile, out_dtype, a_ld=None, k_loop=True)
            _PRECOMPILED.add((n_w, k, tile))
            done += 1
        except Exception as e:                            # noqa: BLE001
            print(f"[residue] kloop precompile failed n_w={n_w} k={k} "
                  f"tile={tile}: {type(e).__name__}: {str(e)[:70]}",
                  flush=True)
    return done











# for the numbers that justified it.

_DEGRADED = [False]
_ANNOUNCED = [False]


def _kloop_valid_tactics(inputs, tuning):
    if _is_sm10x():
        if tuning:
            return list(range(len(KLOOP_SM100_CFGS)))
        return [i for i, (t, c) in enumerate(KLOOP_SM100_CFGS)
                if ("sm100", t, c) in _PRECOMPILED]
    w = inputs[1]
    n_w, k = int(w.shape[0]), int(w.shape[1]) * 2
    if tuning:
        # Warmup window: compiling is legal (and lru-cached before the
        # timed reps), so every candidate competes.
        return list(range(len(KLOOP_TILES)))
    
    
    
    
    return [i for i, tile in enumerate(KLOOP_TILES)
            if (n_w, k, tile) in _PRECOMPILED]


def _make_runners(fallback_forward, fallback_key, output_dtype):
    from .fi_tuner_base import make_runner_pair

    return make_runner_pair(
        _CUSTOM_OP,
        fallback_forward=fallback_forward,
        fallback_extras=("two_gemm", 2, fallback_key, str(output_dtype)),
        valid_tactics=_kloop_valid_tactics,
        run_tactic=_run_kloop,
        candidate_extras=lambda inputs: ("k_loop", 2, fallback_key, str(output_dtype)),
        degraded_flag=_DEGRADED,
    )


_RUNNERS = {}


def tuned_mext_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    force_kloop: bool = False,
    *,
    fallback_forward: Callable[[list], torch.Tensor],
    fallback_key: str,
) -> torch.Tensor:
    """Route prefill through FlashInfer with a caller-owned two-GEMM fallback.

    ``fallback_forward(inputs)`` receives [x, weight, scale_inv, weight_sf,
    alpha] and returns the result. ``fallback_key`` must be a stable,
    versioned backend identifier: changing the fallback changes the race.
    Current K-loop candidates require output_dtype == x.dtype.
    """
    if (not callable(fallback_forward)
            or not isinstance(fallback_key, str) or not fallback_key):
        raise ValueError("provide a callable fallback_forward and a nonempty fallback_key")
    inputs = [x, weight, input_global_scale_inv, weight_scale_base, alpha]
    if output_dtype != x.dtype:
        return fallback_forward(inputs)

    if force_kloop:
        if _is_sm10x():
            for i, (t, c) in enumerate(KLOOP_SM100_CFGS):
                if ("sm100", t, c) in _PRECOMPILED:
                    return _run_kloop(inputs, i)
            return fallback_forward(inputs)
        n_w, k = int(weight.shape[0]), int(weight.shape[1]) * 2
        for i, tile in enumerate(KLOOP_TILES):
            if (n_w, k, tile) in _PRECOMPILED:
                return _run_kloop(inputs, i)
        return fallback_forward(inputs)

    from .fi_tuner_base import tuned_call

    def _runners():
        key = (fallback_forward, fallback_key, output_dtype)
        if key not in _RUNNERS:
            _RUNNERS[key] = _make_runners(fallback_forward, fallback_key, output_dtype)
        return _RUNNERS[key]

    return tuned_call(
        _CUSTOM_OP,
        runners_getter=_runners,
        config_getter=_tuning_config,
        inputs=inputs,
        fallback_forward=fallback_forward,
        kill_env_value=None,       # no kill switch: two_gemm IS the fallback
        announce_flag=_ANNOUNCED,
        announce=None,
    )
