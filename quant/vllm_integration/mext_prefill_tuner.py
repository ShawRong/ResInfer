# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'FlashInfer-autotuned dispatch for the mext_r1 prefill GEMM on sm120.'

from __future__ import annotations

import os

import torch

# TUNING buckets only: FlashInfer profiles each bucket once and keys its
# cache on the bucketed m. They are NOT compile keys on
# either arch -- the sm120 kernel takes the token count at runtime (one
# compile per (shape, tile)), same as the sm100 symbolic-shape kernel.
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


def _run_two_gemm(inputs):
    from .residue_linear_op import _run_mext_r1_two_gemm

    x, w, gs_inv, wsb, alpha = inputs
    return _run_mext_r1_two_gemm(x, w, gs_inv, wsb, alpha, x.dtype)


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
    """Compile the kloop kernels for one mext_r1 shape -- one per tile.

    The token count is a runtime argument, so the old
    per-bucket compile grid collapsed to len(KLOOP_TILES) compiles and the
    kernel serves ANY m, including above the top tuning bucket.

    Called at PWAL (register_sm120_fold_shape) so serving never compiles.
    Returns the number of kernels compiled; best-effort like every warmup.
    """
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


def _make_runners():
    from flashinfer.autotuner import AutoTuner, TunableRunner

    class TwoGemmRunner(TunableRunner):
        """runners[0]: FlashInfer's designated fallback. Any cache miss or
        skipped op serves this -- the never-JIT guarantee."""

        def get_valid_tactics(self, inputs, profile):
            return [0]

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            return _run_two_gemm(inputs)

        def get_cache_key_extras(self, inputs):
            return ("two_gemm", 1)

    
    
    
    
    
    
    class KloopRunner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            tuning = AutoTuner.get().is_tuning_mode
            if _is_sm10x():
                if tuning:
                    return list(range(len(KLOOP_SM100_CFGS)))
                return [i for i, (t, c) in enumerate(KLOOP_SM100_CFGS)
                        if ("sm100", t, c) in _PRECOMPILED]
            w = inputs[1]
            n_w, k = int(w.shape[0]), int(w.shape[1]) * 2
            if tuning:
                # Warmup window: compiling is legal (and lru-cached before
                # the timed reps), so every candidate competes.
                return list(range(len(KLOOP_TILES)))
            # One compiled kernel per (shape, tile) serves EVERY m (the token
            # count is a runtime argument) -- including m
            # above the top tuning bucket, which the bucketed compile keys
            # used to have no kernel for.
            return [i for i, tile in enumerate(KLOOP_TILES)
                    if (n_w, k, tile) in _PRECOMPILED]

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            if tactic < 0:
                return _run_two_gemm(inputs)
            return _run_kloop(inputs, tactic)

        def get_cache_key_extras(self, inputs):
            return ("k_loop", 1)

    return [TwoGemmRunner(), KloopRunner()]


_RUNNERS = None


def tuned_mext_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    force_kloop: bool = False,
) -> torch.Tensor:
    """Route one mext_r1 prefill GEMM through FlashInfer's tuned choice."""
    global _RUNNERS
    inputs = [x, weight, input_global_scale_inv, weight_scale_base, alpha]

    if force_kloop:
        if _is_sm10x():
            for i, (t, c) in enumerate(KLOOP_SM100_CFGS):
                if ("sm100", t, c) in _PRECOMPILED:
                    return _run_kloop(inputs, i)
            return _run_two_gemm(inputs)
        n_w, k = int(weight.shape[0]), int(weight.shape[1]) * 2
        for i, tile in enumerate(KLOOP_TILES):
            if (n_w, k, tile) in _PRECOMPILED:
                return _run_kloop(inputs, i)
        return _run_two_gemm(inputs)

    from flashinfer.autotuner import AutoTuner

    if _RUNNERS is None:
        _RUNNERS = _make_runners()
    runner, tactic = AutoTuner.get().choose_one(
        _CUSTOM_OP, _RUNNERS, _tuning_config(), inputs)
    return runner.forward(inputs, tactic=tactic)
