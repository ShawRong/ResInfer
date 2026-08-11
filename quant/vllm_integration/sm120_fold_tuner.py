# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'FlashInfer-autotuned tile selection for the sm120 CuTeDSL decode fold.'

from __future__ import annotations

import os

import torch

from .fi_tuner_base import (bucket_of, make_runner_pair, make_tuning_config,
                            tuned_call, weight_geometry)

# Tuning-bucket grid for the decode band (cache keys only -- the kernel takes
# the token count at runtime, so no bucket can mint a compile).
FOLD_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)







FOLD_TILES = (
    (64, 64, 128), (64, 64, 256), (64, 128, 128),
    (128, 64, 128), (128, 64, 256), (128, 128, 128), (128, 128, 256),
)

_CUSTOM_OP = "residue_fold_tile_sm120"

# (n_w, k_logical, a_ld, tile) whose kernel was compiled at PWAL. Outside the
# tuning window only these tactics are offered; inside it compilation is
# legal (lru-cached by the host before the timed reps).
_PRECOMPILED: set = set()

_ANNOUNCED = [False]
_DEGRADED = [False]

# The ACTUAL source of the most recent call's tactic, for benches and probes:
# "autotuned" | "offline_table" | "fallback" (degrade) | None (never called).
# Under cudagraphs this reflects capture time -- replay never runs Python.
_LAST_SOURCE: list = [None]


def last_tactic_source() -> str | None:
    return _LAST_SOURCE[0]


def _record_source(src: str) -> None:
    _LAST_SOURCE[0] = src

# Kept as a module name for callers/tests; geometry itself is arch-generic.
_weight_geometry = weight_geometry


def _bucket_of(m: int) -> int:
    return bucket_of(FOLD_BUCKETS, m)


def _map_to_tuning_buckets(x: int) -> int:
    # Module-level named function: TuningConfig is hashed into the cache key,
    # so this must be ONE stable object, not a fresh lambda per call.
    return _bucket_of(int(x))


def fold_tiles_hash() -> str:
    import hashlib

    return hashlib.md5(repr(FOLD_TILES).encode()).hexdigest()[:12]


_TUNING_CONFIG = None


def _tuning_config():
    global _TUNING_CONFIG
    if _TUNING_CONFIG is None:
        _TUNING_CONFIG = make_tuning_config(FOLD_BUCKETS,
                                            _map_to_tuning_buckets)
    return _TUNING_CONFIG


def _run_fold_tile(inputs, tile, out_dtype) -> torch.Tensor:
    """Pad to THIS tile's boundary, quantize row_pair, run the fold GEMM."""
    import nvfp4
    from nvfp4.cute_fold_sm120.host import mext_fold_gemm_sm120

    x, weight, gs_inv, wsb, alpha = inputs
    m = int(x.shape[0])
    pad = tile[1] // 2
    m_pad = (m + pad - 1) // pad * pad
    x_in = x
    if m_pad != m:
        x_in = torch.zeros(m_pad, x.shape[1], dtype=x.dtype, device=x.device)
        x_in[:m] = x
    x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(x_in, gs_inv, layout_mode=1)
    out = mext_fold_gemm_sm120(
        weight, x_fp4, wsb, x_sf.view(torch.float8_e4m3fn), alpha, out_dtype,
        tile_shape_mnk=tile, m_tok_real=m,
    )
    # THE WHITELIST LEARNS FROM EXECUTION: this kernel just ran, so it is
    # compiled in this process's cache -- serving it cannot JIT. Without
    # this, only serving's PWAL ever fed the whitelist, and any caller that
    # tunes without registering shapes (the kernel bench) degraded every
    # tile-runner win back to the table -- 7 of 32 rows on the field's
    # first day out, caught by fold_tactic_source=fallback.
    _note_compiled(weight, tile)
    return out


def _note_compiled(weight: torch.Tensor, tile: tuple[int, int, int]) -> None:
    n_w, k_logical, a_ld = _weight_geometry(weight)
    _PRECOMPILED.add((n_w, k_logical, a_ld, tile))


def _table_tile(inputs) -> tuple[int, int, int]:
    from nvfp4.cute_fold_sm120.tactics import select_tile

    x, weight = inputs[0], inputs[1]
    n_w, k_logical, _ = _weight_geometry(weight)
    return select_tile(int(x.shape[0]), n_w, k_logical)


def run_fold_table(inputs, out_dtype) -> torch.Tensor:
    """The untuned path: the offline-swept static table picks the tile."""
    return _run_fold_tile(inputs, _table_tile(inputs), out_dtype)


def precompile_fold_tiles(n_w: int, k_logical: int, a_ld: int | None = None,
                          out_dtype=torch.bfloat16) -> int:
    """Compile every candidate tile for one registered shape; whitelist each.

    Called at PWAL next to `warmup_fold_default`, so a persisted cache entry
    restored on a later boot finds its winner already compiled -- without
    this, a JSON cache hit would JIT at serving time, which is exactly the
    incident class the whitelist exists to prevent.
    """
    from nvfp4.cute_fold_sm120.host import compile_fold_gemm_sm120

    # cute.compile needs a live CURRENT context (see precompile_kloop_sm120).
    torch.zeros(1, device="cuda")
    done = 0
    for tile in FOLD_TILES:
        if k_logical % tile[2]:
            continue
        try:
            compile_fold_gemm_sm120(n_w, k_logical, tile, out_dtype, a_ld=a_ld)
            _PRECOMPILED.add((n_w, k_logical, a_ld, tile))
            done += 1
        except Exception as e:                            # noqa: BLE001
            print(f"[residue] fold tile precompile failed n_w={n_w} "
                  f"k={k_logical} a_ld={a_ld} tile={tile}: "
                  f"{type(e).__name__}: {str(e)[:70]}", flush=True)
    return done


def _make_runners(out_dtype):
    def _fallback(inputs):
        return run_fold_table(inputs, out_dtype)

    def _valid(inputs, tuning):
        n_w, k_logical, a_ld = _weight_geometry(inputs[1])
        if tuning:
            # Warmup window: compiling is legal (lru-cached by the host
            # before the timed reps), so every legal candidate competes.
            return [i for i, t in enumerate(FOLD_TILES)
                    if k_logical % t[2] == 0]
        return [i for i, t in enumerate(FOLD_TILES)
                if (n_w, k_logical, a_ld, t) in _PRECOMPILED]

    def _run(inputs, i):
        return _run_fold_tile(inputs, FOLD_TILES[i], out_dtype)

    def _extras(inputs):
        _, _, a_ld = _weight_geometry(inputs[1])
        # fold_tiles_hash(): an edited/reordered FOLD_TILES invalidates
        # persisted entries instead of decoding to a different kernel.
        return ("row_pair", "sm120", fold_tiles_hash(), a_ld, str(out_dtype))

    return make_runner_pair(
        _CUSTOM_OP,
        fallback_forward=_fallback,
        fallback_extras=("table", 1),
        valid_tactics=_valid,
        run_tactic=_run,
        candidate_extras=_extras,
        degraded_flag=_DEGRADED,
        on_degrade=lambda: _record_source("fallback"),
    )


_RUNNERS_BY_DTYPE: dict = {}


def _announce(runner, tactic, runners) -> str:
    if runner is runners[1] and tactic >= 0:
        return f"tactic source: autotuned tile={FOLD_TILES[tactic]}"
    if tactic >= 0:
        return "tactic source: autotuned (the table's own tile won the profile)"
    return "tactic source: offline_table (no tuned entry -- fallback)"


def tuned_sm120_fold(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Route one sm120 decode fold through FlashInfer's tuned tile choice.

    `weight` may be an ext-K base-K prefix VIEW; its stride carries a_ld.
    Returns D[x.shape[0], n_w] -- padding rows are dropped by the epilogue.
    """
    inputs = [x, weight, input_global_scale_inv, weight_scale_base, alpha]

    def _runners():
        runners = _RUNNERS_BY_DTYPE.get(output_dtype)
        if runners is None:
            runners = _RUNNERS_BY_DTYPE[output_dtype] = _make_runners(
                output_dtype)
        return runners

    return tuned_call(
        _CUSTOM_OP,
        runners_getter=_runners,
        config_getter=_tuning_config,
        inputs=inputs,
        fallback_forward=lambda ins: run_fold_table(ins, output_dtype),
        kill_env_value=os.environ.get("RESIDUE_SM120_FOLD_TUNER", "1"),
        announce_flag=_ANNOUNCED,
        announce=_announce,
        record=_record_source,
    )
