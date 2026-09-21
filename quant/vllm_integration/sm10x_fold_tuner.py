# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'FlashInfer-autotuned producer+family+tactic selection for SM100 Fold.'

from __future__ import annotations

import os

import torch

from .fi_tuner_base import (bucket_of, make_runner_pair, make_tuning_config,
                            tuned_call, weight_geometry)

FOLD_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512)
_FAMILY = ("row_pair", "std")
_CUSTOM_OP = "residue_fold_tactic_sm10x"
_LAYOUT_ROW_PAIR = 1
_LAYOUT_TILE8 = 2
_M_PAD_MULTIPLE = 8

# warmup_fold_default compiled the row_pair candidates (symbolic shapes:
# one compile set serves every (n_w, k, m), strided and contiguous).
_READY = [False]
# Tactic indices that EXECUTED in this process (tuning window or serving):
# a kernel that just ran is compiled, so it may serve even without the full
# precompile -- the same semantic weld as sm120's _note_compiled (family
# bug #7: two individually-correct modules whose semantics did not join).
_EXECUTED: set = set()
_ANNOUNCED = [False]
_DEGRADED = [False]

# Actual source of the most recent call's tactic (see sm120_fold_tuner).
_LAST_SOURCE: list = [None]
# The tactic that ACTUALLY ran (see sm120_fold_tuner). None on the legacy
# path, which resolves its tile inside nvfp4.fold and does not report it.
_LAST_TILE: list = [None]


def last_tactic_source() -> str | None:
    return _LAST_SOURCE[0]


def last_tactic_tile():
    return _LAST_TILE[0]


def _record_source(src: str) -> None:
    _LAST_SOURCE[0] = src

_CANDIDATES: list | None = None
_INDEX_BY_TACTIC: dict | None = None


def _bucket_of(m: int) -> int:
    return bucket_of(FOLD_BUCKETS, m)


def _map_to_tuning_buckets(x: int) -> int:
    # Module-level named function: TuningConfig is hashed into the cache key.
    return _bucket_of(int(x))


_TUNING_CONFIG = None


def _tuning_config():
    global _TUNING_CONFIG
    if _TUNING_CONFIG is None:
        _TUNING_CONFIG = make_tuning_config(FOLD_BUCKETS,
                                            _map_to_tuning_buckets)
    return _TUNING_CONFIG


def _families() -> tuple:
    from nvfp4.fold import candidate_families

    if _arch() != "sm100":
        return (_FAMILY,)
    return tuple(candidate_families())


def _candidates() -> list:
    """Normalized (family, mode, tiler, cluster, ab), index-stable."""
    global _CANDIDATES, _INDEX_BY_TACTIC
    if _CANDIDATES is None:
        from nvfp4.cute_fold.tactics import FAMILY_TACTICS, _norm_tactic

        _CANDIDATES = [
            (family, *_norm_tactic(tactic))
            for family in _families()
            for tactic in FAMILY_TACTICS[family]
        ]
        _INDEX_BY_TACTIC = {
            (candidate[0], candidate[1:]): i
            for i, candidate in enumerate(_CANDIDATES)
        }
    return _CANDIDATES


def candidates_hash() -> str:
    import hashlib

    return hashlib.md5(repr(_candidates()).encode()).hexdigest()[:12]


def _arch() -> str | None:
    from nvfp4.cute_fold.tactics import kernel_arch_for_capability

    major, minor = torch.cuda.get_device_capability()
    return kernel_arch_for_capability(int(major), int(minor))


def _legal_indices(m_tok: int) -> list[int]:
    """m-dependent legality, delegated to the ranking the dict tuner used:
    narrow tiles must cover 2*m_tok outright, smem-infeasible entries are
    already absent from the table. Ranking order is irrelevant here (the
    profiler measures everything offered); only membership matters."""
    from nvfp4.cute_fold.tactics import _rank_tactics

    _candidates()
    legal = []
    for family in _families():
        ranked = _rank_tactics(family, m_tok, _arch())
        legal.extend(
            _INDEX_BY_TACTIC[(family, tactic)]
            for tactic in ranked
            if (family, tactic) in _INDEX_BY_TACTIC
        )
    return sorted(legal)


def tile8_pad_multiple(mode: str, tiler: tuple) -> int:
    # A6's direct store has no partial-tile clamp. Make real M legal even
    # when its autotune bucket is a different multiple of tile_M / 2.
    if mode == "direct_mext":
        return max(_M_PAD_MULTIPLE, tiler[0] // 2)
    return _M_PAD_MULTIPLE


def _quant_for_family(inputs, family, pad_multiple: int = _M_PAD_MULTIPLE):
    import nvfp4
    from nvfp4.cute_fold.tactics import FAMILY_ACT_LAYOUT

    x, _, gs_inv, _, _ = inputs
    if FAMILY_ACT_LAYOUT[family] == "tile8":
        x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(
            x, gs_inv, layout_mode=_LAYOUT_TILE8,
            m_pad_multiple=pad_multiple)
    else:
        x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(
            x, gs_inv, layout_mode=_LAYOUT_ROW_PAIR)
    return x_fp4, x_sf.view(torch.float8_e4m3fn)


def _run_tactic(inputs, idx, out_dtype) -> torch.Tensor:
    from nvfp4.cute_fold.host import mext_fold_gemm_sm103
    from nvfp4.cute_fold.tactics import _needs_pi_layout

    family, mode, tiler, cluster, ab = _candidates()[idx]
    _LAST_TILE[0] = (family, mode, tiler, cluster, ab)
    x_fp4, x_sf = _quant_for_family(
        inputs, family, pad_multiple=tile8_pad_multiple(mode, tiler))
    _, weight, _, wsb, alpha = inputs
    out = mext_fold_gemm_sm103(
        weight, x_fp4, wsb, x_sf, alpha, out_dtype,
        mma_tiler_mn=tiler, cluster_shape_mn=cluster, store_mode=mode,
        kernel_arch=_arch(), ab_stage_override=ab,
        pi_via_layout=_needs_pi_layout(family),
    )
    _EXECUTED.add(idx)
    return out[: inputs[0].shape[0]]


def _run_fallback(inputs, out_dtype) -> torch.Tensor:
    """Static row-pair fallback: never nest the old tuner inside FI tuning."""
    from nvfp4.cute_fold.tactics import run_fold_default

    _LAST_TILE[0] = None      # legacy path picks internally and stays mute
    x_fp4, x_sf = _quant_for_family(inputs, _FAMILY)
    _, weight, _, wsb, alpha = inputs
    major, minor = torch.cuda.get_device_capability()
    return run_fold_default(
        int(major), int(minor), weight, x_fp4, wsb, x_sf, alpha, out_dtype)


def precompile_sm10x_families() -> bool:
    """Compile all enabled family candidates with symbolic activation shapes."""
    from nvfp4.cute_fold.tactics import warmup_fold_default, warmup_specs

    major, minor = torch.cuda.get_device_capability()
    torch.zeros(1, device="cuda")
    try:
        families = _families()
        expected = len(warmup_specs(_arch(), families=families))
        compiled = warmup_fold_default(
            int(major), int(minor), families=families)
        _READY[0] = compiled == expected
    except Exception as e:                                # noqa: BLE001
        print(f"[residue] sm10x fold tuner precompile failed: "
              f"{type(e).__name__}: {str(e)[:70]}", flush=True)
    return _READY[0]


def precompile_row_pair() -> bool:
    """Compatibility name for the existing PWAL warmup hook."""
    return precompile_sm10x_families()


def _make_runners(out_dtype):
    def _fallback(inputs):
        return _run_fallback(inputs, out_dtype)

    def _valid(inputs, tuning):
        legal = _legal_indices(int(inputs[0].shape[0]))
        if tuning:
            return legal
        if _READY[0]:
            return legal
        return [i for i in legal if i in _EXECUTED]

    def _run(inputs, i):
        return _run_tactic(inputs, i, out_dtype)

    def _extras(inputs):
        _, _, a_ld = weight_geometry(inputs[1])
        return ("family_v2", _arch(), candidates_hash(), a_ld, str(out_dtype))

    return make_runner_pair(
        _CUSTOM_OP,
        fallback_forward=_fallback,
        fallback_extras=("row_pair_table", 2),
        valid_tactics=_valid,
        run_tactic=_run,
        candidate_extras=_extras,
        degraded_flag=_DEGRADED,
        on_degrade=lambda: _record_source("fallback"),
    )


_RUNNERS_BY_DTYPE: dict = {}


def _announce(runner, tactic, runners) -> str:
    if runner is runners[1] and tactic >= 0:
        return f"tactic source: autotuned {_candidates()[tactic]}"
    if tactic >= 0:
        return "tactic source: autotuned (the legacy path won the profile)"
    return "tactic source: legacy dict/table (no tuned entry -- fallback)"


def tuner_enabled() -> bool:
    
    
    
    
    
    
    return os.environ.get("RESIDUE_SM10X_FOLD_TUNER", "1") == "1"


def tuned_sm10x_fold(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Route one sm10x decode fold through FlashInfer's tuned tactic choice.

    `weight` may be an ext-K base-K prefix VIEW; its stride carries a_ld.
    Returns D[x.shape[0], n_w].
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
        fallback_forward=lambda ins: _run_fallback(ins, output_dtype),
        kill_env_value=("1" if tuner_enabled() else "0"),
        announce_flag=_ANNOUNCED,
        announce=_announce,
        record=_record_source,
    )
