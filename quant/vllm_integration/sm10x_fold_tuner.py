# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer-autotuned tactic selection for the sm10x CuTeDSL decode fold.

Phase 3 of the fold-autotune-via-flashinfer plan, row_pair family only --
the default build's sole family (A4/A6 stay on the in-process tuner until
Level 2 lands). Same scaffold as the sm120 fold and the prefill tuner:
runners[0] is the pre-existing serving path (dict-cache hit or static
table via `fold.run_fold` with the family pinned), candidates are indices
into FAMILY_TACTICS[("row_pair","std")] with the list's hash in the
cache-key extras, and outside the tuning window only warmed tactics are
offered.

DEFAULT OFF (RESIDUE_SM10X_FOLD_TUNER=0): this machine has no sm10x GPU,
so the switch ships dark and a pod task validates it on B200 before the
default flips. Until then the in-process dict tuner keeps serving sm100
decode exactly as before -- at no point is sm10x without a tuned path.
After the flip, the dict machinery in cute_fold/tactics.py is deleted
(the convergence this module exists for).

What sm10x gains over the dict tuner, same as everywhere else: JSON
persistence (no per-boot retune), TP consistency (leader tunes, one cache
broadcast), and the shared prewarm/audit tooling.
"""

from __future__ import annotations

import os

import torch

from .fi_tuner_base import (bucket_of, make_runner_pair, make_tuning_config,
                            tuned_call, weight_geometry)

FOLD_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
_FAMILY = ("row_pair", "std")
_CUSTOM_OP = "residue_fold_tactic_sm10x"
_LAYOUT_ROW_PAIR = 1

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


def last_tactic_source() -> str | None:
    return _LAST_SOURCE[0]


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


def _candidates() -> list:
    """Normalized (mode, tiler, cluster, ab) list, order = tactic index.

    The order is FAMILY_TACTICS' order, which therefore becomes part of the
    persisted cache contract -- candidates_hash() invalidates on any edit.
    """
    global _CANDIDATES, _INDEX_BY_TACTIC
    if _CANDIDATES is None:
        from nvfp4.cute_fold.tactics import FAMILY_TACTICS, _norm_tactic

        _CANDIDATES = [_norm_tactic(t) for t in FAMILY_TACTICS[_FAMILY]]
        _INDEX_BY_TACTIC = {t: i for i, t in enumerate(_CANDIDATES)}
    return _CANDIDATES


def candidates_hash() -> str:
    import hashlib

    from nvfp4.cute_fold.tactics import FAMILY_TACTICS

    return hashlib.md5(repr(FAMILY_TACTICS[_FAMILY]).encode()).hexdigest()[:12]


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

    cands = _candidates()
    ranked = _rank_tactics(_FAMILY, m_tok, _arch())
    return sorted(_INDEX_BY_TACTIC[t] for t in ranked if t in _INDEX_BY_TACTIC)


def _quant_row_pair(inputs):
    import nvfp4

    x, _, gs_inv, _, _ = inputs
    x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(
        x, gs_inv, layout_mode=_LAYOUT_ROW_PAIR)
    return x_fp4, x_sf.view(torch.float8_e4m3fn)


def _run_tactic(inputs, idx, out_dtype) -> torch.Tensor:
    from nvfp4.cute_fold.host import mext_fold_gemm_sm103

    mode, tiler, cluster, ab = _candidates()[idx]
    x_fp4, x_sf = _quant_row_pair(inputs)
    _, weight, _, wsb, alpha = inputs
    out = mext_fold_gemm_sm103(
        weight, x_fp4, wsb, x_sf, alpha, out_dtype,
        mma_tiler_mn=tiler, cluster_shape_mn=cluster, store_mode=mode,
        kernel_arch=_arch(), ab_stage_override=ab, pi_via_layout=False,
    )
    _EXECUTED.add(idx)
    return out


def _run_fallback(inputs, out_dtype) -> torch.Tensor:
    """The pre-existing sm10x serving path, family pinned to row_pair:
    run_fold consults the in-process tuned cache, then the static table."""
    from nvfp4.fold import run_fold

    x_fp4, x_sf = _quant_row_pair(inputs)
    _, weight, _, wsb, alpha = inputs
    major, minor = torch.cuda.get_device_capability()
    return run_fold(int(major), int(minor), weight, x_fp4, wsb, x_sf,
                    alpha, out_dtype, family=_FAMILY)


def precompile_row_pair() -> bool:
    """Compile the row_pair candidate set (symbolic shapes -- one set serves
    every layer shape, contiguous and strided weights alike). Called at PWAL
    when the tuner is enabled; the tuning window may also compile lazily."""
    from nvfp4.cute_fold.tactics import warmup_fold_default

    major, minor = torch.cuda.get_device_capability()
    torch.zeros(1, device="cuda")
    try:
        warmup_fold_default(int(major), int(minor), families=(_FAMILY,))
        _READY[0] = True
    except Exception as e:                                # noqa: BLE001
        print(f"[residue] sm10x fold tuner precompile failed: "
              f"{type(e).__name__}: {str(e)[:70]}", flush=True)
    return _READY[0]


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
        return ("row_pair", _arch(), candidates_hash(), a_ld, str(out_dtype))

    return make_runner_pair(
        _CUSTOM_OP,
        fallback_forward=_fallback,
        fallback_extras=("dict_or_table", 1),
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
    return os.environ.get("RESIDUE_SM10X_FOLD_TUNER", "0") == "1"


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
