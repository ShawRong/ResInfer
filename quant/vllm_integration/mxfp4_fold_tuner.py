# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer AutoTuner selection for the full MXFP4 quantize + Fold path.

The NVFP4 tactic table is only a source of candidate tile descriptions.  An
MXFP4 candidate must pass a shape-specific output check *before* it is offered
to FlashInfer's profiler.  A cache hit without that check serves the safe
row-pair fallback instead of launching an unverified tactic.
"""

from __future__ import annotations

import hashlib
import os

import torch

from .fi_tuner_base import make_runner_pair, make_tuning_config, tuned_call

_OP = "residue_mxfp4_fold_sm100_group32_v1"
_VERSION = 1
_FAMILIES = (("row_pair", "std"), ("tile8", "std"), ("tile8", "pi"))
_FALLBACK = (("row_pair", "std"), "tma", (128, 128), (1, 1), None)

_CANDIDATES: list[tuple] | None = None
_TUNING_CONFIGS: dict[int, object] = {}
_RUNNERS: dict[tuple, list] = {}
_PREFLIGHT: dict[tuple, tuple[int, ...]] = {}
_STRICT_PREFLIGHT: set[tuple] = set()
_EXECUTED: set[tuple[tuple, int]] = set()
_ANNOUNCED = [False]
_DEGRADED = [False]
_LAST_SOURCE: list[str | None] = [None]
_LAST_TILE: list[tuple | None] = [None]
_WARMUP_REQUESTED = False


def _candidates() -> list[tuple]:
    """Stable indices; changes invalidate persisted FlashInfer cache entries."""
    global _CANDIDATES
    if _CANDIDATES is None:
        from nvfp4.cute_fold.tactics import FAMILY_TACTICS, _norm_tactic

        _CANDIDATES = [
            (family, *_norm_tactic(raw))
            for family in _FAMILIES
            for raw in FAMILY_TACTICS[family]
        ]
    return _CANDIDATES


def candidates_hash() -> str:
    return hashlib.sha256(repr((_VERSION, _candidates())).encode()).hexdigest()[:16]


def _map_to_tuning_buckets(m: int) -> int:
    # One exact-M profile per shape: a precheck at M=1 must not green-light a
    # synthetic M=2/4/... profile generated in the same choose_one call.
    return int(m)


def _tuning_config(m: int):
    m = int(m)
    if m not in _TUNING_CONFIGS:
        _TUNING_CONFIGS[m] = make_tuning_config((m,), _map_to_tuning_buckets)
    return _TUNING_CONFIGS[m]


def tuner_enabled() -> bool:
    """Experimental MXFP4 service route; opt in only after a GPU smoke."""
    value = os.environ.get("RESIDUE_MXFP4_FOLD_TUNER", "0")
    if value not in ("0", "1"):
        raise ValueError("RESIDUE_MXFP4_FOLD_TUNER must be 0 or 1")
    return value == "1"


def request_vllm_warmup() -> None:
    global _WARMUP_REQUESTED
    _WARMUP_REQUESTED = True


def vllm_warmup_requested() -> bool:
    return _WARMUP_REQUESTED and tuner_enabled()


def is_tuning_mode() -> bool:
    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError:
        return False
    return bool(AutoTuner.get().is_tuning_mode)


def _geometry(inputs, pdl_mode: str, out_dtype) -> tuple:
    x, weight, weight_sf = inputs
    return (
        "mxfp4", "ceil", 32, _VERSION, candidates_hash(), pdl_mode,
        int(x.shape[0]), int(x.shape[1]), int(weight.shape[0]),
        tuple(x.stride()), tuple(weight.stride()), tuple(weight_sf.stride()),
        int(weight_sf.numel()), str(x.dtype), str(weight.dtype),
        str(weight_sf.dtype), str(out_dtype), str(x.device),
    )


def preflighted(inputs, *, pdl_mode="early", out_dtype=torch.bfloat16,
                strict_reference=False) -> bool:
    """Whether this process validated candidates for this exact shape/config."""
    key = _geometry(inputs, pdl_mode, out_dtype)
    return key in (_STRICT_PREFLIGHT if strict_reference else _PREFLIGHT)


def _static_legal(candidate: tuple, m: int) -> bool:
    family, mode, tile, _, _ = candidate
    pad = _pad_multiple(candidate)
    physical_m = 2 * (m if pad is None else ((m + pad - 1) // pad) * pad)
    if mode == "direct_mext" and physical_m < tile[0]:
        return False
    # The N-ext token axis cannot use a narrow tile that misses real rows.
    if family[0] == "row_pair" and tile[1] < 64 and 2 * m > tile[1]:
        return False
    return True


def _pad_multiple(candidate: tuple) -> int | None:
    family, mode, tile, _, _ = candidate
    if family[0] != "tile8":
        return None
    # A4's TMA store clamps M tails; A6's direct store needs a full MMA tile.
    return max(8, tile[0] // 2) if mode == "direct_mext" else 8


def _run_tactic(inputs, candidate: tuple, pdl_mode: str, out_dtype):
    from nvfp4.cute_fold.host import mext_fold_gemm_sm103
    from nvfp4.mxfp4 import quantize_mext

    x, weight, weight_sf = inputs
    family, mode, tile, cluster, ab = candidate
    pad = _pad_multiple(candidate)
    packed, scales = quantize_mext(
        x, recipe="ceil", mext_layout=family[0], m_pad_multiple=pad,
        pdl_mode=pdl_mode)
    _LAST_TILE[0] = candidate
    out = mext_fold_gemm_sm103(
        weight, packed, weight_sf, scales, None,
        out_dtype=out_dtype, mma_tiler_mn=tile,
        cluster_shape_mn=cluster, store_mode=mode, kernel_arch="sm100",
        ab_stage_override=ab, pi_via_layout=family == ("tile8", "pi"),
        fp4_format="mxfp4")
    return out[: x.shape[0]]


def _fallback(inputs, pdl_mode: str, out_dtype):
    return _run_tactic(inputs, _FALLBACK, pdl_mode, out_dtype)


def _check_output(actual, reference) -> float:
    if tuple(actual.shape) != tuple(reference.shape):
        raise ValueError(f"output shape {tuple(actual.shape)} != {tuple(reference.shape)}")
    if not bool(torch.isfinite(actual).all()):
        raise ValueError("non-finite output")
    denom = float(reference.float().abs().max()) + 1e-9
    relmax = float((actual.float() - reference.float()).abs().max()) / denom
    if relmax > 0.25:
        raise ValueError(f"BF16 relative max {relmax:.5f} > 0.25")
    return relmax


def _check_against_fixed(actual, reference) -> None:
    # Reference is the same MXFP4 quantizer + fixed Fold implementation, not
    # an unquantized BF16 matmul. Require pointwise agreement, including tails.
    torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.125)


def preflight_tactics(inputs, reference, *, pdl_mode="early",
                      out_dtype=torch.bfloat16,
                      strict_reference=False) -> dict:
    """Verify every listed tactic for this exact logical shape before tuning.

    Only input reads and fresh output allocations occur; the caller's x,
    weight, scale and reference tensors are never changed.  Rechecking after
    a cache hit is intentional: the process may have a new compiled binary.
    """
    if pdl_mode not in ("off", "tail", "early"):
        raise ValueError(f"invalid MXFP4 PDL mode {pdl_mode!r}")
    if len(inputs) != 3:
        raise ValueError("expected (raw activation, packed weight, flat scales)")
    inputs = tuple(inputs)
    key = _geometry(inputs, pdl_mode, out_dtype)
    _PREFLIGHT.pop(key, None)
    _STRICT_PREFLIGHT.discard(key)
    m = int(inputs[0].shape[0])
    valid: list[int] = []
    failed: dict[int, str] = {}
    max_relative: dict[int, float] = {}
    # A broken fallback would poison both cache misses and the tuning result.
    try:
        fallback_output = _fallback(inputs, pdl_mode, out_dtype)
        torch.cuda.synchronize(inputs[0].device)
        fallback_relative = _check_output(fallback_output, reference)
        if strict_reference:
            _check_against_fixed(fallback_output, reference)
    except Exception as exc:
        raise RuntimeError(f"MXFP4 row-pair fallback failed preflight: {exc}") from exc
    for index, candidate in enumerate(_candidates()):
        if not _static_legal(candidate, m):
            failed[index] = "static legality filter"
            continue
        try:
            actual = _run_tactic(inputs, candidate, pdl_mode, out_dtype)
            torch.cuda.synchronize(inputs[0].device)
            max_relative[index] = _check_output(actual, reference)
            if strict_reference:
                _check_against_fixed(actual, reference)
        except Exception as exc:  # CUDA launch and correctness failures are candidates, not winners.
            # A sticky asynchronous failure cannot be safely ignored.
            try:
                torch.cuda.synchronize(inputs[0].device)
            except Exception:
                raise
            failed[index] = f"{type(exc).__name__}: {str(exc)[:160]}"
            continue
        valid.append(index)
        _EXECUTED.add((key, index))
    # Publish the set atomically, after every candidate has been checked.
    _PREFLIGHT[key] = tuple(valid)
    if strict_reference:
        _STRICT_PREFLIGHT.add(key)
    return {"op": _OP, "candidate_hash": candidates_hash(),
            "logical_m": m, "candidate_count": len(_candidates()),
            "valid_indices": valid, "failed": failed,
            "relative_max": max_relative, "fallback_relative_max": fallback_relative,
            "fallback": _FALLBACK, "strict_reference": strict_reference}


def _make_runners(pdl_mode: str, out_dtype):
    def fallback(inputs):
        return _fallback(inputs, pdl_mode, out_dtype)

    def valid(inputs, tuning):
        key = _geometry(inputs, pdl_mode, out_dtype)
        checked = _PREFLIGHT.get(key, ())
        return list(checked) if tuning else [
            i for i in checked if (key, i) in _EXECUTED]

    def run(inputs, index):
        candidate = _candidates()[index]
        output = _run_tactic(inputs, candidate, pdl_mode, out_dtype)
        _EXECUTED.add((_geometry(inputs, pdl_mode, out_dtype), index))
        return output

    def extras(inputs):
        return _geometry(inputs, pdl_mode, out_dtype)

    return make_runner_pair(
        _OP, fallback_forward=fallback,
        fallback_extras=("mxfp4", 32, "row_pair", _VERSION, pdl_mode,
                         str(out_dtype)),
        valid_tactics=valid, run_tactic=run, candidate_extras=extras,
        degraded_flag=_DEGRADED,
        on_degrade=lambda: _set_source("fallback"),
    )


def _set_source(source: str):
    _LAST_SOURCE[0] = source


def last_tactic_source() -> str | None:
    return _LAST_SOURCE[0]


def last_tactic_tile() -> tuple | None:
    return _LAST_TILE[0]


def record_static_fallback() -> None:
    """Record the existing fixed Fold path when normal serving skips tuning."""
    _LAST_SOURCE[0] = "offline_table"
    _LAST_TILE[0] = _FALLBACK


def _announce(runner, tactic, runners):
    if runner is runners[1] and tactic >= 0:
        return f"tactic source: autotuned {_candidates()[tactic]}"
    return "tactic source: row-pair fallback"


def tuned_mxfp4_fold(x, packed_w, flat_ws, *, pdl_mode="early",
                     out_dtype=torch.bfloat16):
    """Tune and run MXFP4 producer+Fold using FlashInfer's cache/scaffold."""
    if pdl_mode not in ("off", "tail", "early"):
        raise ValueError(f"invalid MXFP4 PDL mode {pdl_mode!r}")
    inputs = [x, packed_w, flat_ws]
    runner_key = (pdl_mode, out_dtype)

    def runners():
        if runner_key not in _RUNNERS:
            _RUNNERS[runner_key] = _make_runners(pdl_mode, out_dtype)
        return _RUNNERS[runner_key]

    return tuned_call(
        _OP, runners_getter=runners,
        config_getter=lambda: _tuning_config(int(x.shape[0])),
        inputs=inputs,
        fallback_forward=lambda args: _fallback(args, pdl_mode, out_dtype),
        kill_env_value=None, announce_flag=_ANNOUNCED,
        announce=_announce, record=_set_source,
    )
