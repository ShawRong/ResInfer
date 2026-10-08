# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'FlashInfer selection for MXFP4 quantize + Fold.'

from __future__ import annotations

import hashlib
import os
import torch

from .linear_tuner import LinearSpec, tuned_linear

_VERSION = 3
_FAMILIES = (("row_pair", "std"), ("tile8", "std"), ("tile8", "pi"))
_DEFAULT = (("row_pair", "std"), "tma", (128, 128), (1, 1), None)

_CANDIDATES: list[tuple] | None = None
_LAST_SOURCE: list[str | None] = [None]
_LAST_TILE: list[tuple | None] = [None]


def _candidates() -> list[tuple]:
    """Stable indices; changes invalidate persisted FlashInfer cache entries."""
    global _CANDIDATES
    if _CANDIDATES is None:
        from nvfp4.cute_fold.tactics import FAMILY_TACTICS, _norm_tactic

        _CANDIDATES = [
            (family, *_norm_tactic(raw))
            for family in _FAMILIES
            for raw in FAMILY_TACTICS[family]
            # M1_ALL_TACTICS.md: invalid launches and A4 wrong output.
            if not (family[0] == "row_pair" and raw[1] in ((128, 24), (256, 256)))
            if not (family == ("tile8", "std") and raw[1] == (128, 32))
        ]
    return _CANDIDATES


def candidates_hash() -> str:
    return hashlib.sha256(repr((_VERSION, _candidates(), _DEFAULT)).encode()).hexdigest()[:16]


def tuner_enabled() -> bool:
    """Enable FlashInfer selection by default; allow explicit static fallback."""
    value = os.environ.get("RESIDUE_MXFP4_FOLD_TUNER", "1")
    if value not in ("0", "1"):
        raise ValueError("RESIDUE_MXFP4_FOLD_TUNER must be 0 or 1")
    return value == "1"


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
    from nvfp4.mxfp4 import fold_gemm, quantize_mext

    x, weight, weight_sf = inputs
    family, mode, tile, cluster, ab = candidate
    pad = _pad_multiple(candidate)
    packed, scales = quantize_mext(
        x, recipe="ceil", mext_layout=family[0], m_pad_multiple=pad,
        pdl_mode=pdl_mode)
    _LAST_TILE[0] = candidate
    out = fold_gemm(
        packed, weight, scales, weight_sf,
        out_dtype=out_dtype, mma_tiler_mn=tile,
        cluster_shape_mn=cluster, store_mode=mode,
        ab_stage_override=ab, pi_via_layout=family == ("tile8", "pi"))
    return out[: x.shape[0]]


def _fallback(inputs, pdl_mode: str, out_dtype):
    return _run_tactic(inputs, _DEFAULT, pdl_mode, out_dtype)


def _valid(inputs, policy, _tuning):
    out_dtype, _pdl_mode, _hash = policy
    # vLLM eager warmup loads cached winners before capture, even when tuning
    # is off. Capture itself may only select resident specializations.
    allow_uncompiled = not torch.cuda.is_current_stream_capturing()
    from nvfp4.cute_fold.host import _compile_fold_gemm, STORE_MODE_TO_MODE

    return [i for i, (family, mode, tile, cluster, ab) in enumerate(_candidates())
            if _static_legal(_candidates()[i], int(inputs[0].shape[0]))
            and (allow_uncompiled or _compile_fold_gemm(
                tile, cluster, out_dtype, mode=STORE_MODE_TO_MODE[mode],
                kernel_arch="sm100", ab_stage_override=ab,
                pi_via_layout=family == ("tile8", "pi"), fp4_format="mxfp4",
                compile_if_missing=False) is not None)]


def _run(inputs, policy, index):
    out_dtype, pdl_mode, _hash = policy
    return _run_tactic(inputs, _candidates()[index], pdl_mode, out_dtype)


def _default(inputs, policy):
    out_dtype, pdl_mode, _hash = policy
    return _fallback(inputs, pdl_mode, out_dtype)


def _route(m, policy):
    # The legality of direct stores and padded quantization changes at exact
    # logical M, so a graph-capture size must never borrow another M's winner.
    return int(m)


def _set_source(source: str):
    _LAST_SOURCE[0] = source


def last_tactic_source() -> str | None:
    return _LAST_SOURCE[0]


def last_tactic_tile() -> tuple | None:
    return _LAST_TILE[0]


def tuned_mxfp4_fold(x, packed_w, flat_ws, *, pdl_mode="early",
                     out_dtype=torch.bfloat16, tune=True,
                     tuning_buckets=None):
    """Tune full MXFP4 quantize + Fold over exact logical token counts."""
    if pdl_mode not in ("off", "tail", "early"):
        raise ValueError(f"invalid MXFP4 PDL mode {pdl_mode!r}")
    if not tune:
        _set_source("offline_table")
        return _fallback([x, packed_w, flat_ws], pdl_mode, out_dtype)
    m = int(x.shape[0])
    buckets = tuple(sorted({*(int(b) for b in (tuning_buckets or ())), m}))
    spec = LinearSpec(
        "mxfp4", (out_dtype, pdl_mode, candidates_hash()), buckets,
        bool(tune), _default, _valid, _run, _route,
        "mxfp4_row_pair_v3", _set_source,
    )
    return tuned_linear([x, packed_w, flat_ws], spec)
