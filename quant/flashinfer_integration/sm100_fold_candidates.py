# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 SM10x producer + Fold selection; defaults and family gates are unchanged."""

from __future__ import annotations

from functools import lru_cache

import torch

from .fi_tuner_base import weight_geometry

_FAMILY = ("row_pair", "std")
_LAYOUT_ROW_PAIR = 1
_LAYOUT_TILE8 = 2
_M_PAD_MULTIPLE = 8

_LAST_SOURCE: list = [None]
_LAST_TILE: list = [None]


def last_tactic_source() -> str | None:
    return _LAST_SOURCE[0]


def last_tactic_tile():
    return _LAST_TILE[0]


def _record_source(src: str) -> None:
    _LAST_SOURCE[0] = src


def _families() -> tuple:
    from nvfp4.fold import candidate_families

    if _arch() != "sm100":
        return (_FAMILY,)
    return tuple(candidate_families())


@lru_cache(maxsize=None)
def _candidate_table(families):
    from nvfp4.cute_fold.tactics import FAMILY_TACTICS, _norm_tactic

    return [(family, *_norm_tactic(tactic)) for family in families
            for tactic in FAMILY_TACTICS[family]]


def _candidates():
    return _candidate_table(_families())


def candidates_hash() -> str:
    import hashlib

    return hashlib.md5(repr(_candidates()).encode()).hexdigest()[:12]


def _arch() -> str | None:
    from nvfp4.cute_fold.tactics import kernel_arch_for_capability

    major, minor = torch.cuda.get_device_capability()
    return kernel_arch_for_capability(int(major), int(minor))


def _legal_indices(m_tok: int) -> list[int]:
    """Filter the shared tactic table by logical M and known output restrictions."""
    from nvfp4.cute_fold.tactics import _rank_tactics

    arch = _arch()
    if arch != "sm100":
        return []
    legal = {(family, tactic) for family in _families()
             for tactic in _rank_tactics(family, m_tok, arch)
             # A4 (128,32) returned incomplete output at N1152/K8192.
             if not (family == ("tile8", "std") and tactic[1] == (128, 32))}
    return [i for i, c in enumerate(_candidates()) if (c[0], c[1:]) in legal]


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
    from nvfp4.cute_fold.host import mext_fold_gemm
    from nvfp4.cute_fold.tactics import _needs_pi_layout

    family, mode, tiler, cluster, ab = _candidates()[idx]
    _LAST_TILE[0] = (family, mode, tiler, cluster, ab)
    x_fp4, x_sf = _quant_for_family(
        inputs, family, pad_multiple=tile8_pad_multiple(mode, tiler))
    _, weight, _, wsb, alpha = inputs
    out = mext_fold_gemm(
        weight, x_fp4, wsb, x_sf, alpha, out_dtype,
        mma_tiler_mn=tiler, cluster_shape_mn=cluster, store_mode=mode,
        kernel_arch=_arch(), ab_stage_override=ab,
        pi_via_layout=_needs_pi_layout(family),
    )
    return out[: inputs[0].shape[0]]


def _run_fallback(inputs, out_dtype) -> torch.Tensor:
    """Static row-pair fallback: never nest the old tuner inside FI tuning."""
    from nvfp4.cute_fold.tactics import run_fold_default, select_arch_tactic

    _, tiler, cluster = select_arch_tactic(_arch(), int(inputs[0].shape[0]))
    _LAST_TILE[0] = (_FAMILY, "tma", tiler, cluster, None)
    x_fp4, x_sf = _quant_for_family(inputs, _FAMILY)
    _, weight, _, wsb, alpha = inputs
    major, minor = torch.cuda.get_device_capability()
    return run_fold_default(
        int(major), int(minor), weight, x_fp4, wsb, x_sf, alpha, out_dtype)


def valid_tactics(inputs, out_dtype, tuning):
    legal = _legal_indices(int(inputs[0].shape[0]))
    if tuning:
        return legal
    from nvfp4.cute_fold.host import _compile_fold_gemm, STORE_MODE_TO_MODE

    _, _, a_ld = weight_geometry(inputs[1])
    return [i for i in legal if _compile_fold_gemm(
        _candidates()[i][2], _candidates()[i][3], out_dtype,
        mode=STORE_MODE_TO_MODE[_candidates()[i][1]], kernel_arch=_arch(),
        ab_stage_override=_candidates()[i][4],
        pi_via_layout=_candidates()[i][0] == ("tile8", "pi"),
        strided_weight=a_ld is not None, dual_alpha=inputs[4].numel() == 2,
        compile_if_missing=False) is not None]
