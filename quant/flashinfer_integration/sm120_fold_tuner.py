# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 SM120 tile selection; profile padding, quantization, and Fold together."""

from __future__ import annotations


import torch

from .fi_tuner_base import precompile, weight_geometry


FOLD_TILES = (
    (64, 64, 128), (64, 64, 256), (64, 128, 128),
    (128, 64, 128), (128, 64, 256), (128, 128, 128), (128, 128, 256),
)

_CUSTOM_OP = "residue_fold_tile_sm120"

_LAST_SOURCE: list = [None]
_LAST_TILE: list = [None]


def last_tactic_source() -> str | None:
    return _LAST_SOURCE[0]


def last_tactic_tile() -> tuple[int, int, int] | None:
    return _LAST_TILE[0]


def _record_source(src: str) -> None:
    _LAST_SOURCE[0] = src


def fold_tiles_hash() -> str:
    import hashlib

    return hashlib.md5(repr(FOLD_TILES).encode()).hexdigest()[:12]


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
    _LAST_TILE[0] = tile
    x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(x_in, gs_inv, layout_mode=1)
    out = mext_fold_gemm_sm120(
        weight, x_fp4, wsb, x_sf.view(torch.float8_e4m3fn), alpha, out_dtype,
        tile_shape_mnk=tile, m_tok_real=m,
    )
    return out


def _table_tile(inputs) -> tuple[int, int, int]:
    from nvfp4.cute_fold_sm120.tactics import select_tile

    x, weight = inputs[0], inputs[1]
    n_w, k_logical, _ = weight_geometry(weight)
    return select_tile(int(x.shape[0]), n_w, k_logical)


def run_fold_table(inputs, out_dtype) -> torch.Tensor:
    """The untuned path: the offline-swept static table picks the tile."""
    return _run_fold_tile(inputs, _table_tile(inputs), out_dtype)


def precompile_fold_tiles(n_w: int, k_logical: int, a_ld: int | None = None,
                          out_dtype=torch.bfloat16) -> int:
    """Prepare all legal candidate tiles for one weight geometry."""
    from nvfp4.cute_fold_sm120.host import compile_fold_gemm_sm120

    specs = [((n_w, k_logical, tile, out_dtype), {"a_ld": a_ld})
             for tile in FOLD_TILES if k_logical % tile[2] == 0]
    return precompile(_CUSTOM_OP, compile_fold_gemm_sm120, specs)


def valid_tactics(inputs, out_dtype, tuning):
    n_w, k_logical, a_ld = weight_geometry(inputs[1])
    from nvfp4.cute_fold_sm120.host import compile_fold_gemm_sm120

    return [i for i, tile in enumerate(FOLD_TILES)
            if k_logical % tile[2] == 0 and (tuning or compile_fold_gemm_sm120(
                n_w, k_logical, tile, out_dtype, a_ld=a_ld,
                compile_if_missing=False) is not None)]
