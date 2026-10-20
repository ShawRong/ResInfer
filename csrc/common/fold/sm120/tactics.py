# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""SM120 Fold default tile selection."""

from __future__ import annotations

import torch


_N_WIDE = 8192
_N_MIDWIDE = 6144
_N_MID = 4096


_TILE_TABLE = {
    "wide": ((32, (128, 64, 256)), (1 << 30, (128, 128, 128))),
    "midwide": ((32, (64, 64, 256)), (64, (128, 64, 256)),
                (1 << 30, (128, 128, 128))),
    "mid": ((32, (64, 64, 256)), (1 << 30, (128, 64, 256))),
    "narrow": ((32, (64, 64, 128)), (64, (64, 64, 256)),
               (1 << 30, (64, 128, 128))),
}

def supports_capability(major: int, minor: int) -> bool:
    """True on GeForce Blackwell (sm_120 / sm_121), where this kernel is validated."""
    return major == 12


def select_tile(m_tok: int, n_w: int, k: int) -> tuple[int, int, int]:
    """Select a static tile by (n_w, m_tok), then enforce K divisibility."""
    bucket = ("wide" if n_w >= _N_WIDE else
              "midwide" if n_w >= _N_MIDWIDE else
              "mid" if n_w >= _N_MID else "narrow")
    for bound, tile in _TILE_TABLE[bucket]:
        if m_tok <= bound:
            tile_m, tile_n, tile_k = tile
            break
    if k % tile_k:
        tile_k = 128
    if k % tile_k:
        raise ValueError(f"K={k} is not a multiple of 128")
    return (tile_m, tile_n, tile_k)


def act_pad_multiple(m_tok: int, n_w: int, k: int) -> int:
    """Return the activation padding multiple for the selected tile."""
    return select_tile(m_tok, n_w, k)[1] // 2


def run_fold_default(
    major: int,
    minor: int,
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    m_tok_real: int | None = None,
) -> torch.Tensor:
    """Run the fold GEMM with the measured-best tile for this GPU / m_tok."""
    from .host import mext_fold_gemm_sm120

    if not supports_capability(major, minor):
        raise RuntimeError(f"no sm120 fold kernel for capability ({major},{minor})")
    return mext_fold_gemm_sm120(
        weight_fp4, act_fp4_rowpair, weight_sf, act_sf, alpha, out_dtype,
        m_tok_real=m_tok_real,
    )


def warmup_fold_default(major: int, minor: int, shapes) -> None:
    """Compile default Fold kernels for the requested shapes."""
    from .host import compile_fold_gemm_sm120

    if not supports_capability(major, minor):
        return

    torch.zeros(1, device="cuda")

    for entry in shapes:
        m_tok, n_w, k, out_dtype = entry[:4]
        k_ext = entry[4] if len(entry) > 4 else k
        tile = select_tile(m_tok, n_w, k)
        compile_fold_gemm_sm120(n_w, k, tile, out_dtype,
                                a_ld=None if k_ext == k else k_ext)
