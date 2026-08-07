# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Tile selection for the SM120 CuTeDSL fold GEMM, by (n_w, m_tok).'

from __future__ import annotations

import os

import torch

# Width buckets. `n_w` is the GEMM's N (output features).
_N_WIDE = 8192      # gate_up-class: needs tile_m=128 even at m_tok=1
_N_MID = 4096       # qkv-class: needs tile_m=128 once m_tok >= 64

# (n_w bucket, m_tok upper bound inclusive) -> (tile_m, tile_n, tile_k)
_TILE_TABLE = {
    "wide": ((32, (128, 64, 128)), (1 << 30, (128, 128, 128))),
    
    
    "mid": ((32, (64, 64, 256)), (1 << 30, (128, 64, 256))),
    "narrow": ((32, (64, 64, 128)), (64, (64, 64, 256)),
               (1 << 30, (64, 128, 128))),
}


def supports_capability(major: int, minor: int) -> bool:
    """True on GeForce Blackwell (sm_120 / sm_121), where this kernel is validated."""
    return major == 12


def select_tile(m_tok: int, n_w: int, k: int) -> tuple[int, int, int]:
    'Implementation details.'
    bucket = ("wide" if n_w >= _N_WIDE else
              "mid" if n_w >= _N_MID else "narrow")
    for bound, tile in _TILE_TABLE[bucket]:
        if m_tok <= bound:
            tile_m, tile_n, tile_k = tile
            break
    if k % tile_k:
        tile_k = 128          # K=768/1536/... cannot use the 256 k-tile
    if k % tile_k:
        raise ValueError(f"K={k} is not a multiple of 128")
    return (tile_m, tile_n, tile_k)


def act_pad_multiple(m_tok: int, n_w: int, k: int) -> int:
    'Token-count multiple the activation should be padded to.'
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
    'Implementation details.'
    from .host import mext_fold_gemm_sm120

    if not supports_capability(major, minor):
        raise RuntimeError(f"no sm120 fold kernel for capability ({major},{minor})")
    if os.environ.get("RESIDUE_FOLD_SILENCE_DEFAULT", "0") != "1":
        from ..fold import warn_default_path

        m_tok = act_fp4_rowpair.shape[0] // 2
        warn_default_path("sm120", m_tok, f"tile={select_tile(m_tok, weight_fp4.shape[0], weight_fp4.shape[1] * 2)}")
    return mext_fold_gemm_sm120(
        weight_fp4, act_fp4_rowpair, weight_sf, act_sf, alpha, out_dtype,
        m_tok_real=m_tok_real,
    )


def warmup_fold_default(major: int, minor: int, shapes) -> None:
    """Pre-compile the kernels a server will need (call at engine init, i.e.
    OUTSIDE cudagraph capture; runtime dispatch then never JITs).

    shapes: iterable of (m_tok, n_w, k, out_dtype) or
            (m_tok, n_w, k, out_dtype, k_ext).

    k_ext is the STORED width of an extended_k residue weight. Such a layer is
    served from a `weight_ext[:, :k_base_packed]` view, whose row stride makes
    it a DIFFERENT compiled kernel from the contiguous one (a_ld is part of the
    compile cache key). Omitting k_ext here would leave that variant to JIT on
    first use -- i.e. inside cudagraph capture, which is the one place warmup
    exists to prevent. Pass k_ext == k (or omit it) for a contiguous weight.
    """
    from .host import compile_fold_gemm_sm120

    if not supports_capability(major, minor):
        return
    # Compiling queries the device through CUTLASS HardwareInfo, which needs a
    # live CUDA context; without one it fails with CUDA_ERROR_INVALID_CONTEXT.
    # A caller that warms up before touching CUDA (a standalone script, or a
    # future engine that warms before loading weights) would hit that, so
    # force the context here rather than depending on call order.
    torch.zeros(1, device="cuda")
    
    
    
    
    for entry in shapes:
        m_tok, n_w, k, out_dtype = entry[:4]
        k_ext = entry[4] if len(entry) > 4 else k
        tile = select_tile(m_tok, n_w, k)
        compile_fold_gemm_sm120(n_w, k, tile, out_dtype,
                                a_ld=None if k_ext == k else k_ext)
