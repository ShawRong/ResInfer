# SPDX-License-Identifier: Apache-2.0
"""Ratio-1 residue prefill K-loop candidates used by the shared Linear tuner.

SM100 has two candidates; SM120 has one. Kernel readiness is checked before
using a persisted tactic outside the FlashInfer tuning window.
"""
from __future__ import annotations


import torch

from .fi_tuner_base import precompile, weight_geometry

KLOOP_TILES = ((128, 128, 128),)
KLOOP_SM100_CFGS = (((128, 128), (1, 1)), ((256, 128), (2, 1)))
_CUSTOM_OP = "residue_ratio1_prefill_fp4_gemm"


def _is_sm10x():
    return torch.cuda.get_device_capability()[0] == 10


def _kernel(sm10x, index, dtype, n_w=0, k=0, a_ld=None, *, compile_if_missing=True):
    if sm10x:
        from nvfp4.cute_fold.host import _compile_fold_gemm

        tile, cluster = KLOOP_SM100_CFGS[index]
        return _compile_fold_gemm(tile, cluster, dtype, mode="kloop_tma",
                                  kernel_arch="sm100", compile_if_missing=compile_if_missing)
    from nvfp4.cute_fold_sm120.host import compile_fold_gemm_sm120

    return compile_fold_gemm_sm120(n_w, k, KLOOP_TILES[index], dtype,
                                   a_ld=a_ld, k_loop=True,
                                   compile_if_missing=compile_if_missing)


def _run_kloop(inputs, tactic_idx):
    import nvfp4

    x, w, gs_inv, wsb, alpha = inputs
    m = x.shape[0]
    with torch.cuda.device(x.device):
        if _is_sm10x():
            from nvfp4.cute_fold.host import kext_kloop_gemm_sm100

            tile, cluster = KLOOP_SM100_CFGS[tactic_idx]
            f, s = nvfp4.scaled_fp4_quant_mext_r1(x, gs_inv, layout_mode="concat_k")
            return kext_kloop_gemm_sm100(w, f, wsb, s, alpha, x.dtype,
                                        mma_tiler_mn=tile, cluster_shape_mn=cluster)
        from nvfp4.cute_fold_sm120.host import kext_kloop_gemm_sm120

        # SM120 requires a full activation tile; stores use the real row count.
        rows = (m + 127) // 128 * 128
        xp = x
        if rows != m:
            xp = torch.zeros(rows, x.shape[1], dtype=x.dtype, device=x.device)
            xp[:m] = x
        f, s = nvfp4.scaled_fp4_quant_mext_r1(xp, gs_inv, layout_mode="concat_k")
        return kext_kloop_gemm_sm120(w, f, wsb, s, alpha, x.dtype,
                                    tile_shape_mnk=KLOOP_TILES[tactic_idx], m_tok_real=m)


def precompile_kloop_sm100(out_dtype=torch.bfloat16):
    return precompile(_CUSTOM_OP, _kernel,
                      [((True, i, out_dtype), {}) for i in range(len(KLOOP_SM100_CFGS))])


def precompile_kloop_sm120(n_w, k, out_dtype=torch.bfloat16):
    return precompile(_CUSTOM_OP, _kernel,
                      [((False, i, out_dtype, n_w, k), {}) for i in range(len(KLOOP_TILES))])


def _kloop_valid_tactics(inputs, tuning):
    x, weight = inputs[:2]
    with torch.cuda.device(x.device):
        sm10x = _is_sm10x()
        indices = range(len(KLOOP_SM100_CFGS) if sm10x else len(KLOOP_TILES))
        n_w, k, a_ld = weight_geometry(weight)
        if sm10x and a_ld is not None:
            return []  # The SM100 K-loop host requires contiguous weights.
        if tuning:
            return list(indices)
        try:
            return [i for i in indices if _kernel(sm10x, i, x.dtype, n_w, k, a_ld,
                                                 compile_if_missing=False) is not None]
        except ImportError:
            return []
