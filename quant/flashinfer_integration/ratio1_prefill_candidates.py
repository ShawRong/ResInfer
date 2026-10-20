# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Ratio-1 prefill K-loop candidates for FlashInfer autotuning."""
from __future__ import annotations

import warnings

import torch

from .fi_tuner_base import weight_geometry

KLOOP_TILES = ((128, 128, 128),)
KLOOP_SM100_CFGS = (((128, 128), (1, 1)), ((256, 128), (2, 1)))


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


def prepare_kloop(inputs):
    """Prepare explicit k_loop mode on eager first use; failures retain two-GEMM."""
    x, weight = inputs[:2]
    n, k, a_ld = weight_geometry(weight)
    sm10x = _is_sm10x()
    torch.zeros(1, device="cuda")  # CuTe needs a current CUDA context.
    done = 0
    for i in _kloop_valid_tactics(inputs, True):
        try:
            _kernel(sm10x, i, x.dtype, n, k, a_ld)
            done += 1
        except Exception as exc:
            warnings.warn(
                f"[residue] ratio1 prefill K-loop tactic {i} compilation failed: "
                f"{type(exc).__name__}: {str(exc)[:160]}", RuntimeWarning)
    return done
