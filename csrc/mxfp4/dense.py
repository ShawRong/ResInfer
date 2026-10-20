# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""MXFP4 dense and fused MExt-R1 GEMM on the validated SM100 target."""
from __future__ import annotations

import torch

from ..cute_fold.host import _compile_fold_gemm, mext_fold_gemm


def _validate(a, b, sa, sb, out_dtype):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected A[M,K/2], B[N,K/2]")
    if not a.is_cuda or any(t.device != a.device for t in (b, sa, sb)):
        raise ValueError("all tensors must share a CUDA device")
    if any(t.dtype != torch.uint8 or not t.is_contiguous() for t in (a, b, sa, sb)):
        raise ValueError("packed data and 128x4 scales must be contiguous uint8")
    m, kp = a.shape
    n = b.shape[0]
    k = kp * 2
    if min(m, n, k) <= 0 or k % 128 or n % 32:
        raise ValueError("positive dimensions with K%128==0 and N%32==0 required")
    for rows, sf in ((m, sa), (n, sb)):
        size = ((rows + 127) // 128) * ((k + 127) // 128) * 512
        if sf.ndim != 1 or sf.numel() != size:
            raise ValueError(f"expected 128x4 E8M0 scale buffer of {size} bytes")
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("output must be bf16 or fp16")
    capability = torch.cuda.get_device_capability(a.device)
    if capability != (10, 0):
        raise ValueError(f"MXFP4 dense kernel requires SM100; got {capability}")
    return m, n, k, "sm100"


def gemm(a, b, a_scale, b_scale, *, out_dtype=torch.bfloat16,
         mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1)):
    """Compute dense MXFP4 GEMM."""
    m,n,k,arch = _validate(a,b,a_scale,b_scale,out_dtype)
    with torch.cuda.device(a.device):
        compiled = _compile_fold_gemm(
            tuple(mma_tiler_mn),tuple(cluster_shape_mn),out_dtype,
            mode="plain_mn",kernel_arch=arch,fp4_format="mxfp4")
        out = torch.empty((m,n),device=a.device,dtype=out_dtype)
        compiled(a,b,out,(m+127)//128,(n+127)//128,(k+127)//128,
                 a_scale.data_ptr(),b_scale.data_ptr(),None)
        return out


def fold_gemm(a_rowpair, b, a_scale, b_scale, *, out_dtype=torch.bfloat16,
              mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1),
              store_mode="tma", pi_via_layout=False, ab_stage_override=None):
    """Compute MXFP4 GEMM with full-channel residue compensation."""
    m2, _, _, arch = _validate(a_rowpair, b, a_scale, b_scale, out_dtype)
    if m2 % 2:
        raise ValueError("MExt input needs an even number of physical rows")
    if store_mode not in ("tma", "tma_mext", "direct_mext"):
        raise ValueError("unsupported MXFP4 Fold store mode")
    with torch.cuda.device(a_rowpair.device):
        return mext_fold_gemm(
            b, a_rowpair, b_scale, a_scale,
            None,
            out_dtype=out_dtype, mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=tuple(cluster_shape_mn), kernel_arch=arch,
            store_mode=store_mode, pi_via_layout=pi_via_layout,
            ab_stage_override=ab_stage_override,
            fp4_format="mxfp4")
