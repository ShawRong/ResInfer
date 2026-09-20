# SPDX-License-Identifier: Apache-2.0
"""MXFP4 quantization through the official FlashInfer kernel used by vLLM.

FP16/BF16 standard quant calls the unmodified official producer. FP32,
full-R1 and selective paths use the NVFP4 pack16 CUDA specialization.
All paths use block32 and official ceil E8M0 zero/subnormal policy.
FP32 input and residual retain FP32 precision; no intermediate BF16 cast.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _ceildiv(n, d):
    return (n+d-1)//d


def _check_recipe(recipe):
    if recipe != "ceil":
        raise ValueError("MXFP4 uses vLLM/FlashInfer native ceil quantization; legacy ocp is unsupported")


def _native_2d(x, layout):
    """Call the unmodified upstream producer; preserve our public byte shapes."""
    from flashinfer.quantization.kernels.mxfp4_quantize import (
        mxfp4_quantize_cute_dsl, SF_LAYOUT_LINEAR)
    m,k=x.shape
    if not m:
        sf_shape=(m,k//32) if layout=="linear" else (0,)
        return (torch.empty((m,k//2),device=x.device,dtype=torch.uint8),
                torch.empty(sf_shape,device=x.device,dtype=torch.uint8))
    x=x.contiguous()
    if layout=="linear":
        q,sf=mxfp4_quantize_cute_dsl(x,sf_layout=SF_LAYOUT_LINEAR)
        return q,sf.reshape(m,k//32)
    # Public vLLM wrapper selects this same FlashInfer entry/backend.
    from flashinfer import mxfp4_quantize
    # Upstream's swizzled result reshape requires complete 128-value tiles.
    kp=_ceildiv(k,128)*128
    if kp!=k:
        x=F.pad(x,(0,kp-k))
    q,sf=mxfp4_quantize(x,backend="cute-dsl")
    return q[:,:k//2].contiguous(),sf.flatten()


def _validate_2d(x, layout, recipe):
    _check_recipe(recipe)
    if x.ndim!=2 or not x.is_cuda or x.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise ValueError("expected CUDA fp16/bf16/fp32 [M,K]")
    _,k=x.shape
    if k<=0 or k%32:
        raise ValueError("K must be a positive multiple of 32")
    if layout not in ("linear","128x4"):
        raise ValueError("layout must be linear or 128x4")
    return k


def quantize(x, *, layout="128x4", recipe="ceil"):
    _validate_2d(x,layout,recipe)
    if x.dtype!=torch.float32:
        return _native_2d(x,layout)
    from .residue_quant import quantize_fused
    return quantize_fused(x,residue=False,layout=layout)


def quantize_mext(x, *, layout="128x4", recipe="ceil"):
    _validate_2d(x,layout,recipe)
    from .residue_quant import quantize_fused
    return quantize_fused(x,residue=True,layout=layout)


def _layout(scales, rows, k, inverse):
    if rows<0 or k<=0 or k%32 or scales.dtype!=torch.uint8 or not scales.is_cuda or not scales.is_contiguous():
        raise ValueError("expected contiguous CUDA uint8 scales, rows>=0, K%32==0")
    rp,gp=_ceildiv(rows,128)*128,_ceildiv(k//32,4)*4
    stride=rp*gp
    if inverse:
        if scales.ndim not in (1,2) or scales.shape[-1]!=stride:
            raise ValueError("expected [swizzled_bytes] or [E,swizzled_bytes]")
        batched=scales.ndim==2
        e=scales.shape[0] if batched else 1
        result=(scales.reshape(e,rp//128,gp//4,32,4,4)
                .permute(0,1,4,3,2,5).reshape(e,rp,gp)[:,:rows,:k//32].contiguous())
    else:
        if scales.ndim not in (2,3) or scales.shape[-2:]!=(rows,k//32):
            raise ValueError("expected linear [M,K/32] or [E,M,K/32]")
        batched=scales.ndim==3
        e=scales.shape[0] if batched else 1
        padded=F.pad(scales.reshape(e,rows,k//32),(0,gp-k//32,0,rp-rows))
        result=(padded.reshape(e,rp//128,4,32,gp//4,4)
                .permute(0,1,4,3,2,5).contiguous().reshape(e,stride))
    return result if batched else result[0]


def swizzle_scales(scales):
    if scales.ndim not in (2,3):
        raise ValueError("expected [M,K/32] or [E,M,K/32]")
    return _layout(scales,scales.shape[-2],scales.shape[-1]*32,False)


def unswizzle_scales(scales, rows, k):
    return _layout(scales,rows,k,True)
