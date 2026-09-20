# SPDX-License-Identifier: Apache-2.0
"""Selective residue adapters around vLLM's official FlashInfer MXFP4 quant.

NVFP4 pack16 dataflow: fixed selection per8, register QDQ, warp-only residue
gather and final stores in one launch. No intermediate residual rounding.
"""
from __future__ import annotations

import operator
import torch
import torch.nn.functional as F

from .quantization import quantize, _ceildiv, _validate_2d


def prepare_indices(indices,k,device):
    """Prepare immutable indices/masks with exactly 1/2/4 selections per8.

    Keep the returned tensor for forward calls; cloned/moved indices must be
    prepared again. Mask generation and validation never run during capture.
    """
    if isinstance(indices,torch.Tensor):
        if indices.ndim!=1 or indices.dtype not in (torch.int32,torch.int64):
            raise ValueError("indices must be a 1D integer tensor")
        indices=indices.cpu().tolist()
    values=[operator.index(i) for i in indices]
    _dimensions(k,len(values))
    if not values or values[0]<0 or values[-1]>=k:
        raise ValueError("indices must lie in [0,K)")
    if any(b<=a for a,b in zip(values,values[1:])):
        raise ValueError("indices must be strictly increasing and unique")
    per8=len(values)*8//k
    masks=[0]*(k//8)
    for i,value in enumerate(values):
        if value//8 != i//per8:
            raise ValueError('selective requires exactly 1/2/4 selected channels in every group of 8')
        masks[value//8] |= 1 << (value%8)
    result=torch.tensor(values,dtype=torch.int32,device=device)
    # Setup-only immutable metadata; no mask construction in the hot path.
    result._mxfp4_masks=torch.tensor(masks,dtype=torch.uint8,device=device)
    return result


def _dimensions(k,selected):
    if k<=0 or k%32 or selected<=0:
        raise ValueError("K must be a positive multiple of 32")
    if k not in (selected*8,selected*4,selected*2):
        raise ValueError("selective residue supports ratios 0.125, 0.25, 0.5")
    return _ceildiv(k+selected,128)*128


def quantize_selective(x,indices,*,masks=None,layout="128x4",recipe="ceil",k_alignment=128):
    """Quantize dense [M,K] base and selected FP32 reconstruction error."""
    k=_validate_2d(x,layout,recipe)
    if indices.ndim!=1 or indices.dtype!=torch.int32 or indices.device!=x.device or not indices.is_contiguous():
        raise ValueError("indices must be prepared contiguous int32 [S] on input device")
    s=indices.numel()
    kp=_dimensions(k,s)
    if k_alignment not in (128,256):
        raise ValueError("selective K alignment must be 128 or 256")
    kp=_ceildiv(kp,k_alignment)*k_alignment
    masks=getattr(indices,'_mxfp4_masks',None) if masks is None else masks
    if masks is None:
        raise ValueError('call prepare_indices at setup to validate per-8 selection and build masks')
    if (masks.ndim!=1 or masks.dtype!=torch.uint8 or masks.device!=x.device
            or not masks.is_contiguous() or masks.numel()!=k//8):
        raise ValueError('masks must be prepared contiguous uint8 [K/8] on input device')
    from .residue_quant import quantize_fused
    return quantize_fused(x,masks=masks,per8=s*8//k,residue=True,
                          layout=layout,kp=kp)


def prepare_selective_weight(weight,indices,*,recipe="ceil"):
    """Quantize selected original weight columns with the official producer."""
    if weight.ndim!=2:
        raise ValueError("expected weight [N,K]")
    k=weight.shape[1]
    kp=_dimensions(k,indices.numel())
    extended=torch.cat((weight,weight.index_select(1,indices)),dim=1)
    return quantize(F.pad(extended,(0,kp-extended.shape[1])),recipe=recipe)
