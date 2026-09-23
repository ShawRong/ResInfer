# SPDX-License-Identifier: Apache-2.0
"""Selective residue metadata and fused activation quantization."""
from __future__ import annotations

import operator

import torch

from .quantization import _ceildiv, _validate_2d


def _dimensions(k, selected):
    if k <= 0 or k % 32 or selected <= 0:
        raise ValueError("K must be a positive multiple of 32")
    if k not in (selected * 8, selected * 4, selected * 2):
        raise ValueError("selective residue supports ratios 0.125, 0.25, 0.5")
    return _ceildiv(k + selected, 128) * 128


def prepare_indices(indices, k, device):
    """Validate fixed per-eight selection and return explicit indices/masks."""
    if isinstance(indices, torch.Tensor):
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be a 1D integer tensor")
        indices = indices.cpu().tolist()
    values = [operator.index(index) for index in indices]
    _dimensions(k, len(values))
    if not values or values[0] < 0 or values[-1] >= k:
        raise ValueError("indices must lie in [0,K)")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("indices must be strictly increasing and unique")

    per8 = len(values) * 8 // k
    masks = [0] * (k // 8)
    for position, value in enumerate(values):
        if value // 8 != position // per8:
            raise ValueError(
                "selective requires exactly 1/2/4 selected channels "
                "in every group of 8"
            )
        masks[value // 8] |= 1 << (value % 8)
    return (
        torch.tensor(values, dtype=torch.int32, device=device),
        torch.tensor(masks, dtype=torch.uint8, device=device),
    )


def quantize_selective(
    source,
    indices,
    masks,
    *,
    layout="128x4",
    recipe="ceil",
    k_alignment=128,
):
    """Quantize dense input plus selected FP32 reconstruction error."""
    k = _validate_2d(source, layout, recipe)
    if (indices.ndim != 1 or indices.dtype != torch.int32
            or indices.device != source.device or not indices.is_contiguous()):
        raise ValueError("indices must be contiguous int32 [S] on input device")
    selected = indices.numel()
    padded_k = _dimensions(k, selected)
    if k_alignment not in (128, 256):
        raise ValueError("selective K alignment must be 128 or 256")
    padded_k = _ceildiv(padded_k, k_alignment) * k_alignment
    if (masks.ndim != 1 or masks.dtype != torch.uint8
            or masks.device != source.device or not masks.is_contiguous()
            or masks.numel() != k // 8):
        raise ValueError("masks must be contiguous uint8 [K/8] on input device")

    from .residue_quant import quantize_residue

    return quantize_residue(
        source,
        masks=masks,
        per8=selected * 8 // k,
        layout=layout,
        kp=padded_k,
    )
