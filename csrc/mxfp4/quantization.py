# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""MXFP4 activation quantization."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def _check_recipe(recipe):
    if recipe != "ceil":
        raise ValueError(
            "MXFP4 uses vLLM/FlashInfer native ceil quantization; "
            "legacy ocp is unsupported"
        )


def _native_2d(source, layout):
    """Quantize with the upstream producer and return two-dimensional outputs."""
    from flashinfer.quantization.kernels.mxfp4_quantize import (
        SF_LAYOUT_LINEAR,
        mxfp4_quantize_cute_dsl,
    )

    rows, k = source.shape
    if not rows:
        scale_shape = (rows, k // 32) if layout == "linear" else (0,)
        return (
            torch.empty((rows, k // 2), device=source.device, dtype=torch.uint8),
            torch.empty(scale_shape, device=source.device, dtype=torch.uint8),
        )
    source = source.contiguous()
    if layout == "linear":
        packed, scales = mxfp4_quantize_cute_dsl(
            source, sf_layout=SF_LAYOUT_LINEAR
        )
        return packed, scales.reshape(rows, k // 32)

    from flashinfer import mxfp4_quantize

    padded_k = _ceildiv(k, 128) * 128
    if padded_k != k:
        source = F.pad(source, (0, padded_k - k))
    packed, scales = mxfp4_quantize(source, backend="cute-dsl")
    return packed[:, :k // 2].contiguous(), scales.flatten()


def _validate_2d(source, layout, recipe):
    _check_recipe(recipe)
    if (source.ndim != 2 or not source.is_cuda
            or source.dtype not in (torch.float16, torch.bfloat16)):
        raise ValueError("expected CUDA fp16/bf16 [M,K]")
    k = source.shape[1]
    if k <= 0 or k % 32:
        raise ValueError("K must be a positive multiple of 32")
    if layout not in ("linear", "128x4"):
        raise ValueError("layout must be linear or 128x4")
    return k


def quantize(source, *, layout="128x4", recipe="ceil"):
    _validate_2d(source, layout, recipe)
    return _native_2d(source, layout)


def quantize_mext(source, *, layout="128x4", recipe="ceil",
                  mext_layout="row_pair", m_pad_multiple=None, pdl_mode="off"):
    """Quantize activations and their full-channel residue."""
    _check_recipe(recipe)
    from .residue_quant import quantize_residue

    return quantize_residue(source, layout=layout,
                          mext_layout=mext_layout,
                          m_pad_multiple=m_pad_multiple, pdl_mode=pdl_mode)
