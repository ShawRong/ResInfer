# SPDX-License-Identifier: Apache-2.0
"""Allocation wrapper for the wheel-built SM100 MXFP4 residue operator."""
import torch

from ..extension import ensure_loaded


def _run(source, packed, scales, masks, per8, residue, swizzle):
    ensure_loaded()
    torch.ops.nvfp4.mxfp4_residue_quant(
        source, packed, scales, masks, per8, residue, swizzle
    )


def quantize_fused(
    source,
    *,
    masks=None,
    per8=0,
    residue=False,
    layout="linear",
    kp=None,
):
    rows, k = source.shape
    padded_k = k if kp is None else kp
    output_rows = rows * (2 if residue and masks is None else 1)
    packed = torch.empty(
        (output_rows, padded_k // 2), device=source.device, dtype=torch.uint8
    )
    if layout == "linear":
        scale_shape = (output_rows, padded_k // 32)
    else:
        scale_shape = (
            ((output_rows + 127) // 128) * ((padded_k + 127) // 128) * 512,
        )
    scales = torch.empty(scale_shape, device=source.device, dtype=torch.uint8)
    if rows:
        _run(
            source,
            packed,
            scales,
            masks,
            per8,
            residue,
            layout == "128x4",
        )
    return packed, scales
