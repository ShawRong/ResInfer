# SPDX-License-Identifier: Apache-2.0
"""Allocation wrapper for the wheel-built SM100 MXFP4 residue operator."""
import torch

from ..extension import ensure_loaded


def _run(source, packed, scales, masks, per8, swizzle, tile8=False, pdl_mode=0):
    ensure_loaded()
    torch.ops.nvfp4.mxfp4_residue_quant(
        source, packed, scales, masks, per8, swizzle, tile8, pdl_mode
    )


def quantize_residue(
    source,
    *,
    masks=None,
    per8=0,
    layout="linear",
    kp=None,
    mext_layout="row_pair",
    m_pad_multiple=None,
    pdl_mode="off",
):
    from .quantization import _validate_2d

    _validate_2d(source, layout, "ceil")
    modes = {"off": 0, "tail": 1, "early": 2}
    if pdl_mode not in modes:
        raise ValueError("pdl_mode must be off, tail or early")
    rows, k = source.shape
    padded_k = k if kp is None else kp
    if mext_layout not in ("row_pair", "tile8"):
        raise ValueError("mext_layout must be row_pair or tile8")
    tile8 = mext_layout == "tile8"
    if tile8:
        if masks is not None or layout != "128x4":
            raise ValueError("tile8 requires full residue and 128x4 scales")
        multiple = 8 if m_pad_multiple is None else int(m_pad_multiple)
        if multiple <= 0 or multiple % 8:
            raise ValueError("tile8 m_pad_multiple must be a positive multiple of 8")
        output_m = ((rows + multiple - 1) // multiple) * multiple
    else:
        if m_pad_multiple is not None:
            raise ValueError("m_pad_multiple is supported only for tile8")
        output_m = rows
    output_rows = output_m * (2 if masks is None else 1)
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
            layout == "128x4",
            tile8,
            modes[pdl_mode],
        )
    return packed, scales
