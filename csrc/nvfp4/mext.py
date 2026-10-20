# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Python allocation wrappers for M-extension FP4 quantization."""

from __future__ import annotations

import os

import torch

from .extension import ensure_loaded

ensure_loaded(required=False)


_MEXT_R1_ELTS_MODE_AUTO = 0
_MEXT_R1_ELTS_MODE_8 = 8
_MEXT_R1_ELTS_MODE_16 = 16
_MEXT_R1_LAYOUT_CONCAT = 0
_MEXT_R1_LAYOUT_ROW_PAIR = 1
_MEXT_R1_LAYOUT_TILE8 = 2

_MEXT_R1_LAYOUT_CONCAT_K = 3

_MEXT_R1_LAYOUT_ATTN_TILE8 = 4


def _resolve_mext_r1_elts_mode(elts_mode=None) -> int:
    if elts_mode is None:
        elts_mode = os.getenv("NVFP4_MEXT_R1_ELTS_PER_THREAD", "auto")

    if isinstance(elts_mode, str):
        normalized = elts_mode.strip().lower()
        if normalized in {"", "auto", "default"}:
            return _MEXT_R1_ELTS_MODE_AUTO
        if normalized in {"8", "elts8", "elts_8"}:
            return _MEXT_R1_ELTS_MODE_8
        if normalized in {"16", "elts16", "elts_16"}:
            return _MEXT_R1_ELTS_MODE_16
        raise ValueError(
            f"Unsupported mext_r1 elts_mode '{elts_mode}'. Expected one of: auto, 8, 16."
        )

    if elts_mode in (
        _MEXT_R1_ELTS_MODE_AUTO,
        _MEXT_R1_ELTS_MODE_8,
        _MEXT_R1_ELTS_MODE_16,
    ):
        return int(elts_mode)

    raise ValueError(
        f"Unsupported mext_r1 elts_mode {elts_mode!r}. Expected one of: auto, 8, 16."
    )


def _resolve_mext_r1_layout_mode(layout_mode=None) -> int:
    if layout_mode is None:
        layout_mode = os.getenv("RESIDUE_MEXT_R1_LAYOUT", "concat")

    if isinstance(layout_mode, str):
        normalized = layout_mode.strip().lower().replace("-", "_")
        if normalized in {"", "concat", "default", "base_residue"}:
            return _MEXT_R1_LAYOUT_CONCAT
        if normalized in {"row_pair", "rowpair", "pair", "interleaved"}:
            return _MEXT_R1_LAYOUT_ROW_PAIR
        if normalized in {"tile8", "tile_8", "tile8_pair", "tile8_row_pair"}:
            return _MEXT_R1_LAYOUT_TILE8
        if normalized in {
            "attn_tile8",
            "attention_tile8",
            "qk_tile8",
            "gqa_tile8",
        }:
            return _MEXT_R1_LAYOUT_ATTN_TILE8
        if normalized in {"concat_k", "kext", "k_concat", "kloop"}:
            return _MEXT_R1_LAYOUT_CONCAT_K
        raise ValueError(
            f"Unsupported mext_r1 layout_mode '{layout_mode}'. "
            "Expected one of: concat, row_pair, tile8, attn_tile8, concat_k."
        )

    if layout_mode in (
        _MEXT_R1_LAYOUT_CONCAT,
        _MEXT_R1_LAYOUT_ROW_PAIR,
        _MEXT_R1_LAYOUT_TILE8,
        _MEXT_R1_LAYOUT_CONCAT_K,
        _MEXT_R1_LAYOUT_ATTN_TILE8,
    ):
        return int(layout_mode)

    raise ValueError(
        f"Unsupported mext_r1 layout_mode {layout_mode!r}. "
        "Expected one of: concat, row_pair, tile8, attn_tile8, concat_k."
    )


def scaled_fp4_quant_mext_r1(
    input_tensor: torch.Tensor,
    input_scale: torch.Tensor,
    *,
    elts_mode=None,
    layout_mode=None,
    m_pad_multiple=None,
    enable_pdl: bool | None = None,
):
    """Quantize activations and their full-channel residue into packed FP4 outputs."""
    resolved_elts_mode = _resolve_mext_r1_elts_mode(elts_mode)
    resolved_layout_mode = _resolve_mext_r1_layout_mode(layout_mode)
    if enable_pdl is None:
        configured = os.getenv("NVFP4_MEXT_R1_PDL", "auto").strip().lower()
        if configured in {"1", "true", "on", "yes"}:
            enable_pdl = True
        elif configured in {"0", "false", "off", "no"}:
            enable_pdl = False
        elif configured in {"", "auto", "default"}:
            major, _ = torch.cuda.get_device_capability(input_tensor.device)
            enable_pdl = major >= 9
        else:
            raise ValueError(
                "NVFP4_MEXT_R1_PDL must be auto/on/off (or 1/0), got "
                f"{configured!r}"
            )

    m, k = input_tensor.shape
    if k % 16 != 0:
        raise ValueError(f"scaled_fp4_quant_mext_r1 requires K multiple of 16, got {k}")

    if m_pad_multiple is None:
        if resolved_layout_mode in (
            _MEXT_R1_LAYOUT_TILE8,
            _MEXT_R1_LAYOUT_ATTN_TILE8,
        ) and m % 8 != 0:
            raise ValueError(
                "tile8 layouts require M to be divisible by 8, "
                "or pass m_pad_multiple=8 to request padded row+8 output."
            )
        output_m = m
    else:
        if resolved_layout_mode not in (
            _MEXT_R1_LAYOUT_TILE8,
            _MEXT_R1_LAYOUT_ATTN_TILE8,
        ):
            raise ValueError(
                "m_pad_multiple is supported only with tile8 layouts"
            )
        m_pad_multiple = int(m_pad_multiple)
        if m_pad_multiple <= 0:
            raise ValueError(f"m_pad_multiple must be positive, got {m_pad_multiple}")
        output_m = ((m + m_pad_multiple - 1) // m_pad_multiple) * m_pad_multiple

    output = torch.empty(
        (2 * output_m) * k // 2,
        dtype=torch.uint8,
        device=input_tensor.device,
    )
    if resolved_layout_mode == _MEXT_R1_LAYOUT_CONCAT_K:


        num_m_tiles = (output_m + 127) // 128
        num_k_tiles = (2 * k + 63) // 64
    elif resolved_layout_mode == _MEXT_R1_LAYOUT_ATTN_TILE8:


        num_m_tiles = output_m // 8
        num_k_tiles = (k + 63) // 64
    else:
        num_m_tiles = (2 * output_m + 127) // 128
        num_k_tiles = (k + 63) // 64
    scale_shape = (num_m_tiles * num_k_tiles * 512,)
    output_scale = (
        torch.zeros(*scale_shape, dtype=torch.uint8, device=input_tensor.device)
        if resolved_layout_mode == _MEXT_R1_LAYOUT_ATTN_TILE8
        else torch.empty(*scale_shape, dtype=torch.uint8, device=input_tensor.device)
    )


    ensure_loaded()


    torch.ops.nvfp4.scaled_fp4_quant_mext_r1(
        output,
        input_tensor,
        output_scale,
        input_scale,
        resolved_elts_mode,
        resolved_layout_mode,
        bool(enable_pdl),
    )
    if resolved_layout_mode == _MEXT_R1_LAYOUT_CONCAT_K:

        return output.view(output_m, k), output_scale
    return output.view(2 * output_m, k // 2), output_scale


__all__ = ["scaled_fp4_quant_mext_r1"]
