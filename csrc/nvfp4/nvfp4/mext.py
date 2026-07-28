"""Python allocation wrappers for M-extension FP4 quantization."""

from __future__ import annotations

import os

import torch

from .extension import ensure_loaded

ensure_loaded()

_MEXT_R1_ELTS_MODE_AUTO = 0
_MEXT_R1_ELTS_MODE_8 = 8
_MEXT_R1_ELTS_MODE_16 = 16
_MEXT_R1_LAYOUT_CONCAT = 0
_MEXT_R1_LAYOUT_ROW_PAIR = 1
_MEXT_R1_LAYOUT_TILE8 = 2


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
        layout_mode = os.getenv(
            "RESIDUE_MEXT_R1_LAYOUT",
            os.getenv("NVFP4_MEXT_R1_LAYOUT", "concat"),
        )

    if isinstance(layout_mode, str):
        normalized = layout_mode.strip().lower().replace("-", "_")
        if normalized in {"", "concat", "default", "base_residue"}:
            return _MEXT_R1_LAYOUT_CONCAT
        if normalized in {"row_pair", "rowpair", "pair", "interleaved"}:
            return _MEXT_R1_LAYOUT_ROW_PAIR
        if normalized in {"tile8", "tile_8", "tile8_pair", "tile8_row_pair"}:
            return _MEXT_R1_LAYOUT_TILE8
        raise ValueError(
            f"Unsupported mext_r1 layout_mode '{layout_mode}'. "
            "Expected one of: concat, row_pair, tile8."
        )

    if layout_mode in (
        _MEXT_R1_LAYOUT_CONCAT,
        _MEXT_R1_LAYOUT_ROW_PAIR,
        _MEXT_R1_LAYOUT_TILE8,
    ):
        return int(layout_mode)

    raise ValueError(
        f"Unsupported mext_r1 layout_mode {layout_mode!r}. "
        "Expected one of: concat, row_pair, tile8."
    )


def scaled_fp4_quant_mext_r1(
    input_tensor: torch.Tensor,
    input_scale: torch.Tensor,
    salient_indices,
    use_swapped_residue: bool = False,
    elts_mode=None,
    layout_mode=None,
    m_pad_multiple=None,
):
    """Ratio=1.0 M-extension FP4 quantization entrypoint.

    The public signature intentionally matches scaled_fp4_quant() for easier
    call-site migration, but this path does not consume salient_indices or
    use_swapped_residue internally. The underlying kernel emits a row-doubled
    layout directly:

      - output data:  [2M, K/2] packed uint8
      - output scale: tiled layout over (2M, K)
    """
    del salient_indices, use_swapped_residue

    resolved_elts_mode = _resolve_mext_r1_elts_mode(elts_mode)
    resolved_layout_mode = _resolve_mext_r1_layout_mode(layout_mode)

    m, k = input_tensor.shape
    if k % 16 != 0:
        raise ValueError(f"scaled_fp4_quant_mext_r1 requires K multiple of 16, got {k}")

    if m_pad_multiple is None:
        if resolved_layout_mode == _MEXT_R1_LAYOUT_TILE8 and m % 8 != 0:
            raise ValueError(
                "layout_mode='tile8' requires M to be divisible by 8, "
                "or pass m_pad_multiple=8 to request padded row+8 output."
            )
        output_m = m
    else:
        if resolved_layout_mode != _MEXT_R1_LAYOUT_TILE8:
            raise ValueError(
                "m_pad_multiple is currently supported only with layout_mode='tile8'"
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
    num_m_tiles = (2 * output_m + 127) // 128
    num_k_tiles = (k + 63) // 64
    output_scale = torch.empty(
        num_m_tiles * num_k_tiles * 512,
        dtype=torch.uint8,
        device=input_tensor.device,
    )

    # output_m is intentionally not passed: the C++ op derives it from
    # input_tensor.size(0). Passing it as an int op-arg made torch.compile
    # specialize it to a constant while output stayed symbolic in M.
    torch.ops.nvfp4.scaled_fp4_quant_mext_r1(
        output,
        input_tensor,
        output_scale,
        input_scale,
        resolved_elts_mode,
        resolved_layout_mode,
    )
    return output.view(2 * output_m, k // 2), output_scale


__all__ = ["scaled_fp4_quant_mext_r1"]
