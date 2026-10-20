# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Python allocation wrappers for NVFP4 quantization kernels."""

from __future__ import annotations

import os

import torch

from .extension import ensure_loaded

ensure_loaded(required=False)


SUPPORTED_RATIOS = {
    0.0: 0,
    0.125: 1,
    0.25: 2,
    0.5: 4,
}


def _resolve_residue_per_8(num_salient: int, k: int) -> int:
    ratio = num_salient / k
    for supported_ratio, residue_per_8 in SUPPORTED_RATIOS.items():
        if abs(ratio - supported_ratio) < 0.001:
            return residue_per_8
    supported = ", ".join(f"{r:.3f}" for r in sorted(SUPPORTED_RATIOS))
    raise ValueError(
        f"Unsupported salient ratio {ratio:.4f} ({num_salient}/{k}). "
        f"Only the following ratios are supported: {supported}."
    )


def indices_to_channel_masks(
    salient_indices: torch.Tensor,
    num_channels: int,
    layer_name: str | None = None,
) -> torch.Tensor:
    assert num_channels % 8 == 0, "num_channels must be multiple of 8"

    num_masks = num_channels // 8
    channel_masks = torch.zeros(
        num_masks,
        dtype=torch.uint8,
        device=salient_indices.device,
    )

    mask_idx = salient_indices // 8
    bit_pos = salient_indices % 8
    bit_masks = (1 << bit_pos).to(torch.uint8)
    channel_masks.index_add_(0, mask_idx, bit_masks)

    if os.getenv("RESIDUE_DEBUG", "0") == "1":
        layer_info = f" [{layer_name}]" if layer_name else ""
        sample = salient_indices[:16].detach().cpu()
        print(f"[RESIDUE_DEBUG_MASK]{layer_info} salient_indices[:16]={sample.tolist()}")
        print(
            f"[RESIDUE_DEBUG_MASK]{layer_info} "
            f"num_channels={num_channels}, num_masks={num_masks}"
        )
        print(
            f"[RESIDUE_DEBUG_MASK]{layer_info} "
            f"masks[:16]={channel_masks[:16].detach().cpu().tolist()}"
        )

    return channel_masks


def _alloc_residue_quant_outputs(
    input_tensor: torch.Tensor,
    n_ext: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = input_tensor.shape[0]
    output = torch.empty(m * n_ext // 2, dtype=torch.uint8, device=input_tensor.device)
    num_m_tiles = (m + 127) // 128
    num_k_tiles = (n_ext + 63) // 64
    output_scale = torch.empty(
        num_m_tiles * num_k_tiles * 512,
        dtype=torch.uint8,
        device=input_tensor.device,
    )
    return output, output_scale


def _run_scaled_fp4_quant(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    output_scale: torch.Tensor,
    input_scale: torch.Tensor,
    channel_masks: torch.Tensor,
    n_ext: int,
    residue_per_8: int,
    use_swapped_residue: bool,
) -> None:


    ensure_loaded()
    try:
        torch.ops.nvfp4.scaled_fp4_quant(
            output,
            input_tensor,
            output_scale,
            input_scale,
            channel_masks,
            n_ext,
            residue_per_8,
            use_swapped_residue,
        )
    except RuntimeError as err:


        if "expected at most 7 argument" not in str(err):
            raise
        torch.ops.nvfp4.scaled_fp4_quant(
            output,
            input_tensor,
            output_scale,
            input_scale,
            channel_masks,
            n_ext,
            residue_per_8,
        )


def scaled_fp4_quant(
    input_tensor: torch.Tensor,
    input_scale: torch.Tensor,
    salient_indices: torch.Tensor,
    use_swapped_residue: bool = False,
):
    """Quantize activations with residue channels selected by indices."""
    m, k = input_tensor.shape
    channel_masks = indices_to_channel_masks(salient_indices, k)
    num_salient = len(salient_indices)
    n_ext = k + num_salient
    residue_per_8 = _resolve_residue_per_8(num_salient, k)
    output, output_scale = _alloc_residue_quant_outputs(input_tensor, n_ext)
    _run_scaled_fp4_quant(
        output,
        input_tensor,
        output_scale,
        input_scale,
        channel_masks,
        n_ext,
        residue_per_8,
        use_swapped_residue,
    )
    return output.view(m, n_ext // 2), output_scale


def scaled_fp4_quant_with_mask(
    input_tensor: torch.Tensor,
    input_scale: torch.Tensor,
    channel_masks: torch.Tensor,
    num_salient: int,
    use_swapped_residue: bool = False,
):
    """FP4 quantization with pre-computed channel masks."""
    m, k = input_tensor.shape
    n_ext = k + num_salient
    residue_per_8 = _resolve_residue_per_8(num_salient, k)
    output, output_scale = _alloc_residue_quant_outputs(input_tensor, n_ext)
    _run_scaled_fp4_quant(
        output,
        input_tensor,
        output_scale,
        input_scale,
        channel_masks,
        n_ext,
        residue_per_8,
        use_swapped_residue,
    )
    return output.view(m, n_ext // 2), output_scale


__all__ = [
    "indices_to_channel_masks",
    "scaled_fp4_quant",
    "scaled_fp4_quant_with_mask",
]
