"""Python allocation wrappers for NVFP4 quantization kernels."""

from __future__ import annotations

import os

import torch

from .extension import ensure_loaded

# 软加载:缺扩展只警告,真正调用 op 时才致命(见 extension.ensure_loaded)
ensure_loaded(required=False)

# ratio 1.0 (residue_per_8=8) K-extension was removed: it doubled the weight
# (K_ext = 2K). Full-rank residue is served by the M-extension path instead
# (scaled_fp4_quant_mext_r1 + fold GEMM), which keeps the weight at K.
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
        # Older locally-built extensions did not expose the use_swapped_residue
        # argument. Keep this compatibility path so benchmark scripts can still
        # inspect old wheels, but runtime builds should use the full signature.
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
    """Convenience helper that accepts salient indices.

    This converts salient_indices to channel_masks on every call, which launches
    extra CUDA work. Runtime and benchmarks should use
    scaled_fp4_quant_with_mask() with precomputed masks instead.
    """
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


def _scaled_fp4_quant_pack8_no_swap_with_mask(
    input_tensor: torch.Tensor,
    input_scale: torch.Tensor,
    channel_masks: torch.Tensor,
    num_salient: int,
    expected_ratio: float,
    residue_per_8: int,
    op_name: str,
):
    m, k = input_tensor.shape
    n_ext = k + num_salient
    ratio = num_salient / k
    if abs(ratio - expected_ratio) >= 0.001:
        raise ValueError(
            f"pack8 r{expected_ratio:g} requires salient ratio {expected_ratio:g}, "
            f"got {ratio:.4f} ({num_salient}/{k})."
        )

    output, output_scale = _alloc_residue_quant_outputs(input_tensor, n_ext)
    getattr(torch.ops.nvfp4, op_name)(
        output,
        input_tensor,
        output_scale,
        input_scale,
        channel_masks,
        n_ext,
        residue_per_8,
    )
    return output.view(m, n_ext // 2), output_scale


def scaled_fp4_quant_pack8_r025_no_swap_with_mask(
    input_tensor,
    input_scale,
    channel_masks,
    num_salient,
):
    """Experimental pack8 no-swap residue quantization specialized for ratio=0.25."""
    return _scaled_fp4_quant_pack8_no_swap_with_mask(
        input_tensor,
        input_scale,
        channel_masks,
        num_salient,
        0.25,
        2,
        "scaled_fp4_quant_pack8_r025_no_swap",
    )


def scaled_fp4_quant_pack8_r0125_no_swap_with_mask(
    input_tensor,
    input_scale,
    channel_masks,
    num_salient,
):
    """Experimental pack8 no-swap residue quantization specialized for ratio=0.125."""
    return _scaled_fp4_quant_pack8_no_swap_with_mask(
        input_tensor,
        input_scale,
        channel_masks,
        num_salient,
        0.125,
        1,
        "scaled_fp4_quant_pack8_r0125_no_swap",
    )


def scaled_fp4_quant_pack8_r050_no_swap_with_mask(
    input_tensor,
    input_scale,
    channel_masks,
    num_salient,
):
    """Experimental pack8 no-swap residue quantization specialized for ratio=0.5."""
    return _scaled_fp4_quant_pack8_no_swap_with_mask(
        input_tensor,
        input_scale,
        channel_masks,
        num_salient,
        0.5,
        4,
        "scaled_fp4_quant_pack8_r050_no_swap",
    )


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


def official_scaled_fp4_quant(input_tensor: torch.Tensor, input_scale: torch.Tensor):
    """Standalone vLLM-style swizzled-scale NVFP4 quantization helper."""
    m, k = input_tensor.shape
    if k % 16 != 0:
        raise ValueError(f"official scaled_fp4_quant requires K multiple of 16, got {k}")

    output = torch.empty(m * k // 2, dtype=torch.uint8, device=input_tensor.device)
    num_m_tiles = (m + 127) // 128
    num_k_tiles = (k + 63) // 64
    output_scale = torch.empty(
        num_m_tiles * num_k_tiles * 512,
        dtype=torch.uint8,
        device=input_tensor.device,
    )
    torch.ops.nvfp4.official_scaled_fp4_quant_pack8(
        output,
        input_tensor,
        output_scale,
        input_scale,
    )
    return output.view(m, k // 2), output_scale


__all__ = [
    "indices_to_channel_masks",
    "official_scaled_fp4_quant",
    "scaled_fp4_quant",
    "scaled_fp4_quant_pack8_r0125_no_swap_with_mask",
    "scaled_fp4_quant_pack8_r025_no_swap_with_mask",
    "scaled_fp4_quant_pack8_r050_no_swap_with_mask",
    "scaled_fp4_quant_with_mask",
]
