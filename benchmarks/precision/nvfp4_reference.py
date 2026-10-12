# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unmodified FP32 KV helpers extracted from vLLM tests (no vLLM dependency).

Source: tests/kernels/quantization/nvfp4_utils.py
Commit: c9e62f13c2b52619d8bc99f0c76cf04e77bff3c0
Only the E2M1 constant and these two functions are included.
"""
import torch

kE2M1ToFloat = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)

def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


def dequant_nvfp4_kv_cache(
    fp4_data: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: float,
    head_size: int,
    block_size: int,
    swizzled_scales: bool = True,
) -> torch.Tensor:
    """Dequantize an NVFP4 KV cache.

    The input must be in HND layout so that the last two dims are
    (block_size, last_dim).  For NHD caches, permute to HND first.

    Args:
        fp4_data: [..., num_heads, block_size, head_size//2] uint8 packed fp4.
        block_scale: [..., num_heads, block_size, head_size//16] fp8 block
            scales (as uint8 or float8_e4m3fn).
        global_scale: checkpoint dequant scale (k_scale or v_scale).
        head_size: head dimension.
        block_size: page size.
        swizzled_scales: whether ``block_scale`` is in the 4x4-swizzled
            layout of the SM100 trtllm-gen kernel. ``reshape_and_cache_flash``
            writes K scales linearly and V scales swizzled, so pass ``False``
            for K and ``True`` for V.

    Returns:
        [..., num_heads, block_size, head_size] float32.

    """
    data_dim = head_size // 2
    scale_dim = head_size // 16

    fp4_packed = fp4_data
    sf_bytes = block_scale.view(torch.uint8)
    batch_shape = sf_bytes.shape[:-2]
    T, S = block_size, scale_dim

    if swizzled_scales:
        # Unswizzle 4x4 block scales on (block_size, scale_dim) plane.
        # [..., T, S] → [..., T//4, 4, sg, 4] → permute → [..., T, S]
        sg = S // 4
        sf_reshape = sf_bytes.reshape(*batch_shape, T // 4, 4, sg, 4)
        ndim = sf_reshape.ndim
        # Swap the last four dims: (..., T//4, 4, sg, 4) → (..., T//4, 4, 4, sg)
        perm = list(range(ndim - 4)) + [ndim - 4, ndim - 1, ndim - 3, ndim - 2]
        sf_linear = sf_reshape.permute(*perm).reshape(*batch_shape, T, S)
    else:
        sf_linear = sf_bytes
    sf_f32 = sf_linear.view(torch.float8_e4m3fn).to(torch.float32)

    # Unpack fp4
    shape = fp4_packed.shape  # [..., T, data_dim]
    fp4_flat = fp4_packed.reshape(-1, data_dim)
    fp4_vals = break_fp4_bytes(fp4_flat, torch.float32)
    fp4_vals = fp4_vals.reshape(*shape[:-1], head_size)

    # Dequant: fp4_val * block_scale * global_scale per 16-element group
    return (
        fp4_vals.reshape(*shape[:-1], scale_dim, 16)
        * (sf_f32 * global_scale).unsqueeze(-1)
    ).reshape(*shape[:-1], head_size)
