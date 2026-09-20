# SPDX-License-Identifier: Apache-2.0
"""Opaque MXFP4 linear op: preserves quant/GEMM dispatch under torch.compile."""
from __future__ import annotations

import torch


@torch.library.custom_op("residue::mxfp4_linear", mutates_args=())
def mxfp4_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                 indices: torch.Tensor, masks: torch.Tensor, mode: str, output_dtype: torch.dtype,
                 bias: torch.Tensor | None = None, activation_recipe: str = "ceil") -> torch.Tensor:
    from nvfp4.mxfp4 import quantize, quantize_mext, quantize_selective, gemm, fold_gemm

    if mode not in ("standard", "selective", "mext_r1"):
        raise ValueError(f"unknown MXFP4 linear mode {mode}")
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("linear input must have a positive feature dimension")
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (10, 0):
        raise ValueError("MXFP4 linear currently requires SM100")
    shape = (*x.shape[:-1], weight.shape[0])
    rows = x.numel() // x.shape[-1]
    if rows == 0:
        return torch.empty(shape, device=x.device, dtype=output_dtype)
    flat = x.reshape(rows, x.shape[-1])
    if mode == "selective":
        a, sa = quantize_selective(flat, indices, masks=masks, recipe=activation_recipe)
        y = gemm(a, weight, sa, weight_scale, out_dtype=output_dtype)
    elif mode == "mext_r1":
        a, sa = quantize_mext(flat, recipe=activation_recipe)
        y = fold_gemm(a, weight, sa, weight_scale, out_dtype=output_dtype)
    else:
        a, sa = quantize(flat, recipe=activation_recipe)
        y = gemm(a, weight, sa, weight_scale, out_dtype=output_dtype)
    if bias is not None:
        # Match the fake op's declared output dtype even with an FP32 bias.
        y = (y + bias).to(output_dtype)
    return y.reshape(shape)


@mxfp4_linear.register_fake
def _fake_mxfp4_linear(x, weight, weight_scale, indices, masks, mode, output_dtype, bias=None, activation_recipe="ceil"):
    return torch.empty((*x.shape[:-1], weight.shape[0]), device=x.device, dtype=output_dtype)
