# SPDX-License-Identifier: Apache-2.0
"""Opaque MXFP4 linear op preserving quant/GEMM dispatch under compile."""
from __future__ import annotations

import torch


@torch.library.custom_op("residue::mxfp4_linear", mutates_args=())
def mxfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    indices: torch.Tensor,
    masks: torch.Tensor,
    mode: str,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    activation_recipe: str = "ceil",
    tuning_buckets: list[int] | None = None,
) -> torch.Tensor:
    from nvfp4.mxfp4 import (
        gemm,
        quantize,
        quantize_selective,
    )

    if mode not in ("standard", "selective", "mext_r1"):
        raise ValueError(f"unknown MXFP4 linear mode {mode}")
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("linear input must have a positive feature dimension")
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (10, 0):
        raise ValueError("MXFP4 linear currently requires SM100")

    output_shape = (*x.shape[:-1], weight.shape[0])
    rows = x.numel() // x.shape[-1]
    if rows == 0:
        return torch.empty(output_shape, device=x.device, dtype=output_dtype)
    source = x.reshape(rows, x.shape[-1])
    if mode == "mext_r1":
        from quant.flashinfer_integration import mxfp4_fold_tuner as fold_tuner

        if activation_recipe != "ceil":
            raise ValueError("MXFP4 uses native ceil quantization")
        tune = fold_tuner.tuner_enabled()
        output = fold_tuner.tuned_mxfp4_fold(
            source, weight, weight_scale, out_dtype=output_dtype,
            pdl_mode="early" if tune else "off", tune=tune,
            tuning_buckets=tuning_buckets,
        )
    else:
        if mode == "selective":
            activation, activation_scale = quantize_selective(
                source, indices, masks, recipe=activation_recipe
            )
        else:
            activation, activation_scale = quantize(source, recipe=activation_recipe)
        output = gemm(
            activation, weight, activation_scale, weight_scale,
            out_dtype=output_dtype,
        )
    if bias is not None:
        output = (output + bias).to(output_dtype)
    return output.reshape(output_shape)


@mxfp4_linear.register_fake
def _fake_mxfp4_linear(
    x,
    weight,
    weight_scale,
    indices,
    masks,
    mode,
    output_dtype,
    bias=None,
    activation_recipe="ceil",
    tuning_buckets=None,
):
    del weight_scale, indices, masks, mode, bias, activation_recipe, tuning_buckets
    return torch.empty(
        (*x.shape[:-1], weight.shape[0]), device=x.device, dtype=output_dtype
    )
