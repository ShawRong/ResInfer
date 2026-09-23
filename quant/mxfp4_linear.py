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
) -> torch.Tensor:
    from nvfp4.mxfp4 import (
        fold_gemm,
        gemm,
        quantize,
        quantize_mext,
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
    if mode == "selective":
        activation, activation_scale = quantize_selective(
            source, indices, masks, recipe=activation_recipe
        )
        output = gemm(
            activation, weight, activation_scale, weight_scale,
            out_dtype=output_dtype,
        )
    elif mode == "mext_r1":
        def fixed_fold():
            activation, activation_scale = quantize_mext(
                source, recipe=activation_recipe
            )
            return fold_gemm(
                activation, weight, activation_scale, weight_scale,
                out_dtype=output_dtype,
            )

        # The independent residue_mxfp4 config installs the shared vLLM warmup
        # wrapper only when this opt-in route is enabled. Only an actual
        # FlashInfer tuning window may preflight/profile tactics; a normal
        # request without this process's preflight keeps the existing Fold.
        from quant.vllm_integration import mxfp4_fold_tuner as fold_tuner

        if not fold_tuner.tuner_enabled() or activation_recipe != "ceil":
            fold_tuner.record_static_fallback()
            output = fixed_fold()
        else:
            args = (source, weight, weight_scale)
            pdl_mode = "early"
            checked = fold_tuner.preflighted(
                args, pdl_mode=pdl_mode, out_dtype=output_dtype,
                strict_reference=True)
            fixed_reference = None
            if fold_tuner.is_tuning_mode() and not checked:
                # This is the *existing* MXFP4 path's output, not an FP32
                # ideal. Preflight never runs during ordinary serving/capture.
                fixed_reference = fixed_fold()
                try:
                    fold_tuner.preflight_tactics(
                        args, fixed_reference, pdl_mode=pdl_mode,
                        out_dtype=output_dtype, strict_reference=True)
                    checked = True
                except Exception as exc:  # keep the known fixed path available
                    print("[residue] MXFP4 Fold tuner preflight failed: "
                          f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
            if checked:
                output = fold_tuner.tuned_mxfp4_fold(
                    source, weight, weight_scale, pdl_mode=pdl_mode,
                    out_dtype=output_dtype)
            else:
                fold_tuner.record_static_fallback()
                output = fixed_reference if fixed_reference is not None else fixed_fold()
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
):
    del weight_scale, indices, masks, mode, bias, activation_recipe
    return torch.empty(
        (*x.shape[:-1], weight.shape[0]), device=x.device, dtype=output_dtype
    )
