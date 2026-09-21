# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Dense TensorRT-LLM NVFP4-weight/FP8-activation paper arm.

The contract follows TensorRT-LLM's W4A8NVFP4FP8LinearMethod: block-32
weight scales, shuffled weight rows/scales, per-tensor FP8 activation
quantization, and fp4_fp8_gemm_trtllmgen. Imports remain lazy so dry-run and
CPU-only aggregation do not depend on TensorRT-LLM.
"""

from __future__ import annotations


def require_dense_ops():
    try:
        import tensorrt_llm._torch  # noqa: F401 - registers torch.ops
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
        import torch
    except ImportError as exc:
        raise RuntimeError("TensorRT-LLM dense W4A8 Runtime is unavailable") from exc
    for namespace, name in (("trtllm", "fp4_quantize"),
                            ("trtllm", "shuffle_matrix"),
                            ("trtllm", "fp4_fp8_gemm_trtllmgen"),
                            ("tensorrt_llm", "static_quantize_e4m3_per_tensor")):
        if not hasattr(getattr(torch.ops, namespace), name):
            raise RuntimeError(f"TensorRT-LLM dense W4A8 op missing: {namespace}.{name}")
    return fp4_utils


def prepare_weight(weight, fp4_utils):
    """Quantize a BF16 [N,K] weight once, as in checkpoint preparation."""
    import torch

    n, k = weight.shape
    if n % 128 or k % 128:
        raise ValueError(f"TensorRT-LLM W4A8 needs N,K multiples of 128: {n},{k}")
    global_scale = (448.0 / weight.float().abs().max()).reshape(1).float()
    packed, scales = torch.ops.trtllm.fp4_quantize(
        weight.contiguous(), global_scale, 32, False, False)
    if packed.shape != (n, k // 2):
        raise RuntimeError(f"unexpected TensorRT-LLM FP4 weight shape: {packed.shape}")
    scales = scales.view(n, k // 32).view(fp4_utils.float4_sf_dtype)
    packed = fp4_utils.shuffle_matrix_a(packed, 128)
    scales = fp4_utils.shuffle_matrix_sf_a(scales, 128, 32)
    return packed.contiguous(), scales.contiguous().view(torch.float8_e4m3fn), global_scale


def linear(x, weight, weight_scale, input_global_scale, alpha):
    """Time this whole callable: online FP8 quantization plus dense GEMM."""
    import torch

    x_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        x, input_global_scale)
    return torch.ops.trtllm.fp4_fp8_gemm_trtllmgen(
        x_fp8, weight, weight_scale, alpha, torch.bfloat16)
