# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Dense TensorRT-LLM W4A8 benchmark callables.

NVFP4/FP8 follows W4A8NVFP4FP8LinearMethod with block-32 weight scales.
MXFP4/MXFP8 shares the existing packed MXFP4 weights and uses TRT's native
quantizer/GEMM runner, with tactic selection delegated to FlashInfer.
Imports stay lazy so CPU aggregation and dry-run need no GPU packages.
"""

from __future__ import annotations


def require_dense_ops():
    import importlib.metadata
    import importlib.util

    import torch

    try:
        package = importlib.metadata.distribution("tensorrt-llm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("TensorRT-LLM dense W4A8 Runtime is unavailable") from exc
    # The full tensorrt_llm package import eagerly loads its serving stack.
    # The dense ops and this standalone layout utility only need Torch.
    op_library = package.locate_file("tensorrt_llm/libs/libth_common.so")
    utils_path = package.locate_file("tensorrt_llm/quantization/utils/fp4_utils.py")
    if not op_library.is_file() or not utils_path.is_file():
        raise RuntimeError("TensorRT-LLM dense W4A8 library or layout utility missing")
    if not hasattr(torch.ops.trtllm, "fp4_fp8_gemm_trtllmgen"):
        try:
            torch.ops.load_library(str(op_library))
        except OSError as exc:
            raise RuntimeError("TensorRT-LLM dense W4A8 binary is incompatible") from exc
    spec = importlib.util.spec_from_file_location("mext_trtllm_fp4_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load TensorRT-LLM FP4 layout utility")
    fp4_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fp4_utils)
    for namespace, name in (("trtllm", "shuffle_matrix"),
                            ("trtllm", "fp4_fp8_gemm_trtllmgen"),
                            ("tensorrt_llm", "static_quantize_e4m3_per_tensor")):
        if not hasattr(getattr(torch.ops, namespace), name):
            raise RuntimeError(f"TensorRT-LLM dense W4A8 op missing: {namespace}.{name}")
    return fp4_utils


def prepare_weight(weight, fp4_utils):
    """Encode a BF16/FP16 [N,K] weight in the ModelOpt W4A8 layout."""
    import torch

    n, k = weight.shape
    if n % 128 or k % 128:
        raise ValueError(f"TensorRT-LLM W4A8 needs N,K multiples of 128: {n},{k}")
    if weight.dtype not in (torch.bfloat16, torch.float16) or not weight.is_cuda:
        raise ValueError("TensorRT-LLM W4A8 weight must be CUDA BF16/FP16")
    # ModelOpt W4A8 uses E2M1 values and E4M3 scales per 32 elements. TRT's
    # online fp4_quantize op supports E4M3 scales only at block 16, so prepare
    # this checkpoint-format synthetic weight offline before any timing.
    global_scale = (448.0 / weight.float().abs().max()).reshape(1).float()
    blocks = weight.float().contiguous().reshape(n, k // 32, 32)
    scales = (blocks.abs().amax(dim=-1) * (global_scale / 6.0))
    scales = torch.where(scales == 0, 1.0, scales)
    scales = scales.clamp(min=2**-9, max=448.0).to(torch.float8_e4m3fn).contiguous()
    scaled = (blocks * global_scale / scales.float().clamp_min(1e-12).unsqueeze(-1))
    # E2M1 magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6. The midpoints give
    # nearest-value rounding. Sign occupies the high bit of each nibble.
    edges = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
                         device=weight.device, dtype=torch.float32)
    magnitude = scaled.abs()
    ordinal = torch.bucketize(magnitude, edges)
    # ModelOpt rounds exact odd-indexed midpoints to the even E2M1 code.
    odd_midpoint = ((magnitude == 0.75) | (magnitude == 1.75) |
                    (magnitude == 3.5)).to(torch.int64)
    nibble = (ordinal + odd_midpoint + (scaled < 0).to(torch.int64) * 8).to(torch.uint8)
    pair = nibble.reshape(n, k // 2, 2)
    packed = (pair[..., 0] | (pair[..., 1] << 4)).contiguous()
    scales = scales.view(fp4_utils.float4_sf_dtype)
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


def make_mxfp8_runner(device):
    """Use TRT's native MXFP4/MXFP8 runner with the shared FlashInfer tuner.

    Loading only the native library avoids importing TRT's serving/MPI stack.
    Quantization is inside the runner, so tuning and timing cover the same work.
    """
    import torch
    from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig

    fp4_utils = require_dense_ops()
    native = torch.classes.trtllm.FP4GemmRunner(
        torch.bfloat16, int(fp4_utils.FP4GemmType.W4A8_MXFP4_MXFP8))
    alpha = torch.ones(1, device=device, dtype=torch.float32)

    class Runner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return list(range(native.get_num_configs()))

        def forward(self, inputs, tactic=-1, **kwargs):
            x, weight, weight_scale = inputs
            x_fp8, x_scale = torch.ops.trtllm.mxfp8_quantize(x, True)
            return native.run_gemm(
                x_fp8, weight, x_scale, weight_scale, alpha,
                0, tactic, None, None)  # TRT BufferKind.DEFAULT

    runners, config = [Runner()], TuningConfig()

    def linear_mxfp8(x, weight, weight_scale):
        inputs = [x, weight, weight_scale]
        runner, tactic = AutoTuner.get().choose_one(
            "resinfer_bench::trtllm_mxfp4_mxfp8", runners, config, inputs)
        return runner.forward(inputs, tactic=tactic)

    return linear_mxfp8
