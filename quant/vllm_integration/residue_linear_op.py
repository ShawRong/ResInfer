"""Opaque NVFP4 Linear op: shared routing, quantization and GEMM.

Python chooses a route during eager execution or CUDA graph capture.
Replay executes the recorded kernels. Keeping the route inside this op
prevents torch.compile from fixing every decode graph to its large dummy M.
"""

from __future__ import annotations

import torch

import nvfp4
from .env_contract import settings
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    slice_nvfp4_output,
)

# Row-pair layout used by the two-GEMM prefill fallback.
_LAYOUT_ROW_PAIR = 1


def fold_max_m_for(arch: str | None) -> int:
    configured = settings().decode_max_m
    return configured if configured is not None else (128 if arch is not None else 64)


def mext_fold_max_m_for(arch: str | None) -> int:
    decode_max = fold_max_m_for(arch)
    if arch is None:
        return decode_max
    configured = settings().ratio1_max_m
    return max(decode_max, configured if configured is not None else 512)


def _sm10x_fold_arch() -> str | None:
    if settings().disable_sm10x:
        return None
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        return None
    from nvfp4.cute_fold.tactics import kernel_arch_for_capability
    return kernel_arch_for_capability(major, minor)


def _run_ratio1_prefill_two_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """ratio-1 prefill: one backend GEMM at 2M rows + pair-sum epilogue.

    out[i] = W·base_i + W·res_i, exactly the fold's math. The mext_r1 quant's
    row_pair SF is bytewise the cutlass 128x4 swizzle over the 2M rows (the
    kernel places SFs with the same cvt_quant_to_fp4_get_sf_out_offset the
    stock quant uses), so the doubled activation feeds cutlass_scaled_fp4_mm
    directly and base/residue rows come back interleaved at even/odd indices.

    Every op here has a static compile space: no CuTeDSL, nothing to JIT at
    serving time, and cudagraph capture sees ordinary torch/cutlass calls.
    """
    x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(
        x, input_global_scale_inv, layout_mode=_LAYOUT_ROW_PAIR,
    )
    two_m, k = x_fp4.shape[0], x.shape[-1]
    num_m_tiles = (two_m + 127) // 128
    num_k_tiles = (k + 63) // 64
    x_sf_v = x_sf.view(torch.float8_e4m3fn).view(
        num_m_tiles * 128, num_k_tiles * 4)
    y = cutlass_scaled_fp4_mm(
        x_fp4, weight, x_sf_v, weight_scale_base, alpha, output_dtype)
    return y[0::2] + y[1::2]


def _linear_policy(k_base, num_salient, padding, output_size,
                   fold_eligible, use_swap, ratio1):
    from ..flashinfer_integration.nvfp4_linear_adapter import NVFP4Policy

    cfg = settings()
    sm10x = _sm10x_fold_arch()
    arch = sm10x or ("sm120" if torch.cuda.get_device_capability()[0] == 12 else None)
    return NVFP4Policy(
        int(k_base), int(num_salient), int(padding), int(output_size),
        bool(fold_eligible), bool(use_swap), bool(ratio1), arch,
        fold_max_m_for(sm10x), mext_fold_max_m_for(sm10x), cfg.force_fold,
        cfg.ratio1_prefill,
        cfg.sm10x_tuner,
        cfg.sm120_tuner,
    )


def _nvfp4_linear_impl(inputs, policy):
    """Execute the static implementation of the shared numerical route."""
    from ..flashinfer_integration.nvfp4_linear_adapter import (
        _fold_inputs, _route,
    )

    x, weight, scale_inv, weight_scale, _base_sf, mask, alpha = inputs
    m, k = x.shape
    if m == 0:
        return x.new_empty((0, policy.output_size))
    route = _route(m, policy)
    if route == "invalid_mext":
        raise RuntimeError("Ratio-1 residue requires a prepared Fold path")
    if route in ("ratio1_prefill", "sm10x_fold", "sm120_fold"):
        args = _fold_inputs(inputs, policy)
        if route == "ratio1_prefill":
            if policy.prefill == "k_loop":
                from ..flashinfer_integration.ratio1_prefill_candidates import (
                    _kloop_valid_tactics, _run_kloop, prepare_kloop,
                )
                ready = _kloop_valid_tactics(args, False)
                if not ready and not torch.cuda.is_current_stream_capturing():
                    prepare_kloop(args)
                    ready = _kloop_valid_tactics(args, False)
                if ready:
                    return _run_kloop(args, ready[0])
            return _run_ratio1_prefill_two_gemm(*args, x.dtype)
        if route == "sm10x_fold":
            from ..flashinfer_integration.sm100_fold_candidates import _run_fallback
            return _run_fallback(args, x.dtype)[:m]
        from ..flashinfer_integration.sm120_fold_candidates import run_fold_table
        return run_fold_table(args, x.dtype)[:m]

    if route == "plain":
        x_fp4, x_sf = scaled_fp4_quant(
            x, scale_inv, is_sf_swizzled_layout=True, backend="cutlass")
    else:
        x_fp4, x_sf = nvfp4.scaled_fp4_quant_with_mask(
            x, scale_inv, mask, policy.num_salient,
            use_swapped_residue=policy.use_swap)
        x_sf = x_sf.view(torch.float8_e4m3fn).view(
            ((m + 127) // 128) * 128,
            ((k + policy.num_salient + 63) // 64) * 4)
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, policy.padding)
    out = cutlass_scaled_fp4_mm(
        x_fp4, weight, x_sf, weight_scale, alpha, x.dtype)
    return slice_nvfp4_output(out, policy.output_size)


@torch.library.custom_op(
    "residue::nvfp4_linear", mutates_args=(), device_types="cuda",
)
def nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_base: torch.Tensor,
    channel_mask: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    num_salient: int,
    weights_padding_cols: int,
    output_size: int,
    fold_eligible: bool,
    use_swap: bool,
    is_mext_r1: bool,
) -> torch.Tensor:
    """Dispatch all NVFP4 Linear sizes through one FlashInfer operation."""
    if x.shape[0] == 0:
        return x.new_empty((0, weight.shape[0]))
    from ..flashinfer_integration.nvfp4_linear_adapter import tuned_nvfp4_linear

    with torch.cuda.device(x.device):
        policy = _linear_policy(k_base, num_salient, weights_padding_cols,
                                output_size, fold_eligible, use_swap, is_mext_r1)
        return tuned_nvfp4_linear(
            [x, weight, input_global_scale_inv, weight_scale,
             weight_scale_base, channel_mask, alpha], policy,
        )


@nvfp4_linear.register_fake
def _nvfp4_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_base: torch.Tensor,
    channel_mask: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    num_salient: int,
    weights_padding_cols: int,
    output_size: int,
    fold_eligible: bool,
    use_swap: bool,
    is_mext_r1: bool,
) -> torch.Tensor:
    """Meta/abstract implementation for shape inference during torch.compile."""
    M = x.shape[0]
    N = int(output_size) if int(output_size) > 0 else weight.shape[0]
    return torch.empty(M, N, dtype=x.dtype, device=x.device)
