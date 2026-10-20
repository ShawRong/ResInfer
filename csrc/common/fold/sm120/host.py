# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""SM120 Fold kernel compilation and launch."""

from __future__ import annotations

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import make_ptr

SF_VEC_SIZE = 16

_KERNEL_CACHE: dict = {}


def _hw_max_active_clusters() -> int:
    return cutlass.utils.HardwareInfo().get_max_active_clusters(1)


@cute.jit
def _fold_launch(
    gemm: cutlass.Constexpr,
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    d_ptr: cute.Pointer,
    alpha_ptr: cute.Pointer,
    m: cutlass.Constexpr,
    n: cutlass.Int32,
    k: cutlass.Constexpr,
    d_rows: cutlass.Int32,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
    a_ld: cutlass.Constexpr = None,
    b_k: cutlass.Constexpr = None,
):
    """Construct operand views and launch GEMM."""

    _a_ld = k if a_ld is None else a_ld

    _b_k = k if b_k is None else b_k
    mA = cute.make_tensor(
        a_ptr, cute.make_layout((m, k, 1), stride=(_a_ld, 1, m * _a_ld)))
    mB = cute.make_tensor(
        b_ptr, cute.make_layout((n, _b_k, 1), stride=(_b_k, 1, n * _b_k)))

    mSFA = cute.make_tensor(sfa_ptr, cute.make_layout((m, k // SF_VEC_SIZE, 1)))
    mSFB = cute.make_tensor(sfb_ptr, cute.make_layout((n, _b_k // SF_VEC_SIZE, 1)))
    mD = cute.make_tensor(
        d_ptr, cute.make_layout((d_rows, m, 1), stride=(m, 1, d_rows * m))
    )
    mAlpha = cute.make_tensor(alpha_ptr, cute.make_layout((1,)))
    gemm(mA, mB, mSFA, mSFB, mD, mAlpha, max_active_clusters, stream)


def compile_fold_gemm_sm120(
    m: int,
    k: int,
    tile_shape_mnk: tuple[int, int, int],
    out_dtype: torch.dtype,
    a_ld: int | None = None,
    k_loop: bool = False,
    compile_if_missing: bool = True,
):
    """Compile or retrieve an SM120 Fold kernel."""
    a_ld = None if a_ld == k else a_ld
    key = (torch.cuda.current_device(), m, k, tile_shape_mnk, out_dtype, a_ld, k_loop)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    if not compile_if_missing:
        return None
    from .kernel import Sm120BlockScaledGemmKernel

    c_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[out_dtype]
    gemm = Sm120BlockScaledGemmKernel(
        cutlass.Float32, SF_VEC_SIZE, tile_shape_mnk,
        k_loop=k_loop,
    )
    a_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfb_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    d_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    alpha_ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4)
    compiled = cute.compile(
        _fold_launch, gemm, a_ptr, b_ptr, sfa_ptr, sfb_ptr, d_ptr, alpha_ptr,
        m, cutlass.Int32(256), k, cutlass.Int32(256),
        _hw_max_active_clusters(),
        cutlass_torch.default_stream(),
        a_ld,
        (2 * k if k_loop else None),
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


def kext_kloop_gemm_sm120(
    weight_fp4: torch.Tensor,
    act_fp4_kext: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    tile_shape_mnk: tuple[int, int, int] = (128, 128, 128),
    m_tok_real: int | None = None,
) -> torch.Tensor:
    """Compute base and residue GEMM using K-loop."""
    n_w, k_packed = weight_fp4.shape
    k = k_packed * 2
    a_ld = weight_fp4.stride(0)
    if a_ld != k_packed:
        if weight_fp4.stride(1) != 1 or a_ld < k_packed or a_ld % 16:
            raise ValueError(
                f"unsupported weight view for k_loop: stride={weight_fp4.stride()}")
    n_rows = act_fp4_kext.shape[0]
    if act_fp4_kext.shape[1] != k:
        raise ValueError(
            f"activation must be [rows, {k}] bytes (2K elements), got "
            f"{tuple(act_fp4_kext.shape)}")
    m_tok = n_rows if m_tok_real is None else int(m_tok_real)
    compiled = compile_fold_gemm_sm120(
        n_w, k, tile_shape_mnk, out_dtype,
        a_ld=(None if a_ld == k_packed else a_ld * 2), k_loop=True)
    d_t = torch.empty(m_tok, n_w, dtype=out_dtype, device=weight_fp4.device)

    def _p(t, dtype, align=16):
        return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

    compiled(
        _p(weight_fp4, cutlass.Float4E2M1FN),
        _p(act_fp4_kext, cutlass.Float4E2M1FN),
        _p(weight_sf, cutlass.Float8E4M3FN),
        _p(act_sf, cutlass.Float8E4M3FN),
        _p(d_t, {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16,
                 torch.float32: cutlass.Float32}[out_dtype]),
        _p(alpha.reshape(1).to(torch.float32), cutlass.Float32, align=4),
        n_rows,
        m_tok,
        cutlass_torch.current_stream(),
    )
    return d_t


def mext_fold_gemm_sm120(
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    tile_shape_mnk: tuple[int, int, int] | None = None,
    m_tok_real: int | None = None,
) -> torch.Tensor:
    """Compute GEMM with full-channel residue compensation."""
    from .tactics import select_tile

    n_w, k_packed = weight_fp4.shape
    k = k_packed * 2

    a_ld = weight_fp4.stride(0)
    if a_ld != k_packed:
        if weight_fp4.stride(1) != 1:
            raise ValueError(
                "weight must stay K-contiguous within a row; got stride "
                f"{weight_fp4.stride()}")
        if a_ld < k_packed:
            raise ValueError(
                f"weight row stride {a_ld} < width {k_packed}: rows overlap")

        if a_ld % 16:
            raise ValueError(
                f"strided weight row stride {a_ld}B is not 16B-aligned; "
                "materialise a contiguous base-K prefix for this layer")
    n_phys = act_fp4_rowpair.shape[0]
    if n_phys % 2:
        raise ValueError(f"activation rows must be even (row-pair), got {n_phys}")
    m_tok = n_phys // 2 if m_tok_real is None else int(m_tok_real)
    if tile_shape_mnk is None:
        tile_shape_mnk = select_tile(m_tok, n_w, k)

    if 2 * m_tok > n_phys:
        raise ValueError(
            f"m_tok_real={m_tok} needs {2 * m_tok} activation rows, got {n_phys}")
    compiled = compile_fold_gemm_sm120(
        n_w, k, tile_shape_mnk, out_dtype,
        a_ld=(None if a_ld == k_packed else a_ld * 2))
    d_t = torch.empty(m_tok, n_w, dtype=out_dtype, device=weight_fp4.device)

    def _p(t, dtype, align=16):
        return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

    compiled(
        _p(weight_fp4, cutlass.Float4E2M1FN),
        _p(act_fp4_rowpair, cutlass.Float4E2M1FN),
        _p(weight_sf, cutlass.Float8E4M3FN),
        _p(act_sf, cutlass.Float8E4M3FN),
        _p(d_t, {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16,
                 torch.float32: cutlass.Float32}[out_dtype]),
        _p(alpha.reshape(1).to(torch.float32), cutlass.Float32, align=4),
        n_phys,
        m_tok,
        cutlass_torch.current_stream(),
    )
    return d_t
