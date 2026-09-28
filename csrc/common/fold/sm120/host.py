# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Torch entry point + compile cache for the SM120 CuTeDSL fold GEMM.

The compiled artifact takes raw device POINTERS (problem shape and tile
config are baked in at compile time), so a cached kernel can be replayed
inside a CUDA graph with different buffers -- the same contract the
FlashInfer CuTeDSL GEMM uses.
"""

from __future__ import annotations

import torch

from ..diagnostics import log_compilation

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import make_ptr

SF_VEC_SIZE = 16

# Keep warmed kernels resident for graph capture; queries use the same keys.
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
    """Build the CuTe views over caller-owned memory and run the kernel.

    m = n_w   (weight rows, GEMM-M)      A: [m, k] k-major, packed FP4
    n = act rows (extended, may be PADDED)  B: [n, k] k-major, packed FP4
    d_rows = REAL token count            D: [d_rows, m] m-major

    n and d_rows are RUNTIME Int32 values (sm100-style): one compiled kernel
    serves every token count for its (n_w, k, tile). Everything m-dependent
    downstream is dynamic-safe -- the grid is cute.ceil_div, the TMA
    descriptors are built at launch from the live extents, and the epilogue
    store already predicates on D's extent (the m_tok_real contract).

    d_rows is the real (unpadded) token count, NOT n // 2. The activation may
    be padded past the real tokens for the tile-N fast path, and D's extent is
    exactly what the epilogue predicate clamps against -- building mD from
    n // 2 would let the padding rows write past the caller's buffer.
    """
    # a_ld is A's row stride in ELEMENTS. It equals k for a contiguous weight,
    # and k_ext for an ext-K residue layer whose weight is a
    # `weight_ext[:, :k_base_packed]` view: the problem's K stays k_base while
    # the rows sit k_ext apart, so the fold reads the base-K prefix in place
    # with no copy. Same trick the deleted SM120 C++ path used --
    #   stride_B = make_int_tuple_from<StrideB>(physical_k_ext, 0)
    # with the problem shape at k_base. A's SF needs nothing special:
    # `weight_scale_base` is already the contiguous re-swizzled base-K prefix.
    _a_ld = k if a_ld is None else a_ld
    # b_k: B's contraction extent. Equal to k for the fold; 2k for the
    # wrapped-K-loop variant, whose activation is the K-concat [m_tok, 2K]
    # quant output while A (the weight) stays at k and wraps in the kernel.
    _b_k = k if b_k is None else b_k
    mA = cute.make_tensor(
        a_ptr, cute.make_layout((m, k, 1), stride=(_a_ld, 1, m * _a_ld)))
    mB = cute.make_tensor(
        b_ptr, cute.make_layout((n, _b_k, 1), stride=(_b_k, 1, n * _b_k)))
    # SF views: the kernel re-lays these out with tile_atom_to_shape_SF, so only
    # the iterator + element type matter here.
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
    'Compile (and cache) ONE fold kernel per (n_w, k, tile) -- the token'
    a_ld = None if a_ld == k else a_ld
    key = (torch.cuda.current_device(), m, k, tile_shape_mnk, out_dtype, a_ld, k_loop)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    if not compile_if_missing:
        return None
    if log_compilation():
        print(f"[fold-compile] n_w={m} k={k} (dynamic m) "
              f"tile={tile_shape_mnk} a_ld={a_ld} k_loop={k_loop}", flush=True)
    from .kernel import Sm120BlockScaledGemmKernel

    c_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[out_dtype]
    gemm = Sm120BlockScaledGemmKernel(
        cutlass.Float32, SF_VEC_SIZE, tile_shape_mnk,
        (tile_shape_mnk[0], tile_shape_mnk[1]),
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
    """Wrapped-K-loop GEMM: D[m_tok, n_w] = act[m_tok, 2K] @ [W | W]^T with the
    weight stored ONCE (the kernel wraps its k coordinate at the midpoint).

    weight_fp4:   [n_w, k/2] uint8 packed FP4, K-major (base-K view of an
                  ext-K weight works, row stride is read off the tensor).
    act_fp4_kext: [n_rows, k] uint8 -- the K-concat quant output
                  (`scaled_fp4_quant_mext_r1(layout_mode="concat_k")`): 2K
                  elements per row, base in element cols [0, K), residue in
                  [K, 2K). Rows may be padded past the real token count.
    act_sf:       SF bytes in the canonical (n_rows, 2K) atom order (the
                  concat_k quant emits exactly this).
    """
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
    """SM120 CuTeDSL fold GEMM. Returns D[m_tok, n_w] (contiguous).

    weight_fp4:      [n_w, k_base // 2] uint8 packed FP4, K-major. May be a
                     `weight_ext[:, :k_base_packed]` VIEW of an ext-K residue
                     weight -- the row stride is read off the tensor, so the
                     base-K prefix is consumed in place with no copy. Must stay
                     contiguous within a row.
    act_fp4_rowpair: [2 * m_tok, k // 2] uint8 packed FP4, row-pair layout
                     (adjacent rows = base/residue pair) -- exactly the
                     `scaled_fp4_quant_mext_r1(layout_mode=row_pair)` output.
                     May be PADDED past the real token count: rows beyond
                     2 * m_tok_real are ignored (the store is predicated on
                     the output extent), and padding to the tile-N boundary
                     is the fast path.
    weight_sf/act_sf: swizzled 128x4 FP8 scale factors (uint8 or e4m3 view).
    alpha:           float32 scalar tensor (global dequant scale product).
    m_tok_real:      real token count when act rows are padded; defaults to
                     act rows // 2.
    """
    from .tactics import select_tile

    n_w, k_packed = weight_fp4.shape
    k = k_packed * 2
    # ext-K: shape gives the problem K (k_base); stride says how far apart the
    # rows are (k_ext). Read it off the tensor rather than taking it as an arg
    # -- a narrowed view already carries the fact, and a caller-supplied value
    # could silently disagree with the memory being read.
    a_ld = weight_fp4.stride(0)
    if a_ld != k_packed:
        if weight_fp4.stride(1) != 1:
            raise ValueError(
                "weight must stay K-contiguous within a row; got stride "
                f"{weight_fp4.stride()}")
        if a_ld < k_packed:
            raise ValueError(
                f"weight row stride {a_ld} < width {k_packed}: rows overlap")
        # A's pointer is declared assumed_align=16 and the rows are FP4 nibbles
        # (2 elements per byte), so a row start must land on a 16-byte boundary:
        # a_ld (in packed BYTES) must be a multiple of 16. Every ext layer in
        # the exported checkpoints satisfies this, but a coarser salient
        # granularity would not -- refuse instead of reading misaligned.
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
