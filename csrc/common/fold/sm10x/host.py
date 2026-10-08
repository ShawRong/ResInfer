# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Host entry for the SM100 MExt R1 fold GEMM (CuTeDSL).

Computes D[m_tok, n_w] = fold(A_act[2*m_tok, K] @ W[n_w, K]^T) where the
activation rows are row-pair interleaved [base0, res0, base1, res1, ...]
(nvfp4.scaled_fp4_quant_mext_r1 layout_mode=row_pair) and fold sums each
(base, residue) output pair. The kernel writes row-major D[m_tok, n_w]
directly, using a transposed TMA view for the row-pair store.

Compile/call marshaling mirrors flashinfer's _cute_dsl_gemm_fp4_runner
(fake tensors + constexpr baking + TVM-FFI env stream).
"""

from __future__ import annotations

import functools

import torch

_KERNEL_CACHE: dict[tuple, tuple] = {}

# The tactic tables speak `store_mode`; _compile_fold_gemm speaks `mode`.
# Module-level so the warmup can enumerate the same space the call path
# reaches, instead of keeping a second copy that drifts.
STORE_MODE_TO_MODE = {
    "tma": "fold_tma",
    "tma_mext": "fold_tma_mext",
    "direct_mext": "fold_mext_direct",
}

# Default SM100 Fold tile.
DEFAULT_MMA_TILER_MN = (128, 128)
DEFAULT_CLUSTER_SHAPE_MN = (1, 1)
SF_VEC_SIZE = 16


@functools.lru_cache(maxsize=1)
def _cutlass_modules():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    from .kernel_sm100_fold import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    return cutlass, cute, make_ptr, get_max_active_clusters, {
        "sm100": Sm100BlockScaledPersistentDenseGemmKernel,
    }


def _compile_fold_gemm(
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    out_dtype: torch.dtype,
    mode: str = "fold_tma",
    kernel_arch: str = "sm100",
    ab_stage_override: int | None = None,
    dual_alpha: bool = False,
    pi_via_layout: bool = False,
    strided_weight: bool = False,
    fp4_format: str = "nvfp4",
    compile_if_missing: bool = True,
):
    """Compile a fold-GEMM kernel variant:

    - "fold_tma":         N-ext fold, register fold + TMA store
    - "fold_tma_mext":    M-ext fold (A4), compact-smem alias + TMA store
    - "fold_mext_direct": M-ext fold (A6), pi alias + 128-bit direct stores
    - "plain_mn":         SM100 plain GEMM, activation on M, row-major output

    compile_if_missing=False queries residency without compiling; returns None
    on a miss. The current device is part of the cache identity.

    (The former "fold_direct" tiny-m mode was removed: dominated at every
    m_tok after the transposed-staging fold_tma optimization -- see
    fold_vs_plain_latency_results.md.)
    """
    assert mode in ("fold_tma", "fold_tma_mext",
                    "fold_mext_direct", "kloop_tma", "plain_mn"), mode
    if kernel_arch != "sm100":
        raise ValueError("production Fold supports SM100; SM103 is in .dev/tools/fold/sm103_experimental")
    # kloop_tma: wrapped-K-loop -- act (K-concat, 2K) as A, weight as B with a
    # wrapped k coordinate, PLAIN epilogue writing C[m_tok, n_w]. sm100-only,
    # single-alpha.
    assert mode != "kloop_tma" or (
        kernel_arch == "sm100" and not dual_alpha
        and not strided_weight), "kloop_tma: sm100, single-alpha"
    # dual_alpha only makes sense for fold modes (plain has no (base,res) pairs).
    assert not dual_alpha or mode not in ("plain_mn",), "dual_alpha requires a fold mode"
    if fp4_format not in ("nvfp4", "mxfp4"):
        raise ValueError(f"unknown FP4 format: {fp4_format}")
    sf_vec_size = 32 if fp4_format == "mxfp4" else SF_VEC_SIZE
    key = (torch.cuda.current_device(), fp4_format, mma_tiler_mn, cluster_shape_mn,
           out_dtype, mode, kernel_arch, ab_stage_override, dual_alpha,
           pi_via_layout, strided_weight)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    if not compile_if_missing:
        return None

    cutlass, cute, make_ptr, get_max_active_clusters, kernel_classes = (
        _cutlass_modules()
    )
    kernel_cls = kernel_classes[kernel_arch]
    c_cutlass_dtype = (
        cutlass.BFloat16 if out_dtype == torch.bfloat16 else cutlass.Float16
    )

    # sm100 kernel always TMA-stores; no use_tma_store knob.
    gemm = kernel_cls(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        enable_pdl=True,
        fold_pairs_tma=(mode in ("fold_tma", "fold_tma_mext",
                                 "fold_mext_direct")),
        fold_on_m=(mode in ("fold_tma_mext", "fold_mext_direct")),
        mext_direct_store=(mode == "fold_mext_direct"),
        pi_via_layout=pi_via_layout,
        ab_stage_override=ab_stage_override,
        dual_alpha=dual_alpha,
        k_loop=(mode == "kloop_tma"),
    )

    # A6 (fold_mext_direct) needs PROVABLE 16B alignment of the direct-store
    # pointers: annotate the dynamic dims' divisibility (n_w%32 is already a
    # hard pi requirement; m_tok%8 comes with tile8). Runtime asserts below.
    _a6 = mode == "fold_mext_direct"
    sym_m = cute.sym_int(divisibility=32 if _a6 else 1)  # n_w (weight rows)
    sym_n = cute.sym_int(divisibility=16 if _a6 else 1)  # m2 = 2 * m_tok
    sym_k = cute.sym_int()
    sym_mtok = cute.sym_int(divisibility=8 if _a6 else 1)

    # Operand row dims. N-ext: mA=weight(sym_m), mB=act(sym_n). M-ext swaps them
    # (runtime passes act as mA, weight as mB) so the baked mC.shape[1]==(mB rows)
    # constraint ties the output's n_w dim to the weight operand, not the act.
    _a_rows, _b_rows = (
        (sym_n, sym_m)
        if mode in ("fold_tma_mext", "fold_mext_direct", "kloop_tma", "plain_mn")
        else (sym_m, sym_n)
    )
    # `strided_weight` supports an ext-K residue layer WITHOUT materialising a
    # base-K copy: the problem's K is k_base, but the weight rows are k_ext
    # apart because the tensor is a `weight_ext[:, :k_base_packed]` view. This
    # mirrored what the SM120 C++ path did --
    #   stride_B = make_int_tuple_from<StrideB>(physical_k_ext, 0)
    # with the problem shape still at k_base. That kernel
    # (nvfp4_scaled_mm_sm120_epilogue_kernels.cu) has since been deleted; the
    # technique is described here rather than cross-referenced so the comment
    # survives its removal. Recoverable from git history if needed.
    # The B-side scale needs no special handling: `weight_scale_base` is
    # already the re-swizzled base-K prefix, and sf_k derives from k_base.
    #
    # Which operand carries the weight depends on the orientation: N-ext
    # ("fold_tma") passes weight as mA, M-ext passes it as mB -- same swap as
    # _a_rows/_b_rows above. Only that operand gets a dynamic leading dim; the
    # activation is freshly quantised and always compact.
    #
    # A compact tensor keeps its own cache entry (strided_weight=False), so the
    # validated contiguous path compiles to exactly the same binary as before.
    _w_is_a = mode not in ("fold_tma_mext", "fold_mext_direct", "kloop_tma", "plain_mn")

    def _operand(rows, is_weight):
        if strided_weight and is_weight:
            return cute.runtime.make_fake_tensor(
                cutlass.Uint8, (rows, sym_k), (cute.sym_int(), 1),
                assumed_align=32,
            )
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (rows, sym_k), stride_order=(1, 0), assumed_align=32
        )

    a_fake = _operand(_a_rows, _w_is_a)
    b_fake = _operand(_b_rows, not _w_is_a)
    if mode == "kloop_tma":
        # A is the K-concat activation at 2K -- its own symbolic K, unrelated
        # to B's (the weight's) sym_k. The kernel takes trip count from A and
        # wraps B; runtime shapes enforce the 2x relation.
        a_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (_a_rows, cute.sym_int()), stride_order=(1, 0),
            assumed_align=32,
        )
    if mode in ("kloop_tma", "plain_mn"):
        # Plain epilogue, C = [act_rows(m_tok), n_w] row-major -- no fold,
        # no transpose: the operand swap already put tokens on GEMM-M.
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype, (sym_n, sym_m), stride_order=(1, 0),
            assumed_align=16,
        )
    elif mode in ("fold_tma", "fold_tma_mext", "fold_mext_direct"):
        # Physical output: row-major D[m_tok, n_w]. N-ext writes it via a
        # transposed TMA view; M-ext writes it naturally (no transpose) -- either
        # way the physical buffer is [m_tok, n_w].
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype, (sym_mtok, sym_m), stride_order=(1, 0),
            assumed_align=16,
        )
    else:
        # Stock C [n_w, m2] row-major.
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype, (sym_m, sym_n), stride_order=(1, 0),
            assumed_align=16,
        )
    sf_dtype = cutlass.Float8E8M0FNU if fp4_format == "mxfp4" else cutlass.Float8E4M3FN
    a_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
    b_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = None if fp4_format == "mxfp4" and kernel_arch == "sm100" else cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (2,) if dual_alpha else (1,), assumed_align=4
    )
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # sm100 wrapper keeps current_stream but has NO fold_pairs param
    # (the fork keys off self.fold_pairs_tma set in the constructor).
    compiled = cute.compile(
        gemm.wrapper,
        a_fake,
        b_fake,
        c_fake,
        1,
        1,
        1,
        1,  # l
        a_sf_ptr,
        b_sf_ptr,
        alpha_fake,
        max_active_clusters,
        stream_fake,
        False,  # swap_ab
        lambda x: x,  # epilogue_op (alpha applied inside the kernel)
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def mext_fold_gemm(
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    mma_tiler_mn: tuple[int, int] = DEFAULT_MMA_TILER_MN,
    cluster_shape_mn: tuple[int, int] = DEFAULT_CLUSTER_SHAPE_MN,
    store_mode: str = "tma",
    kernel_arch: str = "sm100",
    ab_stage_override: int | None = None,
    pi_via_layout: bool = False,
    alpha_res: torch.Tensor | None = None,
    fp4_format: str = "nvfp4",
) -> torch.Tensor:
    """Run the fold GEMM. Returns D[m_tok, n_w].

    alpha_res: optional residue-half global scale (dual-alpha). None keeps the
    legacy single-alpha path bit-for-bit. When set, the fold computes
    alpha*base + alpha_res*res in f32 (base rows = even/pair-first submode).

    Args:
        weight_fp4: [n_w, k_base // 2] uint8 packed FP4, K-major. May be a
            `weight_ext[:, :k_base_packed]` VIEW of an ext-K residue weight;
            the row stride is read from the tensor, so no base-K copy is
            needed. Must stay contiguous within a row.
        act_fp4_rowpair: [2 * m_tok, k // 2] uint8 packed FP4, row-pair layout.
        weight_sf: swizzled 128x4 FP8 scale for the k_base prefix
            (quant.vllm_integration layout_utils weight_scale_base).
        act_sf: swizzled 128x4 FP8 scale from scaled_fp4_quant_mext_r1.
        alpha: float32 scalar tensor (global dequant scale product).
    """
    n_w, k_packed = weight_fp4.shape
    m2 = act_fp4_rowpair.shape[0]
    assert m2 % 2 == 0, "activation must be row-pair interleaved (even rows)"
    m_tok = m2 // 2
    k = k_packed * 2

    # ext-K residue: `weight_fp4` may be a `weight_ext[:, :k_base_packed]` view,
    # so its row stride is k_ext_packed while its width is k_base_packed. The
    # shape already gives the problem K (= k_base); the stride is what tells the
    # kernel how far apart the rows sit. Detect it here rather than making the
    # caller pass k_ext -- a narrowed view carries the fact already, and a
    # caller-supplied value could disagree with the tensor silently.
    _w_ld = weight_fp4.stride(0)
    _strided_w = _w_ld != k_packed
    if _strided_w:
        assert weight_fp4.stride(1) == 1, (
            "weight must stay K-contiguous within a row; got stride "
            f"{weight_fp4.stride()}")
        assert _w_ld > k_packed, (
            f"weight row stride {_w_ld} < width {k_packed}: rows would overlap")
        # TMA needs the row starts aligned, and the operands are declared
        # assumed_align=32. A contiguous weight satisfies this via k_packed;
        # a strided one must satisfy it via the EXT width. Every ext layer in
        # the exported checkpoints does (k_ext in {2880,3200,3840,12160,14592}
        # -> packed strides all %32 == 0), but a coarser salient granularity
        # could break it, so refuse loudly instead of corrupting the read.
        assert _w_ld % 32 == 0, (
            f"strided weight row stride {_w_ld} is not 32B-aligned; "
            "materialise a contiguous base-K prefix for this layer")
    if act_fp4_rowpair.stride(0) != k_packed or act_fp4_rowpair.stride(1) != 1:
        # The activation is produced by scaled_fp4_quant_mext_r1 and is always
        # compact; only the weight operand has a dynamic leading dim compiled in.
        raise ValueError(
            f"activation must be compact [{m2}, {k_packed}]; got stride "
            f"{act_fp4_rowpair.stride()}")


    _mode = STORE_MODE_TO_MODE[store_mode]
    # Dual-alpha hot path: pass alpha as a PRE-COMBINED (2,) f32 tensor
    # ([alpha_base, alpha_res]) -- zero per-call allocations. The alpha_res
    # convenience arg concatenates per call (extra kernel; init-time use only).
    no_alpha = fp4_format == "mxfp4" and kernel_arch == "sm100"
    if no_alpha and alpha_res is not None:
        raise ValueError("MXFP4 has no global residue scale")
    _dual = not no_alpha and (alpha_res is not None or alpha.numel() == 2)

    compiled = _compile_fold_gemm(
        mma_tiler_mn, cluster_shape_mn, out_dtype,
        mode=_mode, kernel_arch=kernel_arch,
        ab_stage_override=ab_stage_override,
        pi_via_layout=pi_via_layout,
        dual_alpha=_dual,
        strided_weight=_strided_w,
        fp4_format=fp4_format,
    )

    if no_alpha:
        alpha_t = None
    elif alpha_res is not None:
        alpha_t = torch.cat(
            [alpha.reshape(1), alpha_res.reshape(1)]).to(torch.float32)
    elif _dual:
        alpha_t = alpha.reshape(2).to(torch.float32)
    else:
        alpha_t = alpha.reshape(1).to(torch.float32)
    sf_m = _ceil_div(n_w, 128)   # weight scale-factor tiles
    sf_n = _ceil_div(m2, 128)    # activation scale-factor tiles
    sf_k = _ceil_div(k, (32 if fp4_format == "mxfp4" else SF_VEC_SIZE) * 4)

    if store_mode in ("tma_mext", "direct_mext"):
        # direct_mext (A6) additionally requires the WEIGHT rows pre-permuted
        # by pi unless pi_via_layout carries the permutation; output is standard order.
        # A6 dereferences the m2-congruent alias: the peer half of a 2SM mma
        # tile must exist (2*m_tok >= mma M) or its stores land out of bounds.
        assert store_mode != "direct_mext" or m2 >= mma_tiler_mn[0], (
            f"direct_mext needs 2*m_tok >= mma_tiler M "
            f"({m2} < {mma_tiler_mn[0]}); use a 1SM (128,x) tile at small m")
        assert store_mode != "direct_mext" or (
            n_w % 32 == 0 and m_tok % 8 == 0
        ), (n_w, m_tok)
        # M-ext non-swap: activation on M, weight on N; output D[m_tok, n_w]
        # written naturally (no transpose). Swap operands (mA=act, mB=weight)
        # and SF (sf_m arg = act tiles, sf_n arg = weight tiles).
        d_t = torch.empty(m_tok, n_w, dtype=out_dtype,
                          device=weight_fp4.device)
        compiled(
            act_fp4_rowpair, weight_fp4, d_t,
            sf_n, sf_m, sf_k,
            act_sf.data_ptr(), weight_sf.data_ptr(),
            alpha_t,
        )
        return d_t

    if store_mode == "tma":
        # N-ext: kernel writes row-major D[m_tok, n_w] via a transposed TMA view.
        d_t = torch.empty(m_tok, n_w, dtype=out_dtype, device=weight_fp4.device)
        compiled(
            weight_fp4, act_fp4_rowpair, d_t,
            sf_m, sf_n, sf_k,
            weight_sf.data_ptr(), act_sf.data_ptr(),
            alpha_t,
        )
        return d_t

    raise ValueError(f"unknown store_mode {store_mode!r}")

def kext_kloop_gemm_sm100(
    weight_fp4: "torch.Tensor",
    act_fp4_kext: "torch.Tensor",
    weight_sf: "torch.Tensor",
    act_sf: "torch.Tensor",
    alpha: "torch.Tensor",
    out_dtype=None,
    mma_tiler_mn: tuple[int, int] = DEFAULT_MMA_TILER_MN,
    cluster_shape_mn: tuple[int, int] = DEFAULT_CLUSTER_SHAPE_MN,
) -> "torch.Tensor":
    """Wrapped-K-loop GEMM on sm100: D[m_tok, n_w] = act[m_tok, 2K] @ [W|W]^T
    with the weight stored ONCE (its k coordinate wraps in the kernel).

    weight_fp4:   [n_w, k/2] uint8 packed FP4, K-major, contiguous.
    act_fp4_kext: [m_tok, k] uint8 -- the K-concat quant output
                  (`scaled_fp4_quant_mext_r1(layout_mode="concat_k")`): 2K
                  elements per row, base in cols [0, K), residue in [K, 2K).
    act_sf:       SF bytes in canonical (m_tok, 2K) atom order.
    weight_sf:    swizzled SF for (n_w, K).

    Compiles ONCE per (tile, dtype): every dim is symbolic, so serving m is
    a free variable -- no bucket grid needed on this arch.
    """
    import torch

    out_dtype = out_dtype or torch.bfloat16
    n_w, k_packed = weight_fp4.shape
    k = k_packed * 2
    assert weight_fp4.stride(0) == k_packed and weight_fp4.stride(1) == 1, (
        "kloop_tma expects a contiguous weight (mext_r1 weights are)")
    m_tok = act_fp4_kext.shape[0]
    assert act_fp4_kext.shape[1] == k, (
        f"activation must be [rows, {k}] bytes (2K elements), got "
        f"{tuple(act_fp4_kext.shape)}")

    compiled = _compile_fold_gemm(
        tuple(mma_tiler_mn), tuple(cluster_shape_mn), out_dtype,
        mode="kloop_tma", kernel_arch="sm100")
    d_t = torch.empty(m_tok, n_w, dtype=out_dtype, device=weight_fp4.device)
    alpha_t = alpha.reshape(1).to(torch.float32).contiguous()
    # wrapper (sf_m, sf_n, sf_k) = A-side rows/128, B-side rows/128, BASE-K
    # groups (the wrapper doubles the A side under k_loop).
    compiled(
        act_fp4_kext, weight_fp4, d_t,
        _ceil_div(m_tok, 128), _ceil_div(n_w, 128),
        _ceil_div(k // SF_VEC_SIZE, 4),
        act_sf.data_ptr(), weight_sf.data_ptr(),
        alpha_t,
    )
    return d_t
