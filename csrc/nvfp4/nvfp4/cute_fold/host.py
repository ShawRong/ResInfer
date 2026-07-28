# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Host entry for the SM103 MExt R1 fold GEMM (CuTeDSL).

Computes D[m_tok, n_w] = fold(A_act[2*m_tok, K] @ W[n_w, K]^T) where the
activation rows are row-pair interleaved [base0, res0, base1, res1, ...]
(nvfp4.scaled_fp4_quant_mext_r1 layout_mode=row_pair) and fold sums each
(base, residue) output pair. The kernel writes the transposed compact
D_t[n_w, m_tok]; callers get [m_tok, n_w] back via a transpose-copy.

Compile/call marshaling mirrors flashinfer's _cute_dsl_gemm_fp4_runner
(fake tensors + constexpr baking + TVM-FFI env stream).
"""

from __future__ import annotations

import functools

import torch

_KERNEL_CACHE: dict[tuple, tuple] = {}

# Valid sm103 tilers: (128|256, 128|256) (flashinfer sm103 candidates).
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
    from .kernel_sm103_fold import (
        Sm103BlockScaledPersistentDenseGemmKernel,
    )

    return cutlass, cute, make_ptr, get_max_active_clusters, {
        "sm100": Sm100BlockScaledPersistentDenseGemmKernel,
        "sm103": Sm103BlockScaledPersistentDenseGemmKernel,
    }


def _compile_fold_gemm(
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    out_dtype: torch.dtype,
    mode: str = "fold_tma",
    kernel_arch: str = "sm103",
    acc_stage_override: int | None = None,
    ab_stage_override: int | None = None,
    use_prefetch: bool = False,
):
    """Compile a fold-GEMM kernel variant:

    - "fold_tma":         N-ext fold, register fold + TMA store
    - "fold_tma_mext":    M-ext fold (A4), compact-smem alias + TMA store
    - "fold_mext_direct": M-ext fold (A6), pi alias + 128-bit direct stores
    - "plain_tma":        stock kernel/epilogue baseline (no fold)

    (The former "fold_direct" tiny-m mode was removed: dominated at every
    m_tok after the transposed-staging fold_tma optimization -- see
    fold_vs_plain_latency_results.md.)
    """
    assert mode in ("fold_tma", "plain_tma", "fold_tma_mext",
                    "fold_mext_direct"), mode
    assert kernel_arch in ("sm100", "sm103"), kernel_arch
    # sm100 kernel supports the TMA fold paths (N-ext + M-ext) and plain.
    assert kernel_arch == "sm103" or mode in (
        "fold_tma", "fold_tma_mext", "plain_tma", "fold_mext_direct")
    # M-ext (fold_on_m) is implemented on the sm100 kernel only for now.
    assert mode not in ("fold_tma_mext", "fold_mext_direct") or kernel_arch == "sm100", (
        "M-ext modes are sm100-only")
    # Exp2 depth knobs are sm100-only (the sm103 ctor has no such params).
    assert kernel_arch == "sm100" or (
        acc_stage_override is None and ab_stage_override is None
    ), "stage overrides are sm100-only"
    key = (mma_tiler_mn, cluster_shape_mn, out_dtype, mode, kernel_arch,
           acc_stage_override, ab_stage_override, use_prefetch)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    cutlass, cute, make_ptr, get_max_active_clusters, kernel_classes = (
        _cutlass_modules()
    )
    kernel_cls = kernel_classes[kernel_arch]
    c_cutlass_dtype = (
        cutlass.BFloat16 if out_dtype == torch.bfloat16 else cutlass.Float16
    )

    if kernel_arch == "sm100":
        # sm100 kernel always TMA-stores; no use_tma_store knob.
        gemm = kernel_cls(
            sf_vec_size=SF_VEC_SIZE,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_prefetch=use_prefetch,
            enable_pdl=True,
            fold_pairs_tma=(mode in ("fold_tma", "fold_tma_mext",
                                     "fold_mext_direct")),
            fold_on_m=(mode in ("fold_tma_mext", "fold_mext_direct")),
            mext_direct_store=(mode == "fold_mext_direct"),
            acc_stage_override=acc_stage_override,
            ab_stage_override=ab_stage_override,
        )
    else:
        gemm = kernel_cls(
            sf_vec_size=SF_VEC_SIZE,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_tma_store=True,
            fold_pairs_tma=(mode == "fold_tma"),
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
        if mode in ("fold_tma_mext", "fold_mext_direct")
        else (sym_m, sym_n)
    )
    a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (_a_rows, sym_k), stride_order=(1, 0), assumed_align=32
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (_b_rows, sym_k), stride_order=(1, 0), assumed_align=32
    )
    if mode in ("fold_tma", "fold_tma_mext", "fold_mext_direct"):
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
    a_sf_ptr = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16)
    b_sf_ptr = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if kernel_arch == "sm100":
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
            1,  # l (constexpr batch)
            a_sf_ptr,
            b_sf_ptr,
            alpha_fake,
            max_active_clusters,
            stream_fake,
            False,  # swap_ab
            lambda x: x,  # epilogue_op (alpha applied inside the kernel)
            options="--opt-level 2 --enable-tvm-ffi",
        )
    else:
        compiled = cute.compile(
            gemm.wrapper,
            a_fake,
            b_fake,
            c_fake,
            # Int64-typed dynamic args: any concrete int marshals at compile
            # time; the real per-call values are passed at runtime.
            1,
            1,
            1,
            1,  # l (constexpr batch)
            a_sf_ptr,
            b_sf_ptr,
            alpha_fake,
            max_active_clusters,
            stream_fake,
            False,  # swap_ab
            lambda x: x,  # epilogue_op (alpha applied inside the kernel)
            mode == "fold_tma",  # wrapper fold_pairs
            options="--opt-level 2 --enable-tvm-ffi",
        )
    _KERNEL_CACHE[key] = compiled
    return compiled


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def mext_fold_gemm_sm103(
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    mma_tiler_mn: tuple[int, int] = DEFAULT_MMA_TILER_MN,
    cluster_shape_mn: tuple[int, int] = DEFAULT_CLUSTER_SHAPE_MN,
    store_mode: str = "tma",
    kernel_arch: str = "sm103",
    acc_stage_override: int | None = None,
    ab_stage_override: int | None = None,
    use_prefetch: bool = False,
) -> torch.Tensor:
    """Run the fold GEMM. Returns D[m_tok, n_w] (contiguous).

    Args:
        weight_fp4: [n_w, k // 2] uint8 packed FP4, K-major, k_base prefix only.
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


    _mode = {
        "tma": "fold_tma",
        "tma_mext": "fold_tma_mext",
        "direct_mext": "fold_mext_direct",
    }[store_mode]
    compiled = _compile_fold_gemm(
        mma_tiler_mn, cluster_shape_mn, out_dtype,
        mode=_mode, kernel_arch=kernel_arch,
        acc_stage_override=acc_stage_override,
        ab_stage_override=ab_stage_override,
        use_prefetch=use_prefetch,
    )

    alpha_t = alpha.reshape(1).to(torch.float32)
    sf_m = _ceil_div(n_w, 128)   # weight scale-factor tiles
    sf_n = _ceil_div(m2, 128)    # activation scale-factor tiles
    sf_k = _ceil_div(k, SF_VEC_SIZE * 4)

    if store_mode in ("tma_mext", "direct_mext"):
        # direct_mext (A6) additionally requires the WEIGHT rows pre-permuted
        # by pi (see a6_weight_perm_indices); output D is standard order.
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
        d_t = torch.empty(m_tok, n_w, dtype=out_dtype, device=weight_fp4.device)
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

def a6_weight_perm_indices(n_w: int) -> "torch.Tensor":
    """Row-permutation for the A6 (direct_mext) weight: kernel position p must
    hold logical weight row f(p), per 32-row block. Derived from the .16x256b
    atom dst TV ((4,8),(2,2,4)):((2,32),(1,256,8)) at epi (128,32): position
    p = va + 2*ta + 8*vc  ->  f(p) = va + 8*ta + 2*vc  (va in 2, ta in 4, vc in 4).
    Usage: w_kernel = w_logical[a6_weight_perm_indices(n_w)] (permute the bf16
    weight BEFORE quantization so its SF stays consistent). Output D is
    standard order; no downstream compensation.
    """
    assert n_w % 32 == 0, n_w
    f = torch.empty(32, dtype=torch.long)
    for va in range(2):
        for ta in range(4):
            for vc in range(4):
                f[va + 2 * ta + 8 * vc] = va + 8 * ta + 2 * vc
    blocks = torch.arange(0, n_w, 32).unsqueeze(1)
    return (blocks + f.unsqueeze(0)).reshape(-1)
