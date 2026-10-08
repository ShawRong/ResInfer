# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors (modifications)
# SPDX-License-Identifier: BSD-3-Clause
"ResAttn: paged NVFP4 GQA decode attention (Blackwell tcgen05, CuTeDSL).\n\nDerived from FlashInfer's ``flashinfer/cute_dsl/attention/gqa_decode.py``\n(NVIDIA, BSD-3-Clause). The production mode is ``qmul4_tmem``: FP4 QK with Q\nresidue, conventional token-major V, QMUL4 conversion, and BF16 or packed\nNVFP4 output. Paged addressing, padded/residue head mapping and fused output\nreducers are local extensions."

import math
from typing import Literal, Type, Tuple, cast
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    NamedBarrier as nbar,
)

import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.typing import (
    Float32,
    BFloat16,
    Int32,
    Int64,
    Optional,
)

from .output import quantize_nvfp4_block16, output_scale_offset
from .qmul4 import (
    dequantize_nvfp4x8_ldsm,
)

# Kernel invariants
mma_modes = (0, 1, 2)

def _sf_s2t_partition(sf_dtype, sSF, tSF):
    """SF smem→TMEM(UTCCP)分区,移植自 cutlass 示例 mainloop_s2t_copy_and_partition。"""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype
    )
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_s2t_)
    tCtSF_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_s2t, tCtSF_s2t
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
cpy_dice = (None,) + mma_dice  # (CPY, #CPY_MMA, #CPY_M, #CPY_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128
max_reduction_iters = 4  # log2(16)
max_reduction_splits = 1 << max_reduction_iters

# Math helpers
log2_e = math.log2(math.e)  # change exponential base
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")

class GroupedQueryAttentionDecodePagedFp4:
    """分页 nvfp4 变体(MMA-native KV 池,page_size=128)。

    Single-token residue decode: physical Q8/Q16, D128, linear K scales.
    Derived from dense ``GroupedQueryAttentionDecode``; paged addressing and
    compact logical-head output are local extensions.

    页几何(page_size=128 定理):
      - page_size = SF atom 行粒度 = KQ mma_tile_m ⇒ K 的 sm 子循环即页循环;
      - 页局部 SF = dense atom 序按页切块 ⇒ SF TMA 结构不变,只换 l 坐标;
      - V 以 FP4 载入 SMEM，QMUL4 转换后写入 TMEM 供 VP 使用。
    """

    def __init__(
        self,
        headdim,
        grouped_head_tile,
        prediction_tile=1,
        sequence_tile=256,
        reduction_mode: Literal[
            "kernel", "gmem_fused", "cga_sharded", "none"
        ] = "kernel",
        *,
        page_size=128,
        logical_grouped_heads: int | None = None,
        single_softmax_wg: bool = False,
        smem_correction_broadcast: bool = False,
    ):
        """
        Parameters
        ----------
        headdim
            Head dimension.
        grouped_head_tile
            Grouped heads per threadblock (GQA packing factor).
        prediction_tile
            Predicted tokens per threadblock.
        sequence_tile
            KV tokens per threadblock per loop iteration.
        reduction_mode
            Split-K reduction algorithm:
              - ``"kernel"``: deterministic two-kernel reduction with partial workspace.
              - ``"gmem_fused"``: TRT-style last-arriving-CTA global-memory
                reduction, currently restricted to direct NVFP4 output.
              - ``"cga_sharded"``: remote-SMEM partial-O reduction;
                output slices are reduced by different cluster CTAs and stored once.
              - ``"none"``: no split-K, flash decoding disabled.
        """
        if (grouped_head_tile not in (8, 16)
                or prediction_tile != 1 or headdim != 128):
            raise NotImplementedError("production ResAttn requires residue Q8/Q16, D128, decode")
        if reduction_mode not in ("none", "kernel", "gmem_fused", "cga_sharded"):
            raise ValueError(f"unsupported production reducer: {reduction_mode}")
        self.headdim = headdim
        self.grouped_head_tile = grouped_head_tile
        self.prediction_tile = prediction_tile
        self.sequence_tile = sequence_tile
        self.do_kernel_red = reduction_mode == "kernel"
        # TRT-LLM Gen mode-1 style split-K: every CTA publishes its partial
        # numerator/statistics to global memory, then the dynamically last CTA
        # performs the final merge in this same kernel. Keep this separate
        # from ``kernel`` until silicon A/B proves that removing the second
        # launch pays for the extra in-kernel synchronization.
        self.do_gmem_fused_red = reduction_mode == "gmem_fused"
        self.do_partial_red = self.do_kernel_red or self.do_gmem_fused_red
        self.do_cga_red = reduction_mode == "cga_sharded"
        self.do_none_red = reduction_mode == "none"
        # Production residue-Q packs base and residual rows per KV head.
        # Fold their QK scores before softmax; the VP operand remains FP8.
        # Store only the model's logical heads in the compact output.
        self.logical_grouped_heads = logical_grouped_heads
        # The selected scheduler uses one or two softmax warpgroups.
        self.single_softmax_wg = single_softmax_wg
        # TRT's exact Q16 decode cubin broadcasts correction factors with far
        # fewer indexed shuffles than the current one-shuffle-per-head path.
        # The single-softmax scheduler reuses correction WG's private sL column as a
        # 16-float warp-local carrier, then issues four 128-bit shared loads.
        # Arithmetic, TMEM ownership, and the final sL reduction stay fixed.
        self.smem_correction_broadcast = smem_correction_broadcast
        # TRT-LLM's generation Q8 tactic keeps the skinny head dimension on
        # MMA-N but changes the FP8 P operand to K-major. SM100 permits N=8
        # for a K-major B operand; the MN-major form requires N>=16. Limit
        # this specialization to the compact GQA4 residue geometry first.
        self.swap_ab_for_gen = grouped_head_tile == 8
        # The direct-to-TMEM path needs a conversion warpgroup in addition to
        # the historical MMA/TMA, 2x softmax, and correction warpgroups.
        self.threads_per_cta = (4 if single_softmax_wg else 5) * warpgroup_threads
        self.page_size = page_size

        assert page_size == 128, (
            "MMA-native profile 冻结 page=128(SF atom 对齐定理);"
            "64 需 cp.async SF 拼装路径,未实现"
        )
        assert sequence_tile > 0 and sequence_tile % 128 == 0
        if logical_grouped_heads is not None:
            assert 0 < logical_grouped_heads <= grouped_head_tile
        if smem_correction_broadcast:
            assert single_softmax_wg

    ##############################
    # Launch helpers
    ##############################
    # Pack grouped heads with predicted tokens (s_q)
    @staticmethod
    def gqa_pack(t_bshd: cute.Tensor, h_k: int):
        d, h_q, s_q, b = tuple(reversed(t_bshd.shape))[:4]
        stride_d, stride_h, stride_s, stride_b = tuple(reversed(t_bshd.stride))[:4]
        # Batch + partial stride must be coalescible
        # to get 5 independent TMA modes (TMA limitation)
        has_partial = cute.rank(t_bshd) == 5
        b_partial = b * t_bshd.shape[0] if has_partial else b
        h_g = h_q // h_k  # grouped heads
        gqa_shape = (b_partial, (h_g, s_q), h_k, d)
        gqa_stride = (stride_b, (stride_h, stride_s), stride_h * h_g, stride_d)
        gqa_layout = cute.make_layout(gqa_shape, stride=gqa_stride)
        return cute.make_tensor(t_bshd.iterator, gqa_layout)

    @staticmethod
    def _sf_layout_hb(m, k, l_shape, sf_vec_size):
        # 坑30:tile_atom_to_shape_SF 会把嵌套 l=(h,b) 折叠成 (1,?)(h 模
        # extent-1/stride-0 → 全部 head 读 h=0 切片)。绕开:先建 l=1 布局,
        # 再手工拼 (h,b) 模(h-fastest,与平铺缓冲的 per-(b,h) 切片序一致)。
        base = blockscaled_utils.tile_atom_to_shape_SF((m, k, 1), sf_vec_size)
        slice_sz = cute.cosize(base)
        h_ext = l_shape[0]
        return cute.make_layout(
            (base.shape[0], base.shape[1], l_shape),
            stride=(
                base.stride[0],
                base.stride[1],
                (slice_sz, h_ext * slice_sz),
            ),
        )

    # Reorder and group modes for GEMM
    @staticmethod
    def gemm_view(t_bshd: cute.Tensor, s_first: bool):
        sdhb = (1, 3, 2, 0)  # GEMM1 MKL
        dshb = (3, 1, 2, 0)  # GEMM2 MKL
        reorder = sdhb if s_first else dshb
        mT_layout = cute.select(t_bshd.layout, reorder)
        mT_layout = cute.group_modes(mT_layout, 2, 4)
        return cute.make_tensor(t_bshd.iterator, mT_layout)

    # Pack, reorder, and group modes for GEMM workspace
    @staticmethod
    def gemm_view_bsh(t_bsh: cute.Tensor, h_k: int):
        h_q, s_q, b = tuple(reversed(t_bsh.shape))[:3]
        stride_h, stride_s, stride_b = tuple(reversed(t_bsh.stride))[:3]
        h_g = h_q // h_k
        mT_shape = ((h_g, s_q), (h_k, b))
        mT_stride = ((stride_h, stride_s), (stride_h * h_g, stride_b))
        has_partial = cute.rank(t_bsh) == 4
        mT_shape += (t_bsh.shape[0],) if has_partial else ()
        mT_stride += (t_bsh.stride[0],) if has_partial else ()
        mT_layout = cute.make_layout(mT_shape, stride=mT_stride)
        return cute.make_tensor(t_bsh.iterator, mT_layout)

    ##############################
    # Decode Kernel launch
    ##############################
    @cute.jit
    def __call__(
        self,
        kv_splits: Int32,
        q_bshd: cute.Tensor,
        k_bshd: cute.Tensor,
        v_bshd: cute.Tensor,
        q_sf: cute.Tensor,  # Q scale factors in SF-atom order
        k_sf: cute.Tensor,  # K scales in the selected cache layout
        v_sf: cute.Tensor,  # D-contiguous V scales
        o_bshd: cute.Tensor,
        o_block_scale: Optional[cute.Tensor],
        o_global_scale: Optional[cute.Tensor],
        o_partial_bshd: Optional[cute.Tensor],
        l_partial_bsh: Optional[cute.Tensor],
        m_partial_bsh: Optional[cute.Tensor],
        completion_counter: Optional[cute.Tensor],
        scale_s: Float32,
        scale_o: Float32,
        seq_lens: cute.Tensor,
        block_tables: cute.Tensor,
        stream: cuda.CUstream,
        enable_pdl: bool = True,
    ):
        """
        Parameters
        ----------
        kv_splits
            Threadblocks per sequence (flash decoding).
        q_bshd
            Q tensor in BSHD logical view.
        k_bshd, v_bshd
            Paged KV views with logical shape (pages, page_size, h_k, d).
            Both borrow conventional HND packed-FP4 storage. The block table
            maps each request's logical pages to physical cache pages.
        seq_lens
            Int32 KV lengths, shape (batch,). Only live pages are read.
        block_tables
            Int32 physical page IDs, shape (batch, max_pages_per_seq).
            Columns have unit stride; the row stride may include padding.
        o_bshd
            Output tensor. A
            Float4E2M1FN tensor selects the fused NVFP4 epilogue in the
            no-split attention CTA, separate reducer, or last-CTA reducer.
        o_block_scale
            E4M3 output block scales in CUTLASS's 128x4 swizzled layout.
        o_global_scale
            One-element Float32 tensor containing the activation global-scale
            inverse used by the downstream NVFP4 GEMM.
        o_partial_bshd
            Partial O per kv split (kernel-red workspace).
        l_partial_bsh
            Partial ``colsum_p`` per kv split (kernel-red workspace).
        m_partial_bsh
            Partial ``colmax_s`` per kv split (kernel-red workspace).
        completion_counter
            Persistent Int32 counters used only by ``gmem_fused``. One entry
            is consumed per output tile and reset by the last CTA.
        scale_s
            Softmax scale.
        scale_o
            Output scale, applied in the reduction epilogue.
        stream
            CUDA stream to launch on.
        enable_pdl
            Programmatic Dependent Launch. Runtime-dynamic — no recompile on
            toggle.
        """
        ##############################
        # TiledMma creation
        ##############################
        # QK uses blockscaled FP4; V is converted to FP8 in TMEM.
        assert q_bshd.dtype == cutlass.Float4E2M1FN
        assert q_bshd.shape[1] == 1, "production decode requires one Q token"
        assert q_sf is not None and k_sf is not None
        quantize_o = cutlass.const_expr(
            o_bshd.dtype == cutlass.Float4E2M1FN
        )
        if cutlass.const_expr(quantize_o):
            assert (
                self.do_none_red
                or self.do_kernel_red
                or self.do_gmem_fused_red
            ), (
                "fused NVFP4 output requires no reduction, the deterministic "
                "kernel reducer, or the fused last-CTA reducer"
            )
            assert o_block_scale is not None and o_global_scale is not None
            assert o_block_scale.dtype == cutlass.Float8E4M3FN
            assert o_global_scale.dtype == Float32
            assert self.prediction_tile == 1
            if cutlass.const_expr(self.do_gmem_fused_red):
                assert completion_counter is not None
                assert completion_counter.dtype == Int32
        else:
            assert o_block_scale is None and o_global_scale is None
            assert not self.do_gmem_fused_red, (
                "gmem_fused V0 is restricted to direct NVFP4 output"
            )
        kq_dtype = q_bshd.dtype
        v_cache_dtype = v_bshd.dtype

        # The conventional cache remains packed e2m1, while the VP operands
        # are E4M3 after the side conversion.  Keeping these as two distinct
        # compile-time types prevents the old "dtype implies layout" bug.
        mma_dtype = cutlass.Float8E4M3FN
        sf_dtype = cutlass.Float8E4M3FN
        sf_vec_size = 16
        acc_dtype = Float32
        assert k_bshd.dtype == kq_dtype

        # Block tile sets the granularity at which threadblocks consume work (BMM1/BMM2)
        blk_tile_s = self.sequence_tile
        blk_tile_h = self.grouped_head_tile
        blk_tile_p = self.prediction_tile
        blk_tile_d = self.headdim
        blk_tile_shpd = (blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d)

        # MMA tile sets the granularity at which TMAs + MMAs are staged in smem
        mma_tile_m = 128
        # KQ 的收缩深度按 head_dim 封顶(fp4 时 128*8/4=256 会超出 D=128)
        mma_tile_k = min(128 * 8 // mma_dtype.width, self.headdim)
        # N-major 8b B in SMEM requires N multiple of 16. The generation
        # swap tactic stores P K-major instead, for which SM100 supports N8.
        min_mma_tile_n = (
            8 if self.swap_ab_for_gen or mma_dtype.width < 8 else 16
        )
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_n = max(min_mma_tile_n, blk_tile_n)
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)

        # MMA tiles per block tile
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_dk = math.ceil(blk_tile_d / mma_tile_k)
        assert blk_tile_s % mma_tile_m == 0
        assert mma_tile_n % blk_tile_n == 0

        # GEMM1: (S_K, (H_R, S_Q), D, (H_K, B))
        # blockscaled mxf4nvf4:SFA=K-scale,SFB=Q-scale(硬件在 MMA 内乘 SF)
        tiled_mma_kq = sm100_utils.make_blockscaled_trivial_tiled_mma(
            kq_dtype,
            OperandMajorMode.K,  # K
            OperandMajorMode.K,  # Q
            sf_dtype,
            sf_vec_size,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
        )
        # SFB 的 TMEM/TMA 视图需要独立的 tiled_mma;UTCCP 要求 SFB 按 N=128 粒度
        # 搬运(cutlass 示例 :245 规则:(M//cta组, round_up(N,128)))→ Q-SF 缓冲
        # 必须按 N_padded=128 分配(实际 N=16,其余为垫)。
        mma_tile_n_sfb = max(128, mma_tile_n)
        tiled_mma_kq_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            kq_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            sf_dtype,
            sf_vec_size,
            tcgen05.CtaGroup.ONE,
            (mma_tile_mnk[0], mma_tile_n_sfb),
        )

        # GEMM2: (D, (H_R, S_Q), S_K, (H_K, B))
        assert v_cache_dtype == cutlass.Float4E2M1FN
        assert v_sf is not None
        vp_dtype = cutlass.Float8E4M3FN
        # One converted FP8 V tile is consumed per VP MMA stage.
        mma_tile_k_vp = mma_tile_k
        mma_tile_m_vp = mma_tile_m
        mma_tile_mnk_vp = (mma_tile_m_vp, mma_tile_n, mma_tile_k_vp)
        tiles_dm = math.ceil(blk_tile_d / mma_tile_m_vp)
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,   # V:token 最内(收缩轴)
            (
                OperandMajorMode.K
                if self.swap_ab_for_gen
                else OperandMajorMode.MN
            ),  # P: generation Q8 uses K-major to make N8 legal
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            (
                tcgen05.OperandSource.TMEM
            ),
        )

        ##############################
        # Calculate stage counts
        ##############################
        # Fixed stage counts
        self.p_stages = p_stages = 4  # smem P (BMM2 B)
        self.o_stages = o_stages = 2  # tmem O (BMM2 C)

        # Calculate tmem alloc
        tmem_capacity_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        tmem_s_stage_cols = tiles_sm * mma_tile_n
        tmem_alloc_cols = mma_tile_n * o_stages  # per-thread colsum
        tmem_alloc_cols += tiles_dm * mma_tile_n * o_stages  # O
        # SF 列先记账再给 S 定级 —— 原顺序(S 先吃满、SF 后追加)在 N=64 时
        # 让总量冲破 512 列。SF 是刚性开销,必须优先预留。
        # SF TMEM 列(cutlass 示例 :373 公式:tile/sf_atom_mn(=32) × mma_inst_tile_k(=4))
        sf_cols = (mma_tile_m // 32) * 4  # SFK
        sf_cols += (mma_tile_n_sfb // 32) * 4  # SFQ
        # TRT-style direct conversion stores one FP8 V stage in TMEM.  A
        # 128x128 E4M3 tile consumes 128*8/32 = 32 TMEM columns.  Reserve this
        # ring before allowing S to consume the remaining columns; otherwise
        # the old greedy S-stage calculation leaves no legal V allocation.
        cvt_stage_cols = mma_tile_k_vp * vp_dtype.width // 32
        cvt_budget = tmem_capacity_cols - tmem_alloc_cols - sf_cols
        cvt_stages = min(8, cvt_budget // cvt_stage_cols)
        assert cvt_stages >= 2
        self.cvt_stages = cvt_stages
        tmem_alloc_cols += cvt_stages * cvt_stage_cols
        # po2 向上取整会再吃一截,故按 po2 后的额度反推 S 的可用列
        _budget = tmem_capacity_cols - tmem_alloc_cols - sf_cols
        max_s_stages = max(1, _budget // tmem_s_stage_cols)
        self.s_stages = s_stages = min(max_s_stages, p_stages)

        tmem_alloc_cols += tmem_s_stage_cols * s_stages + sf_cols
        tmem_alloc_cols = 2 ** math.ceil(math.log2(tmem_alloc_cols))  # po2
        self.tmem_alloc_cols = tmem_alloc_cols
        assert tmem_alloc_cols <= tmem_capacity_cols

        # Calculate smem alloc
        smem_alloc_bits = 0
        mbarrier_bits = Int64.width
        pipe_stage_bits = mbarrier_bits * 2  # producer + consumer
        # ``mk_stage_bits`` is the MMA-ready V stage. In qmul4 mode this is
        # FP8 and a separate packed raw-V stage is accounted below.
        mk_stage_bits = mma_tile_m_vp * mma_tile_k_vp * vp_dtype.width
        k_stage_bits = mma_tile_m * mma_tile_k * kq_dtype.width  # K stage(fp4 时减半)
        nk_stage_bits = mma_tile_n * mma_tile_k * kq_dtype.width  # Q(与 K 同 dtype)
        mn_stage_bits = mma_tile_m * mma_tile_n * mma_dtype.width
        # SF stage 尺寸(仅 nvfp4):SFK 每 stage M×(K/16);SFQ 一次性 N_pad×(K/16)
        sfk_stage_bits = mma_tile_m * (mma_tile_k // sf_vec_size) * sf_dtype.width
        sfk_stage_bits *= 2
        sfq_total_bits = (
            mma_tile_n_sfb * (mma_tile_k // sf_vec_size) * sf_dtype.width * tiles_dk
        )
        # tmem ptr
        smem_alloc_bits += Int32.width
        if cutlass.const_expr(self.do_gmem_fused_red):
            smem_alloc_bits += Int32.width  # last-arriving CTA flag
        # colmax + colsum
        smem_alloc_bits += blk_tile_n * acc_dtype.width
        smem_alloc_bits += blk_tile_n * warpgroup_warps * acc_dtype.width
        if cutlass.const_expr(self.do_cga_red):
            # Each CTA is the owner of 1/kv_splits of the output threads.  All
            # split partials for those threads occupy exactly one full output
            # tile per target CTA, independent of the cluster width.  Stats
            # are broadcast because every output slice needs all head max/sum
            # pairs.  One transaction barrier covers partial O + stats; the
            # owning reducer threads write their disjoint final fragments
            # straight to global memory.
            # Partial numerators stay FP32.  Besides preserving the online
            # softmax accumulator, DSMEM st.async has a 32-bit scalar path;
            # attempting a BF16 scalar copy crashes the CuTe 4.6 backend.
            smem_alloc_bits += blk_tile_n * blk_tile_d * acc_dtype.width
            smem_alloc_bits += (
                max_reduction_splits * warp_threads * 2 * acc_dtype.width
            )
            smem_alloc_bits += mbarrier_bits
        # Raw V scales cover contiguous groups of 16 D values.
        sfv_stage_bits = (
            (mma_tile_m_vp // sf_vec_size)
            * mma_tile_k_vp
            * sf_dtype.width
        )
        # Q(+SFQ), S, P, O
        smem_alloc_bits += tiles_dk * nk_stage_bits + mbarrier_bits  # 1 mbar for Q
        smem_alloc_bits += sfq_total_bits
        smem_alloc_bits += s_stages * pipe_stage_bits  # s in tmem
        smem_alloc_bits += p_stages * (tiles_sm * mn_stage_bits + pipe_stage_bits)
        smem_alloc_bits += o_stages * pipe_stage_bits  # o in tmem
        smem_alloc_bits += cvt_stages * pipe_stage_bits
        alignment_bits = 1024 - (smem_alloc_bits % 1024)
        # K(+SFK)与 V:两条独立流水线,等 stage 数(坑1:拆分决策见 plan doc)
        smem_capacity_bits = (
            utils.get_smem_capacity_in_bytes("sm_100") // 1 * 8
        )
        # The dynamic-smem allocator inserts alignment between the packed
        # pipeline/barrier prefix and several separately aligned tensor
        # allocations.  A single end-of-prefix round-up under-counted that
        # padding by 448 B for qmul4_tmem on B200, selecting eight KV stages
        # and requesting 232768 B against the 232448 B launch limit.  Keep a
        # small explicit guard here so stage selection reflects the final
        # allocator rather than failing at launch.  The next legal choice is
        # seven stages for the validated head-dimension-128 geometry.
        smem_allocator_guard_bits = 512 * 8
        remaining_bits = (
            smem_capacity_bits
            - smem_alloc_bits
            - alignment_bits
            - smem_allocator_guard_bits
        )
        kv_pipe_count = 2
        kv_pair_bits = (
            k_stage_bits
            + sfk_stage_bits
            + mk_stage_bits
            + sfv_stage_bits
            + kv_pipe_count * pipe_stage_bits
        )
        kv_stages = remaining_bits // kv_pair_bits
        if cutlass.const_expr(self.swap_ab_for_gen):
            
            
            
            
            
            kv_stages = min(kv_stages, 7)
        # 原实现的 mbar 挤占修正沿用(两条流水线共 2×pipe_stage_bits/stage 已计入 pair)

        ##############################
        # TMA creation
        ##############################
        h_k = k_bshd.shape[2]
        o_bshd_ = o_partial_bshd if self.do_partial_red else o_bshd

        # ((h_g, s_q), d, (h_k, b))
        mQ_nkl = self.gemm_view(self.gqa_pack(q_bshd, h_k), True)
        mK_mkl = self.gemm_view(k_bshd, True)  # (s_k, d, (h_k, b))
        mV_mkl = self.gemm_view(v_bshd, False)  # (d, s_k, (h_k, b))
        # Keep the final output view even when the main CTA writes split
        # partials. ``gmem_fused`` consumes it in its last-arriving CTA.
        mO_final_mnl = self.gemm_view(self.gqa_pack(o_bshd, h_k), False)
        # (d, (h_g, s_q), (h_k, b_partial))
        mO_mnl = self.gemm_view(self.gqa_pack(o_bshd_, h_k), False)

        # (MMA, MMA_M/N, MMA_K, stages)
        smem_layout_q = sm100_utils.make_smem_layout_b(
            tiled_mma_kq, mma_tile_mnk, kq_dtype, tiles_dk
        )
        smem_layout_k = sm100_utils.make_smem_layout_a(
            tiled_mma_kq, mma_tile_mnk, kq_dtype, kv_stages
        )
        # Packed FP4 V travels through an SMEM byte carrier, then
        # LDSM.U4 -> QMUL4 writes the converted FP8 operand to TMEM.
        smem_layout_v = sm100_utils.make_smem_layout_a(
            tiled_mma_vp,
            mma_tile_mnk_vp,
            cutlass.Int8,
            kv_stages,
            is_k_major=False,
        )
        smem_layout_v_sf_d = None
        _sf_blocks = mma_tile_m_vp // 16
        _tile_sf = _sf_blocks * mma_tile_k_vp
        smem_layout_v_sf_d = cute.make_layout(
            (16, _tile_sf // 16, kv_stages),
            stride=(1, 16, _tile_sf),
        )
        # SF smem 布局(仅 nvfp4;SFQ 单份、SFK 随 K stage)
        smem_layout_sfk_raw = None
        smem_layout_sfk = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_kq, mma_tile_mnk, sf_vec_size, kv_stages
        )
        assert self.headdim == 128
        _ksf_stage_bytes = mma_tile_m * (mma_tile_k // sf_vec_size)
        smem_layout_sfk_raw = cute.make_layout(
            (16, _ksf_stage_bytes // 16, kv_stages),
            stride=(1, 16, _ksf_stage_bytes),
        )
        smem_layout_sfq = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_kq_sfb,
            (mma_tile_mnk[0], mma_tile_n_sfb, mma_tile_mnk[2]),
            sf_vec_size,
            tiles_dk,
        )

        # In the no-split specialization the otherwise-idle TMA warp owns the
        # quantizer.  Split-K must first publish FP32 partials, so only its
        # unique combine kernel sees the final NVFP4 output contract.
        quantize_o_main = cutlass.const_expr(quantize_o and self.do_none_red)
        o_smem_dtype = Float32 if quantize_o_main else mO_mnl.dtype
        smem_layout_atom_o = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, o_smem_dtype
        )
        smem_layout_o = cute.tile_to_shape(
            smem_layout_atom_o, (max(blk_tile_d, mma_tile_m), mma_tile_n), order=(1, 0)
        )
        smem_layout_o = cute.flat_divide(
            smem_layout_o, (mma_tile_m_vp, mma_tile_n)
        )
        # (MMA_TILE_M, MMA_TILE_N, #TILE_DM)
        smem_layout_o = cute.select(smem_layout_o, (0, 1, 2))

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # Construct multimode gmem tiler
        tma_tile_n = (blk_tile_h, mma_tile_n // blk_tile_h)
        tma_tile_mnk = (mma_tile_m, tma_tile_n, mma_tile_k)
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_nkl,
            cute.select(smem_layout_q, mma_modes),
            tma_tile_mnk,
            tiled_mma_kq,
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK_mkl,
            cute.select(smem_layout_k, mma_modes),
            tma_tile_mnk,
            tiled_mma_kq,
        )
        tma_tile_mnk_vp = (mma_tile_m_vp, tma_tile_n, mma_tile_k_vp)
        tma_atom_v_sf_d = tma_tensor_v_sf_d = None
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV_mkl,
            cute.select(smem_layout_v, mma_modes),
            tma_tile_mnk_vp,
            tiled_mma_vp,
            internal_type=cutlass.Int8,
        )

        _tsf = (self.headdim // 16) * self.page_size
        _hkv = v_sf.shape[1]
        mVSF_d = cute.make_tensor(
            v_sf.iterator,
            cute.make_layout(
                (16, _tsf // 16, (_hkv, v_sf.shape[0])),
                stride=(
                    1,
                    16,
                    (v_sf.stride[1], v_sf.stride[0]),
                ),
            ),
        )
        tma_atom_v_sf_d, tma_tensor_v_sf_d = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op,
                mVSF_d,
                cute.slice_(smem_layout_v_sf_d, (None, None, 0)),
                (16, ((mma_tile_m_vp // 16) * mma_tile_k_vp) // 16),
            )
        )
        # SF gmem 视图 + TMA(仅 nvfp4)。k_sf/q_sf 为平铺 e4m3 缓冲,按 SF-atom 布局解释:
        # K-SF Shape=(S, D, (h_k,b));Q-SF Shape=(N_pad, D, (h_k,b)),N_pad=mma_tile_n_sfb。
        _ksf_bytes = self.page_size * (self.headdim // sf_vec_size)
        mSFK = cute.make_tensor(
            k_sf.iterator,
            cute.make_layout(
                (16, _ksf_bytes // 16, (k_sf.shape[1], k_sf.shape[0])),
                stride=(
                    1,
                    16,
                    (k_sf.stride[1], k_sf.stride[0]),
                ),
            ),
        )
        tma_atom_sfk, tma_tensor_sfk = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op,
                mSFK,
                cute.slice_(smem_layout_sfk_raw, (None, None, 0)),
                (16, _ksf_bytes // 16),
            )
        )
        sfq_layout = self._sf_layout_hb(
            mma_tile_n_sfb, self.headdim, mQ_nkl.shape[2], sf_vec_size
        )
        mSFQ = cute.make_tensor(q_sf.iterator, sfq_layout)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            (1, 1), tiled_mma_kq_sfb.thr_id
        )
        tma_atom_sfq, tma_tensor_sfq = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            mSFQ,
            cute.select(smem_layout_sfq, mma_modes),
            (mma_tile_mnk[0], mma_tile_n_sfb, mma_tile_mnk[2]),
            tiled_mma_kq_sfb,
            internal_type=cutlass.Int16,
        )

        tma_atom_o = tma_tensor_o = None
        if cutlass.const_expr(not quantize_o_main and not self.do_cga_red):
            tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO_mnl,
                cute.select(smem_layout_o, mode=[0, 1]),
                tma_tile_mnk[:2],
            )

        # GEMM views for workspace tensors
        mM_partial_nl = mL_partial_nl = None
        if cutlass.const_expr(self.do_partial_red):
            assert m_partial_bsh.dtype == l_partial_bsh.dtype == acc_dtype
            assert o_partial_bshd.dtype in (acc_dtype, BFloat16)

            # ((h_g, s_q), (h_k, b), kv_splits)
            mM_partial_nl = self.gemm_view_bsh(m_partial_bsh, h_k)
            mL_partial_nl = self.gemm_view_bsh(l_partial_bsh, h_k)

        ##############################
        # Launch kernel(s)
        ##############################
        scale_s_log2_e = scale_s * log2_e

        n_tiles = cute.ceil_div(mQ_nkl.shape[0], (blk_tile_h, blk_tile_p))
        grid_y = cute.size(n_tiles)
        grid_z = cute.size(mQ_nkl.shape[2])  # l tiles
        grid_x = 1 if self.do_none_red else kv_splits
        grid = (grid_x, grid_y, grid_z)
        cluster_x = kv_splits if (self.do_cga_red) else 1

        self.decode(
            # MMA
            blk_tile_shpd,
            mma_tile_mnk,
            mma_tile_mnk_vp,
            tiled_mma_kq,
            tiled_mma_vp,
            mma_dtype,
            kq_dtype,
            vp_dtype,
            o_smem_dtype,
            # Q
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            # K
            smem_layout_k,
            tma_atom_k,
            tma_tensor_k,
            # NVFP4 QK scale factors
            tiled_mma_kq_sfb,
            smem_layout_sfk,
            smem_layout_sfk_raw,
            tma_atom_sfk,
            tma_tensor_sfk,
            smem_layout_sfq,
            tma_atom_sfq,
            tma_tensor_sfq,
            # V
            smem_layout_v,
            smem_layout_v_sf_d,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_v_sf_d,
            tma_tensor_v_sf_d,
            # O
            smem_layout_o,
            tma_atom_o,
            mO_mnl if quantize_o_main or self.do_cga_red else tma_tensor_o,
            mO_mnl if self.do_gmem_fused_red else None,
            (
                o_block_scale
                if quantize_o_main or self.do_gmem_fused_red
                else None
            ),
            (
                o_global_scale
                if quantize_o_main or self.do_gmem_fused_red
                else None
            ),
            (
                mO_final_mnl
                if self.do_cga_red or self.do_gmem_fused_red
                else None
            ),
            # Rest
            mL_partial_nl,
            mM_partial_nl,
            completion_counter,
            scale_s_log2_e,
            scale_o,
            seq_lens,
            block_tables,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[cluster_x, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

        if cutlass.const_expr(self.do_kernel_red):
            self.launch_reduction(
                self.headdim,
                o_bshd,
                o_partial_bshd,
                l_partial_bsh,
                m_partial_bsh,
                scale_o,
                o_block_scale,
                o_global_scale,
                self.grouped_head_tile,
                (
                    self.logical_grouped_heads
                    if self.logical_grouped_heads is not None
                    else self.grouped_head_tile
                ),
                stream,
                enable_pdl,
            )

    @cute.jit
    def mma_vp_qmul4_tmem_step(
        self,
        initialize_o: cutlass.Constexpr,
        tiled_mma_vp: cute.TiledMma,
        tAtV_cvt: cute.Tensor,
        tBsP_desc: cute.Tensor,
        tCtO: cute.Tensor,
        p_consumer,
        o_producer,
        v_ready_consumer,
        tiles_sk: cutlass.Constexpr,
        tiles_dm: cutlass.Constexpr,
    ):
        """Issue one direct-TMEM VP sequence tile.

        ``initialize_o`` is compile-time so the first MMA in each O-ring
        phase is a literal !UPT.  Keeping the false->true transition inside
        one jitted linear step also keeps every instruction on the same UMMA
        accumulator descriptor; a runtime UP predicate or a second TiledMma
        object does not provide that guarantee on SM100.
        """
        p_token = p_consumer.try_wait()
        o_token = o_producer.try_acquire()
        v_ready_token = v_ready_consumer.try_wait()

        p_handle = p_consumer.wait_and_advance(p_token)
        o_handle = o_producer.acquire_and_advance(o_token)
        for sk in cutlass.range_constexpr(tiles_sk):
            for dm in cutlass.range_constexpr(tiles_dm):
                v_mma_handle = v_ready_consumer.wait_and_advance(v_ready_token)
                is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                if not is_last_iter:
                    v_ready_token = v_ready_consumer.try_wait()

                mmas_k = cute.size(tAtV_cvt.shape[2])
                for mma_k in cutlass.range_constexpr(mmas_k):
                    tiled_mma_vp.set(
                        tcgen05.Field.ACCUMULATE,
                        not (
                            initialize_o
                            and sk == 0
                            and dm == 0
                            and mma_k == 0
                        ),
                    )
                    cute.gemm(
                        tiled_mma_vp,
                        tCtO[mma_dice + (dm, o_handle.index)],
                        tAtV_cvt[
                            None,
                            None,
                            mma_k,
                            v_mma_handle.index,
                        ],
                        tBsP_desc[
                            None,
                            None,
                            mma_k,
                            sk,
                            p_handle.index,
                        ],
                        tCtO[mma_dice + (dm, o_handle.index)],
                    )
                # PipelineAsyncUmma's empty arrival is ordered after all
                # tensor-core reads of this converted V slot.
                v_mma_handle.release()

        p_handle.release()
        o_handle.commit()
        return tiled_mma_vp, p_consumer, o_producer, v_ready_consumer

    @cute.kernel
    def decode(
        self,
        # MMA
        blk_tile_shpd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        mma_tile_mnk_vp: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        mma_dtype: Type[cutlass.Numeric],
        kq_dtype: Type[cutlass.Numeric],
        vp_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        # Q
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        # K
        smem_layout_k: cute.ComposedLayout,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        # NVFP4 QK scale factors
        tiled_mma_kq_sfb: cute.TiledMma,
        smem_layout_sfk: cute.Layout,
        smem_layout_sfk_raw: Optional[cute.Layout],
        tma_atom_sfk: cute.CopyAtom,
        mSFK: cute.Tensor,
        smem_layout_sfq: cute.Layout,
        tma_atom_sfq: cute.CopyAtom,
        mSFQ: cute.Tensor,
        # V
        smem_layout_v: cute.ComposedLayout,
        smem_layout_v_sf_d: cute.Layout,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        tma_atom_v_sf_d: cute.CopyAtom,
        mVSF_d: cute.Tensor,
        # O
        smem_layout_o: cute.ComposedLayout,
        tma_atom_o: Optional[cute.CopyAtom],
        mO: cute.Tensor,
        mOPartialGlobal: Optional[cute.Tensor],
        mOSF: Optional[cute.Tensor],
        mOGlobalScale: Optional[cute.Tensor],
        mOFinal: Optional[cute.Tensor],
        # Rest
        mL_partial: Optional[cute.Tensor],
        mM_partial: Optional[cute.Tensor],
        mCompletion: Optional[cute.Tensor],
        scale_s_log2_e: Float32,
        scale_o: Float32,
        mSeqLens: cute.Tensor,
        mBlockTables: cute.Tensor,
    ):
        ##############################
        # Static variables
        ##############################
        # Smem alloc helper
        svector_align = 16
        # 坑27:swizzled 操作数的 desc 基址必须对齐到 swizzle 模式周期
        # (S<2,4,3>=512B;取 1024 覆盖 S<3,4,3>),否则硬件对 desc 偏移的
        # 应用失效(mma_k 无效化、环槽偏移落零)。B200 28 轮探针定案。
        stensor_align = 1024
        smem = utils.SmemAllocator()

        # No multicast
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))  # vmnk

        # Alias types(nvfp4:Q/K=e2m1,V=mma_dtype)
        q_dtype = k_dtype = kq_dtype

        sf_dtype = cutlass.Float8E4M3FN
        o_dtype = out_dtype
        acc_dtype = Float32
        quantize_o = cutlass.const_expr(mO.dtype == cutlass.Float4E2M1FN)
        if cutlass.const_expr(quantize_o):
            assert mOSF is not None and mOGlobalScale is not None
            assert tma_atom_o is None

        # Shapes for MMA tile indexing (Read TMA partition for example)
        blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d = blk_tile_shpd
        blk_tile_hp = (blk_tile_h, blk_tile_p)  # multimode tiler
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_m, mma_tile_n, mma_tile_k = mma_tile_mnk
        mma_tile_k_vp = mma_tile_mnk_vp[2]
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_sk = blk_tile_s // mma_tile_k_vp
        mma_tile_m_vp = mma_tile_mnk_vp[0]
        tiles_dm = cute.ceil_div(blk_tile_d, mma_tile_m_vp)
        tiles_dk = cute.ceil_div(blk_tile_d, mma_tile_k)

        # Static control flow
        do_gmem_fused_red = self.do_gmem_fused_red
        do_partial_red = self.do_partial_red
        do_cga_red = self.do_cga_red
        do_none_red = self.do_none_red

        ##############################
        # Warp specialization
        ##############################
        # Warp assignments
        warpgroup_id = 0
        mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        tma_kv_warp_id = warpgroup_id * warpgroup_warps + 2
        tma_qo_warp_id = warpgroup_id * warpgroup_warps + 3
        reduction_warp_id = tma_kv_warp_id
        warpgroup_id += 1

        conversion_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        softmax_warpgroups = 1 if self.single_softmax_wg else 2
        softmax_warpgroup_ids = tuple(
            range(warpgroup_id, warpgroup_id + softmax_warpgroups)
        )
        warpgroup_id += softmax_warpgroups

        correction_warpgroup_id = warpgroup_id
        warpgroup_id += 1
        assert self.threads_per_cta == warpgroup_id * warpgroup_threads

        # Register allocations
        # qmul4_tmem's fifth warpgroup makes the Q16 correction role spill
        # under uniform allocation.  Reuse the existing role budgets there;
        # this changes register ownership only, not the QK/V arithmetic.
        max_sw_regs_per_wg_thread = 256  # CUDA limitation
        mma_tma_regs = 64
        conversion_regs = 64
        # A 640-thread qmul4_tmem CTA starts from a 96-register uniform
        # allocation at min_blocks_per_mp=1.  ``decrease(120)`` is therefore
        # an illegal direction on hardware.  Keep those softmax groups at the
        # uniform ceiling and donate the MMA/conversion registers to correction.
        softmax_regs = 120 if softmax_warpgroups == 1 else 96
        # 640 threads round the uniform launch allocation down to 96 registers
        # per thread (61,440 total).  The unused 4,096 physical registers are
        # not donor registers in this CTA's dynamic pool.  Budget qmul4_tmem
        # against the actual 5*96 pool and leave one 8-register allocation unit
        # idle so correction's TRY_ALLOC cannot wait forever at exact capacity.
        role_reg_budget = (
            (softmax_warpgroups + 3)
            * (64 * 1024 // self.threads_per_cta // 8 * 8) - 8
        )
        correction_regs = min(
            max_sw_regs_per_wg_thread,
            role_reg_budget
            - mma_tma_regs
            - conversion_regs
            - softmax_regs * softmax_warpgroups,
        )
        assert (mma_tma_regs + conversion_regs
                + softmax_regs * softmax_warpgroups
                + correction_regs) <= role_reg_budget

        # Read thread indices
        kv_splits, grid_tiles_hp, tiles_hb = cute.arch.grid_dim()
        kv_split_idx, coord_hp, coord_hb = cute.arch.block_idx()
        coord_hp_linear = coord_hp
        coord_hb_linear = coord_hb
        if cutlass.const_expr(do_none_red):
            kv_splits, kv_split_idx = (1, 0)
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 1  # warp 0 does all mbarrier inits for now

        # Unpack multimodes
        grouped_heads, prediction = mQ.shape[0]
        heads_k, batches = tiles_hb = mQ.shape[2]
        tiles_hp = cute.ceil_div(mQ.shape[0], blk_tile_hp)
        coord_hb = cute.idx2crd(coord_hb, tiles_hb)
        coord_hp = cute.idx2crd(coord_hp, tiles_hp)
        coord_hg, coord_p = coord_hp
        coord_hk, coord_b = coord_hb

        # Lengths bound the live pages in each row. CuTe preserves the source
        # block table's row stride, including any unused trailing columns.
        seqlen = mSeqLens[coord_b]
        page_size = self.page_size
        pages_s = blk_tile_s // page_size  # 每 blk tile 页数(256/128 = 2)
        page_count = cute.ceil_div(seqlen, page_size)

        # Runtime control flow
        tiles_s = cute.ceil_div(seqlen, blk_tile_s)
        iters_s = cute.ceil_div(tiles_s - kv_split_idx, kv_splits)
        prefetch_iters = min(2, self.s_stages - 1)  # MMA KQ iters to hide first softmax
        exit_early = kv_split_idx >= tiles_s
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n

        ##############################
        # Prefetch TMA descriptor
        ##############################
        if warp_idx == init_warp and not exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v_sf_d)
            if cutlass.const_expr(not quantize_o and not do_cga_red):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)
        init_warp += 1

        ##############################
        # Tmem Allocation
        ##############################
        tmem_alloc_cols = self.tmem_alloc_cols
        tmem_ptr_smem_ptr = smem.allocate_array(Int32)
        # Match NVIDIA mixed-input FMHA: an init warp allocates the CTA's TMEM
        # and the VP warp later relinquishes the allocation permit before it
        # deallocates.  ``relinquish_tmem_alloc_permit`` exists precisely for
        # this cross-warp lifetime hand-off; making VP both allocate and free
        # bypasses the reference protocol and leaves reuse state ill-defined.
        if warp_idx == init_warp and not exit_early:
            cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1

        ##############################
        # Pipeline Allocation + Init
        ##############################
        # Initialize named barriers
        softmax_threads = warpgroup_threads
        correction_threads = warpgroup_threads
        reduction_threads = warp_threads
        tma_threads = warp_threads
        sM_producer_nbar = nbar(3, softmax_threads + correction_threads)
        sM_consumer_nbar = nbar(4, softmax_threads + correction_threads)
        tL_producer_nbar = nbar(5, softmax_threads + correction_threads)
        tL_consumer_nbar = nbar(7, softmax_threads + correction_threads)
        sM_final_nbar = nbar(9, correction_threads + reduction_threads)
        sL_final_nbar = nbar(10, correction_threads + reduction_threads)
        sO_final_nbar = nbar(11, correction_threads + tma_threads)
        sM_mutex_nbar = nbar(12, softmax_threads * softmax_warpgroups)
        # qmul4_tmem adds an independent conversion warpgroup and changes the
        # scheduler order.  Explicitly publish correction's initial O/TMEM
        # stores to the VP warp instead of relying on warpgroup placement.
        # IDs 12/13 are both owned by the two phases of sM_mutex_nbar.
        # Keep the qmul4_tmem-only handshake in a remaining slot.
        o_init_nbar = nbar(14, correction_threads + warp_threads)
        # cga_sharded's local reduction warp collapses the four softmax-warp
        # sums before the correction warpgroup broadcasts stats to peers.
        cga_stats_ready_nbar = nbar(15, correction_threads + reduction_threads)
        # gmem_fused and cga_sharded are mutually exclusive specializations,
        # so they can share the last hardware named-barrier slot. Only the
        # specialist WG (TMA store + completion atomic) and correction WG
        # (final merge) participate; conversion and softmax can retire.
        gmem_completion_nbar = nbar(
            15, warpgroup_threads + correction_threads
        )
        # named barrier stage helper
        def with_phase(nbar_, phase):
            return nbar(nbar_.barrier_id + phase, nbar_.num_threads)

        # Alias thread cooperatives
        thr_cg = lambda t: CooperativeGroup(Agent.Thread, t)
        elect_one_cooperative = thr_cg(1)
        warpgroup_cooperative = thr_cg(warpgroup_threads)
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        softmax_group = warpgroup_cooperative
        correction_group = warpgroup_cooperative

        # Initialize cluster colmax + colsum mbar (even if this split exits early)
        cga_partial_mbar = None
        if cutlass.const_expr(do_cga_red):
            cga_partial_mbar = smem.allocate_array(Int64, 1)
            if warp_idx == init_warp:
                with cute.arch.elect_one():
                    partial_o_bytes = (
                        blk_tile_n * blk_tile_d * acc_dtype.width // 8
                    )
                    partial_stats_bytes = (
                        kv_splits * warp_threads * 2 * acc_dtype.width // 8
                    )
                    cute.arch.mbarrier_init(cga_partial_mbar, 1)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        cga_partial_mbar,
                        Int32(partial_o_bytes + partial_stats_bytes),
                    )
            init_warp += 1
            cute.arch.cluster_arrive_relaxed()

        # Initialize Q load mbarrier
        q_load_mbar = smem.allocate_array(Int64, 1)
        sfq_mbar = smem.allocate_array(Int64, 1)
        if warp_idx == init_warp:
            expect_tx_bytes = cute.size_in_bytes(q_dtype, smem_layout_q)
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_load_mbar, 1)
                cute.arch.mbarrier_init(sfq_mbar, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(q_load_mbar, expect_tx_bytes)
                sfq_expect = cutlass.Int32(
                    cute.size_in_bytes(sf_dtype, smem_layout_sfq)
                )
                cute.arch.mbarrier_arrive_and_expect_tx(sfq_mbar, sfq_expect)
        init_warp += 1

        # K: TMA -> UMMA; raw V: TMA -> conversion warpgroup;
        # ready V: conversion warpgroup -> UMMA through TMEM.
        kv_stages = smem_layout_k.shape[-1]
        k_stage_bytes = mma_tile_m * mma_tile_k * k_dtype.width // 8
        k_stage_bytes += cute.size_in_bytes(
            sf_dtype, cute.select(smem_layout_sfk, mma_modes)
        )
        k_producer = k_consumer = None
        v_raw_producer = v_raw_consumer = None
        v_ready_producer = v_ready_consumer = None
        k_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
        k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=kv_stages,
            producer_group=tma_group,
            consumer_group=mma_group,
            tx_count=k_stage_bytes,
            barrier_storage=k_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        v_raw_stage_bytes = cute.size_in_bytes(
            cutlass.Float4E2M1FN,
            cute.select(smem_layout_v, mma_modes),
        )
        v_raw_stage_bytes += cute.size_in_bytes(
            sf_dtype, cute.slice_(smem_layout_v_sf_d, (None, None, 0))
        )
        v_raw_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
        v_raw_producer, v_raw_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=kv_stages,
            producer_group=tma_group,
            consumer_group=warpgroup_cooperative,
            tx_count=v_raw_stage_bytes,
            barrier_storage=v_raw_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            # One signaling lane per conversion warp: four arrivals per
            # empty slot, matching the full conversion warpgroup.
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()
        v_ready_stages = self.cvt_stages
        v_ready_pipeline_ptr = smem.allocate_array(Int64, v_ready_stages * 2)
        v_ready_producer, v_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=v_ready_stages,
            producer_group=warpgroup_cooperative,
            consumer_group=mma_group,
            barrier_storage=v_ready_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()

        s_stages = self.s_stages
        s_pipeline_ptr = smem.allocate_array(Int64, s_stages * 2)
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=s_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=s_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        p_stages = self.p_stages
        p_pipeline_ptr = smem.allocate_array(Int64, p_stages * 2)
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=p_stages,
            producer_group=softmax_group,
            consumer_group=mma_group,
            barrier_storage=p_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        o_stages = self.o_stages
        o_pipeline_ptr = smem.allocate_array(Int64, o_stages * 2)
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=o_stages,
            producer_group=mma_group,
            consumer_group=correction_group,
            barrier_storage=o_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        ##############################
        # Smem Tensor Allocation
        ##############################
        # Threadblock slice
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)

        # Q, K, V
        tAsK = smem.allocate_tensor(
            k_dtype, smem_layout_k.outer, stensor_align, smem_layout_k.inner
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        # 坑5:pristine 的 V 与 K 共享同一块 smem(共享 FIFO 的物理本质);
        # 流水线拆分后必须独立分配,否则两条流水线并发互踩(cos 掉到 0.5 的元凶)。
        v_smem_dtype = cutlass.Int8
        tAsV = smem.allocate_tensor(
            v_smem_dtype,
            smem_layout_v.outer,
            stensor_align,
            smem_layout_v.inner,
        )  # native/ready V, or packed 8-bit carrier for qmul4_tmem
        tAsVRaw = tAsV
        tAsVSFD = smem.allocate_tensor(
            sf_dtype, smem_layout_v_sf_d, stensor_align
        )
        # 索引别名:TMA 按 (16, tile/16) 搬字节,转换循环要 (S, token)
        # 语义。trtllm 的 4-token interleave 在 tile 内是
        #   offset = (t//4)*4*S + s*4 + t%4
        # ≡ shape (S,(4,ps/4)) stride (4,(1,4S)) —— CuTe 把扁平 token
        # 坐标按 colex 拆成 (t%4, t//4),所以 ``scales[d_block, token]``
        # 的写法一个字都不用改。等价性有离线门盯着
        # (test_trtllm_v_scale_interleave_matches_flashinfer_recipe)。
        _sfb = mma_tile_m_vp // 16
        _tsf_s = _sfb * mma_tile_k_vp
        tAsVSFD = cute.make_tensor(
            tAsVSFD.iterator,
            cute.make_layout(
                (_sfb, (4, mma_tile_k_vp // 4), kv_stages),
                stride=(4, (1, 4 * _sfb), _tsf_s),
            ),
        )
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, q_stages)
        # nvfp4:SFK(随 K stage)与 SFQ(单份)的 smem
        tAsSFK = None
        tAsSFKRaw = None
        tBsSFQ = None
        tAsSFK = smem.allocate_tensor(sf_dtype, smem_layout_sfk, stensor_align)
        tAsSFKRaw = smem.allocate_tensor(
            sf_dtype, smem_layout_sfk_raw, stensor_align
        )
        tBsSFQ = smem.allocate_tensor(sf_dtype, smem_layout_sfq, stensor_align)
        tAsSFKAtomU32 = None
        tAsSFKRawU64 = None
        _ksf_stage_bytes = mma_tile_m * (mma_tile_k // 16)
        _tAsSFKAtomBytes = cute.recast_tensor(tAsSFK, cutlass.Uint8)
        tAsSFKAtomU32 = cute.make_tensor(
            cute.recast_ptr(
                _tAsSFKAtomBytes.iterator, dtype=cutlass.Uint32
            ),
            cute.make_layout(
                (_ksf_stage_bytes // 4, kv_stages),
                stride=(1, _ksf_stage_bytes // 4),
            ),
        )
        _tAsSFKRawBytes = cute.recast_tensor(tAsSFKRaw, cutlass.Uint8)
        tAsSFKRawU64 = cute.make_tensor(
            cute.recast_ptr(
                _tAsSFKRawBytes.iterator, dtype=cutlass.Uint64
            ),
            cute.make_layout(
                (mma_tile_m, kv_stages),
                stride=(1, mma_tile_m),
            ),
        )

        # S
        # (MMA_MN, #MMA_M=1, #MMA_N=1, #TILE_SM, s_stages)
        tCtS_shape = tiled_mma_kq.partition_shape_C(
            (mma_tile_m, mma_tile_n, tiles_sm, s_stages)
        )
        tCtS = thrblk_mma_kq.make_fragment_C(tCtS_shape)

        # P - Treat MN C tile of BMM0 as NM B tile of BMM1
        # (MMA_NK, #MMA_N=1, #MMA_K=TILE_S/MMA_K, p_stages)
        blk_tile_nm = (None, mma_tile_n, mma_tile_m * tiles_sm)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_vp, blk_tile_nm, vp_dtype, p_stages
        )
        # FP8 P uses its own shared-memory allocation.
        tBsP_nm = smem.allocate_tensor(
            vp_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )
            
            

        # Tile for NK B tile iteration
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B(
            (mma_tile_n, mma_tile_k)
        )  # (MMA_NK, #MMA_N=1, #MMA_K=MMA_TILE_K/MMA_K, #TILE_SK=TILE_S/MMA_TILE_K, p_stages)
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))

        # Reshape NM B tile of BMM1 to become MN C tile of BMM0
        # (MMA_NK, #MMA_N, #MMA_K=TILE_S/MMA_K, p_stages) ->
        # (MMA_MN, #MMA_M, #MMA_N, #TILE_SM, p_stages)
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4, 5))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)

        # O
        # Reuse KV smem for O TMA store
        sO_iterator = cute.recast_ptr(tAsK.iterator, smem_layout_o.inner, dtype=o_dtype)
        # (MMA_TILE_M, MMA_TILE_N, #TILE_DM)
        sO_mma = cute.make_tensor(sO_iterator, smem_layout_o.outer)
        # (MMA, #MMA_M, #MMA_N, #TILE_DM, o_stages)
        tCsO = thrblk_mma_vp.partition_C(sO_mma)
        tCtO = thrblk_mma_vp.make_fragment_C((*tCsO.shape, o_stages))

        # Converted V lives in TMEM for the TRT-style path.  The layout must
        # come from the TMEM-source MMA atom itself; manufacturing a parallel
        # flat layout would lose the tcgen05 operand addressing contract.
        tAtV_cvt = None
        tAtV_cvt_shape = tiled_mma_vp.partition_shape_A(
            (mma_tile_m_vp, mma_tile_k_vp, self.cvt_stages)
        )
        tAtV_cvt = thrblk_mma_vp.make_fragment_A(tAtV_cvt_shape)

        # M - colmax
        sM_layout = cute.make_layout(blk_tile_n)
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        if warp_idx == init_warp:
            if lane_store_max:
                sM[lane_idx] = -Float32.inf
        init_warp += 1

        # L - colsum
        sL_layout = cute.make_layout((blk_tile_n, warpgroup_warps))
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        if warp_idx == init_warp:
            for i in cutlass.range_constexpr(0, cute.size(sL), warp_threads):
                if i + lane_idx < cute.size(sL):
                    sL[i + lane_idx] = Float32(0)
        init_warp += 1

        # CTA-local publication of the completion-counter result. The whole
        # CTA reaches the two sync points around it, so one Int32 is enough to
        # route the dynamically last CTA into the in-kernel reducer.
        sGmemLast = None
        if cutlass.const_expr(do_gmem_fused_red):
            sGmemLast = smem.allocate_tensor(
                Int32, cute.make_layout(1), svector_align
            )

        # per-thread colsum
        # (MMA_MN, #MMA_M=1, #MMA_N=1, o_stages)
        tCtL_shape = tiled_mma_kq.partition_shape_C((mma_tile_m, mma_tile_n, o_stages))
        tCtL = thrblk_mma_kq.make_fragment_C(tCtL_shape)

        # R - cluster reduction buffers for colmax + colsum
        sCgaO = None
        sCgaStats = None
        if cutlass.const_expr(do_cga_red):
            # Target-CTA layout is source-major after the output-thread
            # permutation.  Its total size is one BF16 output tile for every
            # cluster width: splits * (128/splits) * N.
            sCgaO = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout((blk_tile_d * blk_tile_n,)),
                svector_align,
            )
            sCgaStats = smem.allocate_tensor(
                acc_dtype,
                # Source-major, then lane/head, then (max,sum).  Remote STAS
                # computes this same flat offset explicitly; pinning strides
                # avoids relying on make_layout's default compact order.
                cute.make_layout(
                    (max_reduction_splits, warp_threads, 2),
                    stride=(warp_threads * 2, 2, 1),
                ),
                svector_align,
            )

        ##############################
        # Sync
        ##############################
        # Ensure visibility of cluster mbarriers
        if cutlass.const_expr(do_cga_red):
            cute.arch.cluster_wait()

        # Ensure visibility of local mbarrier inits and tmem alloc
        cute.arch.sync_threads()
        assert init_warp < (self.threads_per_cta // warp_threads), (
            f"used {init_warp + 1} init warps, {self.threads_per_cta // warp_threads} warps available"
        )

        ##############################
        # Tmem tensor allocation
        ##############################
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0

        # Keep converted MMA operands at TMEM column zero, matching NVIDIA's
        # mixed-input decode.  TMEM-source UMMA addressing is more restricted
        # than an accumulator fragment: placing this ring after S/L/O happened
        # to pass a cold launch but produced stale/NaN data on reuse.
        tAtV_cvt = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=vp_dtype),
            tAtV_cvt.layout,
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tAtV_cvt)

        tCtS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)

        tCtL = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtL.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtL)

        tCtO = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

        # nvfp4:SFK/SFQ 的 TMEM tensor(列偏移续接在 O 之后)
        tCtSFK = None
        tCtSFQ = None
        tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma_kq,
            (mma_tile_m, mma_tile_n, mma_tile_k),
            16,
            cute.slice_(smem_layout_sfk, (None, None, None, 0)),
        )
        tCtSFK = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=sf_dtype),
            tCtSFK_layout,
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtSFK)
        tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma_kq_sfb,
            (mma_tile_m, max(128, mma_tile_n), mma_tile_k),  # SFB 按 N=128 粒度(坑2)
            16,
            cute.slice_(smem_layout_sfq, (None, None, None, 0)),
        )
        tCtSFQ = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=sf_dtype),
            tCtSFQ_layout,
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtSFQ)

        assert tmem_offset <= tmem_alloc_cols, (
            f"\t{tmem_offset} tmem cols used, {tmem_alloc_cols} tmem cols allocated"
        )

        # ``setmaxnreg.*.sync.aligned`` is a warpgroup collective: all four
        # specialist warps in WG0 must execute the *same static instruction*.
        # Keeping one call inside each warp-specific dispatch is invalid even
        # when all four use the same target value.

        partial_o = None
        if cutlass.const_expr(do_cga_red):
            partial_o = cute.make_rmem_tensor((blk_tile_n,), acc_dtype)
            partial_o.fill(0)

        ##############################
        # Exit early
        ##############################
        if exit_early:
            if warpgroup_idx == correction_warpgroup_id:
                sM_final_nbar.arrive()
                sL_final_nbar.arrive()
            if cutlass.const_expr(quantize_o):
                if warp_idx == tma_kv_warp_id:
                    # No-split empty rows still own their packed output. Clear
                    # the logical payload/scales here so graph replay cannot
                    # expose the preceding request's values.
                    mO_bytes = cute.recast_tensor(mO, cutlass.Uint8)
                    logical_heads = self.logical_grouped_heads
                    if cutlass.const_expr(logical_heads is None):
                        logical_heads = grouped_heads
                    d_blocks = blk_tile_d // 16
                    hidden = heads_k * logical_heads * blk_tile_d
                    for block in cutlass.range(lane_idx, blk_tile_h * d_blocks, warp_threads):
                        group_head = coord_hg * blk_tile_h + block // d_blocks
                        d_block = block % d_blocks
                        if group_head < logical_heads:
                            for byte in cutlass.range_constexpr(8):
                                mO_bytes[d_block * 8 + byte, (group_head, 0),
                                         (coord_hk, coord_b)] = cutlass.Uint8(0)
                            k_idx = (coord_hk * logical_heads + group_head) * d_blocks + d_block
                            sf_offset = output_scale_offset(coord_b, k_idx, hidden)
                            mOSF[sf_offset] = sf_dtype(0)

        ##############################
        # TMA KV Dispatch
        ##############################
        elif warp_idx == tma_kv_warp_id:
            # 分页视图:页内 tile 一次成型,页号留作最后一个自由模,物理页号做
            # 动态坐标(-1 → TMA OOB 填零,上游同款机制)。
            # K:mK (page_size, d, (h,P));M 模 = page_size = mma_tile_m 恰好
            # 一页一 tile(page=128 定理),coord=0 吃掉 M rest。
            gK = cute.local_tile(
                mK, tiler=(mma_tile_m, mma_tile_k), coord=(0, None, (coord_hk, None))
            )  # (MMA_TILE_M, MMA_TILE_K, #TILE_DK, #P)
            # V:mV (d, page_size, (h,P));K 模 = page_size = mma_tile_k_vp(=128)
            # 恰好一页一 stage。
            gV = cute.local_tile(
                mV,
                tiler=(mma_tile_m_vp, mma_tile_k_vp),
                coord=(None, None, (coord_hk, None)),
            )  # (MMA_TILE_M, MMA_TILE_K, #TILE_DM, #TILE_SK=1, #P)

            tAgK = thrblk_mma_kq.partition_A(
                gK
            )  # (MMA, #MMA_M, #MMA_K, #TILE_DK, #P)
            tAgV = thrblk_mma_vp.partition_A(
                gV
            )  # (MMA, #MMA_M, #MMA_K, #TILE_DM, #TILE_SK, #P)

            # #TILE_SM=TILE_S/MMA_TILE_M, #TILE_HN=TILE_H/MMA_TILE_N, #TILE_DK=TILE_D/MMA_TILE_K
            # #TILE_DM=TILE_D/MMA_TILE_M, #TILE_HN=TILE_H/MMA_TILE_N, #TILE_SK=TILE_S/MMA_TILE_K
            #
            # Example with TILE_S=MMA_TILE_M=128, TILE_H=MMA_TILE_N=8, MMA_TILE_K=64, TILE_D=512
            # BMM1: MMA=128x8x16, #MMA_M=1, #MMA_N=1, #MMA_K=4,
            #       TILE_SM=1, #TILE_HN=1, #TILE_DK=8, #TILE_S=S/128
            # BMM2: MMA=128x8x16, #MMA_M=1, #MMA_N=1, #MMA_K=4,
            #       #TILE_DM=4, #TILE_HN=1, #TILE_SK=2, #TILE_S=S/128

            # TMA partition
            # (MMA, #MMA_M, #MMA_K, Rest...) -> (TMA, Rest...)
            tGSsK, tGSgK = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsK, 0, 3),
                gmem_tensor=cute.group_modes(tAgK, 0, 3),
            )

            tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsV, 0, 3),
                gmem_tensor=cute.group_modes(tAgV, 0, 3),
            )
            gVSFD = cute.local_tile(
                mVSF_d,
                tiler=(16, ((mma_tile_m_vp // 16) * mma_tile_k_vp) // 16),
                coord=(None, None, (coord_hk, None)),
            )
            tGSsVSFD, tGSgVSFD = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v_sf_d,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsVSFD, 0, 2),
                gmem_tensor=cute.group_modes(gVSFD, 0, 2),
            )
            gSFK = cute.local_tile(
                mSFK,
                tiler=(16, self.page_size * (self.headdim // 16) // 16),
                coord=(None, None, (coord_hk, None)),
            )
            tGSsSFK, tGSgSFK = cute.nvgpu.cpasync.tma_partition(
                tma_atom_sfk,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsSFKRaw, 0, 2),
                gmem_tensor=cute.group_modes(gSFK, 0, 2),
            )

            cute.arch.griddepcontrol_wait()

            # Sequence loop(pristine 单 FIFO;K/V 各自 expected_tx)
            # 页表直读(v0):每 tile 每子页一次 4B 均匀 ld.global(L2 驻留),
            # producer 流水线的超前量吸收其延迟;smem 环形预取留 M2.5。
            # 越界逻辑页 → phys=-1 → TMA OOB 填零(尾部掩码另行兜底)。
            # The compact generation-Q8 tactic has a dedicated conversion WG.
            # Starting raw-V two tiles behind K leaves that WG waiting on its
            # first TMA stage and becomes visible across multiple grid waves.
            # Keep the KQ/softmax prefetch schedule unchanged, but interleave
            # K0/V0 TMA issue from the first producer iteration.
            v_prefetch_iters = 0 if self.swap_ab_for_gen else prefetch_iters
            prefetch_tiles = v_prefetch_iters * kv_splits
            k_token_tma = True
            v_raw_token_tma = True
            for s in cutlass.range(kv_split_idx, prefetch_tiles + tiles_s, kv_splits):
                # Load K(sm 子循环即页循环:page_size = mma_tile_m)
                if s < tiles_s:
                    for sm in cutlass.range_constexpr(tiles_sm):
                        logical_k = s * pages_s + sm
                        phys_k = Int32(-1)
                        if logical_k < page_count:
                            phys_k = mBlockTables[coord_b, logical_k]
                        for dk in cutlass.range_constexpr(tiles_dk):
                            k_handle = k_producer.acquire_and_advance(k_token_tma)
                            k_token_tma = k_producer.try_acquire()
                            cute.copy(
                                tma_atom_k,
                                tGSgK[None, dk, phys_k],
                                tGSsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                            cute.copy(
                                tma_atom_sfk,
                                tGSgSFK[None, 0, 0, phys_k],
                                tGSsSFK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                # Load V(sk 子循环即页循环:page_size = mma_tile_k_vp)
                if s >= prefetch_tiles:
                    s_v = s - prefetch_tiles
                    for sk in cutlass.range_constexpr(tiles_sk):
                        logical_v = s_v * pages_s + sk
                        phys_v = Int32(-1)
                        if logical_v < page_count:
                            phys_v = mBlockTables[coord_b, logical_v]
                        for dm in cutlass.range_constexpr(tiles_dm):
                            v_handle = v_raw_producer.acquire_and_advance(
                                v_raw_token_tma
                            )
                            v_raw_token_tma = v_raw_producer.try_acquire()
                            cute.copy(
                                tma_atom_v,
                                tGSgV[None, dm, 0, phys_v],
                                tGSsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )
                            cute.copy(
                                tma_atom_v_sf_d,
                                tGSgVSFD[None, dm, 0, phys_v],
                                tGSsVSFD[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )

        ##############################
        # TMA QO Dispatch
        ##############################
        elif warp_idx == tma_qo_warp_id:
            # Slice and partition Q
            # (TILE_H, TILE_D)
            gQ = cute.local_tile(
                mQ,
                tiler=(blk_tile_hp, blk_tile_d),
                coord=(coord_hp, 0, coord_hb),
            )
            # Apply MMA tiler and MMA partition
            # (MMA_TILE_N, MMA_TILE_K, #TILE_HN=1, #TILE_DK)
            gQ_mma = cute.flat_divide(gQ, (mma_tile_n, mma_tile_k))
            gQ_mma = gQ_mma[None, None, 0, None]
            # (MMA, #MMA_N, #MMA_K, #TILE_DK)
            tBgQ = thrblk_mma_kq.partition_B(gQ_mma)
            # (TMA, #TILE_DK)
            tBsQ_tma, tBgQ_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgQ, 0, 3),
            )

            cute.arch.griddepcontrol_wait()

            # Load Q
            cute.copy(
                tma_atom_q,
                tBgQ_tma,
                tBsQ_tma,
                tma_bar_ptr=q_load_mbar,
            )
            # Load SFQ(nvfp4;与 Q 同一 mbar)
            gSFQ = cute.local_tile(
                mSFQ,
                tiler=(max(128, mma_tile_n), blk_tile_d),
                coord=(0, 0, coord_hb),
            )
            gSFQ_mma = cute.flat_divide(
                gSFQ, (max(128, mma_tile_n), mma_tile_k)
            )[None, None, 0, None]
            thrblk_sfb = tiled_mma_kq_sfb.get_slice(0)
            tBgSFQ = thrblk_sfb.partition_B(gSFQ_mma)
            tBsSFQ_tma, tBgSFQ_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_sfq,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsSFQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgSFQ, 0, 3),
            )
            tBsSFQ_tma = cute.filter_zeros(tBsSFQ_tma)
            tBgSFQ_tma = cute.filter_zeros(tBgSFQ_tma)
            cute.copy(
                tma_atom_sfq,
                tBgSFQ_tma,
                tBsSFQ_tma,
                tma_bar_ptr=sfq_mbar,
            )

            # Slice and partition O
            # (TILE_D, TILE_H)
            coord_b_partial = (
                kv_split_idx * batches + coord_b if do_partial_red else coord_b
            )
            gO = None
            if cutlass.const_expr(not quantize_o and not do_cga_red):
                gO = cute.local_tile(
                    mO,
                    tiler=(blk_tile_d, blk_tile_hp),
                    coord=(0, coord_hp, (coord_hk, coord_b_partial)),
                )
            sO_tma = gO_tma = None
            if cutlass.const_expr(not quantize_o and not do_cga_red):
                # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN=1)
                gO_mma = cute.flat_divide(gO, (mma_tile_m, mma_tile_n))
                gO_mma = gO_mma[None, None, None, 0]
                # (TMA, #TILE_DM)
                sO_tma, gO_tma = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_o,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(sO_mma, 0, 2),
                    gmem_tensor=cute.group_modes(gO_mma, 0, 2),
                )

            # Store O to gmem
            if cutlass.const_expr(do_cga_red):
                # CGA reducer threads own disjoint output fragments and store
                # them directly; this warp has no output transaction to issue.
                pass
            elif cutlass.const_expr(quantize_o):
                sO_final_nbar.arrive_and_wait()

                # There are H_tile * D/16 = 128 activation-scale blocks in a
                # 16-head, D=128 decode tile.  The TMA warp owns them in four
                # rounds: one lane loads 16 contiguous FP32 values, computes
                # the dynamic E4M3 scale, packs two E2M1 values per byte, and
                # writes the exact CUTLASS 128x4 scale offset expected by the
                # following O-projection GEMM.
                assert blk_tile_p == 1 and blk_tile_d % 16 == 0
                d_scale_blocks = blk_tile_d // 16
                scale_blocks = blk_tile_h * d_scale_blocks
                mO_bytes = cute.recast_tensor(mO, cutlass.Uint8)
                global_scale = Float32(mOGlobalScale[0])
                logical_grouped_heads = self.logical_grouped_heads
                if cutlass.const_expr(logical_grouped_heads is None):
                    logical_grouped_heads = grouped_heads
                hidden = heads_k * logical_grouped_heads * blk_tile_d
                row_idx = coord_b

                for block_idx in cutlass.range(
                    lane_idx, scale_blocks, warp_threads
                ):
                    local_head = block_idx // d_scale_blocks
                    d_block = block_idx % d_scale_blocks
                    group_head = coord_hg * blk_tile_h + local_head
                    if group_head < logical_grouped_heads:
                        values = cute.make_rmem_tensor((16,), Float32)
                        for value_idx in cutlass.range_constexpr(16):
                            value = Float32(
                                sO_mma[
                                    d_block * 16 + value_idx,
                                    local_head,
                                    0,
                                ]
                            )
                            values[value_idx] = value
                        packed_values, sf_value = quantize_nvfp4_block16(
                            values.load(), global_scale
                        )

                        packed_fp4 = cute.make_rmem_tensor(
                            (16,), cutlass.Float4E2M1FN
                        )
                        packed_fp4.store(packed_values)
                        packed_bytes = cute.recast_tensor(
                            packed_fp4, cutlass.Uint8
                        )
                        for byte_idx in cutlass.range_constexpr(8):
                            mO_bytes[
                                d_block * 8 + byte_idx,
                                (group_head, 0),
                                (coord_hk, coord_b),
                            ] = packed_bytes[byte_idx]

                        global_head = (
                            coord_hk * logical_grouped_heads + group_head
                        )
                        k_idx = global_head * d_scale_blocks + d_block
                        sf_offset = output_scale_offset(row_idx, k_idx, hidden)
                        mOSF[sf_offset] = sf_value
            else:
                sO_final_nbar.arrive_and_wait()
                cute.copy(tma_atom_o, sO_tma, gO_tma)
                if cutlass.const_expr(do_gmem_fused_red):
                    # The completion counter may not be published until this
                    # CTA's partial numerator is globally visible.
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=False)

        ##############################
        # V Convert Dispatch (LDSM.U4 -> QMUL4 -> STTM)
        ##############################
        elif warpgroup_idx == conversion_warpgroup_id:

            # Match the copy geometry used by NVIDIA's mixed-input decode.
            # LDSM transposes a D-major packed FP4 tile into eight nibble
            # containers per lane; QMUL4 applies the independently packed
            # D-scales; STTM writes the E4M3 fragment into the exact TMEM-A
            # layout consumed by the VP MMA.
            mma_k_bits = mma_tile_k_vp * vp_dtype.width
            # Use the same 16x128b TMEM store atom as NVIDIA's
            # mixed-input decode.  16x256b happens to cover the same tile
            # on a cold allocation, but it creates a different TMEM-A
            # lane/column ownership contract and is not the store shape
            # paired with this VP MMA operand in the reference pipeline.
            tmem_store_atom_v = cute.make_copy_atom(
                tcgen05.St16x128bOp(
                    tcgen05.Repetition(mma_k_bits // 128)
                ),
                vp_dtype,
            )
            smem_load_atom_v = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix16x16x8bOp(
                    transpose=True,
                    num_matrices=2,
                    unpack_bits=4,
                ),
                cutlass.Int8,
            )
            tmem_store_v = tcgen05.make_tmem_copy(
                tmem_store_atom_v, tAtV_cvt[mma_dice + (0,)]
            )
            thr_store_v = tmem_store_v.get_slice(warpgroup_tidx)
            tVrV_cvt_shape = thr_store_v.partition_S(tAtV_cvt).shape[:-1]
            tVtV_cvt = thr_store_v.partition_D(tAtV_cvt)


            smem_load_v = cute.make_tiled_copy_S(
                smem_load_atom_v, tmem_store_v
            )
            thr_load_v = smem_load_v.get_slice(warpgroup_tidx)
            tVsV = thr_load_v.partition_S(tAsVRaw)
            # CuTe 4.6 conservatively drops this partition to 8-byte pointer
            # alignment even though the NVIDIA LDSM tiling maps every row
            # start onto the 16-byte-aligned SW128 carrier allocation. Restore
            # the proven atom precondition so the verifier accepts LDSM.U4.
            tVsV = cute.make_tensor(tVsV.iterator.align(16), tVsV.layout)
            tVrV_shape = thr_load_v.partition_D(tAsVRaw).shape[:-1]

            # Reinterpret the TRT 4-token-interleaved scale tile as a
            # broadcast D tensor. This is the scale analogue of NVIDIA's
            # ((block_scaledim, scaledim), token) layout, with the physical
            # token interleave preserved in the strides.
            _sfb = mma_tile_m_vp // 16
            sVS = cute.make_tensor(
                tAsVSFD.iterator,
                cute.make_layout(
                    ((16, _sfb), (4, mma_tile_k_vp // 4), kv_stages),
                    stride=((0, 4), (1, 4 * _sfb), _sfb * mma_tile_k_vp),
                ),
            )
            sVS_mma = cute.group_modes(
                cute.flat_divide(sVS, (mma_tile_m_vp, mma_tile_k_vp)),
                2,
                4,
            )
            tAsVS = thrblk_mma_vp.partition_A(sVS_mma)
            tVsVS = thr_load_v.partition_D(tAsVS)

            v_token = v_raw_consumer.try_wait()
            cvt_token = v_ready_producer.try_acquire()
            for _s in cutlass.range(iters_s):
                for _sk in cutlass.range_constexpr(tiles_sk):
                    for _dm in cutlass.range_constexpr(tiles_dm):
                        v_handle = v_raw_consumer.wait_and_advance(v_token)
                        cvt_handle = v_ready_producer.acquire_and_advance(
                            cvt_token
                        )

                        tVrVS = cute.make_rmem_tensor_like(
                            tVsVS[cpy_dice + (None, 0)]
                        )
                        cute.autovec_copy(
                            tVsVS[cpy_dice + (None, v_handle.index)],
                            tVrVS,
                        )
                        tVrV = cute.make_rmem_tensor(
                            tVrV_shape, cutlass.Int8
                        )
                        tVrV_cvt = cute.make_rmem_tensor(
                            tVrV_cvt_shape, vp_dtype
                        )
                        # The scale partition contains zero-stride
                        # broadcasts.  Materialize it into the compact
                        # register shape paired with the LDSM fragment
                        # before slicing eight-value QMUL4 vectors.
                        tVrVS_expanded = cute.make_rmem_tensor(
                            tVrV_shape, sf_dtype
                        )
                        cute.autovec_copy(tVrVS, tVrVS_expanded)
                        cute.copy(
                            thr_load_v,
                            tVsV[cpy_dice + (v_handle.index,)],
                            tVrV,
                        )
                        cute.arch.fence_view_async_shared()
                        v_handle.release()

                        _frag_values = cute.size(tVrV_shape)
                        assert _frag_values % 8 == 0
                        _src_flat = cute.make_tensor(
                            tVrV.iterator,
                            cute.make_layout(_frag_values),
                        )
                        _sf_flat = cute.make_tensor(
                            tVrVS_expanded.iterator,
                            cute.make_layout(_frag_values),
                        )
                        _dst_flat = cute.make_tensor(
                            tVrV_cvt.iterator,
                            cute.make_layout(_frag_values),
                        )
                        for _block in cutlass.range_constexpr(
                            _frag_values // 8
                        ):
                            _src8 = cute.local_tile(
                                _src_flat, (8,), (_block,)
                            )
                            _sf8 = cute.local_tile(
                                _sf_flat, (8,), (_block,)
                            )
                            _dst8 = cute.local_tile(
                                _dst_flat, (8,), (_block,)
                            )
                            _dst8.store(
                                dequantize_nvfp4x8_ldsm(_src8, _sf8)
                            )
                        cute.copy(
                            thr_store_v,
                            tVrV_cvt,
                            tVtV_cvt[cpy_dice + (cvt_handle.index,)],
                        )
                        cute.arch.fence_view_async_tmem_store()
                        cvt_handle.commit()
                        v_token = v_raw_consumer.try_wait()
                        cvt_token = v_ready_producer.try_acquire()

        ##############################
        # MMA KQ (BMM1) Dispatch
        ##############################
        elif warp_idx == mma_kq_warp_id:
            # Setup mma descriptors
            tAsK_desc = thrblk_mma_kq.make_fragment_A(tAsK)
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)

            copy_sfk, tCsSFK_s2t, tCtSFK_s2t = _sf_s2t_partition(
                sf_dtype, tAsSFK, tCtSFK
            )
            copy_sfq, tCsSFQ_s2t, tCtSFQ_s2t = _sf_s2t_partition(
                sf_dtype, tBsSFQ, tCtSFQ
            )

            # Wait for Q
            cute.arch.mbarrier_wait(q_load_mbar, phase=0)

            # SFQ → TMEM(一次;独立 mbar 正确同步)
            cute.arch.mbarrier_wait(sfq_mbar, phase=0)
            cute.copy(
                copy_sfq,
                tCsSFQ_s2t[None, None, None, None, 0],
                tCtSFQ_s2t,
            )

            # Sequence loop
            for s in cutlass.range(iters_s):
                s_token = s_producer.try_acquire()

                k_token = k_consumer.try_wait()

                s_handle = s_producer.acquire_and_advance(s_token)
                for sm in cutlass.range_constexpr(tiles_sm):
                    tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                    for dk in cutlass.range_constexpr(tiles_dk):
                        k_handle = k_consumer.wait_and_advance(k_token)
                        # SFK → TMEM(每 K stage)
                        sf_stage = k_handle.index
                        for _i in cutlass.range_constexpr(
                            mma_tile_m // warp_threads
                        ):
                            _row = lane_idx + _i * warp_threads
                            _value = tAsSFKRawU64[_row, sf_stage]
                            _atom_lo = (
                                (_row % 32) * 16
                                + (_row // 32) * 4
                            )
                            tAsSFKAtomU32[
                                _atom_lo // 4, sf_stage
                            ] = cutlass.Uint32(_value)
                            tAsSFKAtomU32[
                                (_atom_lo + 512) // 4, sf_stage
                            ] = cutlass.Uint32(_value >> 32)
                        cute.arch.fence_view_async_shared()
                        cute.copy(
                            copy_sfk,
                            tCsSFK_s2t[
                                None,
                                None,
                                None,
                                None,
                                sf_stage,
                            ],
                            tCtSFK_s2t,
                        )
                        is_last_iter = sm == tiles_sm - 1 and dk == tiles_dk - 1
                        if not is_last_iter:
                            k_token = k_consumer.try_wait()

                        mmas_k = cute.size(tAsK.shape[2])
                        for mma_k in cutlass.range_constexpr(mmas_k):
                            tiled_mma_kq.set(
                                tcgen05.Field.SFA,
                                tCtSFK[None, None, mma_k].iterator,
                            )
                            tiled_mma_kq.set(
                                tcgen05.Field.SFB,
                                tCtSFQ[None, None, mma_k].iterator,
                            )
                            cute.gemm(
                                tiled_mma_kq,
                                tCtS[mma_dice + (sm, s_handle.index)],
                                tAsK_desc[None, None, mma_k, k_handle.index],
                                tBsQ_desc[None, None, mma_k, dk],
                                tCtS[mma_dice + (sm, s_handle.index)],
                            )
                            if dk == 0 and mma_k == 0:
                                tiled_mma_kq.set(
                                    tcgen05.Field.ACCUMULATE, True
                                )
                        k_handle.release()
                s_handle.commit()

            # Tail loop

        ##############################
        # MMA VP (BMM2) Dispatch
        ##############################
        elif warp_idx == mma_vp_warp_id:
            # Setup mma descriptors
            # 坑26 候选:set() 必须在 make_fragment 之后(KQ 的顺序)——
            # set 先行会破坏 fp4 desc 原型推导(probe4:双侧 mma_k 偏移失效)
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)
            tiled_mma_vp.set(
                tcgen05.Field.ACCUMULATE,
                True,
            )

            o_init_nbar.arrive_and_wait()

            # Prefetch loop

            (
                tiled_mma_vp,
                p_consumer,
                o_producer,
                v_ready_consumer,
            ) = self.mma_vp_qmul4_tmem_step(
                True,
                tiled_mma_vp,
                tAtV_cvt,
                tBsP_desc,
                tCtO,
                p_consumer,
                o_producer,
                v_ready_consumer,
                tiles_sk,
                tiles_dm,
            )
            if iters_s > 1:
                (
                    tiled_mma_vp,
                    p_consumer,
                    o_producer,
                    v_ready_consumer,
                ) = self.mma_vp_qmul4_tmem_step(
                    True,
                    tiled_mma_vp,
                    tAtV_cvt,
                    tBsP_desc,
                    tCtO,
                    p_consumer,
                    o_producer,
                    v_ready_consumer,
                    tiles_sk,
                    tiles_dm,
                )
            for _s in cutlass.range(2, iters_s):
                (
                    tiled_mma_vp,
                    p_consumer,
                    o_producer,
                    v_ready_consumer,
                ) = self.mma_vp_qmul4_tmem_step(
                        False,
                        tiled_mma_vp,
                    tAtV_cvt,
                    tBsP_desc,
                    tCtO,
                    p_consumer,
                    o_producer,
                    v_ready_consumer,
                    tiles_sk,
                    tiles_dm,
                )

            # Sequence loop

            # Wait for signal to dealloc tmem, then dealloc
            if iters_s == 1:
                # Epilogue still reads the empty buffer
                o_producer.commit()
                o_producer.advance()
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        ##############################
        # Softmax Dispatch
        ##############################
        elif warpgroup_idx in softmax_warpgroup_ids:
            # Free registers

            # Initialize for dual warpgroups
            softmax_phase = (
                warpgroup_idx - softmax_warpgroup_ids[0]
            ) % softmax_warpgroups
            softmax_tL_producer_nbar = with_phase(
                tL_producer_nbar, softmax_phase
            )
            softmax_tL_consumer_nbar = with_phase(
                tL_consumer_nbar, softmax_phase
            )
            sM_acquire_nbar = with_phase(sM_mutex_nbar, softmax_phase)
            sM_release_nbar = with_phase(sM_mutex_nbar, softmax_phase ^ 1)
            if softmax_phase == 1:
                s_consumer.advance()
                p_producer.advance()
                sM_release_nbar.arrive()
                if iters_s == 1:
                    softmax_tL_producer_nbar.arrive()
            elif cutlass.const_expr(softmax_warpgroups == 1):
                # The dual-WG scheduler assigns the otherwise-empty phase-1
                # L slot to softmax WG1 when there is only one sequence tile.
                # Preserve that publication with the single-WG scheduler so
                # correction's two-slot tail load cannot wait forever.  The
                # zero payload itself is initialized and fenced by correction
                # before it consumes this barrier phase.
                if iters_s == 1:
                    with_phase(tL_producer_nbar, 1).arrive()

            # Construct copy atom for S
            tmem_repeat_op_s = blk_tile_n
            if cutlass.const_expr(mma_tile_n == blk_tile_n and tiles_sm in (2, 4)):
                tmem_repeat_op_s *= tiles_sm
            # 坑19(修订):16-dp 原子(1线程=2行,距8)对 decode 形状可行 ——
            # Repetition 单位随原子位宽走:32x32b 每 rep=1 个 f32 列,16x256b 每
            # rep=8 列,须 //8(初版误诊"N 太窄"即漏除的 8 倍)。make_tmem_copy 已
            # 验证通过。但它重划所有权几何(线程从"1 token 整行"变"4 token × 部分
            # 列"),掩码/colmax/colsum/L 写回全须按新几何重写 + P/V π 重排,级联
            # 数百行 —— 正确性优先,decode 先走 v1(shfl 拼字节),v2 留待数值基线
            # 建立后作为独立实验(R2 加宽 N 后收益更大)。
            tmem_load_op_s = tcgen05.Ld32x32bOp(
                tcgen05.Repetition(tmem_repeat_op_s)
            )
            tmem_load_atom_s = cute.make_copy_atom(tmem_load_op_s, acc_dtype)
            # Tile atom and slice
            tCtS_stage = tCtS[mma_dice + (None, 0)]
            tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom_s, tCtS_stage)
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)
            # Partition S and P
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #CPY_SM, stages)
            tStS = thr_load_s.partition_S(tCtS)
            tSsP = thr_load_s.partition_D(tCsP)
            # Slice unused modes
            tStS = tStS[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, s_stages)
            tSsP = tSsP[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, p_stages)

            # Construct copy atom for L
            tmem_repeat_op_l = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_l = tcgen05.Ld32x32bOp(tmem_repeat_op_l)
            tmem_store_op_l = tcgen05.St32x32bOp(tmem_repeat_op_l)
            tmem_load_atom_l = cute.make_copy_atom(tmem_load_op_l, acc_dtype)
            tmem_store_atom_l = cute.make_copy_atom(tmem_store_op_l, acc_dtype)
            # Tile atom and slice
            tCtL_phase = tCtL[mma_dice + (softmax_phase,)]
            tmem_load_l = tcgen05.make_tmem_copy(tmem_load_atom_l, tCtL_phase)
            thr_load_l = tmem_load_l.get_slice(warpgroup_tidx)
            # Partition L
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N)
            tStL = thr_load_l.partition_S(
                tCtL
                if cutlass.const_expr(softmax_warpgroups == 1)
                else tCtL_phase
            )
            tSrL_shape = thr_load_l.partition_D(tCtL_phase).shape
            # Slice unused modes
            # (CPY,)
            tStL = (
                tStL[None, 0, 0, 0, None]
                if cutlass.const_expr(softmax_warpgroups == 1)
                else tStL[None, 0, 0, 0]
            )
            tSrL_shape = tSrL_shape[:1]

            # Single-token decode only masks the final, partial KV tile.
            masked_iters_s = masked_start_s = masked_coord_s = 0
            is_last_split = kv_split_idx == (tiles_s - 1) % kv_splits
            is_last_phase = softmax_phase == (iters_s - 1) % softmax_warpgroups
            if seqlen % blk_tile_s != 0:
                if is_last_split and is_last_phase:
                    masked_iters_s = 1
                    masked_coord_s = tiles_s - 1

            # Sequence loop
            num_loops = 2
            # A single softmax WG owns consecutive tiles. Carry the previous
            # published maximum in registers: reloading sM while other warps
            # atomically update it makes the rescale threshold order-dependent.
            previous_max_lane = -Float32.inf
            for loop in cutlass.range_constexpr(num_loops):
                is_masked_loop = loop == 1
                for s in cutlass.range(
                    masked_start_s if is_masked_loop else softmax_phase,
                    masked_iters_s if is_masked_loop else (iters_s - masked_iters_s),
                    softmax_warpgroups,
                ):
                    # With one WG the masked loop restarts its local ``s`` at
                    # zero, but correction still consumes the global 0/1 L
                    # ring phase. Reconstruct that global iteration index.
                    logical_iter_s = s
                    if cutlass.const_expr(
                        is_masked_loop and softmax_warpgroups == 1
                    ):
                        logical_iter_s = iters_s - masked_iters_s + s
                    s_token = s_consumer.try_wait()
                    p_token = p_producer.try_acquire()

                    # Load S from tmem and notify BMM1
                    s_handle = s_consumer.wait_and_advance(s_token)
                    tStS_s = tStS[None, None, s_handle.index]
                    tSrS_s = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                    cute.copy(tmem_load_atom_s, tStS_s, tSrS_s)
                    cute.arch.fence_view_async_tmem_load()
                    s_handle.release()

                    # Apply causal mask
                    if cutlass.const_expr(is_masked_loop):
                        masked = cute.make_rmem_tensor(
                            (blk_tile_h, blk_tile_p, tiles_sm), acc_dtype
                        )
                        masked.store(tSrS_s.load().reshape(masked.shape))
                        offset_s = (masked_coord_s + s) * blk_tile_s + warpgroup_tidx
                        offset_p = seqlen - prediction + coord_p * blk_tile_p
                        for sm in cutlass.range_constexpr(tiles_sm):
                            for p in cutlass.range_constexpr(blk_tile_p):
                                masked_p = masked[None, p, sm]
                                is_oob = offset_s + sm * mma_tile_m > offset_p + p
                                mask = -Float32.inf if is_oob else Float32(0)
                                masked_p.store(masked_p.load() + mask)
                        scores = masked.load().reshape((blk_tile_n, tiles_sm))
                    else:
                        scores = tSrS_s.load().reshape((blk_tile_n, tiles_sm))

                    # Q packs [base; residue]. Add their scores before softmax
                    # and duplicate into both physical halves for the MMA layout.
                    # Final output stores only logical heads, excluding duplicate
                    # residue rows and padding. Masking precedes the fold.
                    _nf = blk_tile_n // 2
                    _s3 = scores.reshape((_nf, 2, tiles_sm))
                    _fold = _s3[None, 0, None] + _s3[None, 1, None]
                    _dup = cute.make_rmem_tensor((_nf, 2, tiles_sm), acc_dtype)
                    _dup_b = _dup[None, 0, None]
                    _dup_b.store(_fold)
                    _dup_r = _dup[None, 1, None]
                    _dup_r.store(_fold)
                    scores = _dup.load().reshape((blk_tile_n, tiles_sm))

                    rM = cute.make_rmem_tensor_like(sM)
                    rM.store(
                        scores.reduce(
                            cute.ReductionOp.MAX,
                            init_val=-Float32.inf,
                            reduction_profile=(None, 0),
                        )
                    )

                    # Reduce colmax in warp RF
                    rM_lane = Float32(0)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        rM[n] = warp_fmax(rM[n])  # warp reduction
                        # Avoid dynamic register indexing (creates spills)
                        if n == lane_idx:
                            rM_lane = rM[n]
                    rM_lane *= scale_s_log2_e  # apply scale

                    # Reduce colmax in smem
                    if cutlass.const_expr(softmax_warpgroups > 1):
                        sM_acquire_nbar.arrive_and_wait()
                    sM_consumer_nbar.arrive_and_wait()
                    if lane_store_max:
                        if cutlass.const_expr(softmax_warpgroups == 1):
                            # TRT-style FP8-P rescale suppression.  Keep
                            # the old normalization origin while the new
                            # tile max is at most 8 log2 units higher.
                            # exp2(delta) <= 256 remains below E4M3 max
                            # 448, so numerator and denominator retain a
                            # common scale without saturating P.
                            if rM_lane - previous_max_lane > Float32(8):
                                smem_fmax(
                                    sM.iterator + sM.layout(lane_idx),
                                    rM_lane,
                                )
                        else:
                            # Alternating WGs cannot use a WG-local prior
                            # maximum. Use ordinary atomic max until a
                            # collective suppression protocol is available.
                            smem_fmax(
                                sM.iterator + sM.layout(lane_idx), rM_lane
                            )

                    # Wait for colmax and load
                    sM_producer_nbar.arrive_and_wait()
                    colmax = sM.load()
                    if cutlass.const_expr(softmax_warpgroups == 1):
                        if lane_store_max:
                            previous_max_lane = sM[lane_idx]
                    if cutlass.const_expr(softmax_warpgroups > 1):
                        sM_release_nbar.arrive()

                    # Handle if we never saw any in-bounds values

                    # Wait for empty P buffer
                    # Here so we can interleave ex2 with convert ops
                    p_handle = p_producer.acquire_and_advance(p_token)
                    tSsP_s = tSsP[None, None, p_handle.index]

                    # Compute online softmax
                    probs = exp2(scale_s_log2_e * scores - colmax)

                    # Store P to smem and notify BMM2
                    _p_quant = probs.to(mma_dtype).reshape(tSsP_s.shape)
                    tSsP_s.store(_p_quant)
                    cute.arch.fence_view_async_shared()
                    p_handle.commit()

                    # Accumulate per-thread colsum
                    tSrL = cute.make_rmem_tensor(tSrL_shape, acc_dtype)
                    colsum = probs[None, 0]
                    for sm in cutlass.range_constexpr(1, tiles_sm, 1):
                        colsum += probs[None, sm]
                    tSrL.store(colsum.reshape(tSrL.shape))

                    # Store per-thread colsum to tmem
                    if cutlass.const_expr(softmax_warpgroups == 1):
                        l_phase = logical_iter_s % o_stages
                        iter_tL_consumer_nbar = with_phase(
                            tL_consumer_nbar, l_phase
                        )
                        iter_tL_producer_nbar = with_phase(
                            tL_producer_nbar, l_phase
                        )
                        tStL_iter = tStL[None, l_phase]
                    else:
                        iter_tL_consumer_nbar = softmax_tL_consumer_nbar
                        iter_tL_producer_nbar = softmax_tL_producer_nbar
                        tStL_iter = tStL
                    iter_tL_consumer_nbar.arrive_and_wait()
                    cute.copy(tmem_store_atom_l, tSrL, tStL_iter)
                    cute.arch.fence_view_async_tmem_store()
                    iter_tL_producer_nbar.arrive()

                    # Advance again for dual warpgroups
                    if cutlass.const_expr(softmax_warpgroups > 1):
                        s_consumer.advance()
                        p_producer.advance()

        ##############################
        # Correction Dispatch
        ##############################
        elif warpgroup_idx == correction_warpgroup_id:
            # Alloc registers

            # Select copy atoms for O and L
            tmem_repeat_op_o = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_o = tcgen05.Ld32x32bOp(tmem_repeat_op_o)
            tmem_store_op_o = tcgen05.St32x32bOp(tmem_repeat_op_o)
            tmem_load_atom_o = cute.make_copy_atom(tmem_load_op_o, acc_dtype)
            tmem_store_atom_o = cute.make_copy_atom(tmem_store_op_o, acc_dtype)
            # Tile atoms and slice
            tCtO_dm = tCtO[mma_dice + (0, 0)]
            tmem_load_o = tcgen05.make_tmem_copy(tmem_load_atom_o, tCtO_dm)
            thr_load_o = tmem_load_o.get_slice(warpgroup_tidx)
            # Partition O and L
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, o_stages)
            tOtO = thr_load_o.partition_S(tCtO)
            tOsO = thr_load_o.partition_D(tCsO)
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, o_stages)
            tOtL = thr_load_o.partition_S(tCtL)
            # Slice unused modes
            tOtO = tOtO[None, 0, 0, 0, None, None]  # (CPY, #TILE_DM, o_stages)
            tOsO = tOsO[None, 0, 0, 0, None]  # (CPY, #TILE_DM)
            tOtL = tOtL[None, 0, 0, 0, None]  # (CPY, o_stages)

            correction_broadcast_atom = None
            if cutlass.const_expr(self.smem_correction_broadcast):
                correction_broadcast_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    acc_dtype,
                    num_bits_per_copy=128,
                )

            def materialize_correction(
                correction_lane,
                should_rescale,
                sL=sL,
                warpgroup_widx=warpgroup_widx,
                lane_idx=lane_idx,
                lane_store_max=lane_store_max,
                correction_broadcast_atom=correction_broadcast_atom,
                blk_tile_n=blk_tile_n,
                smem_correction_broadcast=self.smem_correction_broadcast,
            ):
                correction = cute.make_rmem_tensor((blk_tile_n,), Float32)
                correction.fill(Float32(1))
                if should_rescale:
                    if cutlass.const_expr(smem_correction_broadcast):
                        # Each correction warp owns one contiguous N-float sL
                        # column (N=8/16). The first N lanes publish one factor
                        # each; the warp-local barrier makes those stores
                        # visible before every lane reloads the vectorized row.
                        if lane_store_max:
                            sL[lane_idx, warpgroup_widx] = correction_lane
                        cute.arch.sync_warp()
                        cute.copy(
                            correction_broadcast_atom,
                            sL[None, warpgroup_widx],
                            correction,
                        )
                    else:
                        for n in cutlass.range_constexpr(blk_tile_n):
                            correction[n] = cute.arch.shuffle_sync(
                                correction_lane, n
                            )
                return correction.load()

            # colsum load helper
            def colsum_load(
                phase,
                blk_tile_n=blk_tile_n,
                tOtL=tOtL,
                tOrO_shape=tOsO.shape[:1],
                tmem_load_atom_o=tmem_load_atom_o,
                tL_producer_nbar=tL_producer_nbar,
                tL_consumer_nbar=tL_consumer_nbar,
            ):
                with_phase(tL_producer_nbar, phase).arrive_and_wait()
                tOtL_s = tOtL[None, phase]
                tOrL_s = cute.make_rmem_tensor(tOrO_shape, Float32)
                cute.copy(tmem_load_atom_o, tOtL_s, tOrL_s)
                cute.arch.fence_view_async_tmem_load()
                with_phase(tL_consumer_nbar, phase).arrive()
                return tOrL_s.load().reshape(blk_tile_n)

            # Initialize O and colsum
            tOrO = cute.make_rmem_tensor(tOsO.shape, acc_dtype)
            tOrO.fill(Float32(0))
            for phase in cutlass.range_constexpr(o_stages):
                cute.copy(tmem_store_atom_o, tOrO, tOtO[None, None, phase])
            cute.copy(tmem_store_atom_o, tOrO[None, 0], tOtL[None, 1])
            cute.arch.fence_view_async_tmem_store()
            o_init_nbar.arrive_and_wait()

            # Initialize consumer barriers
            sM_consumer_nbar.arrive()
            for phase in cutlass.range_constexpr(o_stages):
                with_phase(tL_consumer_nbar, phase).arrive()

            # Initialize colsum in RF
            colsum_p = cute.make_rmem_tensor((blk_tile_n, o_stages), Float32)
            colsum_0, colsum_1 = colsum_p[None, 0], colsum_p[None, 1]
            colsum_p.fill(Float32(0))

            # Load colmax of s-2, s-1
            sM_lane_prev_prev = sM_lane_prev = -Float32.inf
            for s in cutlass.range_constexpr(o_stages):
                sM_lane_prev_prev = sM_lane_prev
                if not (s == 1 and iters_s == 1):
                    sM_producer_nbar.arrive_and_wait()
                    if lane_store_max:
                        sM_lane_prev = sM[lane_idx]
                    sM_consumer_nbar.arrive()

            # Sequence loop
            phase = 0
            for s in cutlass.range(iters_s - o_stages, unroll=o_stages):
                # Load colsum of s-2
                colsum_s = colsum_load(phase)

                # Load colmax of s
                sM_producer_nbar.arrive_and_wait()
                if s == iters_s - o_stages - 1:  # Notify for final colmax
                    sM_final_nbar.arrive()
                # Preserve the sentinel on lanes that do not own a head;
                # otherwise their ballot would fabricate a max update.
                sM_lane = sM_lane_prev_prev
                if lane_store_max:
                    sM_lane = sM[lane_idx]
                sM_consumer_nbar.arrive()

                # Wait for O of s-2
                # Here so we can interleave shuffle_sync with correction muls
                o_token = o_consumer.try_wait()
                o_handle = o_consumer.wait_and_advance(o_token)

                # Compute correction of s-2
                should_rescale = (
                    cute.arch.vote_ballot_sync(
                        sM_lane_prev_prev != sM_lane
                    ) != 0
                )
                correction_lane = exp2(sM_lane_prev_prev - sM_lane)
                correction = materialize_correction(
                    correction_lane, should_rescale
                )

                # Correct O of s-2 and notify MMA VP
                correction_o = correction.reshape(tOsO.shape[:1])
                if should_rescale:
                    for dm in cutlass.range_constexpr(tiles_dm):
                        tOtO_dm = tOtO[None, dm, phase]
                        tOrO_dm = cute.make_rmem_tensor(tOsO.shape[:1], acc_dtype)
                        cute.copy(tmem_load_atom_o, tOtO_dm, tOrO_dm)
                        tOrO_dm.store(correction_o * tOrO_dm.load())
                        cute.copy(tmem_store_atom_o, tOrO_dm, tOtO_dm)
                    cute.arch.fence_view_async_tmem_store()
                o_handle.release()

                # Correct and accumulate colsum of s-2
                if should_rescale:
                    colsum_s *= correction
                    if phase == 0:
                        colsum_0.store(correction * colsum_0.load() + colsum_s)
                    elif phase == 1:
                        colsum_1.store(correction * colsum_1.load() + colsum_s)
                else:
                    if phase == 0:
                        colsum_0.store(colsum_0.load() + colsum_s)
                    elif phase == 1:
                        colsum_1.store(colsum_1.load() + colsum_s)

                # Advance loop
                sM_lane_prev_prev, sM_lane_prev = sM_lane_prev, sM_lane
                phase ^= 1


            # Notify for final colmax if we didn't enter loop
            if iters_s <= o_stages:
                sM_final_nbar.arrive()

            # Handle if we never saw any in-bounds values

            # Compute correction of s-1
            should_rescale_tail = (
                cute.arch.vote_ballot_sync(
                    sM_lane_prev_prev != sM_lane_prev
                ) != 0
            )
            correction_lane = exp2(sM_lane_prev_prev - sM_lane_prev)
            correction = materialize_correction(
                correction_lane, should_rescale_tail
            )

            # Correct and accumulate final colsum
            tail_phase = iters_s % o_stages
            for phase in cutlass.range_constexpr(o_stages):
                if tail_phase == phase:
                    # Accumulate in thread RF
                    colsum_prev = colsum_load(phase)
                    colsum_final = colsum_load(phase ^ 1)
                    colsum_prev += colsum_p[None, phase].load()
                    colsum_final += colsum_p[None, phase ^ 1].load()
                    colsum_final += correction * colsum_prev
                    # Reduce colsum in warp RF
                    rL_lane = Float32(0.0)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        rL_n = cute.arch.warp_reduction_sum(colsum_final[n])
                        if n == lane_idx:
                            rL_lane = rL_n
                    # Store partial colsum in smem and notify
                    if lane_store_max:
                        sL[lane_idx, warpgroup_widx] = rL_lane

            sL_final_nbar.arrive_and_wait()

            # Load O of s-1, s
            tOrO_tail = cute.make_rmem_tensor((*tOsO.shape, o_stages), acc_dtype)
            for s in cutlass.range_constexpr(o_stages):
                phase_s = tail_phase ^ s
                tOtO_phase = tOtO[None, None, phase_s]
                tOrO_tail_s = tOrO_tail[None, None, s]
                o_handle = o_consumer.wait_and_advance()
                cute.copy(tmem_load_atom_o, tOtO_phase, tOrO_tail_s)
                cute.arch.fence_view_async_tmem_load()
                o_handle.release()  # Final release signals tmem dealloc

            tOrO_prev = tOrO_tail[None, None, 0].load()
            tOrO_final = tOrO_tail[None, None, 1].load()

            # Correct and accumulate output
            output_prev = tOrO_prev.reshape((blk_tile_n, tiles_dm))
            output_final = tOrO_final.reshape((blk_tile_n, tiles_dm))
            output_final += correction * output_prev

            if cutlass.const_expr(do_cga_red):
                partial_o.store(output_final.reshape((blk_tile_n,)))
            else:
                # No-split normalizes locally; split reducers merge their partials.
                if cutlass.const_expr(do_none_red):
                    sM_final_nbar.arrive_and_wait()
                    normalization = sM.load()
                    output_final *= normalization

                # Store O to smem and notify the TMA warp.
                tOsO.store(output_final.to(o_dtype).reshape(tOsO.shape))
                cute.arch.fence_view_async_shared()
                sO_final_nbar.arrive()

        # Empty splits participate too: their initialized zero numerator and
        # (-inf, 0) statistics must reach every peer's transaction barrier.
        if cutlass.const_expr(do_cga_red):
            if warpgroup_idx == correction_warpgroup_id:
                # The reduction warp has collapsed the four softmax-warp sums
                # into sL[:,0].  Every target output slice needs the same
                # per-head stats, so broadcast the tiny stats vectors while O
                # itself is partitioned by physical correction-thread id.
                cga_stats_ready_nbar.arrive_and_wait()

                dsmem_o_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyDsmemStoreOp(),
                    acc_dtype,
                    num_bits_per_copy=128,
                )
                dsmem_stats_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyDsmemStoreOp(),
                    acc_dtype,
                )

                owner_cta = warpgroup_tidx % kv_splits
                owner_thread = warpgroup_tidx // kv_splits
                owner_threads = warpgroup_threads // kv_splits

                partial_o_offset = (
                    (kv_split_idx * owner_threads + owner_thread)
                    * blk_tile_n
                )
                remote_partial_mbar = cute.arch.map_dsmem_ptr(
                    cga_partial_mbar, owner_cta
                )
                remote_o = cute.make_tensor(
                    cute.arch.map_dsmem_ptr(
                        sCgaO.iterator + partial_o_offset, owner_cta
                    ).align(16),
                    cute.make_layout((blk_tile_n,)),
                )
                cute.copy(
                    dsmem_o_atom,
                    partial_o,
                    remote_o,
                    mbar_ptr=remote_partial_mbar,
                )

                # One physical warp broadcasts 16 float2(max,sum) entries to
                # every reducer CTA.  The O transaction count stays exactly
                # one full output tile per target, independent of split count.
                if warpgroup_widx == 0:
                    local_stat = cute.make_rmem_tensor(
                        cute.make_layout(1), acc_dtype
                    )
                    for target in cutlass.range(kv_splits):
                        stats_offset = (
                            (kv_split_idx * warp_threads + lane_idx) * 2
                        )
                        remote_stats_mbar = cute.arch.map_dsmem_ptr(
                            cga_partial_mbar, target
                        )
                        # Keep these as two scalar st.async operations.  A
                        # hand-populated two-element rmem tensor is not a
                        # reliable CuTe 4.6 round trip (the same pitfall is
                        # documented by cute_fold's silicon-tested DSMEM path).
                        for stat in cutlass.range_constexpr(2):
                            if lane_idx < blk_tile_n:
                                if stat == 0:
                                    local_stat[0] = sM[lane_idx]
                                else:
                                    local_stat[0] = sL[lane_idx, 0]
                            else:
                                # Padding makes the whole warp participate, so
                                # transaction-barrier byte accounting does not
                                # depend on the implementation's treatment of
                                # predicated-off STAS lanes.
                                local_stat[0] = (
                                    -Float32.inf if stat == 0 else Float32(0)
                                )
                            remote_stat_ptr = cute.arch.map_dsmem_ptr(
                                sCgaStats.iterator + stats_offset + stat,
                                target,
                            )
                            cute.copy(
                                dsmem_stats_atom,
                                local_stat,
                                cute.make_tensor(
                                    remote_stat_ptr, cute.make_layout(1)
                                ),
                                mbar_ptr=remote_stats_mbar,
                            )

                # All source CTAs have now populated this target CTA's inbox.
                # The transaction barrier accounts for every producer byte;
                # no extra named barrier is required before the wait.
                cute.arch.fence_acq_rel_cta()
                cute.arch.mbarrier_wait(cga_partial_mbar, phase=0)

                # Across the whole cluster exactly 128 physical threads are
                # reducers.  Each owns the same 16-head fragment that tOsO
                # would ordinarily store, so no coordinate remap is needed.
                if owner_cta == kv_split_idx:
                    # Keep only one head's online-softmax state live at a
                    # time.  The previous vectorized spelling held 48 FP32
                    # values across a runtime split loop and made the CuTe
                    # backend's register/data-flow graph unnecessarily huge.
                    # The final fragment is still written in exactly tOsO's
                    # per-thread order.
                    final_o = cute.make_rmem_tensor((blk_tile_n,), o_dtype)
                    for head in cutlass.range_constexpr(blk_tile_n):
                        reduced_max = sCgaStats[0, head, 0]
                        reduced_sum = sCgaStats[0, head, 1]
                        reduced_o = Float32(
                            sCgaO[owner_thread * blk_tile_n + head]
                        )

                        for source in cutlass.range(1, kv_splits):
                            local_max = sCgaStats[source, head, 0]
                            next_max = cute.arch.fmax(reduced_max, local_max)
                            corr_old = Float32(0)
                            corr_new = Float32(0)
                            if reduced_sum > 0:
                                corr_old = exp2(reduced_max - next_max)
                            if sCgaStats[source, head, 1] > 0:
                                corr_new = exp2(local_max - next_max)
                            local_o = Float32(
                                sCgaO[
                                    (source * owner_threads + owner_thread)
                                    * blk_tile_n
                                    + head
                                ]
                            )
                            reduced_sum = (
                                reduced_sum * corr_old
                                + sCgaStats[source, head, 1] * corr_new
                            )
                            reduced_o = reduced_o * corr_old + local_o * corr_new
                            reduced_max = next_max

                        value = Float32(0)
                        if reduced_sum > 0:
                            value = reduced_o * cute.arch.rcp_approx(reduced_sum) * scale_o
                        final_o[head] = o_dtype(value)

                    # VP is a 128xN tile and the Ld32x32b correction slice
                    # assigns one D row plus all N heads to each physical
                    # correction thread.  The owner predicate selects every
                    # physical thread exactly once across the cluster, so
                    # direct scalar stores cover the output without a CTA0
                    # gather/TMA pass.  A TMEM copy slice cannot partition a
                    # global tensor for CopyUniversalOp: it produces a
                    # coordinate tensor rather than a gmem MemRef.
                    for head in cutlass.range_constexpr(blk_tile_n):
                        group_head = coord_hg * blk_tile_h + head
                        if group_head < mOFinal.shape[1][0]:
                            mOFinal[
                                warpgroup_tidx,
                                (group_head, 0),
                                (coord_hk, coord_b),
                            ] = final_o[head]

        ##############################
        # Reduction Dispatch
        ##############################
        if warp_idx == reduction_warp_id:
            if cutlass.const_expr(do_partial_red):
                self.reduction_epilogue(
                    blk_tile_hp,
                    coord_hp,
                    coord_hb,
                    kv_split_idx,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    sM,
                    sL,
                    mM_partial,
                    mL_partial,
                )
            elif cutlass.const_expr(do_cga_red):
                self.reduction_cga_local_stats(
                    blk_tile_n,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    cga_stats_ready_nbar,
                    sL,
                )
            elif cutlass.const_expr(do_none_red):
                self.reduction_none(
                    lane_store_max,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    sM,
                    sL,
                    scale_o,
                )
            cute.arch.griddepcontrol_launch_dependents()

        ##############################
        # Fused global-memory reduction
        ##############################
        if cutlass.const_expr(do_gmem_fused_red):
            # All role-specific work is complete at this point. In
            # particular, the TMA warp has waited for partial-O's bulk store
            # and the reduction warp has synchronously stored max/sum. TRT's
            # Q16 uses the last four CTAs for four 32-thread output slices.
            # Compact residue GQA4 has only four logical output heads (one
            # 32-thread slice), so its last CTA alone performs the reduction.
            # The final CTA resets the completion word before reading the
            # acquired partials; this also makes the slab replay-safe.
            if (
                warpgroup_idx == 0
                or warpgroup_idx == correction_warpgroup_id
            ):
                gmem_completion_nbar.arrive_and_wait()

                completion_idx = (
                    coord_hb_linear * grid_tiles_hp + coord_hp_linear
                )
                last_reducer_ctas = cutlass.const_expr(
                    1 if self.grouped_head_tile == 8 else 4
                )
                if tidx == 0:
                    cute.arch.fence_acq_rel_gpu()
                    old_count = cute.arch.atomic_add(
                        mCompletion.iterator + completion_idx,
                        Int32(1),
                        sem="acq_rel",
                        scope="gpu",
                    )
                    sGmemLast[0] = Int32(0)
                    if old_count >= kv_splits - Int32(last_reducer_ctas):
                        # Reducer rank + 1; zero remains the non-participant
                        # sentinel. Rank zero is the last CTA.
                        sGmemLast[0] = kv_splits - old_count

                    if old_count == kv_splits - Int32(1):
                        cute.arch.atomic_exch(
                            mCompletion.iterator + completion_idx,
                            Int32(0),
                            sem="release",
                            scope="gpu",
                        )
                    elif old_count >= kv_splits - Int32(last_reducer_ctas):
                        counter_value = Int32(-1)
                        while counter_value != Int32(0):
                            counter_value = cute.arch.atomic_add(
                                mCompletion.iterator + completion_idx,
                                Int32(0),
                                sem="acquire",
                                scope="gpu",
                            )

                gmem_completion_nbar.arrive_and_wait()

                if (
                    warpgroup_idx == correction_warpgroup_id
                    and sGmemLast[0] != Int32(0)
                ):
                    self.reduction_gmem_fused_nvfp4(
                        blk_tile_shpd,
                        coord_hg,
                        coord_p,
                        coord_hk,
                        coord_b,
                        heads_k,
                        batches,
                        kv_splits,
                        warpgroup_tidx,
                        sGmemLast[0] - Int32(1),
                        mOPartialGlobal,
                        mOFinal,
                        mOSF,
                        mOGlobalScale,
                        mL_partial,
                        mM_partial,
                        scale_o,
                    )

        return

    @staticmethod
    @cute.jit
    def reduction_gmem_fused_nvfp4(
        blk_tile_shpd: cute.Tile,
        coord_hg: Int32,
        coord_p: Int32,
        coord_hk: Int32,
        coord_b: Int32,
        heads_k: int,
        batches: int,
        kv_splits: Int32,
        warpgroup_tidx: Int32,
        reducer_rank: Int32,
        mO_partial: cute.Tensor,
        mO_final: cute.Tensor,
        mOSF: cute.Tensor,
        mOGlobalScale: cute.Tensor,
        mL_partial: cute.Tensor,
        mM_partial: cute.Tensor,
        scale_o: Float32,
    ):
        """Merge split partials and emit one direct-NVFP4 output block/thread.

        This is the data-parallel half of TRT's mode-1 protocol. The caller
        dynamically selects the last reducer CTAs. Their 128 correction
        threads map onto at most 16 heads x 8 block16 D slices of a D128 tile;
        compact Q8/GQA4 uses one CTA and predicates the unused threads.
        Statistics are intentionally duplicated across the eight D slices of
        a head, matching TRT's choice to keep the final output loads/stores
        fully distributed rather than serializing on a tiny reducer warp.
        """
        blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d = blk_tile_shpd
        assert blk_tile_p == 1 and blk_tile_d % 16 == 0
        assert mO_final.dtype == cutlass.Float4E2M1FN
        assert mOSF is not None and mOGlobalScale is not None

        d_scale_blocks = blk_tile_d // 16
        local_head = warpgroup_tidx // d_scale_blocks
        d_block = warpgroup_tidx % d_scale_blocks
        group_head = coord_hg * blk_tile_h + local_head
        logical_grouped_heads, prediction = mO_final.shape[1]

        # Match TRT's mode-1 memory contract: each global partial-O
        # transaction is a full 16-byte vector.  The first implementation
        # indexed all sixteen values independently, which compiled to 32K
        # scalar loads for Q16/D128/split16 and erased the launch saving of
        # the fused reducer.  Keep the accumulator in FP32 while allowing the
        # caller to select either FP32 (4 values/vector) or TRT-style BF16
        # (8 values/vector) partial storage.
        partial_vector_elems = 128 // mO_partial.dtype.width
        assert 16 % partial_vector_elems == 0
        partial_o_vector_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO_partial.dtype,
            num_bits_per_copy=128,
        )

        reducer_warp = warpgroup_tidx // 32
        if (
            reducer_warp == reducer_rank
            and group_head < logical_grouped_heads
        ):
            max_final = -Float32.inf
            for source in cutlass.range(kv_splits, unroll=4):
                max_final = cute.arch.fmax(
                    max_final,
                    mM_partial[
                        (group_head, coord_p),
                        (coord_hk, coord_b),
                        source,
                    ],
                )

            final_values = cute.make_rmem_tensor((16,), Float32)
            final_values.fill(Float32(0))
            sum_final = Float32(0)
            if max_final > -Float32.inf:
                for source in cutlass.range(kv_splits, unroll=4):
                    max_partial = mM_partial[
                        (group_head, coord_p),
                        (coord_hk, coord_b),
                        source,
                    ]
                    if max_partial > -Float32.inf:
                        correction = exp2(max_partial - max_final)
                        sum_final += correction * mL_partial[
                            (group_head, coord_p),
                            (coord_hk, coord_b),
                            source,
                        ]
                        coord_b_partial = source * batches + coord_b
                        partial_o_head = mO_partial[
                            None,
                            (group_head, coord_p),
                            (coord_hk, coord_b_partial),
                        ]
                        for vector_idx in cutlass.range_constexpr(
                            16 // partial_vector_elems
                        ):
                            partial_o_vector = cute.local_tile(
                                partial_o_head,
                                (partial_vector_elems,),
                                (
                                    d_block * (16 // partial_vector_elems)
                                    + vector_idx,
                                ),
                            )
                            partial_values = cute.make_rmem_tensor(
                                (partial_vector_elems,), mO_partial.dtype
                            )
                            cute.copy(
                                partial_o_vector_atom,
                                partial_o_vector,
                                partial_values,
                            )
                            partial_values_f32 = partial_values.load().to(
                                Float32
                            )
                            for value_in_vector in cutlass.range_constexpr(
                                partial_vector_elems
                            ):
                                value_idx = (
                                    vector_idx * partial_vector_elems
                                    + value_in_vector
                                )
                                final_values[value_idx] += (
                                    correction
                                    * partial_values_f32[value_in_vector]
                                )

            if sum_final != Float32(0):
                final_values.store(
                    final_values.load()
                    * (cute.arch.rcp_approx(sum_final) * scale_o)
                )

            packed_values, sf_value = quantize_nvfp4_block16(
                final_values.load(), Float32(mOGlobalScale[0])
            )

            packed_fp4 = cute.make_rmem_tensor(
                (16,), cutlass.Float4E2M1FN
            )
            packed_fp4.store(packed_values)
            packed_bytes = cute.recast_tensor(packed_fp4, cutlass.Uint8)
            mO_bytes = cute.recast_tensor(mO_final, cutlass.Uint8)
            for byte_idx in cutlass.range_constexpr(8):
                mO_bytes[
                    d_block * 8 + byte_idx,
                    (group_head, coord_p),
                    (coord_hk, coord_b),
                ] = packed_bytes[byte_idx]

            global_head = coord_hk * logical_grouped_heads + group_head
            k_idx = global_head * d_scale_blocks + d_block
            hidden = heads_k * logical_grouped_heads * blk_tile_d
            row_idx = coord_b * prediction + coord_p
            sf_offset = output_scale_offset(row_idx, k_idx, hidden)
            mOSF[sf_offset] = sf_value

    @staticmethod
    @cute.jit
    def reduction_cga_local_stats(
        blk_tile_n: int,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        cga_stats_ready_nbar: nbar,
        sL: cute.Tensor,
    ):
        """Collapse one CTA's four softmax-warp sums for CGA exchange."""
        sM_final_nbar.arrive_and_wait()
        sL_final_nbar.arrive_and_wait()
        if lane_idx < blk_tile_n:
            local_sum = sL[lane_idx, 0]
            for warp in cutlass.range_constexpr(1, warpgroup_warps):
                local_sum += sL[lane_idx, warp]
            sL[lane_idx, 0] = local_sum
        cute.arch.fence_view_async_shared()
        cga_stats_ready_nbar.arrive()

    @staticmethod
    @cute.jit
    def reduction_none(
        lane_store_max: bool,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        scale_o: Float32,
    ):
        colsum = Float32(0)
        sM_final_nbar.arrive_and_wait()
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane = sL[lane_idx, None]
            colsum = sL_lane[0] + sL_lane[1] + sL_lane[2] + sL_lane[3]
            normalization = cute.arch.rcp_approx(colsum) * scale_o
            sM[lane_idx] = normalization
        sM_final_nbar.arrive()

    @staticmethod
    @cute.jit
    def reduction_epilogue(
        blk_tile_hp: Tuple[int, int],
        coord_hp: Tuple[Int32, Int32],
        coord_hb: Tuple[Int32, Int32],
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        mM_partial: cute.Tensor,
        mL_partial: cute.Tensor,
    ):
        # Get per-split colmax + colsum destinations.  The old implementation
        # also atomically accumulated a final max in mM.  That required a
        # separate -inf fill launch and serialized every split on one address.
        # The combine kernel already reads every partial max, so publishing the
        # extra global maximum is redundant.
        coord_h = (coord_hp, coord_hb, kv_split_idx)
        gM_partial = cute.local_tile(mM_partial, (blk_tile_hp,), coord_h)
        gL_partial = cute.local_tile(mL_partial, (blk_tile_hp,), coord_h)

        # tile predication
        blk_tile_h, blk_tile_p = blk_tile_hp
        blk_tile_n = blk_tile_h * blk_tile_p
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n

        # gmem predication
        grouped_heads, prediction = mM_partial.shape[0]
        cM = cute.make_identity_tensor(mM_partial.shape[0])
        cM = cute.local_tile(cM, blk_tile_hp, coord_hp)
        idx_hg, idx_p = cM[lane_idx]
        lane_store_max &= idx_hg < grouped_heads
        lane_store_max &= idx_p < prediction

        # Load partial colmax and reduce
        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after barrier
        sM_final_nbar.arrive_and_wait()
        if lane_store_max:
            sM_lane = sM[lane_idx]
            gM_partial[lane_idx] = sM_lane

        # Load partial colsum and reduce
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane_wg = sL[lane_idx, None]
            sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
            gL_partial[lane_idx] = sL_lane


    ##############################
    # Reduction Kernel launch
    ##############################
    @staticmethod
    @cute.jit
    def launch_reduction(
        d_per_blk: int,
        o_bshd: cute.Tensor,
        o_partial_bshd: cute.Tensor,  # partial O per kv split
        l_partial_bsh: cute.Tensor,  # partial colsum_p per kv split
        m_partial_bsh: cute.Tensor,  # partial colmax_s per kv split
        scale_o: Float32,
        o_block_scale: Optional[cute.Tensor],
        o_global_scale: Optional[cute.Tensor],
        padded_grouped_heads: int,
        logical_grouped_heads: int,
        stream: cuda.CUstream,
        enable_pdl: bool = True,
    ):
        splits, b, s_q, h_q_partial, d = o_partial_bshd.shape
        h_q = o_bshd.shape[2]
        quantize_o = cutlass.const_expr(
            o_bshd.dtype == cutlass.Float4E2M1FN
        )
        if cutlass.const_expr(quantize_o):
            assert o_block_scale is not None and o_global_scale is not None
            assert d_per_blk % 16 == 0
            assert h_q_partial * logical_grouped_heads == (
                h_q * padded_grouped_heads
            )
        else:
            assert o_block_scale is None and o_global_scale is None

        # Tile in headdim first
        def reverse(t: cute.Tensor):
            modes = tuple(reversed(range(cute.rank(t))))
            layout = cute.select(t.layout, modes)
            return cute.make_tensor(t.iterator, layout)

        o_dhsb = reverse(o_bshd)
        o_partial_dhsb = reverse(o_partial_bshd)
        l_partial_hsb = reverse(l_partial_bsh)
        m_partial_hsb = reverse(m_partial_bsh)

        # A quantizing output thread owns exactly one dynamic-scale block.
        # The BF16/FP32 path retains its historical 32-bit vector width.
        d_per_thr = 16 if quantize_o else 32 // o_bshd.dtype.width
        thr_per_blk = d_per_blk // d_per_thr
        d_blks = cute.ceil_div(d, d_per_blk)
        smem_bytes = splits * 2 * Float32.width // 8

        GroupedQueryAttentionDecodePagedFp4.reduction_kernel(
            (thr_per_blk, d_per_thr, d_per_blk),
            o_dhsb,
            o_partial_dhsb,
            l_partial_hsb,
            m_partial_hsb,
            scale_o,
            o_block_scale,
            o_global_scale,
            padded_grouped_heads,
            logical_grouped_heads,
        ).launch(
            grid=[d_blks, h_q * s_q, b],
            block=[thr_per_blk, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

    @staticmethod
    @cute.kernel
    def reduction_kernel(
        tile_d: cute.Tile,
        o_dhsb: cute.Tensor,
        o_partial_dhsb: cute.Tensor,
        l_partial_hsb: cute.Tensor,
        m_partial_hsb: cute.Tensor,
        scale_o: Float32,
        o_block_scale: Optional[cute.Tensor],
        o_global_scale: Optional[cute.Tensor],
        padded_grouped_heads: int,
        logical_grouped_heads: int,
    ):
        thr_per_blk, d_per_thr, d_per_blk = tile_d
        d, h_q_partial, s_q, b, splits = o_partial_dhsb.shape
        _, h_q, _, _ = o_dhsb.shape
        d_blk_idx, coord_hs, coord_b = cute.arch.block_idx()
        coord_h, coord_s = cute.idx2crd(coord_hs, (h_q, s_q))
        tidx, _, _ = cute.arch.thread_idx()

        # Padded GQA tiles are grouped per KV head, not appended at the end of
        # the flattened head dimension. Map the logical O-proj head back to
        # its corresponding FP32 partial row before the softmax merge.
        coord_hk = coord_h // logical_grouped_heads
        coord_hg = coord_h % logical_grouped_heads
        coord_h_partial = coord_hk * padded_grouped_heads + coord_hg

        not_oob_d = True
        if d % d_per_blk != 0:
            not_oob_d = d_blk_idx * d_per_blk + tidx * d_per_thr < d

        coord_o = (d_blk_idx, coord_h, coord_s, coord_b)
        coord_partial = (
            d_blk_idx,
            coord_h_partial,
            coord_s,
            coord_b,
            None,
        )
        gO = cute.local_tile(o_dhsb, (d_per_blk,), coord_o)
        gO_partial = cute.local_tile(
            o_partial_dhsb, (d_per_blk,), coord_partial
        )

        stats_coord = (coord_h_partial, coord_s, coord_b, None)
        gM_partial = cute.local_tile(m_partial_hsb, (1,), stats_coord)
        gL_partial = cute.local_tile(l_partial_hsb, (1,), stats_coord)
        gM_partial_0 = gM_partial[None, tidx]
        gL_partial_0 = gL_partial[None, tidx]

        smem_ptr = cute.arch.get_dyn_smem(Float32)
        partial_layout = cute.make_layout((1, splits))
        sM_partial = cute.make_tensor(smem_ptr, partial_layout)
        sL_partial = cute.make_tensor(smem_ptr + splits, partial_layout)
        sL_partial_0 = sL_partial[None, tidx]
        sM_partial_0 = sM_partial[None, tidx]

        cpasync_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32
        )

        # TRT-LLM Gen publishes partial O in a 16-bit type and keeps only the
        # much smaller max/sum arrays in FP32.  Keep this reducer generic over
        # FP32/BF16 partial O so the memory-contract change can be isolated in
        # a paired benchmark without changing the attention mainloop.
        partial_o_dtype = o_partial_dhsb.dtype
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), partial_o_dtype
        )
        tv_layout = cute.make_ordered_layout((thr_per_blk, d_per_thr), order=(1, 0))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, (d_per_blk,))
        thr_copy = tiled_copy.get_slice(tidx)

        tCgO_partial = thr_copy.partition_S(gO_partial)  # (CPY, #CPY=1, splits)
        tCgO_partial = tCgO_partial[None, 0, None]  # (CPY, splits)
        tCgO = thr_copy.partition_D(gO)  # (CPY, #CPY=1)
        tCgO = tCgO[None, 0]  # (CPY)
        tCrO_final = cute.zeros_like(tCgO, Float32)

        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL wait
        cute.arch.griddepcontrol_wait()

        if tidx < splits:
            cute.copy(cpasync_atom, gL_partial_0, sL_partial_0)
            cute.copy(cpasync_atom, gM_partial_0, sM_partial_0)

        for split_idx in cutlass.range(thr_per_blk + tidx, splits, thr_per_blk):
            gL_partial_n = gL_partial[None, split_idx]
            sL_partial_n = sL_partial[None, split_idx]
            cute.copy(cpasync_atom, gL_partial_n, sL_partial_n)

            gM_partial_n = gM_partial[None, split_idx]
            sM_partial_n = sM_partial[None, split_idx]
            cute.copy(cpasync_atom, gM_partial_n, sM_partial_n)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Derive the final row max from the same partial statistics used for
        # the softmax correction.  Computing this tiny scalar reduction in
        # each output thread avoids another CTA barrier and, more importantly,
        # removes both the main-kernel global atomic and its initialization
        # launch from the critical path.
        max_final = -Float32.inf
        for split_idx in cutlass.range(splits, unroll=8):
            max_final = cute.arch.fmax(max_final, sM_partial[0, split_idx])
        sum_final = Float32(0)
        if max_final > -Float32.inf and not_oob_d:
            for split_idx in cutlass.range(splits, unroll=8):
                max_partial = sM_partial[0, split_idx]
                if max_partial > -Float32.inf:
                    correction = exp2(max_partial - max_final)
                    sum_final += correction * sL_partial[0, split_idx]
                    partial_o = tCgO_partial[None, split_idx].load()
                    tCrO_final += correction * partial_o.to(Float32)
            tCrO_final *= cute.arch.rcp_approx(sum_final) * scale_o

        cute.arch.griddepcontrol_launch_dependents()

        quantize_o = cutlass.const_expr(
            o_dhsb.dtype == cutlass.Float4E2M1FN
        )
        if not_oob_d:
            if cutlass.const_expr(quantize_o):
                # d_per_thr is exactly one block16 here, so the scale is
                # produced once after all split numerators share the same
                # final max/sum origin.  No BF16 output is materialized.
                assert o_block_scale is not None and o_global_scale is not None
                packed_values, sf_value = quantize_nvfp4_block16(
                    tCrO_final, Float32(o_global_scale[0])
                )
                tCgO.store(packed_values)

                d_block = (
                    d_blk_idx * d_per_blk + tidx * d_per_thr
                ) // 16
                d_scale_blocks = d // 16
                k_idx = coord_h * d_scale_blocks + d_block
                hidden = h_q * d
                row_idx = coord_b * s_q + coord_s
                sf_offset = output_scale_offset(row_idx, k_idx, hidden)
                o_block_scale[sf_offset] = sf_value
            else:
                tCgO.store(tCrO_final.to(o_dhsb.dtype))

        return
