# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors (modifications)
# SPDX-License-Identifier: BSD-3-Clause
"ResAttn: paged NVFP4 GQA decode attention (Blackwell tcgen05, CuTeDSL).\n\nDerived from FlashInfer's ``flashinfer/cute_dsl/attention/gqa_decode.py``\n(NVIDIA, BSD-3-Clause), which this file keeps the license of. Modifications\nby the ResInfer project:"

import math
import os
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
    Float16,
    Int32,
    Int64,
    Optional,
)

from .qmul4 import (
    dequantize_nvfp4_tile_d,
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
gmem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="gpu")


class GroupedQueryAttentionDecodePagedFp4:
    '分页 nvfp4 变体(MMA-native KV 池,page_size=128)。'

    def __init__(
        self,
        headdim,
        grouped_head_tile,
        prediction_tile=1,
        sequence_tile=256,
        reduction_mode: Literal[
            "kernel", "gmem_fused", "atomic", "cga_sharded", "none"
        ] = "kernel",
        *,
        page_size=128,
        tma_mask=False,
        residue_fold=False,
        perf_probe=0,
        blocks_per_sm=1,
        v_mode: Literal[
            "native_token", "qmul4_d", "qmul4_tmem"
        ] = "native_token",
        k_sf_mode: Literal["atom", "linear"] = "atom",
        logical_grouped_heads: int | None = None,
        reg_reconfig_q16: bool = False,
        single_softmax_wg: bool = False,
        smem_correction_broadcast: bool = False,
        # 诊断用 qmul4 分段消融:"none" 全量 / "conv" 跳过转换 /
        # "mma" 跳过 VP MMA。每级只摘掉**一个**嫌疑、保持流水线握手完整,
        # 所以 trap 是否消失能直接归因。生产路径恒为 "none";
        # 编译期常量(坑38:绝不再走运行期参数)。
        qmul4_skip: Literal["none", "conv", "mma"] = "none",
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
              - ``"atomic"``: cluster stats reduction plus global O reduce-add.
              - ``"cga_sharded"``: experimental remote-SMEM partial-O reduction;
                output slices are reduced by different cluster CTAs and stored once.
              - ``"none"``: no split-K, flash decoding disabled.
        tma_mask
            Disable causal masking — out-of-bounds scores are masked to 0
            instead of -inf. Mathematically equivalent softmax, but may
            produce subnormal exponentiations if non-oob scores are all
            negative (max goes to 0).
        """
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
        self.do_atomic_red = reduction_mode == "atomic"
        self.do_cga_red = reduction_mode == "cga_sharded"
        self.do_none_red = reduction_mode == "none" or reduction_mode is None
        self.tma_mask = tma_mask
        # Residue fold(R1):Q 按 [base 块; res 块] concat 打包(每 kv 组行数 2×),
        # softmax 前把 S 的两半相加(S_b+S_r)。v0 = 折叠后复制回两半,下游形状零改动。
        self.residue_fold = residue_fold
        # perf 分解:1=跳过 P 字节写(保留转换);2=跳过整个 P 量化+写
        self.perf_probe = perf_probe
        # 2 CTA/SM 实验:smem/寄存器预算均分 + launch bound(延迟型节拍的跨 CTA 填缝)
        self.blocks_per_sm = blocks_per_sm
        # ``native_token`` keeps the original blockscaled VP path and its
        # token-contiguous V page. ``qmul4_d`` consumes conventional
        # D-contiguous NVFP4 pages, expands them to an FP8-ready SMEM FIFO,
        # and executes the ordinary FP8 VP MMA. ``qmul4_tmem`` keeps the same
        # cache contract but follows TRT-LLM decode: LDSM.U4 -> QMUL4 -> STTM,
        # with converted V used as the TMEM A operand of VP. This must be
        # explicit: all three cache modes carry e2m1 bytes but have different
        # layout and pipeline ownership.
        self.v_mode = v_mode
        # ``atom`` is the historical ResAttn cache contract. ``linear`` is
        # vLLM/FlashInfer's TRT-LLM Gen NVFP4 contract: one scale byte per
        # adjacent 16-D group in ordinary [token, d_block] order.  Keeping the
        # choice explicit prevents a standard vLLM cache from being silently
        # interpreted as tcgen05's scale-factor atom order.
        self.k_sf_mode = k_sf_mode
        # Q may be padded to a supported 16/32-wide MMA tile.  BF16 output
        # historically materializes the padded heads and slices afterward;
        # the direct NVFP4 epilogue instead predicates them and packs only the
        # model's logical GQA heads into the following O-proj contract.
        self.logical_grouped_heads = logical_grouped_heads
        # Compile-time A/B switch. Q16 stays off by default because removing
        # its spills did not produce a stable B200 latency/stream improvement.
        self.reg_reconfig_q16 = reg_reconfig_q16
        # Experimental TRT-style scheduler: one sequential softmax WG instead
        # of two interleaved WGs. Keep it Q16/qmul4_tmem-only until the phase
        # protocol has passed the full B200 matrix.
        self.single_softmax_wg = single_softmax_wg
        # TRT's exact Q16 decode cubin broadcasts correction factors with far
        # fewer indexed shuffles than the current one-shuffle-per-head path.
        # This opt-in experiment reuses correction WG's private sL column as a
        # 16-float warp-local carrier, then issues four 128-bit shared loads.
        # Arithmetic, TMEM ownership, and the final sL reduction stay fixed.
        self.smem_correction_broadcast = smem_correction_broadcast
        self.qmul4_skip = qmul4_skip
        # The direct-to-TMEM path needs a conversion warpgroup in addition to
        # the historical MMA/TMA, 2x softmax, and correction warpgroups.
        self.threads_per_cta = (
            4
            if v_mode == "qmul4_tmem" and single_softmax_wg
            else 5
            if v_mode == "qmul4_tmem"
            else 4
        ) * warpgroup_threads
        self.page_size = page_size
        # 页表 smem 环深度(上游同款:> prefetch_iters + 1)
        self.pt_stages = 4

        assert page_size == 128, (
            "MMA-native profile 冻结 page=128(SF atom 对齐定理);"
            "64 需 cp.async SF 拼装路径,未实现"
        )
        assert headdim > 0 and headdim % 64 == 0
        assert grouped_head_tile * prediction_tile in (1, 2, 4, 8, 16, 32)
        assert sequence_tile > 0 and sequence_tile % 128 == 0
        assert sequence_tile % page_size == 0
        assert sum(
            (
                self.do_kernel_red,
                self.do_gmem_fused_red,
                self.do_atomic_red,
                self.do_cga_red,
                self.do_none_red,
            )
        ) == 1
        assert v_mode in ("native_token", "qmul4_d", "qmul4_tmem"), v_mode
        if self.do_cga_red and v_mode != "qmul4_tmem":
            raise ValueError("cga_sharded is implemented only for qmul4_tmem")
        if self.do_gmem_fused_red and (
            v_mode != "qmul4_tmem" or grouped_head_tile != 16
        ):
            raise ValueError(
                "gmem_fused currently requires qmul4_tmem with a Q16 tile"
            )
        assert k_sf_mode in ("atom", "linear"), k_sf_mode
        if logical_grouped_heads is not None:
            assert 0 < logical_grouped_heads <= grouped_head_tile
        if single_softmax_wg:
            assert v_mode == "qmul4_tmem" and grouped_head_tile == 16
        if smem_correction_broadcast:
            assert (
                v_mode == "qmul4_tmem"
                and single_softmax_wg
                and grouped_head_tile == 16
            )
        assert qmul4_skip in ("none", "conv", "mma"), qmul4_skip
        if residue_fold:
            # R1 范围:decode(s_q=1),folded 宽度 = grouped_head_tile/2
            assert prediction_tile == 1 and grouped_head_tile % 2 == 0

    ##############################
    # Launch helpers
    ##############################
    # Runtime implementable check
    def can_implement(
        self, kv_splits, qo_shape_bshd, kv_shape_bshd, qkv_dtype, o_dtype
    ):
        b_k, s_q, h_q, d_q = qo_shape_bshd
        b_q, s_k, h_k, d_k = kv_shape_bshd

        if qkv_dtype is cutlass.Float8E4M3:
            raise TypeError("use Float8E4M3FN instead of Float8E4M3")

        if not (d_q == d_k == self.headdim):
            raise ValueError(
                f"headdim_q({d_q}), headdim_k({d_k}) must be {self.headdim}"
            )

        if h_q % h_k != 0:
            raise ValueError(f"heads_q({h_q}) must be a multiple of heads_k({h_k})")

        if 0 < s_k < s_q:
            raise ValueError(
                f"non-zero seqlen({s_k}) must be at least prediction({s_q})"
            )

        # Only causal mask up to two tiles
        if not self.tma_mask and s_q > self.sequence_tile:
            raise ValueError(
                f"prediction({s_q}) with causal mask can be at most {self.sequence_tile}"
            )

        if self.tma_mask and 0 < s_k < 8:
            raise ValueError(
                f"non-zero seqlen({s_k}) with TMA masking must be at least 8"
            )

        if b_k != b_q:
            raise ValueError(f"batches_k({b_k}) and batches_q({b_q}) mismatch")

        if self.do_atomic_red or self.do_cga_red:
            if kv_splits not in (1, 2, 4, 8, 16):
                raise ValueError(
                    "cluster reduction requires kv_splits po2 <= 16, "
                    f"got {kv_splits}"
                )

            if self.do_cga_red and o_dtype is not BFloat16:
                raise TypeError(
                    "cga_sharded currently requires BF16 output, "
                    f"got {o_dtype}"
                )
            if self.do_atomic_red and o_dtype not in (Float32, BFloat16, Float16):
                raise TypeError(
                    f"atomic reduction requires (Float32, BFloat16, Float16) o_dtype, got {o_dtype}"
                )

        if self.do_gmem_fused_red and kv_splits not in (4, 8, 16, 32, 64):
            raise ValueError(
                "gmem_fused reduction requires kv_splits in "
                f"{{4,8,16,32,64}}, got {kv_splits}"
            )

        if self.do_none_red and kv_splits != 1:
            raise ValueError("KV splits must be 1 if flash decoding is disabled")

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
        q_sf: Optional[cute.Tensor],  # Q scale factors (e4m3, SF-atom order), None=fp8 path
        k_sf: Optional[cute.Tensor],  # K scale factors (e4m3, SF-atom order), None=fp8 path
        v_sf: Optional[cute.Tensor],  # V scale factors 沿 token(e4m3, SF-atom order), None=fp8 VP
        p_sf: Optional[cute.Tensor],  # P 常数 SF buffer(e4m3=0.171875 填充), None=fp8 VP
        o_bshd: cute.Tensor,
        o_block_scale: Optional[cute.Tensor],
        o_global_scale: Optional[cute.Tensor],
        l_bsh: Optional[cute.Tensor],
        m_bsh: Optional[cute.Tensor],
        o_partial_bshd: Optional[cute.Tensor],
        l_partial_bsh: Optional[cute.Tensor],
        m_partial_bsh: Optional[cute.Tensor],
        completion_counter: Optional[cute.Tensor],
        scale_s: Float32,
        scale_o: Float32,
        stream: cuda.CUstream,
        enable_pdl: bool = True,
        sfq_tx: Int32 = Int32(0),  # DEBUG: SFQ expect 字节覆盖(0=用布局账)
        seqlens: Optional[cute.Tensor] = None,  # 必填:per-batch kv 长度 (Int32, (B,))
        table_offsets: Optional[cute.Tensor] = None,  # 必填:kv_indptr[:B] (Int32, (B,))
        page_table: Optional[cute.Tensor] = None,  # 必填:扁平物理页号 (Int32, (Ltot,))
    ):
        """
        Parameters
        ----------
        kv_splits
            Threadblocks per sequence (flash decoding).
        q_bshd
            Q tensor in BSHD logical view.
        k_bshd, v_bshd
            分页 KV 池的逻辑视图,shape 均为 (P, page_size, h_k, d):
            K 物理 (P, h_k, page_size, d/2) HND fp4;V 物理 (P, h_k, d,
            page_size/2)(页内 token-innermost,坑28)。gemm_view 之后分别成
            (page_size, d, (h_k, P)) 与 (d, page_size, (h_k, P)) ——
            页号是 l 模坐标,页表在 kernel 内换算。
        seqlens / table_offsets / page_table
            变长 + 分页元数据(flashinfer indptr/indices 约定);三者必填。
        o_bshd
            Output tensor. Must be zero initialized for atomic reduction. A
            Float4E2M1FN tensor selects the fused NVFP4 epilogue in either the
            no-split attention CTA or the deterministic final reducer.
        o_block_scale
            E4M3 output block scales in CUTLASS's 128x4 swizzled layout.
        o_global_scale
            One-element Float32 tensor containing the activation global-scale
            inverse used by the downstream NVFP4 GEMM.
        l_bsh
            Log-sum-exp output (Float32, log2 base). May be None.
        m_bsh
            Reserved legacy argument. The kernel-reduction path derives the
            final maximum directly from ``m_partial_bsh`` and accepts None.
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
        seqlens
            变长(varlen)模式:per-batch 真实 kv 长度 (Int32, shape (B,))。
            None = 等长,取 ``k_bshd.shape[1]``。非 None 时 K/V/SF 缓冲仍按
            padded S_max 布局,尾部掩码/exit_early 由各 CTA 的运行时长度驱动。
        """
        ##############################
        # TiledMma creation
        ##############################
        # Phase-1 nvfp4: KQ GEMM 走 blockscaled mxf4nvf4(Q/K=e2m1+SF),VP 保持 fp8。
        self.use_nvfp4_kq = use_nvfp4_kq = cutlass.const_expr(
            q_bshd.dtype == cutlass.Float4E2M1FN
        )
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
            assert self.v_mode == "qmul4_tmem"
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
        kq_dtype = q_bshd.dtype  # e2m1(nvfp4)或 fp8(原路径)
        v_cache_dtype = v_bshd.dtype
        self.use_qmul4_v_smem = use_qmul4_v_smem = cutlass.const_expr(
            self.v_mode == "qmul4_d"
        )
        self.use_qmul4_v_tmem = use_qmul4_v_tmem = cutlass.const_expr(
            self.v_mode == "qmul4_tmem"
        )
        self.use_qmul4_v = use_qmul4_v = cutlass.const_expr(
            use_qmul4_v_smem or use_qmul4_v_tmem
        )
        # The conventional cache remains packed e2m1, while the VP operands
        # are E4M3 after the side conversion.  Keeping these as two distinct
        # compile-time types prevents the old "dtype implies layout" bug.
        mma_dtype = (
            cutlass.Float8E4M3FN if use_qmul4_v else v_cache_dtype
        )
        sf_dtype = cutlass.Float8E4M3FN
        sf_vec_size = 16
        acc_dtype = Float32
        assert k_bshd.dtype == kq_dtype
        if cutlass.const_expr(not use_nvfp4_kq):
            assert kq_dtype == mma_dtype

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
        # N-major 8b B in smem requires N multiple of 16
        min_mma_tile_n = 16 if mma_dtype.width == 8 else 8
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_n = max(min_mma_tile_n, blk_tile_n)
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)

        # MMA tiles per block tile
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_dk = math.ceil(blk_tile_d / mma_tile_k)
        assert blk_tile_s % mma_tile_m == 0
        assert mma_tile_n % blk_tile_n == 0

        # GEMM1: (S_K, (H_R, S_Q), D, (H_K, B))
        if cutlass.const_expr(use_nvfp4_kq):
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
        else:
            tiled_mma_kq = sm100_utils.make_trivial_tiled_mma(
                mma_dtype,
                mma_dtype,
                OperandMajorMode.K,  # K
                OperandMajorMode.K,  # Q
                acc_dtype,
                tcgen05.CtaGroup.ONE,
                mma_tile_mnk[:2],
            )
            tiled_mma_kq_sfb = None

        # GEMM2: (D, (H_R, S_Q), S_K, (H_K, B))
        # Phase 2:V=e2m1(来自 cache)触发全原生 VP;否则 Phase 1(VP=fp8)
        self.use_nvfp4_vp = use_nvfp4_vp = cutlass.const_expr(
            self.v_mode == "native_token"
            and v_cache_dtype == cutlass.Float4E2M1FN
        )
        if cutlass.const_expr(use_qmul4_v):
            assert v_cache_dtype == cutlass.Float4E2M1FN
            assert v_sf is not None
        vp_dtype = cutlass.Float8E4M3FN if use_qmul4_v else v_cache_dtype
        # 分页变体:mma_tile_k_vp 固定 128 = page_size ⇒ V/SFV 每 stage 恰好
        # 一页,TMA box 不跨页。dense 曾用 256(128B/行),但坑32 记录表明
        # 128(11 stages/64B 行)与 256(7 stages/128B 行)净收益≈0,
        # 分页取 128 几乎不付性能代价。
        mma_tile_k_vp = min(self.page_size, blk_tile_s) if use_nvfp4_vp else mma_tile_k
        # 坑32(硬件否决,勿重试):VP 的 M 维是 d,理论上可用半 datapath(64)让
        # V stage 回到 8KB 并同时保住 k_vp=256 的 128B/行 TMA —— 但
        # **blockscaled MMA 强制 M=128**(OpError: expects the M-mode to be 128)。
        # 于是 fp4-V 的"TMA 行长"与"流水线深度"在 smem 预算上是硬矛盾:
        # k_vp=128 → 11 stages / 64B 行;k_vp=256 → 7 stages / 128B 行,净收益≈0。
        mma_tile_m_vp = mma_tile_m
        mma_tile_mnk_vp = (mma_tile_m_vp, mma_tile_n, mma_tile_k_vp)
        tiles_sk_host = blk_tile_s // mma_tile_k_vp
        tiles_dm = math.ceil(blk_tile_d / mma_tile_m_vp)  # e2m1(Phase2)或 fp8(Phase1)
        if cutlass.const_expr(use_nvfp4_vp):
            # 坑24:tcgen05 fp4 操作数只能 K-major(坑18 正主)。布局函数会静默
            # 翻转,但 MMA 的 major 声明必须手动跟上 —— 否则硬件按 MN 语义解释
            # 描述符,B 侧读数出区(B200 六轮探针:P 区任意写均不影响输出)。
            # KQ 的 fp4 MMA 声明 K ✓ 正常,此处曾漏改。
            tiled_mma_vp = sm100_utils.make_blockscaled_trivial_tiled_mma(
                vp_dtype,
                OperandMajorMode.K,  # V
                OperandMajorMode.K,  # P
                sf_dtype,
                sf_vec_size,
                tcgen05.CtaGroup.ONE,
                (mma_tile_m_vp, mma_tile_n),
            )
            tiled_mma_vp_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
                vp_dtype,
                OperandMajorMode.K,
                OperandMajorMode.K,
                sf_dtype,
                sf_vec_size,
                tcgen05.CtaGroup.ONE,
                (mma_tile_mnk[0], max(128, mma_tile_n)),
            )
        else:
            
            
            
            
            
            
            tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(
                mma_dtype,
                mma_dtype,
                OperandMajorMode.K,   # V:token 最内(收缩轴)
                OperandMajorMode.MN,  # P
                acc_dtype,
                tcgen05.CtaGroup.ONE,
                mma_tile_mnk[:2],
                (
                    tcgen05.OperandSource.TMEM
                    if use_qmul4_v_tmem
                    else tcgen05.OperandSource.SMEM
                ),
            )
            tiled_mma_vp_sfb = None

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
        sf_cols = 0
        if cutlass.const_expr(use_nvfp4_kq):
            # SF TMEM 列(cutlass 示例 :373 公式:tile/sf_atom_mn(=32) × mma_inst_tile_k(=4))
            sf_cols += (mma_tile_m // 32) * 4  # SFK
            sf_cols += (mma_tile_n_sfb // 32) * 4  # SFQ
        if cutlass.const_expr(use_nvfp4_vp):
            sf_cols += (mma_tile_m_vp // 32) * 4  # SFV
            sf_cols += (mma_tile_n_sfb // 32) * 4  # SFP
        # TRT-style direct conversion stores one FP8 V stage in TMEM.  A
        # 128x128 E4M3 tile consumes 128*8/32 = 32 TMEM columns.  Reserve this
        # ring before allowing S to consume the remaining columns; otherwise
        # the old greedy S-stage calculation leaves no legal V allocation.
        self.cvt_stages = cvt_stages = 0
        cvt_stage_cols = mma_tile_k_vp * vp_dtype.width // 32
        if cutlass.const_expr(use_qmul4_v_tmem):
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
        v_raw_stage_bits = 0
        if cutlass.const_expr(use_qmul4_v_smem):
            v_raw_stage_bits = (
                mma_tile_m_vp * mma_tile_k_vp * v_cache_dtype.width
            )
        k_stage_bits = mma_tile_m * mma_tile_k * kq_dtype.width  # K stage(fp4 时减半)
        nk_stage_bits = mma_tile_n * mma_tile_k * kq_dtype.width  # Q(与 K 同 dtype)
        mn_stage_bits = mma_tile_m * mma_tile_n * mma_dtype.width
        # SF stage 尺寸(仅 nvfp4):SFK 每 stage M×(K/16);SFQ 一次性 N_pad×(K/16)
        sfk_stage_bits = 0
        sfq_total_bits = 0
        if cutlass.const_expr(use_nvfp4_kq):
            sfk_stage_bits = mma_tile_m * (mma_tile_k // sf_vec_size) * sf_dtype.width
            if cutlass.const_expr(self.k_sf_mode == "linear"):
                # Standard scales first land in a contiguous TMA carrier,
                # then the KQ warp permutes them into the tcgen05 SFA atom.
                # Both 1 KiB views are live for the same K pipeline stage.
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
        if cutlass.const_expr(self.do_atomic_red):
            smem_alloc_bits += max_reduction_iters * blk_tile_n * acc_dtype.width * 2
            smem_alloc_bits += max_reduction_iters * mbarrier_bits * 2
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
        
        
        
        sfv_stage_bits = 0
        sfp_total_bits = 0
        if cutlass.const_expr(use_nvfp4_vp):
            sfv_stage_bits = (
                mma_tile_m_vp * (mma_tile_k_vp // sf_vec_size) * sf_dtype.width
            )
            sfp_total_bits = (
                mma_tile_n_sfb
                * (mma_tile_k_vp // sf_vec_size)
                * sf_dtype.width
                * (blk_tile_s // mma_tile_k_vp)
            )
        elif cutlass.const_expr(use_qmul4_v):
            # Conventional V has one D-scale for every 16 adjacent values.
            # A raw 128x128 page therefore carries 128/16 * 128 = 1024 B.
            sfv_stage_bits = (
                (mma_tile_m_vp // sf_vec_size)
                * mma_tile_k_vp
                * sf_dtype.width
            )
        # Q(+SFQ), S, P, O
        smem_alloc_bits += tiles_dk * nk_stage_bits + mbarrier_bits  # 1 mbar for Q
        smem_alloc_bits += sfq_total_bits
        smem_alloc_bits += sfp_total_bits
        smem_alloc_bits += s_stages * pipe_stage_bits  # s in tmem
        smem_alloc_bits += p_stages * (tiles_sm * mn_stage_bits + pipe_stage_bits)
        smem_alloc_bits += o_stages * pipe_stage_bits  # o in tmem
        if cutlass.const_expr(use_qmul4_v_tmem):
            # TMEM-ready V owns an independent AsyncUmma barrier ring whose
            # depth is ``cvt_stages``, not the raw SMEM/K stage count.
            smem_alloc_bits += cvt_stages * pipe_stage_bits
        alignment_bits = 1024 - (smem_alloc_bits % 1024)
        # K(+SFK)与 V:两条独立流水线,等 stage 数(坑1:拆分决策见 plan doc)
        smem_capacity_bits = (
            utils.get_smem_capacity_in_bytes("sm_100") // self.blocks_per_sm * 8
        )
        # The dynamic-smem allocator inserts alignment between the packed
        # pipeline/barrier prefix and several separately aligned tensor
        # allocations.  A single end-of-prefix round-up under-counted that
        # padding by 448 B for qmul4_tmem on B200, selecting eight KV stages
        # and requesting 232768 B against the 232448 B launch limit.  Keep a
        # small explicit guard here so stage selection reflects the final
        # allocator rather than failing at launch.  The next legal choice is
        # seven stages (still one deeper than the legacy qmul4_d path).
        smem_allocator_guard_bits = 512 * 8 if use_qmul4_v_tmem else 0
        remaining_bits = (
            smem_capacity_bits
            - smem_alloc_bits
            - alignment_bits
            - smem_allocator_guard_bits
        )
        kv_pipe_count = 3 if use_qmul4_v_smem else 2
        kv_pair_bits = (
            k_stage_bits
            + sfk_stage_bits
            + mk_stage_bits
            + v_raw_stage_bits
            + sfv_stage_bits
            + kv_pipe_count * pipe_stage_bits
        )
        kv_stages = remaining_bits // kv_pair_bits
        if os.environ.get("RESATTN_DECODE_SMEM_DEBUG") == "1":
            # trace 期的 Python 值,免 GPU。qmul4_d 把每 stage 从
            # (K + ready-V) 变成 (K + raw-V + ready-V + V-SF),stage 数会被
            # 挤小 —— 这里把账目摊开,便于与实际分配对账。
            print(
                f"[smem] v_mode={self.v_mode} kv_stages={kv_stages}\n"
                f"       capacity={smem_capacity_bits // 8}B "
                f"fixed={smem_alloc_bits // 8}B align={alignment_bits // 8}B "
                f"guard={smem_allocator_guard_bits // 8}B "
                f"remaining={remaining_bits // 8}B\n"
                f"       per-stage: k={k_stage_bits // 8} sfk={sfk_stage_bits // 8} "
                f"v_main={mk_stage_bits // 8} raw_v={v_raw_stage_bits // 8} "
                f"sfv={sfv_stage_bits // 8} "
                f"pipes={kv_pipe_count * pipe_stage_bits // 8} "
                f"=> pair={kv_pair_bits // 8}B\n"
                f"       kv total={kv_stages * kv_pair_bits // 8}B "
                f"cvt_tmem={cvt_stages}x{cvt_stage_cols}cols",
                flush=True,
            )
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
        # TMEM-A still needs a packed input tile in SMEM. Match NVIDIA's
        # mixed-input decode geometry: sub-byte V is carried in 8-bit SMEM
        # containers and unpacked by LDSM.U4 before STTM. The legacy qmul4_d
        # path keeps its compact FP4 raw tile plus separate FP8-ready tile.
        if cutlass.const_expr(use_qmul4_v_tmem):
            smem_layout_v = sm100_utils.make_smem_layout_a(
                tiled_mma_vp,
                mma_tile_mnk_vp,
                cutlass.Int8,
                kv_stages,
                is_k_major=False,
            )
        else:
            smem_layout_v = sm100_utils.make_smem_layout_a(
                tiled_mma_vp, mma_tile_mnk_vp, vp_dtype, kv_stages
            )
        smem_layout_v_raw = None
        smem_layout_v_raw_tma = None
        smem_layout_v_ready = None
        smem_layout_v_sf_d = None
        if cutlass.const_expr(use_qmul4_v_smem):
            # Both aliases are logical (D, token, stage) views. The ready
            # layout is physically identical to ``smem_layout_v`` above but
            # exposes complete 16-D slices to the conversion warp. The MMA
            # still consumes the partitioned alias made by make_smem_layout_a.
            raw_atom = tcgen05.make_smem_layout_atom(
                tcgen05.mma.SmemLayoutAtomKind.MN_SW64, v_cache_dtype
            )
            ready_atom = tcgen05.make_smem_layout_atom(
                tcgen05.mma.SmemLayoutAtomKind.MN_SW128, vp_dtype
            )
            smem_layout_v_raw = cute.tile_to_shape(
                raw_atom,
                (mma_tile_m_vp, mma_tile_k_vp, kv_stages),
                order=(1, 0, 2),
            )
            # TMA describes packed bytes, not sub-byte logical scalars. Keep
            # a byte-shaped alias for the transfer while LDS sees the FP4
            # logical alias above; both carry the same 128B MN swizzle.
            raw_tma_atom = tcgen05.make_smem_layout_atom(
                tcgen05.mma.SmemLayoutAtomKind.MN_SW64, cutlass.Uint8
            )
            smem_layout_v_raw_tma = cute.tile_to_shape(
                raw_tma_atom,
                (mma_tile_m_vp // 2, mma_tile_k_vp, kv_stages),
                order=(1, 0, 2),
            )
            smem_layout_v_ready = cute.tile_to_shape(
                ready_atom,
                (mma_tile_m_vp, mma_tile_k_vp, kv_stages),
                order=(1, 0, 2),
            )
        if cutlass.const_expr(use_qmul4_v):
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            _sf_blocks = mma_tile_m_vp // 16
            _tile_sf = _sf_blocks * mma_tile_k_vp
            smem_layout_v_sf_d = cute.make_layout(
                (16, _tile_sf // 16, kv_stages),
                stride=(1, 16, _tile_sf),
            )
        # SF smem 布局(仅 nvfp4;SFQ 单份、SFK 随 K stage)
        smem_layout_sfk = None
        smem_layout_sfk_raw = None
        smem_layout_sfq = None
        if cutlass.const_expr(use_nvfp4_kq):
            smem_layout_sfk = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma_kq, mma_tile_mnk, sf_vec_size, kv_stages
            )
            if cutlass.const_expr(self.k_sf_mode == "linear"):
                # The logical SFA tensor has an illegal 8-byte TMA row stride,
                # but each physical page/head scale tile is one contiguous
                # 1 KiB allocation.  Load that byte tile through a legal
                # (16,64) carrier; the KQ warp performs the small SMEM->SMEM
                # atom permutation after the shared K barrier completes.
                assert self.headdim == 128
                _ksf_stage_bytes = mma_tile_m * (mma_tile_k // sf_vec_size)
                smem_layout_sfk_raw = cute.make_layout(
                    (16, _ksf_stage_bytes // 16, kv_stages),
                    stride=(1, 16, _ksf_stage_bytes),
                )
                if os.environ.get("RESATTN_DECODE_SMEM_DEBUG") == "1":
                    print(f"[ksf-linear] smem_layout_sfk={smem_layout_sfk}")
            smem_layout_sfq = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma_kq_sfb,
                (mma_tile_mnk[0], mma_tile_n_sfb, mma_tile_mnk[2]),
                sf_vec_size,
                tiles_dk,
            )
        # Phase2:SFV(V-scale 沿 token,VP 的 A 侧)随 V stage
        smem_layout_sfv = None
        if cutlass.const_expr(use_nvfp4_vp):
            smem_layout_sfv = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma_vp, mma_tile_mnk_vp, sf_vec_size, kv_stages
            )
        smem_layout_sfp = None
        if cutlass.const_expr(use_nvfp4_vp):
            smem_layout_sfp = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma_vp_sfb,
                (mma_tile_mnk_vp[0], mma_tile_n_sfb, mma_tile_mnk_vp[2]),
                sf_vec_size,
                blk_tile_s // mma_tile_mnk_vp[2],
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
        tma_store_op = (
            cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp()
            if self.do_atomic_red
            else cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        )

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
        if cutlass.const_expr(use_qmul4_v_smem):
            # Standard HND page: mV=(D, token, (head,page)), D contiguous.
            # TMA lands in the raw MN-swizzled alias; no tensor-core operand
            # descriptor is constructed until after the QMUL4 transform.
            mV_bytes = cute.recast_tensor(mV_mkl, cutlass.Uint8)
            tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op,
                mV_bytes,
                cute.slice_(smem_layout_v_raw_tma, (None, None, 0)),
                (mma_tile_m_vp // 2, mma_tile_k_vp),
            )
        elif cutlass.const_expr(use_qmul4_v_tmem):
            # NVIDIA mixed-input decode keeps packed FP4 in an 8-bit SMEM
            # carrier layout, then LDSM.U4 unpacks it into registers.  Using
            # make_tiled_tma_atom_A here is important: its partition matches
            # the later LDSM -> STTM copy geometry exactly.
            tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mV_mkl,
                cute.select(smem_layout_v, mma_modes),
                tma_tile_mnk_vp,
                tiled_mma_vp,
                internal_type=cutlass.Int8,
            )
        else:
            tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mV_mkl,
                cute.select(smem_layout_v, mma_modes),
                tma_tile_mnk_vp,
                tiled_mma_vp,
            )

        if cutlass.const_expr(use_qmul4_v):
            # v_sf 是 trtllm 契约 [P, H, ps, S](tile 内 4-token interleave)。
            # 每 (page,head) 的 tile 连续 ⇒ 直接按扁平 (16, tile/16) 视图搬。
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
        tma_atom_sfk = tma_tensor_sfk = None
        tma_atom_sfq = tma_tensor_sfq = None
        if cutlass.const_expr(use_nvfp4_kq):
            if cutlass.const_expr(self.k_sf_mode == "linear"):
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
            else:
                sfk_layout = self._sf_layout_hb(
                    mK_mkl.shape[0],
                    self.headdim,
                    mK_mkl.shape[2],
                    sf_vec_size,
                )
                mSFK = cute.make_tensor(k_sf.iterator, sfk_layout)
                tma_atom_sfk, tma_tensor_sfk = cute.nvgpu.make_tiled_tma_atom_A(
                    tma_load_op,
                    mSFK,
                    cute.select(smem_layout_sfk, mma_modes),
                    tma_tile_mnk,
                    tiled_mma_kq,
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

        # Phase2:SFV gmem 视图 + TMA。v_sf 平铺 e4m3,SF-atom 布局按
        # (M=D, K=S, l) 解释(VP 的收缩轴 = token)。
        tma_atom_sfv = tma_tensor_sfv = None
        tma_atom_sfp = tma_tensor_sfp = None
        if cutlass.const_expr(use_nvfp4_vp):
            sfv_layout = self._sf_layout_hb(
                mV_mkl.shape[0], mV_mkl.shape[1], mV_mkl.shape[2], sf_vec_size
            )
            mSFV = cute.make_tensor(v_sf.iterator, sfv_layout)
            tma_atom_sfv, tma_tensor_sfv = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mSFV,
                cute.select(smem_layout_sfv, mma_modes),
                tma_tile_mnk_vp,
                tiled_mma_vp,
            )
            sfp_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (mma_tile_n_sfb, blk_tile_s, 1), sf_vec_size
            )
            mSFP = cute.make_tensor(p_sf.iterator, sfp_layout)
            sfb_op_vp = sm100_utils.cluster_shape_to_tma_atom_SFB(
                (1, 1), tiled_mma_vp_sfb.thr_id
            )
            tma_atom_sfp, tma_tensor_sfp = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op_vp,
                mSFP,
                cute.select(smem_layout_sfp, mma_modes),
                (mma_tile_mnk_vp[0], mma_tile_n_sfb, mma_tile_mnk_vp[2]),
                tiled_mma_vp_sfb,
                internal_type=cutlass.Int16,
            )

        tma_atom_o = tma_tensor_o = None
        if cutlass.const_expr(not quantize_o_main):
            tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO_mnl,
                cute.select(smem_layout_o, mode=[0, 1]),
                tma_tile_mnk[:2],
            )

        # GEMM view for LSE output
        # ((h_g, s_q), (h_k, b))
        if cutlass.const_expr(self.use_nvfp4_vp):
            mL_nl = l_bsh  # DBG:Int8 平铺调试缓冲直通(非 LSE)
        else:
            mL_nl = None if l_bsh is None else self.gemm_view_bsh(l_bsh, h_k)
        assert l_bsh is None or l_bsh.dtype == acc_dtype or self.use_nvfp4_vp

        phase_debug = cutlass.const_expr(
            self.v_mode == "qmul4_tmem"
            and os.environ.get("RESATTN_QMUL4_PHASE_DEBUG") == "1"
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
        cluster_x = kv_splits if (self.do_atomic_red or self.do_cga_red) else 1

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
            # nvfp4 KQ(SF;fp8 路径为 None)
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
            smem_layout_v_raw,
            smem_layout_v_raw_tma,
            smem_layout_v_ready,
            smem_layout_v_sf_d,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_v_sf_d,
            tma_tensor_v_sf_d,
            # Phase2 SFV/SFP(fp8 VP 为 None)
            tiled_mma_vp_sfb,
            smem_layout_sfv,
            tma_atom_sfv,
            tma_tensor_sfv,
            smem_layout_sfp,
            tma_atom_sfp,
            tma_tensor_sfp,
            # O
            smem_layout_o,
            tma_atom_o,
            mO_mnl if quantize_o_main else tma_tensor_o,
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
                if phase_debug or self.do_cga_red or self.do_gmem_fused_red
                else None
            ),
            # Rest
            mL_nl,
            mL_partial_nl,
            mM_partial_nl,
            completion_counter,
            scale_s_log2_e,
            scale_o,
            sfq_tx,
            seqlens,
            table_offsets,
            page_table,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[cluster_x, 1, 1],
            stream=stream,
            min_blocks_per_mp=self.blocks_per_sm,
            use_pdl=enable_pdl,
        )

        if cutlass.const_expr(self.do_kernel_red):
            self.launch_reduction(
                self.headdim,
                o_bshd,
                l_bsh,
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
                    mma_selected = cutlass.const_expr(
                        self.qmul4_skip != "mma"
                    )
                    if mma_selected:
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
        # nvfp4 KQ SF(fp8 路径为 None)
        tiled_mma_kq_sfb: Optional[cute.TiledMma],
        smem_layout_sfk: Optional[cute.Layout],
        smem_layout_sfk_raw: Optional[cute.Layout],
        tma_atom_sfk: Optional[cute.CopyAtom],
        mSFK: Optional[cute.Tensor],
        smem_layout_sfq: Optional[cute.Layout],
        tma_atom_sfq: Optional[cute.CopyAtom],
        mSFQ: Optional[cute.Tensor],
        # V
        smem_layout_v: cute.ComposedLayout,
        smem_layout_v_raw: Optional[cute.ComposedLayout],
        smem_layout_v_raw_tma: Optional[cute.ComposedLayout],
        smem_layout_v_ready: Optional[cute.ComposedLayout],
        smem_layout_v_sf_d: Optional[cute.Layout],
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        tma_atom_v_sf_d: Optional[cute.CopyAtom],
        mVSF_d: Optional[cute.Tensor],
        # Phase2 SFV/SFP(fp8 VP 为 None)
        tiled_mma_vp_sfb: Optional[cute.TiledMma],
        smem_layout_sfv: Optional[cute.Layout],
        tma_atom_sfv: Optional[cute.CopyAtom],
        mSFV: Optional[cute.Tensor],
        smem_layout_sfp: Optional[cute.Layout],
        tma_atom_sfp: Optional[cute.CopyAtom],
        mSFP: Optional[cute.Tensor],
        # O
        smem_layout_o: cute.ComposedLayout,
        tma_atom_o: Optional[cute.CopyAtom],
        mO: cute.Tensor,
        mOPartialGlobal: Optional[cute.Tensor],
        mOSF: Optional[cute.Tensor],
        mOGlobalScale: Optional[cute.Tensor],
        mOFinal: Optional[cute.Tensor],
        # Rest
        mL: Optional[cute.Tensor],  # LSE output (Float32, log2 base)
        mL_partial: Optional[cute.Tensor],
        mM_partial: Optional[cute.Tensor],
        mCompletion: Optional[cute.Tensor],
        scale_s_log2_e: Float32,
        scale_o: Float32,
        sfq_tx: Int32 = Int32(0),
        mSeqlens: Optional[cute.Tensor] = None,
        mTableOffsets: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
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
        use_nvfp4_kq = mSFK is not None  # trace 期常量,用 if cutlass.const_expr(...) 守卫
        use_nvfp4_vp = vp_dtype == cutlass.Float4E2M1FN  # Phase2 常量(坑4:裸 bool)
        use_qmul4_v_smem = cutlass.const_expr(self.v_mode == "qmul4_d")
        use_qmul4_v_tmem = cutlass.const_expr(self.v_mode == "qmul4_tmem")
        use_qmul4_v = cutlass.const_expr(
            use_qmul4_v_smem or use_qmul4_v_tmem
        )
        sf_dtype = cutlass.Float8E4M3FN
        o_dtype = out_dtype
        acc_dtype = Float32
        quantize_o = cutlass.const_expr(mO.dtype == cutlass.Float4E2M1FN)
        if cutlass.const_expr(quantize_o):
            assert mOSF is not None and mOGlobalScale is not None
            assert tma_atom_o is None
        quantize_o_final = cutlass.const_expr(
            mOFinal is not None
            and mOFinal.dtype == cutlass.Float4E2M1FN
        )

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
        do_kernel_red = self.do_kernel_red
        do_gmem_fused_red = self.do_gmem_fused_red
        do_partial_red = self.do_partial_red
        do_atomic_red = self.do_atomic_red
        do_cga_red = self.do_cga_red
        do_none_red = self.do_none_red
        store_lse = mL is not None and mL.dtype == acc_dtype  # Int8 DBG 通道除外
        tma_mask = self.tma_mask

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

        conversion_warpgroup_id = -1
        if cutlass.const_expr(use_qmul4_v_tmem):
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
        use_reg_reconfig = self.blocks_per_sm == 1 and (
            blk_tile_n > 16
            or (use_qmul4_v_tmem and self.reg_reconfig_q16)
        )
        max_sw_regs_per_wg_thread = 256  # CUDA limitation
        max_hw_regs_per_wg_thread = (
            64 * 1024 // warpgroup_threads // self.blocks_per_sm
        )  # 64K regs per SM,按驻留 CTA 数均分
        mma_tma_regs = 64 if self.blocks_per_sm == 1 else 24
        conversion_regs = 64 if self.blocks_per_sm == 1 else 24
        # A 640-thread qmul4_tmem CTA starts from a 96-register uniform
        # allocation at min_blocks_per_mp=1.  ``decrease(120)`` is therefore
        # an illegal direction on hardware.  Keep those softmax groups at the
        # uniform ceiling and donate the MMA/conversion registers to correction.
        softmax_regs = (
            (
                120
                if use_qmul4_v_tmem and softmax_warpgroups == 1
                else 96
                if use_qmul4_v_tmem
                else 120
            )
            if self.blocks_per_sm == 1
            else 80
        )
        # 640 threads round the uniform launch allocation down to 96 registers
        # per thread (61,440 total).  The unused 4,096 physical registers are
        # not donor registers in this CTA's dynamic pool.  Budget qmul4_tmem
        # against the actual 5*96 pool and leave one 8-register allocation unit
        # idle so correction's TRY_ALLOC cannot wait forever at exact capacity.
        role_reg_budget = (
            (
                softmax_warpgroups
                + 3  # MMA/TMA + conversion + correction
            )
            * (
                (64 * 1024 // self.threads_per_cta // 8) * 8
            )
            - 8
            if use_qmul4_v_tmem and self.blocks_per_sm == 1
            else max_hw_regs_per_wg_thread
        )
        correction_regs = min(
            max_sw_regs_per_wg_thread,
            role_reg_budget
            - mma_tma_regs
            - (conversion_regs if use_qmul4_v_tmem else 0)
            - softmax_regs * softmax_warpgroups,
        )
        assert (
            mma_tma_regs
            + (conversion_regs if use_qmul4_v_tmem else 0)
            + softmax_regs * softmax_warpgroups
            + correction_regs
        ) <= role_reg_budget

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

        # Opt-in liveness marker for the qmul4_tmem fourth-iteration probe.
        # Each split owns four otherwise ordinary output elements in head 0:
        # softmax phases 0/1, correction, and VP. The hung kernel never reaches
        # its output TMA, so a copy engine on a separate stream can read these
        # monotonically increasing stage codes without changing the launch ABI
        # or adding a barrier. Production specializations compile this away.
        phase_debug = cutlass.const_expr(
            use_qmul4_v_tmem
            and os.environ.get("RESATTN_QMUL4_PHASE_DEBUG") == "1"
        )
        def phase_mark(
            actor,
            code,
            warp_leader=False,
            phase_debug=phase_debug,
            warpgroup_tidx=warpgroup_tidx,
            lane_idx=lane_idx,
            mO=mOFinal,
            kv_split_idx=kv_split_idx,
            coord_hb=coord_hb,
            o_dtype=o_dtype,
        ):
            if cutlass.const_expr(phase_debug):
                is_leader = (
                    lane_idx == 0
                    if cutlass.const_expr(warp_leader)
                    else warpgroup_tidx == 0
                )
                if is_leader:
                    mO[kv_split_idx * 8 + actor, (0, 0), coord_hb] = o_dtype(
                        code
                    )

        # 分页 + 变长:per-batch 元数据直读(均匀地址 4B,L1 广播;host 预写,
        # 不依赖 PDL 护栏 —— 与 K/V 数据读不同)。mK.shape[0] 现在是 page_size,
        # 序列长度唯一来源是 seqlens;table_offset 定位本序列在页表中的起点。
        seqlen = mSeqlens[coord_b]
        table_offset = mTableOffsets[coord_b]
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
            if cutlass.const_expr(use_qmul4_v):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v_sf_d)
            if cutlass.const_expr(not quantize_o):
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
        # Shared KV pipeline requires ordering MMA
        mma_order_kq_nbar = nbar(1, tma_threads + tma_threads)
        mma_order_vp_nbar = nbar(2, tma_threads + tma_threads)
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
        reduction_mbars_ptr = None
        cga_partial_mbar = None
        if cutlass.const_expr(do_atomic_red):
            reduction_mbars_ptr = smem.allocate_array(Int64, max_reduction_iters * 2)
            if warp_idx == init_warp:
                if lane_idx < max_reduction_iters * 2:
                    mbar_ptr = reduction_mbars_ptr + lane_idx
                    arrive_count = 1
                    expect_tx_bytes = blk_tile_n * acc_dtype.width // 8
                    cute.arch.mbarrier_init(mbar_ptr, arrive_count)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expect_tx_bytes)
            init_warp += 1
            cute.arch.cluster_arrive_relaxed()
        elif cutlass.const_expr(do_cga_red):
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
        sfq_mbar = smem.allocate_array(Int64, 1) if use_nvfp4_kq else None
        sfp_mbar = smem.allocate_array(Int64, 1) if use_nvfp4_vp else None
        if warp_idx == init_warp:
            expect_tx_bytes = cute.size_in_bytes(q_dtype, smem_layout_q)
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_load_mbar, 1)
                if cutlass.const_expr(use_nvfp4_kq):
                    cute.arch.mbarrier_init(sfq_mbar, 1)
                if cutlass.const_expr(use_nvfp4_vp):
                    cute.arch.mbarrier_init(sfp_mbar, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(q_load_mbar, expect_tx_bytes)
                if cutlass.const_expr(use_nvfp4_kq):
                    sfq_expect = cutlass.Int32(
                        cute.size_in_bytes(sf_dtype, smem_layout_sfq)
                    )
                    if sfq_tx > 0:
                        sfq_expect = sfq_tx
                    cute.arch.mbarrier_arrive_and_expect_tx(sfq_mbar, sfq_expect)
                if cutlass.const_expr(use_nvfp4_vp):
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        sfp_mbar,
                        cutlass.Int32(cute.size_in_bytes(sf_dtype, smem_layout_sfp)),
                    )
        init_warp += 1

        # Initialize pipelines. Native-token keeps the historical interleaved
        # K/V FIFO. qmul4_d splits ownership into K(TMA->UMMA), raw V
        # (TMA->warp), and ready V(warp->UMMA).
        kv_stages = smem_layout_k.shape[-1]
        k_stage_bytes = mma_tile_m * mma_tile_k * k_dtype.width // 8
        if cutlass.const_expr(use_nvfp4_kq):
            k_stage_bytes += cute.size_in_bytes(
                sf_dtype, cute.select(smem_layout_sfk, mma_modes)
            )
        v_stage_bytes = mma_tile_m_vp * mma_tile_k_vp * vp_dtype.width // 8
        if cutlass.const_expr(use_nvfp4_vp and self.perf_probe != 3):
            v_stage_bytes += cute.size_in_bytes(
                sf_dtype, cute.select(smem_layout_sfv, mma_modes)
            )
        kv_producer = kv_consumer = None
        k_producer = k_consumer = None
        v_raw_producer = v_raw_consumer = None
        v_ready_producer = v_ready_consumer = None
        if cutlass.const_expr(use_qmul4_v):
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
            if cutlass.const_expr(use_qmul4_v_smem):
                v_raw_stage_bytes = cute.size_in_bytes(
                    cutlass.Uint8,
                    cute.slice_(smem_layout_v_raw_tma, (None, None, 0)),
                )
            else:
                # ``smem_layout_v`` is an Int8 carrier layout, but TMA
                # expands each packed FP4 nibble into that carrier.  Barrier
                # transaction bytes are counted at the *gmem source* width,
                # exactly as NVIDIA's mixed-input decode does.  Counting the
                # carrier width here advertised 16 KiB while TMA only
                # completed 8 KiB, so the conversion WG waited forever on
                # the first raw-V stage.
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
                consumer_group=(
                    warpgroup_cooperative
                    if use_qmul4_v_tmem
                    else thr_cg(warp_threads)
                ),
                tx_count=v_raw_stage_bytes,
                barrier_storage=v_raw_pipeline_ptr,
                cta_layout_vmnk=mcast_layout,
                # PipelineTmaAsync releases an empty slot from one signaling
                # lane per consumer warp. Recompute the barrier arrive count
                # from the full consumer group (4 lanes for qmul4_tmem, 1 for
                # qmul4_d); otherwise the first raw-V ring reuse waits for
                # 128/32 arrivals while only 4/1 can ever be issued.
                enable_multicast_signaling=True,
                defer_sync=True,
            ).make_participants()
            v_ready_stages = (
                self.cvt_stages if use_qmul4_v_tmem else kv_stages
            )
            v_ready_pipeline_ptr = smem.allocate_array(Int64, v_ready_stages * 2)
            v_ready_producer, v_ready_consumer = pipeline.PipelineAsyncUmma.create(
                num_stages=v_ready_stages,
                producer_group=(
                    warpgroup_cooperative
                    if use_qmul4_v_tmem
                    else thr_cg(warp_threads)
                ),
                consumer_group=mma_group,
                barrier_storage=v_ready_pipeline_ptr,
                cta_layout_vmnk=mcast_layout,
                defer_sync=True,
            ).make_participants()
        else:
            kv_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
            kv_producer, kv_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=kv_stages,
                producer_group=tma_group,
                consumer_group=mma_group,
                tx_count=v_stage_bytes,
                barrier_storage=kv_pipeline_ptr,
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
        v_smem_dtype = cutlass.Int8 if use_qmul4_v_tmem else vp_dtype
        tAsV = smem.allocate_tensor(
            v_smem_dtype,
            smem_layout_v.outer,
            stensor_align,
            smem_layout_v.inner,
        )  # native/ready V, or packed 8-bit carrier for qmul4_tmem
        tAsVRaw = None
        tAsVRawTma = None
        tAsVReady = None
        tAsVSFD = None
        if cutlass.const_expr(use_qmul4_v_tmem):
            tAsVRaw = tAsV
        if cutlass.const_expr(use_qmul4_v_smem):
            # TODO(坑45):ready 缓冲仍是 **d 最内**,而 VP 的收缩轴是 token
            # —— 与 Phase 1 的 V K-major 修正同源的问题。直接把 stride 翻成
            # (k_vp, 1, ...) 会让 DSL 编译崩溃(core dump):这个别名叠在
            # ``tAsV`` 带 swizzle(S<2,4,3>)的布局上,换 stride 序后描述符失效。
            # 正确做法是让转换按 ``tAsV`` 自身的(带 swizzle 的)布局写,
            # 而不是另造一个扁平别名。
            v_logical_layout = cute.make_layout(
                (mma_tile_m_vp, mma_tile_k_vp, kv_stages),
                stride=(
                    1,
                    mma_tile_m_vp,
                    mma_tile_m_vp * mma_tile_k_vp,
                ),
            )
            tAsVRawTma = smem.allocate_tensor(
                cutlass.Uint8,
                smem_layout_v_raw_tma.outer,
                stensor_align,
                smem_layout_v_raw_tma.inner,
            )
            # 坑46:TMA 按 ``smem_layout_v_raw_tma.inner``(S<2,4,3>)把 raw V
            # 落进 SMEM,而这里原本用**不带 swizzle** 的 recast_ptr 去读 ——
            # 转换一直在读错位的字节。指针类型上直接可见:
            #   rawtma = ptr<i8, smem, align<1024>, S<2,4,3>>
            #   raw    = ptr<f4E2M1FN, smem, align<1024>>        (无 swizzle)
            # 代码里本来就有 fp4 版的布局(``smem_layout_v_raw``,MN_SW64 atom),
            # 按它建视图即可 —— swizzle 经 recast_ptr 挂回,与 P 的字节视图
            # (sP_flat)同一手法。
            _raw_sw = cute.make_tensor(
                cute.recast_ptr(
                    tAsVRawTma.iterator,
                    smem_layout_v_raw.inner,
                    dtype=cutlass.Float4E2M1FN,
                ),
                smem_layout_v_raw.outer,
            )
            # 带 swizzle 的布局是嵌套的 ((128,1),(8,16),(1,6)),而转换里的
            # ``local_tile(src,(16,1),…)`` 的 tiler 是扁平 rank-2 的 ——
            # 与 ready 缓冲同一个坑。所幸映射同样仿射(128*8 = 1024),
            # 用 composition 导出扁平 (d, token, stage) 视图即可,swizzle 保留。
            tAsVRaw = cute.composition(
                _raw_sw,
                cute.make_layout(
                    (mma_tile_m_vp, mma_tile_k_vp, kv_stages),
                    stride=(1, mma_tile_m_vp, mma_tile_m_vp * mma_tile_k_vp),
                ),
            )
            # Logical D/token alias of the FP8 MMA buffer. Both layouts use
            # the same MN_SW128 atom and tile order, so no second allocation.
            # 坑45:不造扁平别名 —— 从 ``tAsV`` **自身**的布局
            # ((128,32),1,4,stages) 组合出 (d, token) 视图,swizzle 随之保留。
            # 目标映射:(d, t) → 域内下标 d + 128*(t%32) + 4096*(t//32),
            # 即 shape (m, (32, k/32)) stride (1, (m, m*32))。
            # 扁平别名那条路两次都让 DSL 崩(裸 make_tensor / recast_ptr 挂
            # swizzle 皆然),原因是它绕过了真实布局而不是从中导出。
            # 目标映射 (d,t) → tAsV 域下标 d + 128*(t%32) + 4096*(t//32)。
            # 因为 128*32 = 4096,它其实是**仿射**的:= d + 128*t —— 所以可以
            # 用**扁平** shape 表达,不必嵌套。这很重要:转换里的
            # ``local_tile(dst, (16,1), …)`` 对嵌套模不能正确分解。
            _m_v, _k_v = mma_tile_m_vp, mma_tile_k_vp
            tAsVReady = cute.composition(
                tAsV,
                cute.make_layout(
                    (_m_v, _k_v, kv_stages),
                    stride=(1, _m_v, _m_v * _k_v),
                ),
            )
        if cutlass.const_expr(use_qmul4_v):
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
        if cutlass.const_expr(use_nvfp4_kq):
            tAsSFK = smem.allocate_tensor(sf_dtype, smem_layout_sfk, stensor_align)
            if cutlass.const_expr(self.k_sf_mode == "linear"):
                tAsSFKRaw = smem.allocate_tensor(
                    sf_dtype, smem_layout_sfk_raw, stensor_align
                )
            tBsSFQ = smem.allocate_tensor(sf_dtype, smem_layout_sfq, stensor_align)
        tAsSFKAtomU32 = None
        tAsSFKRawU64 = None
        if cutlass.const_expr(
            use_nvfp4_kq and self.k_sf_mode == "linear"
        ):
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
        tAsSFV = None
        tBsSFP = None
        if cutlass.const_expr(use_nvfp4_vp):
            tAsSFV = smem.allocate_tensor(sf_dtype, smem_layout_sfv, stensor_align)
            tBsSFP = smem.allocate_tensor(sf_dtype, smem_layout_sfp, stensor_align)

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
        # 坑23:local_tile 视图上的 make_fragment_B 对 fp4+SW128 生成坏描述符。
        # fp4 分支走 Q 同款直连链(std);nm 仅作分区形状用途,别名到 std 同块
        # 内存(fp4 从不经 nm 寻址)——省 8KB smem。fp8 走原 nm 实分配。
        tBsP_std = None
        if cutlass.const_expr(use_nvfp4_vp):
            # 注:v2 的 P store 用手推字节偏移(见下),仅对 blk_tile_s=256 推导过。
            # tile=512 时 P 缓冲槽数与 v3→槽映射均变 → 需重做 swizzle-aware 推导。
            assert blk_tile_s == 256, "v2 P store 偏移仅支持 sequence_tile=256"
            _tiles_sk_p = blk_tile_s // mma_tile_k_vp
            smem_layout_p_std = sm100_utils.make_smem_layout_b(
                tiled_mma_vp, mma_tile_mnk_vp, vp_dtype, _tiles_sk_p * p_stages
            )
            # 坑27:软件手写 swizzled 操作数,基址对齐 ≥ swizzle 周期
            tBsP_std = smem.allocate_tensor(
                vp_dtype,
                smem_layout_p_std.outer,
                stensor_align,
                smem_layout_p_std.inner,
            )
            tBsP_nm = cute.make_tensor(
                cute.recast_ptr(tBsP_std.iterator, dtype=vp_dtype),
                tBsP_nm_layout,
            )
        else:
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
        if cutlass.const_expr(use_qmul4_v_tmem):
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
        if cutlass.const_expr(do_atomic_red):
            sR_layout = cute.make_layout((blk_tile_n, max_reduction_iters, 2))
            sR = smem.allocate_tensor(acc_dtype, sR_layout, svector_align)
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
        if cutlass.const_expr(do_atomic_red or do_cga_red):
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
        if cutlass.const_expr(
            use_qmul4_v_tmem
            and os.environ.get("RESATTN_QMUL4_VALUE_DEBUG") == "1"
        ):
            if tidx == 0:
                cute.printf("TPTR=%d\n", tmem_ptr)
        tmem_offset = 0

        # Keep converted MMA operands at TMEM column zero, matching NVIDIA's
        # mixed-input decode.  TMEM-source UMMA addressing is more restricted
        # than an accumulator fragment: placing this ring after S/L/O happened
        # to pass a cold launch but produced stale/NaN data on reuse.
        if cutlass.const_expr(use_qmul4_v_tmem):
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
        if cutlass.const_expr(use_nvfp4_kq):
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
        tCtSFV = None
        tCtSFP = None
        if cutlass.const_expr(use_nvfp4_vp):
            tCtSFV_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_vp,
                (mma_tile_m, mma_tile_n, mma_tile_k),
                16,
                cute.slice_(smem_layout_sfv, (None, None, None, 0)),
            )
            tCtSFV = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + tmem_offset, dtype=sf_dtype),
                tCtSFV_layout,
            )
            tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtSFV)
            tCtSFP_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_vp_sfb,
                (mma_tile_m, max(128, mma_tile_n), mma_tile_k),
                16,
                cute.slice_(smem_layout_sfp, (None, None, None, 0)),
            )
            tCtSFP = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + tmem_offset, dtype=sf_dtype),
                tCtSFP_layout,
            )
            tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtSFP)

        assert tmem_offset <= tmem_alloc_cols, (
            f"\t{tmem_offset} tmem cols used, {tmem_alloc_cols} tmem cols allocated"
        )
        if cutlass.const_expr(
            use_qmul4_v_tmem
            and os.environ.get("RESATTN_DECODE_SMEM_DEBUG") == "1"
        ):
            print(
                f"[tmem] V={tcgen05.find_tmem_tensor_col_offset(tAtV_cvt)} "
                f"S={tcgen05.find_tmem_tensor_col_offset(tCtS)} "
                f"L={tcgen05.find_tmem_tensor_col_offset(tCtL)} "
                f"O={tcgen05.find_tmem_tensor_col_offset(tCtO)} "
                f"used={tmem_offset} alloc={tmem_alloc_cols}",
                flush=True,
            )

        # ``setmaxnreg.*.sync.aligned`` is a warpgroup collective: all four
        # specialist warps in WG0 must execute the *same static instruction*.
        # Keeping one call inside each warp-specific dispatch is invalid even
        # when all four use the same target value.
        if cutlass.const_expr(use_reg_reconfig):
            if not exit_early and warpgroup_idx == 0:
                cute.arch.setmaxregister_decrease(mma_tma_regs)

        ##############################
        # Exit early
        ##############################
        if exit_early:
            if warpgroup_idx == correction_warpgroup_id:
                sM_final_nbar.arrive()
                sL_final_nbar.arrive()

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
            if cutlass.const_expr(use_qmul4_v_smem):
                gV = cute.local_tile(
                    mV,
                    tiler=(mma_tile_m_vp // 2, mma_tile_k_vp),
                    coord=(None, None, (coord_hk, None)),
                )
            else:
                gV = cute.local_tile(
                    mV,
                    tiler=(mma_tile_m_vp, mma_tile_k_vp),
                    coord=(None, None, (coord_hk, None)),
                )  # (MMA_TILE_M, MMA_TILE_K, #TILE_DM, #TILE_SK=1, #P)

            tAgK = thrblk_mma_kq.partition_A(
                gK
            )  # (MMA, #MMA_M, #MMA_K, #TILE_DK, #P)
            tAgV = None
            if cutlass.const_expr(not use_qmul4_v_smem):
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

            if cutlass.const_expr(use_qmul4_v_smem):
                # Generic 2-D TMA partitions preserve the explicit
                # (D,token) modes used by the conversion helper.
                tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsVRawTma, 0, 2),
                    gmem_tensor=cute.group_modes(gV, 0, 2),
                )
            else:
                tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsV, 0, 3),
                    gmem_tensor=cute.group_modes(tAgV, 0, 3),
                )
            if cutlass.const_expr(use_qmul4_v):
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
            if cutlass.const_expr(use_nvfp4_kq):
                if cutlass.const_expr(self.k_sf_mode == "linear"):
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
                else:
                    # 页局部 SF = dense atom 序按页切块(定理)⇒ 结构与 K 数据同构
                    gSFK = cute.local_tile(
                        mSFK,
                        tiler=(mma_tile_m, mma_tile_k),
                        coord=(0, None, (coord_hk, None)),
                    )  # (M, K, #TILE_DK, #P)
                    tAgSFK = thrblk_mma_kq.partition_A(gSFK)
                    tGSsSFK, tGSgSFK = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_sfk,
                        mcast_coord,
                        mcast_layout,
                        smem_tensor=cute.group_modes(tAsSFK, 0, 3),
                        gmem_tensor=cute.group_modes(tAgSFK, 0, 3),
                    )
                    # SF 布局含 stride-0 广播模式,必须滤掉否则 TMA 零发射(坑13,示例 :932)
                    tGSsSFK = cute.filter_zeros(tGSsSFK)
                    tGSgSFK = cute.filter_zeros(tGSgSFK)
            if cutlass.const_expr(use_nvfp4_vp):
                gSFV = cute.local_tile(
                    mSFV,
                    tiler=(mma_tile_m_vp, mma_tile_k_vp),
                    coord=(None, None, (coord_hk, None)),
                )  # (M, K, #TILE_DM, #TILE_SK=1, #P)
                tAgSFV = thrblk_mma_vp.partition_A(gSFV)
                tGSsSFV, tGSgSFV = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_sfv,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsSFV, 0, 3),
                    gmem_tensor=cute.group_modes(tAgSFV, 0, 3),
                )
                tGSsSFV = cute.filter_zeros(tGSsSFV)
                tGSgSFV = cute.filter_zeros(tGSgSFV)

            cute.arch.griddepcontrol_wait()

            # Sequence loop(pristine 单 FIFO;K/V 各自 expected_tx)
            # 页表直读(v0):每 tile 每子页一次 4B 均匀 ld.global(L2 驻留),
            # producer 流水线的超前量吸收其延迟;smem 环形预取留 M2.5。
            # 越界逻辑页 → phys=-1 → TMA OOB 填零(尾部掩码另行兜底)。
            prefetch_tiles = prefetch_iters * kv_splits
            kv_token = True  # Producer always acquires first
            k_token_tma = True
            v_raw_token_tma = True
            for s in cutlass.range(kv_split_idx, prefetch_tiles + tiles_s, kv_splits):
                # Load K(sm 子循环即页循环:page_size = mma_tile_m)
                if s < tiles_s:
                    for sm in cutlass.range_constexpr(tiles_sm):
                        logical_k = s * pages_s + sm
                        phys_k = Int32(-1)
                        if logical_k < page_count:
                            phys_k = mPageTable[table_offset + logical_k]
                        for dk in cutlass.range_constexpr(tiles_dk):
                            if cutlass.const_expr(use_qmul4_v):
                                k_handle = k_producer.acquire_and_advance(k_token_tma)
                                k_token_tma = k_producer.try_acquire()
                            else:
                                k_handle = kv_producer.acquire_and_advance(
                                    kv_token, expected_tx=Int32(k_stage_bytes)
                                )
                                kv_token = kv_producer.try_acquire()
                            cute.copy(
                                tma_atom_k,
                                tGSgK[None, dk, phys_k],
                                tGSsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                            if cutlass.const_expr(use_nvfp4_kq):
                                if cutlass.const_expr(
                                    self.k_sf_mode == "linear"
                                ):
                                    cute.copy(
                                        tma_atom_sfk,
                                        tGSgSFK[None, 0, 0, phys_k],
                                        tGSsSFK[None, k_handle.index],
                                        tma_bar_ptr=k_handle.barrier,
                                    )
                                else:
                                    cute.copy(
                                        tma_atom_sfk,
                                        tGSgSFK[None, dk, phys_k],
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
                            phys_v = mPageTable[table_offset + logical_v]
                        for dm in cutlass.range_constexpr(tiles_dm):
                            if cutlass.const_expr(use_qmul4_v):
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
                            else:
                                v_handle = kv_producer.acquire_and_advance(
                                    kv_token, expected_tx=Int32(v_stage_bytes)
                                )
                                kv_token = kv_producer.try_acquire()
                                cute.copy(
                                    tma_atom_v,
                                    tGSgV[None, dm, 0, phys_v],
                                    tGSsV[None, v_handle.index],
                                    tma_bar_ptr=v_handle.barrier,
                                )
                            if cutlass.const_expr(
                                use_nvfp4_vp and self.perf_probe != 3
                            ):
                                cute.copy(
                                    tma_atom_sfv,
                                    tGSgSFV[None, dm, 0, phys_v],
                                    tGSsSFV[None, v_handle.index],
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
            if cutlass.const_expr(use_nvfp4_kq):
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
            # Load SFP(Phase2 常数 SF;独立 mbar)
            if cutlass.const_expr(use_nvfp4_vp):
                gSFP = cute.local_tile(
                    mSFP,
                    tiler=(max(128, mma_tile_n), blk_tile_s),
                    coord=(0, 0, 0),
                )
                gSFP_mma = cute.flat_divide(
                    gSFP, (max(128, mma_tile_n), mma_tile_k_vp)
                )[None, None, 0, None]
                thrblk_sfp = tiled_mma_vp_sfb.get_slice(0)
                tBgSFP = thrblk_sfp.partition_B(gSFP_mma)
                tBsSFP_tma, tBgSFP_tma = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_sfp,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tBsSFP, 0, 3),
                    gmem_tensor=cute.group_modes(tBgSFP, 0, 3),
                )
                tBsSFP_tma = cute.filter_zeros(tBsSFP_tma)
                tBgSFP_tma = cute.filter_zeros(tBgSFP_tma)
                cute.copy(
                    tma_atom_sfp,
                    tBgSFP_tma,
                    tBsSFP_tma,
                    tma_bar_ptr=sfp_mbar,
                )

            # Slice and partition O
            # (TILE_D, TILE_H)
            coord_b_partial = (
                kv_split_idx * batches + coord_b if do_partial_red else coord_b
            )
            gO = None
            if cutlass.const_expr(not quantize_o):
                gO = cute.local_tile(
                    mO,
                    tiler=(blk_tile_d, blk_tile_hp),
                    coord=(0, coord_hp, (coord_hk, coord_b_partial)),
                )
            sO_tma = gO_tma = None
            if cutlass.const_expr(not quantize_o):
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
            phase_mark(1, 210, True)
            if cutlass.const_expr(do_cga_red):
                # CGA reducer threads own disjoint output fragments and store
                # them directly; this warp has no output transaction to issue.
                pass
            elif cutlass.const_expr(quantize_o):
                sO_final_nbar.arrive_and_wait()
                phase_mark(1, 211, True)

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
                num_k_tiles = cute.ceil_div(hidden, 64)
                row_idx = coord_b

                for block_idx in cutlass.range(
                    lane_idx, scale_blocks, warp_threads
                ):
                    local_head = block_idx // d_scale_blocks
                    d_block = block_idx % d_scale_blocks
                    group_head = coord_hg * blk_tile_h + local_head
                    if group_head < logical_grouped_heads:
                        values = cute.make_rmem_tensor((16,), Float32)
                        amax = Float32(0)
                        for value_idx in cutlass.range_constexpr(16):
                            value = Float32(
                                sO_mma[
                                    d_block * 16 + value_idx,
                                    local_head,
                                    0,
                                ]
                            )
                            values[value_idx] = value
                            amax = cute.arch.fmax(
                                amax, cute.arch.fmax(value, -value)
                            )

                        sf_value = sf_dtype(
                            amax * global_scale * Float32(1.0 / 6.0)
                        )
                        sf_rounded = Float32(sf_value)
                        quant_multiplier = Float32(0)
                        if sf_rounded != Float32(0):
                            quant_multiplier = global_scale * cute.arch.rcp_approx(
                                sf_rounded
                            )
                        values.store(values.load() * quant_multiplier)

                        packed_fp4 = cute.make_rmem_tensor(
                            (16,), cutlass.Float4E2M1FN
                        )
                        packed_fp4.store(
                            values.load().to(cutlass.Float4E2M1FN)
                        )
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
                        m_tile_idx = row_idx // 128
                        outer_m_idx = row_idx % 32
                        inner_m_idx = (row_idx // 32) % 4
                        k_tile_idx = k_idx // 4
                        inner_k_idx = k_idx % 4
                        sf_offset = (
                            (m_tile_idx * num_k_tiles + k_tile_idx) * 512
                            + outer_m_idx * 16
                            + inner_m_idx * 4
                            + inner_k_idx
                        )
                        mOSF[sf_offset] = sf_value
            else:
                sO_final_nbar.arrive_and_wait()
                phase_mark(1, 211, True)
                cute.copy(tma_atom_o, sO_tma, gO_tma)
                if cutlass.const_expr(do_gmem_fused_red):
                    # The completion counter may not be published until this
                    # CTA's partial numerator is globally visible.
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            phase_mark(1, 212, True)

        ##############################
        # V Convert Dispatch (LDSM.U4 -> QMUL4 -> STTM)
        ##############################
        elif warpgroup_idx == conversion_warpgroup_id:
            if cutlass.const_expr(use_qmul4_v_tmem):
                if cutlass.const_expr(use_reg_reconfig):
                    cute.arch.setmaxregister_decrease(conversion_regs)

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

                tmem_debug_load_v = thr_debug_load_v = None
                tVtV_debug = tVrV_debug_shape = None
                if cutlass.const_expr(
                    os.environ.get("RESATTN_QMUL4_VALUE_DEBUG") == "1"
                ):
                    tmem_debug_load_atom_v = cute.make_copy_atom(
                        tcgen05.Ld16x128bOp(
                            tcgen05.Repetition(mma_k_bits // 128)
                        ),
                        vp_dtype,
                    )
                    tmem_debug_load_v = tcgen05.make_tmem_copy(
                        tmem_debug_load_atom_v,
                        tAtV_cvt[mma_dice + (0,)],
                    )
                    thr_debug_load_v = tmem_debug_load_v.get_slice(
                        warpgroup_tidx
                    )
                    tVtV_debug = thr_debug_load_v.partition_S(tAtV_cvt)
                    tVrV_debug_shape = thr_debug_load_v.partition_D(
                        tAtV_cvt
                    ).shape[:-1]

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
                if os.environ.get("RESATTN_DECODE_SMEM_DEBUG") == "1":
                    print(
                        f"[cvt] src={tVrV_shape} ({cute.size(tVrV_shape)}) "
                        f"dst={tVrV_cvt_shape} "
                        f"({cute.size(tVrV_cvt_shape)})",
                        flush=True,
                    )

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
                if os.environ.get("RESATTN_DECODE_SMEM_DEBUG") == "1":
                    print(
                        f"[cvt] tVsVS={tVsVS.layout} "
                        f"logical={cute.size(tVsVS.shape)}",
                        flush=True,
                    )

                v_token = v_raw_consumer.try_wait()
                cvt_token = v_ready_producer.try_acquire()
                for _s in cutlass.range(iters_s):
                    for _sk in cutlass.range_constexpr(tiles_sk):
                        for _dm in cutlass.range_constexpr(tiles_dm):
                            _cvt_flat = (
                                _s * tiles_sk * tiles_dm
                                + _sk * tiles_dm
                                + _dm
                            )
                            phase_mark(4, 1 + _cvt_flat)
                            v_handle = v_raw_consumer.wait_and_advance(v_token)
                            phase_mark(5, 1 + _cvt_flat)
                            cvt_handle = v_ready_producer.acquire_and_advance(
                                cvt_token
                            )
                            phase_mark(6, 1 + _cvt_flat)

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

                            if cutlass.const_expr(self.qmul4_skip != "conv"):
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
                            else:
                                tVrV_cvt.fill(0)
                            cute.copy(
                                thr_store_v,
                                tVrV_cvt,
                                tVtV_cvt[cpy_dice + (cvt_handle.index,)],
                            )
                            cute.arch.fence_view_async_tmem_store()
                            cvt_handle.commit()
                            phase_mark(7, 1 + _cvt_flat)
                            v_token = v_raw_consumer.try_wait()
                            cvt_token = v_ready_producer.try_acquire()

        ##############################
        # MMA KQ (BMM1) Dispatch
        ##############################
        elif warp_idx == mma_kq_warp_id:
            # Setup mma descriptors
            tAsK_desc = thrblk_mma_kq.make_fragment_A(tAsK)
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)

            if cutlass.const_expr(use_nvfp4_kq):
                copy_sfk, tCsSFK_s2t, tCtSFK_s2t = _sf_s2t_partition(
                    sf_dtype, tAsSFK, tCtSFK
                )
                copy_sfq, tCsSFQ_s2t, tCtSFQ_s2t = _sf_s2t_partition(
                    sf_dtype, tBsSFQ, tCtSFQ
                )

            # Wait for Q
            cute.arch.mbarrier_wait(q_load_mbar, phase=0)

            # SFQ → TMEM(一次;独立 mbar 正确同步)
            if cutlass.const_expr(use_nvfp4_kq):
                cute.arch.mbarrier_wait(sfq_mbar, phase=0)
                cute.copy(
                    copy_sfq,
                    tCsSFQ_s2t[None, None, None, None, 0],
                    tCtSFQ_s2t,
                )

            # Sequence loop
            for s in cutlass.range(iters_s):
                s_token = s_producer.try_acquire()

                if cutlass.const_expr(use_qmul4_v):
                    k_token = k_consumer.try_wait()
                else:
                    # Advance over V entries in the historical interleaved
                    # FIFO, then hand ownership back to KQ.
                    if s >= prefetch_iters + 1:
                        for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                            kv_consumer.advance()
                    mma_order_vp_nbar.arrive_and_wait()
                    k_token = kv_consumer.try_wait()

                s_handle = s_producer.acquire_and_advance(s_token)
                for sm in cutlass.range_constexpr(tiles_sm):
                    tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                    for dk in cutlass.range_constexpr(tiles_dk):
                        if cutlass.const_expr(use_qmul4_v):
                            k_handle = k_consumer.wait_and_advance(k_token)
                        else:
                            k_handle = kv_consumer.wait_and_advance(k_token)
                        # SFK → TMEM(每 K stage)
                        if cutlass.const_expr(use_nvfp4_kq):
                            sf_stage = k_handle.index
                            if cutlass.const_expr(self.k_sf_mode == "linear"):
                                # K's transaction barrier also covers the
                                # contiguous 1 KiB scale TMA.  Reorder that
                                # already-resident SMEM tile into the fixed
                                # tcgen05 SFA atom immediately before UTCCP.
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
                        if is_last_iter:
                            if cutlass.const_expr(not use_qmul4_v):
                                mma_order_kq_nbar.arrive()
                        else:
                            if cutlass.const_expr(use_qmul4_v):
                                k_token = k_consumer.try_wait()
                            else:
                                k_token = kv_consumer.try_wait()

                        mmas_k = cute.size(tAsK.shape[2])
                        for mma_k in cutlass.range_constexpr(mmas_k):
                            if cutlass.const_expr(use_nvfp4_kq):
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
            if cutlass.const_expr(not use_qmul4_v):
                for _s in cutlass.range_constexpr(prefetch_iters):
                    mma_order_vp_nbar.arrive_and_wait()
                    mma_order_kq_nbar.arrive()

        ##############################
        # MMA VP (BMM2) Dispatch
        ##############################
        elif warp_idx == mma_vp_warp_id:
            # Setup mma descriptors
            # 坑26 候选:set() 必须在 make_fragment 之后(KQ 的顺序)——
            # set 先行会破坏 fp4 desc 原型推导(probe4:双侧 mma_k 偏移失效)
            tAsV_desc = None
            if cutlass.const_expr(not use_qmul4_v_tmem):
                tAsV_desc = thrblk_mma_vp.make_fragment_A(tAsV)
            if cutlass.const_expr(use_nvfp4_vp):
                # 坑23:desc 直接建在分配张量上(Q 同款),不经 local_tile 视图
                tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_std)
            else:
                tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)
            tiled_mma_vp.set(
                tcgen05.Field.ACCUMULATE,
                True,
            )
            if os.environ.get("RESATTN_DECODE_SMEM_DEBUG") == "1":
                # trace 期形状对账:VP 的 gemm 用同一个 mma_k 同时索引 V 和 P
                # (mmas_k = size(tAsV.shape[2]))。两者的 K 分块必须一致,
                # 否则 token 与 P 的槽位错配 —— Phase 1 的"V token 轴塌缩"
                # 指纹正需要这一项来证实或排除。
                if cutlass.const_expr(use_qmul4_v_smem):
                    # raw 与 ready 是否**重叠**:坑5 记过"流水线拆分后必须
                    # 独立分配,否则两条流水线并发互踩"。若 raw TMA 落在
                    # ready 的地址上,转换的产物会被下一次 TMA 冲掉 ——
                    # 这与"转换惰性、写读 slot 一致"完全吻合。
                    print(
                        f"[vp] ptr raw={tAsVRaw.iterator} "
                        f"ready={tAsV.iterator} "
                        f"rawtma={tAsVRawTma.iterator}\n"
                        f"[vp] tAsVRaw={tAsVRaw.layout}\n"
                        f"[vp] tAsVReady={tAsVReady.layout}\n"
                        f"[vp] tAsVSFD={tAsVSFD.layout}",
                        flush=True,
                    )
                if cutlass.const_expr(use_qmul4_v_tmem):
                    print(
                        f"[vp] tAtV_cvt={tAtV_cvt.layout}\n"
                        f"[vp] tBsP_desc={tBsP_desc.layout}\n"
                        f"[vp] mmas_k={cute.size(tAtV_cvt.shape[2])}",
                        flush=True,
                    )
                else:
                    print(
                        f"[vp] tAsV={tAsV.layout}\n"
                        f"[vp] tAsV_desc={tAsV_desc.layout}\n"
                        f"[vp] tBsP_desc={tBsP_desc.layout}\n"
                        f"[vp] mmas_k={cute.size(tAsV.shape[2])}",
                        flush=True,
                    )

            copy_sfv = tCsSFV_s2t = tCtSFV_s2t = None
            if cutlass.const_expr(use_nvfp4_vp):
                copy_sfv, tCsSFV_s2t, tCtSFV_s2t = _sf_s2t_partition(
                    sf_dtype, tAsSFV, tCtSFV
                )
                copy_sfp, tCsSFP_s2t, tCtSFP_s2t = _sf_s2t_partition(
                    sf_dtype, tBsSFP, tCtSFP
                )
                # SFP → TMEM(常数,一次)
                cute.arch.mbarrier_wait(sfp_mbar, phase=0)
                cute.copy(
                    copy_sfp,
                    tCsSFP_s2t[None, None, None, None, 0],
                    tCtSFP_s2t,
                )

            if cutlass.const_expr(use_qmul4_v_tmem):
                phase_mark(3, 90, True)
                o_init_nbar.arrive_and_wait()
                phase_mark(3, 91, True)

            # Prefetch loop
            if cutlass.const_expr(not use_qmul4_v):
                mma_order_vp_nbar.arrive()
                for s in cutlass.range_constexpr(prefetch_iters):
                    if s < iters_s:
                        for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                            kv_consumer.advance()
                    mma_order_kq_nbar.arrive_and_wait()
                    mma_order_vp_nbar.arrive()

            if cutlass.const_expr(use_qmul4_v_tmem):
                # Every non-empty split has phase 0. Phase 1 exists only when
                # there is a second sequence tile; later iterations reuse the
                # corrected ring contents and therefore accumulate.
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
                phase_mark(3, 92, True)
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
                    phase_mark(3, 93, True)
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
                    phase_mark(3, 94 + _s, True)

            # Sequence loop
            vp_iters_s = 0 if use_qmul4_v_tmem else iters_s
            for s in cutlass.range(vp_iters_s):
                p_token = p_consumer.try_wait()
                o_token = o_producer.try_acquire()

                if cutlass.const_expr(use_qmul4_v_smem):
                    v_token = v_raw_consumer.try_wait()
                    v_ready_token = v_ready_producer.try_acquire()
                elif cutlass.const_expr(use_qmul4_v_tmem):
                    v_ready_token = v_ready_consumer.try_wait()
                else:
                    # Advance over K entries in the historical interleaved
                    # FIFO, then hand ownership back to VP.
                    if s < iters_s - prefetch_iters:
                        for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                            kv_consumer.advance()
                    mma_order_kq_nbar.arrive_and_wait()
                    v_token = kv_consumer.try_wait()

                p_handle = p_consumer.wait_and_advance(p_token)
                o_handle = o_producer.acquire_and_advance(o_token)
                for sk in cutlass.range_constexpr(tiles_sk):
                    for dm in cutlass.range_constexpr(tiles_dm):
                        if cutlass.const_expr(use_qmul4_v_smem):
                            v_handle = v_raw_consumer.wait_and_advance(v_token)
                            v_ready_handle = v_ready_producer.acquire_and_advance(
                                v_ready_token
                            )
                        elif cutlass.const_expr(use_qmul4_v_tmem):
                            v_mma_handle = v_ready_consumer.wait_and_advance(
                                v_ready_token
                            )
                        else:
                            v_handle = kv_consumer.wait_and_advance(v_token)
                        is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                        if is_last_iter:
                            if cutlass.const_expr(not use_qmul4_v):
                                mma_order_vp_nbar.arrive()
                        else:
                            if cutlass.const_expr(use_qmul4_v_smem):
                                v_token = v_raw_consumer.try_wait()
                                v_ready_token = v_ready_producer.try_acquire()
                            elif cutlass.const_expr(use_qmul4_v_tmem):
                                v_ready_token = v_ready_consumer.try_wait()
                            else:
                                v_token = kv_consumer.try_wait()

                        if cutlass.const_expr(use_qmul4_v_smem):
                            # qmul4_skip="conv":只摘掉转换,握手/MMA 全留 ——
                            # ready 缓冲里是垃圾值,数值必错,但**不应该 trap**;
                            # 若 trap 随之消失,嫌疑就锁定在转换的访存几何上。
                            if cutlass.const_expr(self.qmul4_skip != "conv"):
                                dequantize_nvfp4_tile_d(
                                    tAsVRaw[None, None, v_handle.index],
                                    tAsVSFD[None, None, v_handle.index],
                                    tAsVReady[None, None, v_ready_handle.index],
                                    lane_idx,
                                    warp_threads,
                                )
                            # Orders every lane's STS.128 before its full-bar
                            # arrival. The one-lane UMMA consumer cannot pass
                            # wait until all 32 producer arrivals are visible.
                            cute.arch.fence_view_async_shared()
                            v_handle.release()
                            v_ready_handle.commit()
                            v_mma_handle = v_ready_consumer.wait_and_advance()
                            if cutlass.const_expr(
                                os.environ.get("RESATTN_QMUL4_IDX_DEBUG") == "1"
                            ):
                                # 决定性测量:转换写的 slot 与 MMA 读的 slot
                                # 是否同一个。消融已证明"转换产物惰性、而 MMA
                                # 是 O 的唯一写入者" —— 同内存同布局下,只剩
                                # stage 下标不一致这一种解释。
                                if lane_idx == 0:
                                    cute.printf(
                                        "IDX write=%d read=%d\n",
                                        v_ready_handle.index,
                                        v_mma_handle.index,
                                    )

                        # SFV → TMEM(每 V stage)
                        # perf_probe=4:跳过此 UTCCP(读 TMEM 陈值,数值废、时序真)
                        # —— 给"单 slot 强迫 UTCCP 与 gemm 串行"单独标价
                        if cutlass.const_expr(use_nvfp4_vp and self.perf_probe != 4):
                            cute.copy(
                                copy_sfv,
                                tCsSFV_s2t[None, None, None, None, v_handle.index],
                                tCtSFV_s2t,
                            )
                        mmas_k = (
                            cute.size(tAtV_cvt.shape[2])
                            if use_qmul4_v_tmem
                            else cute.size(tAsV.shape[2])
                        )
                        for mma_k in cutlass.range_constexpr(mmas_k):
                            if cutlass.const_expr(use_nvfp4_vp):
                                tiled_mma_vp.set(
                                    tcgen05.Field.SFA,
                                    tCtSFV[None, None, mma_k].iterator,
                                )
                                tiled_mma_vp.set(
                                    tcgen05.Field.SFB,
                                    tCtSFP[None, None, mma_k].iterator,
                                )
                                # 坑25(probe2/3:每 stage 4 个 gemm 仅 1 个入账):
                                # 显式每 gemm 重申 ACCUMULATE(KQ 同款防御),
                                # 防 set(SF*)/desc 路径隐性复位
                                tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
                            if cutlass.const_expr(use_nvfp4_vp):
                                _pdesc = tBsP_desc[
                                    None, None, mma_k,
                                    sk + tiles_sk * p_handle.index,
                                ]
                            else:
                                _pdesc = tBsP_desc[None, None, mma_k, sk, p_handle.index]
                            # qmul4_skip="mma":只摘掉 VP MMA,转换与全部握手
                            # 保留 —— 若 trap 随之消失,嫌疑锁定在 FP8 VP MMA
                            # 的 descriptor / TMEM 几何上。
                            if cutlass.const_expr(self.qmul4_skip != "mma"):
                                if cutlass.const_expr(use_qmul4_v_tmem):
                                    _voperand = tAtV_cvt[
                                        None,
                                        None,
                                        mma_k,
                                        v_mma_handle.index,
                                    ]
                                else:
                                    _voperand = tAsV_desc[
                                        None,
                                        None,
                                        mma_k,
                                        (
                                            v_mma_handle.index
                                            if use_qmul4_v_smem
                                            else v_handle.index
                                        ),
                                    ]
                                if cutlass.const_expr(
                                    use_qmul4_v_tmem
                                    and sk == 0
                                    and dm == 0
                                    and mma_k == 0
                                ):
                                    # Each O ring phase is initialized once.
                                    # Branch around two constant ``set`` calls
                                    # on the *same* MMA object: this emits
                                    # literal !UPT/UPT while preserving the
                                    # accumulator descriptor.  A second MMA
                                    # object writes through distinct idesc
                                    # state and therefore cannot initialize
                                    # the accumulator used below.
                                    if s < o_stages:
                                        tiled_mma_vp.set(
                                            tcgen05.Field.ACCUMULATE,
                                            False,
                                        )
                                        cute.gemm(
                                            tiled_mma_vp,
                                            tCtO[mma_dice + (dm, o_handle.index)],
                                            _voperand,
                                            _pdesc,
                                            tCtO[mma_dice + (dm, o_handle.index)],
                                        )
                                    else:
                                        tiled_mma_vp.set(
                                            tcgen05.Field.ACCUMULATE,
                                            True,
                                        )
                                        cute.gemm(
                                            tiled_mma_vp,
                                            tCtO[mma_dice + (dm, o_handle.index)],
                                            _voperand,
                                            _pdesc,
                                            tCtO[mma_dice + (dm, o_handle.index)],
                                        )
                                else:
                                    if cutlass.const_expr(use_qmul4_v_tmem):
                                        tiled_mma_vp.set(
                                            tcgen05.Field.ACCUMULATE,
                                            True,
                                        )
                                    cute.gemm(
                                        tiled_mma_vp,
                                        tCtO[mma_dice + (dm, o_handle.index)],
                                        _voperand,
                                        _pdesc,
                                        tCtO[mma_dice + (dm, o_handle.index)],
                                    )
                        if cutlass.const_expr(use_qmul4_v):
                            # PipelineAsyncUmma's empty arrival is tied to the
                            # tensor-core scoreboard, so this protects the
                            # ready slot even though release follows issue.
                            v_mma_handle.release()
                        else:
                            v_handle.release()
                p_handle.release()
                o_handle.commit()

            # Wait for signal to dealloc tmem, then dealloc
            if iters_s == 1:
                # Epilogue still reads the empty buffer
                o_producer.commit()
                o_producer.advance()
            phase_mark(3, 98, True)
            o_producer.tail()
            phase_mark(3, 99, True)
            cute.arch.relinquish_tmem_alloc_permit()
            phase_mark(3, 100, True)
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)
            phase_mark(3, 101, True)

        ##############################
        # Softmax Dispatch
        ##############################
        elif warpgroup_idx in softmax_warpgroup_ids:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(softmax_regs)

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
            if cutlass.const_expr(use_nvfp4_vp):
                # v2:16-dp 原子,1 线程持 2 行(距8);rep 单位=256bit=8 列 → //8
                tmem_load_op_s = tcgen05.Ld16x256bOp(
                    tcgen05.Repetition(tmem_repeat_op_s // 8)
                )
            else:
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
            if cutlass.const_expr(use_nvfp4_vp):
                # v2:mode1=GRP=2(两个 16-dp 实例 = 两个 16-token SF 组),必须保留。
                # 所有权几何(identity-tensor 破译,详见 plan doc "v2 所有权几何"):
                #   token = 32*w + t1 + 8*v1 + 16*grp + mma_tile_m*v3
                #   col   = 2*t0 + v0 + 8*v2      (lane = t0 + 4*t1, w = warp)
                tStS = tStS[None, None, 0, 0, None, None]  # (CPY, GRP, #CPY_SM, s_stages)
                tSsP = tSsP[None, None, 0, 0, None, None]
                # perf:u32 视图(裸指针,手工 XOR)—— 向量化 P store 用
                _sPb32 = cute.recast_tensor(tBsP_std, cutlass.Int32)
                sP_u32 = cute.make_tensor(
                    _sPb32.iterator, cute.make_layout(cute.size(_sPb32.layout))
                )
                # v2 P store 用字节视图直写(π 槽位),不经 tSsP 寻址
                # 坑22/23:写路径与 desc 必须同源。tBsP_std(S<2,4,3>)的字节视图,
                # recast 丢 swizzle 后经 recast_ptr 挂回 —— 标量写应用指针 swizzle
                # (本地 5090 探针 1024/1024 实证,字节空间)。
                _sPb = cute.recast_tensor(tBsP_std, cutlass.Int8)
                sP_flat = cute.make_tensor(
                    cute.recast_ptr(
                        _sPb.iterator,
                        smem_layout_p_std.inner,  # 布局自带的 swizzle(k_vp 决定)
                        dtype=cutlass.Int8,
                    ),
                    cute.make_layout(cute.size(_sPb.layout)),
                )
            else:
                tStS = tStS[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, s_stages)
                tSsP = tSsP[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, p_stages)

            # Construct copy atom for L
            if cutlass.const_expr(use_nvfp4_vp):
                # v2:L 与 S 同 16-dp 几何(rep 单位 256bit=8 列),colsum 只归约
                # v3(SM tile),保留 (v1,grp) → 每 warp 写满 32 DP,correction 零改动
                tmem_repeat_op_l = tcgen05.Repetition(blk_tile_n // 8)
                tmem_load_op_l = tcgen05.Ld16x256bOp(tmem_repeat_op_l)
                tmem_store_op_l = tcgen05.St16x256bOp(tmem_repeat_op_l)
            else:
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
            if cutlass.const_expr(use_nvfp4_vp):
                
                tStL = (
                    tStL[None, None, 0, 0, None]
                    if cutlass.const_expr(softmax_warpgroups == 1)
                    else tStL[None, None, 0, 0]
                )
                tSrL_shape = tSrL_shape[:2]
            else:
                tStL = (
                    tStL[None, 0, 0, 0, None]
                    if cutlass.const_expr(softmax_warpgroups == 1)
                    else tStL[None, 0, 0, 0]
                )
                tSrL_shape = tSrL_shape[:1]

            # Causal masking for spec decode verification
            masked_iters_s = masked_start_s = masked_coord_s = 0
            check_safe_max = False
            if cutlass.const_expr(not tma_mask):
                is_last_split = kv_split_idx == (tiles_s - 1) % kv_splits
                is_prev_split = kv_split_idx == (tiles_s - 2) % kv_splits
                is_last_phase = softmax_phase == (iters_s - 1) % softmax_warpgroups
                is_prev_phase = softmax_phase == (iters_s - 2) % softmax_warpgroups
                # 2 masked tiles
                if 0 < seqlen % blk_tile_s < prediction:
                    # 1 split masks 2 tiles
                    if kv_splits == 1:
                        masked_start_s = 0 if is_prev_phase else 1
                        masked_iters_s = 2
                        masked_coord_s = tiles_s - 2
                    # 2 splits mask 1 tile each
                    elif (is_last_split or is_prev_split) and is_last_phase:
                        masked_start_s = 0
                        masked_iters_s = 1
                        masked_coord_s = (
                            (tiles_s - 1) if is_last_split else (tiles_s - 2)
                        )
                        # if last split only has 1 tile then some cols will never
                        # see inbounds values, so P = exp(-inf + inf) -> nan
                        check_safe_max = is_last_split and iters_s == 1
                # 1 masked tile
                elif prediction > 1 or seqlen % blk_tile_s != 0:
                    # 1 split masks 1 tile
                    if is_last_split and is_last_phase:
                        masked_start_s = 0
                        masked_iters_s = 1
                        masked_coord_s = tiles_s - 1

            # Sequence loop
            num_loops = 2 if not tma_mask else 1
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
                    phase_mark(softmax_phase, 10 + s)
                    s_token = s_consumer.try_wait()
                    p_token = p_producer.try_acquire()

                    # Load S from tmem and notify BMM1
                    s_handle = s_consumer.wait_and_advance(s_token)
                    if cutlass.const_expr(use_nvfp4_vp):
                        tStS_s = tStS[None, None, None, s_handle.index]
                    else:
                        tStS_s = tStS[None, None, s_handle.index]
                    tSrS_s = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                    cute.copy(tmem_load_atom_s, tStS_s, tSrS_s)
                    cute.arch.fence_view_async_tmem_load()
                    s_handle.release()

                    # Apply causal mask
                    if cutlass.const_expr(use_nvfp4_vp):
                        # ---- v2 几何(identity 破译,见 plan doc):线程持有
                        # token = 32*w + t1 + 8*v1 + 16*grp + mma_tile_m*v3
                        # col   = 2*t0 + v0 + 8*v2
                        _lane = warpgroup_tidx % 32
                        _t0 = _lane % 4
                        _t1 = _lane // 4
                        _w = warpgroup_tidx // 32
                        _tok_base = _w * 32 + _t1
                        _col_base = 2 * _t0
                        _b4 = _t1 % 4   # 4x4 转置组内位置
                        _k4 = _t1 // 4  # 组序(每组 4 个连续 token 字节)
                        # v3(SM tile)模在 tiles_sm=4 时被 DSL 拆成嵌套 (2,2)
                        # → 用 flat 视图统一成一维,索引照旧线性
                        _fshape = tSrS_s.shape[0][0]
                        _v0e, _v1e, _v2e = _fshape[0], _fshape[1], _fshape[2]
                        _v3e = cute.size(_fshape[3])
                        _tSrS_f = cute.make_tensor(
                            tSrS_s.iterator,
                            cute.make_layout(
                                (((_v0e, _v1e, _v2e, _v3e), 1), tSrS_s.shape[1], 1)
                            ),
                        )
                        tSrS_s = _tSrS_f
                        _grpe = tSrS_s.shape[1]
                        assert _v1e == 2, "v1 应为 16-dp 行对"
                        if cutlass.const_expr(is_masked_loop):
                            _off_s = (masked_coord_s + s) * blk_tile_s + _tok_base
                            offset_p = seqlen - prediction + coord_p * blk_tile_p
                            for _g in cutlass.range_constexpr(_grpe):
                                for _v3 in cutlass.range_constexpr(_v3e):
                                    for _v1 in cutlass.range_constexpr(_v1e):
                                        _tok = _off_s + 8 * _v1 + 16 * _g + mma_tile_m * _v3
                                        for _v2 in cutlass.range_constexpr(_v2e):
                                            for _v0 in cutlass.range_constexpr(_v0e):
                                                _p = (_col_base + _v0 + 8 * _v2) // blk_tile_h
                                                _oob = _tok > offset_p + _p
                                                _m = -Float32.inf if _oob else Float32(0)
                                                tSrS_s[((_v0, _v1, _v2, _v3), 0), _g, 0] = (
                                                    tSrS_s[((_v0, _v1, _v2, _v3), 0), _g, 0] + _m
                                                )
                        scores = tSrS_s.load()
                    elif cutlass.const_expr(is_masked_loop):
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

                    # Residue fold(R1 v0):Q 按 [base; res] concat,S 列 [0:nf)=base、
                    # [nf:2nf)=res;折叠 = 两半相加(在 softmax 之前)。v0 把折叠结果
                    # 复制回两半 → 下游(colmax/P/VP/O)形状与字节零改动,res 半列输出
                    # 为 base 的副本,host 侧只读 base 半。mask 为加性 {0,-inf},与
                    # 折叠可交换(已在插入点之前施加)。
                    if cutlass.const_expr(self.residue_fold):
                        assert not use_nvfp4_vp, "R1+fp4P 折叠待 R2 接线(v2 几何:v2 子模高半)"
                        _nf = blk_tile_n // 2
                        _s3 = scores.reshape((_nf, 2, tiles_sm))
                        _fold = _s3[None, 0, None] + _s3[None, 1, None]
                        _dup = cute.make_rmem_tensor((_nf, 2, tiles_sm), acc_dtype)
                        _dup_b = _dup[None, 0, None]
                        _dup_b.store(_fold)
                        _dup_r = _dup[None, 1, None]
                        _dup_r.store(_fold)
                        scores = _dup.load().reshape((blk_tile_n, tiles_sm))

                    if cutlass.const_expr(use_nvfp4_vp):
                        # v2 colmax:线程内归约 8 个 token 槽/列(v1,v3,grp)
                        _rmax = cute.make_rmem_tensor((_v0e, _v2e), acc_dtype)
                        _rmax.fill(-Float32.inf)
                        for _v0 in cutlass.range_constexpr(_v0e):
                            for _v2 in cutlass.range_constexpr(_v2e):
                                for _g in cutlass.range_constexpr(_grpe):
                                    for _v3 in cutlass.range_constexpr(_v3e):
                                        for _v1 in cutlass.range_constexpr(_v1e):
                                            _x = tSrS_s[((_v0, _v1, _v2, _v3), 0), _g, 0]
                                            _cur = _rmax[_v0, _v2]
                                            _rmax[_v0, _v2] = _x if _x > _cur else _cur
                        # t1 组蝶式归约(lane = t0 + 4*t1;offset 16/8/4 = t1 的三个位,
                        # t0 位不混 → 4 个列组各自独立归约,FA2 Allreduce 的跨步版)
                        for _off in (16, 8, 4):
                            for _v0 in cutlass.range_constexpr(_v0e):
                                for _v2 in cutlass.range_constexpr(_v2e):
                                    _cur = _rmax[_v0, _v2]
                                    _oth = cute.arch.shuffle_sync_bfly(_cur, offset=_off)
                                    _rmax[_v0, _v2] = _oth if _oth > _cur else _cur

                        # smem 归约(跨 warp/跨 warpgroup):t1==0 的 lane 写自己的 4 列
                        phase_mark(softmax_phase, 20 + s)
                        if cutlass.const_expr(softmax_warpgroups > 1):
                            sM_acquire_nbar.arrive_and_wait()
                        phase_mark(softmax_phase, 30 + s)
                        sM_consumer_nbar.arrive_and_wait()
                        if _t1 == 0:
                            for _v0 in cutlass.range_constexpr(_v0e):
                                for _v2 in cutlass.range_constexpr(_v2e):
                                    smem_fmax(
                                        sM.iterator + sM.layout(_col_base + _v0 + 8 * _v2),
                                        _rmax[_v0, _v2] * scale_s_log2_e,
                                    )

                        # 等 colmax 就绪,按动态列号从 smem 读 4 列(避免寄存器动态索引)
                        sM_producer_nbar.arrive_and_wait()
                        _cmax = cute.make_rmem_tensor((_v0e, _v2e), acc_dtype)
                        for _v0 in cutlass.range_constexpr(_v0e):
                            for _v2 in cutlass.range_constexpr(_v2e):
                                _cmax[_v0, _v2] = sM[_col_base + _v0 + 8 * _v2]
                        if cutlass.const_expr(softmax_warpgroups > 1):
                            sM_release_nbar.arrive()

                        # Handle if we never saw any in-bounds values
                        if cutlass.const_expr(is_masked_loop and do_atomic_red):
                            if check_safe_max:
                                for _v0 in cutlass.range_constexpr(_v0e):
                                    for _v2 in cutlass.range_constexpr(_v2e):
                                        _c = _cmax[_v0, _v2]
                                        _cmax[_v0, _v2] = (
                                            Float32(0) if _c == -Float32.inf else _c
                                        )

                        # Wait for empty P buffer
                        p_handle = p_producer.acquire_and_advance(p_token)

                        # perf:colmax 用 stride-0 广播视图(替代 32 次标量写),
                        # 直接进 TensorSSA 向量 exp2
                        _cm_bc = cute.make_tensor(
                            _cmax.iterator,
                            cute.make_layout(
                                (((_v0e, _v1e, _v2e, _v3e), 1), _grpe, 1),
                                stride=(((1, 0, _v0e, 0), 0), 0, 0),
                            ),
                        )
                        probs = exp2(scale_s_log2_e * scores - _cm_bc.load())
                    else:
                        # Reduce colmax in thread RF
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
                        phase_mark(softmax_phase, 20 + s)
                        if cutlass.const_expr(softmax_warpgroups > 1):
                            sM_acquire_nbar.arrive_and_wait()
                        phase_mark(softmax_phase, 30 + s)
                        sM_consumer_nbar.arrive_and_wait()
                        if lane_store_max:
                            if cutlass.const_expr(use_qmul4_v_tmem):
                                # TRT-style FP8-P rescale suppression.  Keep
                                # the old normalization origin while the new
                                # tile max is at most 8 log2 units higher.
                                # exp2(delta) <= 256 remains below E4M3 max
                                # 448, so numerator and denominator retain a
                                # common scale without saturating P.
                                old_max = sM[lane_idx]
                                if rM_lane - old_max > Float32(8):
                                    smem_fmax(
                                        sM.iterator + sM.layout(lane_idx),
                                        rM_lane,
                                    )
                            else:
                                smem_fmax(
                                    sM.iterator + sM.layout(lane_idx), rM_lane
                                )

                        # Wait for colmax and load
                        sM_producer_nbar.arrive_and_wait()
                        colmax = sM.load()
                        if cutlass.const_expr(softmax_warpgroups > 1):
                            sM_release_nbar.arrive()

                        # Handle if we never saw any in-bounds values
                        if cutlass.const_expr(is_masked_loop and do_atomic_red):
                            if check_safe_max:
                                rM = cute.make_rmem_tensor_like(colmax)
                                rM.store(colmax)
                                for n in cutlass.range_constexpr(blk_tile_n):
                                    if rM[n] == -Float32.inf:
                                        rM[n] = Float32(0)
                                colmax = rM.load()

                        # Wait for empty P buffer
                        # Here so we can interleave ex2 with convert ops
                        p_handle = p_producer.acquire_and_advance(p_token)
                        tSsP_s = tSsP[None, None, p_handle.index]

                        # Compute online softmax
                        probs = exp2(scale_s_log2_e * scores - colmax)

                    # Store P to smem and notify BMM2
                    if cutlass.const_expr(use_nvfp4_vp):
                        
                        
                        
                        
                        
                        
                        _probs_r = cute.make_rmem_tensor(tSrS_s.shape, acc_dtype)
                        _probs_r.store(probs)
                        _do_pstore = cutlass.const_expr(self.perf_probe < 2)

                        for _g in cutlass.range_constexpr(_grpe if _do_pstore else 0):
                            for _v3 in cutlass.range_constexpr(_v3e):
                                # 坑20:窄精度 rmem 向量须 ≥32bit → 每 (grp,v3) 8 值
                                # 一次转换。v1 放 stride-1 内模 → 字节两 nibble 恰为
                                # (tok, tok+8) 对(低 nibble=v1=0=物理偶槽)
                                _pr = cute.make_rmem_tensor((2, _v0e, _v2e), acc_dtype)
                                for _v2 in cutlass.range_constexpr(_v2e):
                                    for _v0 in cutlass.range_constexpr(_v0e):
                                        for _v1 in cutlass.range_constexpr(2):
                                            _pr[_v1, _v0, _v2] = (
                                                _probs_r[((_v0, _v1, _v2, _v3), 0), _g, 0]
                                                * 5.818181818181818
                                            )
                                _pr4 = cute.make_rmem_tensor((2, _v0e, _v2e), vp_dtype)
                                _pr4.store(_pr.load().to(vp_dtype))
                                _pb = cute.recast_tensor(_pr4, cutlass.Int8)
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                _hps = max(mma_tile_k_vp // mma_tile_m, 1)
                                _sk_of_v3 = _v3 // _hps
                                _v3_in_stage = _v3 % _hps
                                for _v2 in cutlass.range_constexpr(_v2e):
                                    for _v0 in cutlass.range_constexpr(_v0e):
                                        _f = (
                                            (16 * _w + 8 * _g + _t1)
                                            + _v3_in_stage * (mma_tile_m // 2)
                                            + (_col_base + _v0 + 8 * _v2)
                                            * (mma_tile_k_vp // 2)
                                            + (_sk_of_v3 + tiles_sk * p_handle.index)
                                            * (mma_tile_n * mma_tile_k_vp // 2)
                                        )
                                        if cutlass.const_expr(self.perf_probe == 0):
                                            sP_flat[_f] = _pb[0, _v0, _v2]
                    else:
                        _p_quant = probs.to(mma_dtype).reshape(tSsP_s.shape)
                        if cutlass.const_expr(
                            use_qmul4_v_tmem
                            and os.environ.get(
                                "RESATTN_QMUL4_VALUE_DEBUG"
                            )
                            == "1"
                        ):
                            _p_debug = cute.make_rmem_tensor(
                                tSsP_s.shape, mma_dtype
                            )
                            _p_debug.store(_p_quant)
                            _p_debug_flat = cute.make_tensor(
                                _p_debug.iterator,
                                cute.make_layout(cute.size(tSsP_s.shape)),
                            )
                            _p_has_nan = cutlass.Boolean(False)
                            for _pi in cutlass.range_constexpr(
                                cute.size(tSsP_s.shape)
                            ):
                                _px = Float32(_p_debug_flat[_pi])
                                _p_has_nan |= _px != _px
                            if _p_has_nan:
                                cute.printf(
                                    "P_NAN tid=%d stage=%d\n",
                                    warpgroup_tidx,
                                    p_handle.index,
                                )
                            tSsP_s.store(_p_debug.load())
                        else:
                            tSsP_s.store(_p_quant)
                    cute.arch.fence_view_async_shared()
                    p_handle.commit()
                    phase_mark(softmax_phase, 40 + s)

                    # Accumulate per-thread colsum
                    tSrL = cute.make_rmem_tensor(tSrL_shape, acc_dtype)
                    if cutlass.const_expr(use_nvfp4_vp):
                        # v2:只归约 v3(SM tile,同旧代码的 sm-fold),保留 (v1,grp)
                        # → St16x256b 每 warp 写满 32 DP;L 的 DP 语义 = partial 槽,
                        # correction 归约全部 128 DP,不关心槽里是哪个 token
                        for _g in cutlass.range_constexpr(_grpe):
                            for _v2 in cutlass.range_constexpr(_v2e):
                                for _v1 in cutlass.range_constexpr(_v1e):
                                    for _v0 in cutlass.range_constexpr(_v0e):
                                        _acc = Float32(0)
                                        for _v3 in cutlass.range_constexpr(_v3e):
                                            _acc = _acc + _probs_r[
                                                ((_v0, _v1, _v2, _v3), 0), _g, 0
                                            ]
                                        tSrL[((_v0, _v1, _v2), 0), _g] = _acc
                    else:
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
                    phase_mark(softmax_phase, 50 + s)

                    # Advance again for dual warpgroups
                    if cutlass.const_expr(softmax_warpgroups > 1):
                        s_consumer.advance()
                        p_producer.advance()

        ##############################
        # Correction Dispatch
        ##############################
        elif warpgroup_idx == correction_warpgroup_id:
            # Alloc registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_increase(correction_regs)

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
                        # Each correction warp owns one contiguous 16-float
                        # sL column. Lanes 0..15 publish one factor each; the
                        # warp-local barrier makes those stores visible before
                        # every lane reloads the vector as four 128-bit chunks.
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
            if cutlass.const_expr(use_qmul4_v_tmem):
                phase_mark(2, 60)
                o_init_nbar.arrive_and_wait()
                phase_mark(2, 61)

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
                phase_mark(2, 70 + s)
                o_token = o_consumer.try_wait()
                o_handle = o_consumer.wait_and_advance(o_token)
                phase_mark(2, 80 + s)

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
                phase_mark(2, 84 + s)

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

            phase_mark(2, 86)

            # Notify for final colmax if we didn't enter loop
            if iters_s <= o_stages:
                sM_final_nbar.arrive()

            # Handle if we never saw any in-bounds values
            if cutlass.const_expr(not tma_mask and do_atomic_red):
                if sM_lane_prev == -Float32.inf:
                    sM_lane_prev = Float32(0)

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
                    phase_mark(2, 100)
                    colsum_prev = colsum_load(phase)
                    phase_mark(2, 101)
                    colsum_final = colsum_load(phase ^ 1)
                    phase_mark(2, 102)
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

            phase_mark(2, 103)
            sL_final_nbar.arrive_and_wait()
            phase_mark(2, 104)

            # Load O of s-1, s
            tOrO_tail = cute.make_rmem_tensor((*tOsO.shape, o_stages), acc_dtype)
            for s in cutlass.range_constexpr(o_stages):
                phase_s = tail_phase ^ s
                tOtO_phase = tOtO[None, None, phase_s]
                tOrO_tail_s = tOrO_tail[None, None, s]
                phase_mark(2, 110 + s)
                o_handle = o_consumer.wait_and_advance()
                phase_mark(2, 120 + s)
                cute.copy(tmem_load_atom_o, tOtO_phase, tOrO_tail_s)
                phase_mark(2, 130 + s)
                cute.arch.fence_view_async_tmem_load()
                phase_mark(2, 140 + s)
                o_handle.release()  # Final release signals tmem dealloc
                phase_mark(2, 150 + s)

            tOrO_prev = tOrO_tail[None, None, 0].load()
            tOrO_final = tOrO_tail[None, None, 1].load()
            phase_mark(2, 160)

            # Correct and accumulate output
            output_prev = tOrO_prev.reshape((blk_tile_n, tiles_dm))
            output_final = tOrO_final.reshape((blk_tile_n, tiles_dm))
            output_final += correction * output_prev
            phase_mark(2, 161)

            if cutlass.const_expr(do_cga_red):
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

                partial_o = cute.make_rmem_tensor((blk_tile_n,), acc_dtype)
                partial_o.store(output_final.reshape((blk_tile_n,)))
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
                            corr_old = exp2(reduced_max - next_max)
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

                        final_o[head] = o_dtype(
                            reduced_o
                            * cute.arch.rcp_approx(reduced_sum)
                            * scale_o
                        )

                    # VP is a 128xN tile and the Ld32x32b correction slice
                    # assigns one D row plus all N heads to each physical
                    # correction thread.  The owner predicate selects every
                    # physical thread exactly once across the cluster, so
                    # direct scalar stores cover the output without a CTA0
                    # gather/TMA pass.  A TMEM copy slice cannot partition a
                    # global tensor for CopyUniversalOp: it produces a
                    # coordinate tensor rather than a gmem MemRef.
                    for head in cutlass.range_constexpr(blk_tile_n):
                        mOFinal[
                            warpgroup_tidx,
                            (coord_hg * blk_tile_h + head, 0),
                            (coord_hk, coord_b),
                        ] = final_o[head]
            else:
                # Apply final normalization.  atomic's cluster reduction and
                # the no-split path publish it through sM.
                if cutlass.const_expr(do_atomic_red or do_none_red):
                    phase_mark(2, 170)
                    sM_final_nbar.arrive_and_wait()
                    phase_mark(2, 171)
                    normalization = sM.load()
                    output_final *= normalization
                    phase_mark(2, 172)

                # Store O to smem and notify the TMA warp.
                tOsO.store(output_final.to(o_dtype).reshape(tOsO.shape))
                phase_mark(2, 180)
                cute.arch.fence_view_async_shared()
                phase_mark(2, 181)
                sO_final_nbar.arrive()
                phase_mark(2, 182)

        ##############################
        # Reduction Dispatch
        ##############################
        if warp_idx == reduction_warp_id:
            phase_mark(0, 200, True)
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
            elif cutlass.const_expr(do_atomic_red):
                gL = None
                if cutlass.const_expr(store_lse):
                    gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))
                self.reduction_cluster(
                    blk_tile_n,
                    kv_splits,
                    kv_split_idx,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    reduction_mbars_ptr,
                    sM,
                    sL,
                    sR,
                    gL,
                    scale_o,
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
                gL = None
                if cutlass.const_expr(store_lse):
                    gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))
                self.reduction_none(
                    lane_store_max,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    sM,
                    sL,
                    gL,
                    scale_o,
                )
            phase_mark(0, 201, True)
            cute.arch.griddepcontrol_launch_dependents()

        ##############################
        # Fused global-memory reduction
        ##############################
        if cutlass.const_expr(do_gmem_fused_red):
            # All role-specific work is complete at this point. In
            # particular, the TMA warp has waited for partial-O's bulk store
            # and the reduction warp has synchronously stored max/sum. TRT's
            # mode-1 protocol does not serialize the output on one CTA: the
            # last four finishing CTAs split the four 32-thread output slices.
            # The first three wait until the final CTA resets the completion
            # word, which also acquires all published partials.
            if (
                warpgroup_idx == 0
                or warpgroup_idx == correction_warpgroup_id
            ):
                gmem_completion_nbar.arrive_and_wait()

                completion_idx = (
                    coord_hb_linear * grid_tiles_hp + coord_hp_linear
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
                    if old_count >= kv_splits - Int32(4):
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
                    elif old_count >= kv_splits - Int32(4):
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
                        mL,
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
        mL: Optional[cute.Tensor],
        mL_partial: cute.Tensor,
        mM_partial: cute.Tensor,
        scale_o: Float32,
    ):
        """Merge split partials and emit one direct-NVFP4 output block/thread.

        This is the data-parallel half of TRT's mode-1 protocol. The caller
        dynamically selects the last CTA; its 128 correction threads then map
        exactly onto the 16 heads x 8 block16 D slices of a Q16/D128 tile.
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

            amax = Float32(0)
            for value_idx in cutlass.range_constexpr(16):
                value = final_values[value_idx]
                amax = cute.arch.fmax(
                    amax, cute.arch.fmax(value, -value)
                )

            global_scale = Float32(mOGlobalScale[0])
            sf_value = cutlass.Float8E4M3FN(
                amax * global_scale * Float32(1.0 / 6.0)
            )
            sf_rounded = Float32(sf_value)
            quant_multiplier = Float32(0)
            if sf_rounded != Float32(0):
                quant_multiplier = global_scale * cute.arch.rcp_approx(
                    sf_rounded
                )
            final_values.store(final_values.load() * quant_multiplier)

            packed_fp4 = cute.make_rmem_tensor(
                (16,), cutlass.Float4E2M1FN
            )
            packed_fp4.store(
                final_values.load().to(cutlass.Float4E2M1FN)
            )
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
            num_k_tiles = cute.ceil_div(hidden, 64)
            row_idx = coord_b * prediction + coord_p
            m_tile_idx = row_idx // 128
            outer_m_idx = row_idx % 32
            inner_m_idx = (row_idx // 32) % 4
            k_tile_idx = k_idx // 4
            inner_k_idx = k_idx % 4
            sf_offset = (
                (m_tile_idx * num_k_tiles + k_tile_idx) * 512
                + outer_m_idx * 16
                + inner_m_idx * 4
                + inner_k_idx
            )
            mOSF[sf_offset] = sf_value

            if cutlass.const_expr(mL is not None and mL.dtype == Float32):
                if d_block == 0:
                    mL[
                        (group_head, coord_p), (coord_hk, coord_b)
                    ] = max_final + cute.math.log2(sum_final)

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
        gL: Optional[cute.Tensor],
        scale_o: Float32,
    ):
        store_lse = gL is not None
        colmax = Float32(0)
        colsum = Float32(0)
        sM_final_nbar.arrive_and_wait()
        if cutlass.const_expr(store_lse):
            if lane_store_max:
                colmax = sM[lane_idx]
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane = sL[lane_idx, None]
            colsum = sL_lane[0] + sL_lane[1] + sL_lane[2] + sL_lane[3]
            normalization = cute.arch.rcp_approx(colsum) * scale_o
            sM[lane_idx] = normalization
        sM_final_nbar.arrive()
        if cutlass.const_expr(store_lse):
            if lane_store_max:
                gL[lane_idx] = colmax + cute.math.log2(colsum)

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

    @staticmethod
    @cute.jit
    def reduction_cluster(
        blk_tile_n: int,
        kv_splits: Int32,
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        reduction_mbars_ptr: cute.Pointer,
        sM: cute.Tensor,
        sL: cute.Tensor,
        sR: cute.Tensor,
        gL: Optional[cute.Tensor],
        scale_o: Float32,
    ):
        acc_dtype = sM.dtype
        colmax_bits = blk_tile_n * acc_dtype.width
        copy_vec_bits = min(colmax_bits, 128)
        dsmem_store_threads = colmax_bits // copy_vec_bits
        dsmem_store_values = copy_vec_bits // acc_dtype.width
        dsmem_store_atom_r = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyDsmemStoreOp(),
            acc_dtype,
            num_bits_per_copy=copy_vec_bits,
        )
        dsmem_store_r = cute.make_tiled_copy(
            dsmem_store_atom_r,
            cute.make_ordered_layout(
                (dsmem_store_threads, dsmem_store_values), order=(1, 0)
            ),
            (blk_tile_n,),
        )
        thr_store_r = dsmem_store_r.get_slice(lane_idx)
        tRsM = thr_store_r.partition_S(sM)  # (CPY, #CPY)
        tRsL = thr_store_r.partition_S(sL)  # (CPY, #CPY, warpgroup_warps)
        tRsR = thr_store_r.partition_S(sR)  # (CPY, #CPY, max_red_iters, 2)

        tRrM_shape = thr_store_r.partition_D(sM).shape
        tRrM_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrM_prev = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrL_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)

        # Wait for last colmax
        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after barrier
        sM_final_nbar.arrive_and_wait()
        is_reduction_lane = lane_idx < dsmem_store_threads
        if is_reduction_lane:
            tRrM_prev.store(tRsM.load())
            tRrM_final.store(tRrM_prev.load())

            # Cluster butterfly reduction
            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 0x01 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRsR_local = tRsR[None, None, i, 0]
                    tRsR_peer = cute.make_tensor(
                        cute.arch.map_dsmem_ptr(tRsR_local.iterator, peer_idx),
                        tRsR_local.layout,
                    )
                    local_mbar = reduction_mbars_ptr + i
                    peer_mbar = cute.arch.map_dsmem_ptr(local_mbar, peer_idx)
                    cute.copy(
                        dsmem_store_atom_r, tRrM_final, tRsR_peer, mbar_ptr=peer_mbar
                    )
                    cute.arch.fence_acq_rel_cta()  # dont reorder dsmem store after wait
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    tRrR = tRsR_local.load()
                    for j in cutlass.range_constexpr(cute.size(tRrM_final)):
                        tRrM_final[j] = cute.arch.fmax(tRrM_final[j], tRrR[j])

        # Wait for last colsum
        sL_final_nbar.arrive_and_wait()
        if is_reduction_lane:
            # Warpgroup reduction
            colsum = tRsL[None, None, 0].load()
            for i in cutlass.range_constexpr(1, warpgroup_warps, 1):
                colsum += tRsL[None, None, i].load()

            # Compute final correction and correct local colsum
            correction = exp2(tRrM_prev.load() - tRrM_final.load())
            correction = correction.reshape(colsum.shape)
            colsum *= correction

            # Cluster butterfly reduction
            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 0x01 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRrL_local = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
                    tRrL_local.store(colsum)
                    tRsR_local = tRsR[None, None, i, 1]
                    tRsR_peer = cute.make_tensor(
                        cute.arch.map_dsmem_ptr(tRsR_local.iterator, peer_idx),
                        tRsR_local.layout,
                    )
                    local_mbar = reduction_mbars_ptr + max_reduction_iters + i
                    peer_mbar = cute.arch.map_dsmem_ptr(local_mbar, peer_idx)
                    cute.copy(
                        dsmem_store_atom_r, tRrL_local, tRsR_peer, mbar_ptr=peer_mbar
                    )
                    cute.arch.fence_acq_rel_cta()  # dont reorder dsmem store after wait
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    colsum += tRsR_local.load()

            # Divide by final colsum and store
            rcp_colsum = cute.make_rmem_tensor(colsum.shape, acc_dtype)
            for i in cutlass.range(cute.size(colsum.shape)):
                rcp_colsum[i] = cute.arch.rcp_approx(colsum[i])
            tRsM.store(correction * rcp_colsum.load() * scale_o)

            # Save final colsum for LSE
            if cutlass.const_expr(gL is not None):
                tRrL_final.store(colsum)

        # Notify for final correction
        sM_final_nbar.arrive()

        # Compute and store LSE
        if cutlass.const_expr(gL is not None):
            tRgL = thr_store_r.partition_D(gL)  # (CPY, #CPY=1)
            if kv_split_idx == 0 and is_reduction_lane:
                lse = tRrM_final.load() + cute.math.log2(tRrL_final.load())
                tRgL.store(lse)

    ##############################
    # Reduction Kernel launch
    ##############################
    @staticmethod
    @cute.jit
    def launch_reduction(
        d_per_blk: int,
        o_bshd: cute.Tensor,
        l_bsh: Optional[cute.Tensor],  # LSE output (log2 base) or None
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
        l_hsb = reverse(l_bsh) if l_bsh is not None else None
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
            l_hsb,
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
        l_hsb: Optional[cute.Tensor],
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
                amax = Float32(0)
                for value_idx in cutlass.range_constexpr(16):
                    value = Float32(tCrO_final[value_idx])
                    amax = cute.arch.fmax(
                        amax, cute.arch.fmax(value, -value)
                    )

                global_scale = Float32(o_global_scale[0])
                sf_value = cutlass.Float8E4M3FN(
                    amax * global_scale * Float32(1.0 / 6.0)
                )
                sf_rounded = Float32(sf_value)
                quant_multiplier = Float32(0)
                if sf_rounded != Float32(0):
                    quant_multiplier = global_scale * cute.arch.rcp_approx(
                        sf_rounded
                    )
                scaled_o = tCrO_final * quant_multiplier
                tCgO.store(scaled_o.to(cutlass.Float4E2M1FN))

                d_block = (
                    d_blk_idx * d_per_blk + tidx * d_per_thr
                ) // 16
                d_scale_blocks = d // 16
                k_idx = coord_h * d_scale_blocks + d_block
                hidden = h_q * d
                num_k_tiles = cute.ceil_div(hidden, 64)
                row_idx = coord_b * s_q + coord_s
                m_tile_idx = row_idx // 128
                outer_m_idx = row_idx % 32
                inner_m_idx = (row_idx // 32) % 4
                k_tile_idx = k_idx // 4
                inner_k_idx = k_idx % 4
                sf_offset = (
                    (m_tile_idx * num_k_tiles + k_tile_idx) * 512
                    + outer_m_idx * 16
                    + inner_m_idx * 4
                    + inner_k_idx
                )
                o_block_scale[sf_offset] = sf_value
            else:
                tCgO.store(tCrO_final.to(o_dhsb.dtype))

        if cutlass.const_expr(l_hsb is not None and l_hsb.dtype == Float32):
            if d_blk_idx == 0 and tidx == 0:
                l_hsb[coord_h, coord_s, coord_b] = max_final + cute.math.log2(sum_final)

        return



# 注:dense 文件末尾的独立 __main__ 驱动(run/parser)在分页变体中已删除 ——
# 它面向 dense 语义,分页验证统一走 wrappers/batch_decode.py 的
# _get_compiled_paged_fp4_decode_kernel + harness 的 pagedfp4 模式。
