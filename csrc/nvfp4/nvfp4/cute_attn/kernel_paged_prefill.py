# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors (modifications)
# SPDX-License-Identifier: BSD-3-Clause
#
# 派生自 CUTLASS `examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/
# fmha.py` 的 `BlackwellFusedMultiHeadAttentionForward`(NVIDIA,BSD-3-Clause),
# 见同目录 `_vendor_fmha_fwd.py` 的完整 BSD-3 正文。BSD-3 第 1 条要求源码再
# 分发保留原版权声明与条件,所以本文件**沿用 BSD-3**,与同样派生的
# `kernel_paged_decode.py` 一致 —— 此前它标的是 Apache-2.0 且没有 NVIDIA
# 版权行,那是从 BSD-3 派生物单方面改标,开源前修正。
'DEPRECATED FOR PRODUCTION: historical in-house context attention kernel.'

from __future__ import annotations

from typing import Optional, Tuple, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.typing import Int32, Int64, Float32
from cutlass.base_dsl.arch import Arch

from . import _vendor_fmha_helpers as fmha_utils
from ._vendor_fmha_fwd import (
    BlackwellFusedMultiHeadAttentionForward,
    make_thread_cooperative_group,
)

from .kv_layout import E2M1_MAX, PAGE_SIZE, PI16, SF_VEC

#: SF 的存储 dtype(nvfp4 = e4m3 block scale)
SF_DTYPE = cutlass.Float8E4M3FN
#: KV / Q / P 的元素 dtype
FP4_DTYPE = cutlass.Float4E2M1FN


def _sf_s2t_partition(sf_dtype, sSF, tSF):
    """SMEM SF → TMEM 的 UTCCP 分区(decode kernel 逐字同款)。"""
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


class PDynamicSf:
    """P 的逐 16-token 组动态 scale factor —— 本 kernel 唯一没有先例的部分。

    softmax 从 TMEM 读 S 用 ``Ld32x32bOp(Repetition(32))``:线程 t 独占
    datapath t(= 一个 query 行)的连续 N 元素(N=64 chunk ⇒ 64 个,两次
    Repetition(32))。**16 元素一组的 amax 是纯线程内、连续寄存器的归约:
    无 SHFL、无跨 lane 通信。**

    数学与 ``reference.quantize_dequantize`` 同源(不得各写一份):

        sf   = e4m3(amax(p[g*16:(g+1)*16]) / 6.0)      # 6.0 = E2M1_MAX
        p_q  = e2m1(p / max(float(sf), 1e-12))

    两处容易写错、且都会静默降精度:

    1. **除数必须是 e4m3 舍入之后的值**,不是舍入前的 ``amax/6`` ——
       反量化乘回的是存进 SMEM 的那个 e4m3,不一致就重建不回原值。
    2. ``p`` 是 ``exp2`` 的输出恒非负,不需要 ``abs``;别照抄 K/V 侧带
       ``abs`` 的量化路径(那边输入有符号)。
    """

    #: 每线程的 N 片段元素数(= kv chunk 宽,Design E 下为 64)
    ELTS_PER_THREAD = 64
    #: 每线程产出的 SF 个数
    SF_PER_THREAD = ELTS_PER_THREAD // SF_VEC  # = 4

    assert ELTS_PER_THREAD % SF_VEC == 0, (
        "线程片段必须能被 SF 组整除,否则 amax 要跨 lane —— 见类注释"
    )

    @staticmethod
    @cute.jit
    def normalize_and_emit_sf(p_frag: cute.Tensor, sf_frag: cute.Tensor):
        """就地把 ``p_frag`` 归一化到 e2m1 的量程,并填出每组的 e4m3 SF。

        ``p_frag``:线程私有 fp32 寄存器片段,**flat 视图**、沿 N 连续;
        ``sf_frag``:``size(p_frag)//16`` 个 e4m3。
        """
        n_groups = cutlass.const_expr(cute.size(p_frag) // SF_VEC)
        for g in cutlass.range_constexpr(n_groups):
            amax = Float32(0.0)
            for j in cutlass.range_constexpr(SF_VEC):
                amax = cute.arch.fmax(amax, p_frag[g * SF_VEC + j])
            sf_e4m3 = SF_DTYPE(amax * cutlass.const_expr(1.0 / E2M1_MAX))
            sf_frag[g] = sf_e4m3
            # 关键:用**舍入后**的 e4m3 反算除数(见 docstring 第 1 条)
            inv = cute.arch.rcp_approx(
                cute.arch.fmax(Float32(sf_e4m3), Float32(1e-12))
            )
            # Normalize two lanes per instruction.  This is on every
            # 64-token softmax step's critical path; scalar multiplies made
            # the dynamic-P path unnecessarily serial on SM100.
            for j in cutlass.range_constexpr(0, SF_VEC, 2):
                i = g * SF_VEC + j
                p_frag[i], p_frag[i + 1] = cute.arch.mul_packed_f32x2(
                    (p_frag[i], p_frag[i + 1]), (inv, inv)
                )


class ResAttnPagedPrefillFp4(BlackwellFusedMultiHeadAttentionForward):
    """分页 NVFP4 prefill attention(Q 在 M / KV 在 N,原生 ``mxf4nvf4``)。

    继承 vendor 的 warp 特化骨架、四 gemm 轮转(KQ0→PV0→KQ1→PV1)、在线
    softmax 状态机、correction/epilogue、FusedMask 与 tile scheduler;fork
    ``__call__`` / ``kernel`` / ``mma_qk`` / ``mma_pv`` / ``softmax`` /
    ``softmax_step`` 以接入 blockscaled SF 通路、分页 KV 取址与动态 P SF。
    """

    def __init__(
        self,
        head_dim: int,
        *,
        h_q: int = 1,
        h_kv: int = 1,
        causal: bool = True,
        page_size: int = PAGE_SIZE,
        mma_tiler: Tuple[int, int] = (128, 64),
        qk_acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        pv_acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        is_persistent: bool = False,
        mask_type: "fmha_utils.MaskEnum" = None,  # type: ignore[assignment]
        use_tma_store: bool = True,
    ):
        if mask_type is None:
            mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
        super().__init__(
            qk_acc_dtype=qk_acc_dtype,
            pv_acc_dtype=pv_acc_dtype,
            mma_tiler=mma_tiler,
            head_dim=head_dim,
            is_persistent=is_persistent,
            mask_type=mask_type,
            enable_ex2_emulation=False,
            enable_skip_correction=False,
            use_tma_store=use_tma_store,
        )
        self.page_size = page_size
        self.sf_vec_size = SF_VEC
        self.sf_dtype = SF_DTYPE
        self.k_dtype = FP4_DTYPE
        self.p_dtype = FP4_DTYPE
        # 头数/因果是**编译期**常量(坑38:单一来源,不走运行期参数)
        assert h_q % h_kv == 0, (h_q, h_kv)
        self.h_q = h_q
        self.h_kv = h_kv
        self.h_r_const = h_q // h_kv
        self.causal = causal

        assert head_dim == 128, f"当前只支持 head_dim=128,得到 {head_dim}"
        assert mma_tiler[0] == 128, "blockscaled MMA 强制 M=128(坑32 同源)"
        
        
        self.chunks_per_page = page_size // mma_tiler[1]
        assert self.chunks_per_page == 2 and page_size == PAGE_SIZE, (
            f"MMA-native profile 冻结 page={PAGE_SIZE} = 2×kv_chunk;"
            f"得到 page={page_size}, chunk={mma_tiler[1]}"
        )

        # ------------------------------------------------------------------
        # TMEM 列图(Design E)。S/O 缩半后 0..383;384+ 是专用区,
        # 不再有任何 inplace。skip-softmax 未启用,其偏移指向 vec(不会被写)。
        # SF 的偏移在 kernel 内按 find_tmem_tensor_col_offset 链式排,
        # 从 tmem_sf_base 起,trace 时断言 ≤512。
        # ------------------------------------------------------------------
        n = mma_tiler[1]
        self.tmem_s0_offset = 0
        self.tmem_s1_offset = n
        self.tmem_o0_offset = 2 * n
        self.tmem_o1_offset = 2 * n + head_dim
        spare = 2 * n + 2 * head_dim
        p_cols = mma_tiler[1] * FP4_DTYPE.width // 32
        self.tmem_p0_offset = spare
        self.tmem_p1_offset = spare + p_cols
        self.tmem_vec0_offset = spare + 2 * p_cols
        self.tmem_vec1_offset = spare + 2 * p_cols + 2
        self.tmem_skip_softmax0_offset = self.tmem_vec0_offset
        self.tmem_skip_softmax1_offset = self.tmem_vec1_offset
        self.tmem_sf_base = spare + 2 * p_cols + 4

    # ------------------------------------------------------------------
    # MMA 构造(不交换;上游把这两个方法明写成 "Override in subclasses")
    # ------------------------------------------------------------------

    def _make_qk_tiled_mma(self, cta_group):
        """KQ:A=Q(M 轴)、B=K(N 轴),两操作数都沿 head_dim 收缩。"""
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.q_dtype,  # A = Q
            self.k_dtype,  # B = K
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.qk_mma_tiler[:2],
        )

    def _make_qk_sfb_tiled_mma(self, cta_group):
        """SFK(B 侧 SF)专用:UTCCP 要求 SFB 按 N=128 粒度(decode 坑2)。"""
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            (self.qk_mma_tiler[0], max(128, self.qk_mma_tiler[1])),
        )

    def _make_pv_tiled_mma(self, cta_group, p_major_mode, p_source):
        """VP:A=P(TMEM 源,M 轴)、B=V(N=head_dim),沿 kv token 收缩。

        坑28:``mxf4nvf4`` 操作数只能 K-major ⇒ 收缩轴(token)必须 gmem
        连续,V 池 ``[P,H,D,ps/2]`` 本来就满足。基类 ``__call__`` 会以
        "V 必须 MN-major" 拒绝这个布局 —— 这正是 ``__call__`` fork 的
        原因之一,fork 里断言方向已反转。
        """
        assert self.v_major_mode is None or self.v_major_mode == OperandMajorMode.K, (
            f"fp4 路径要求 V 为 K-major(token 最内,坑28),得到 {self.v_major_mode}"
        )
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.p_dtype,  # A = P(softmax 产出,核内量化)
            self.v_dtype,  # B = V
            p_major_mode,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

    # ------------------------------------------------------------------
    # dtype / 形状校验
    # ------------------------------------------------------------------

    def check_supported_dtypes(
        self,
        qk_dtype: Type[cutlass.Numeric],
        pv_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
    ) -> None:
        """只走原生 fp4;输出 bf16/fp16(与 decode 契约一致)。"""
        if qk_dtype is not FP4_DTYPE or pv_dtype is not FP4_DTYPE:
            raise ValueError(
                f"ResAttn prefill 只支持原生 nvfp4 KV(得到 qk={qk_dtype}, "
                f"pv={pv_dtype})。若要 fp8,那是 NVIDIA 的 QMUL4 路线。"
            )
        if out_dtype not in (cutlass.BFloat16, cutlass.Float16):
            raise ValueError(f"out_dtype 需为 bf16/fp16,得到 {out_dtype}")
        if qk_acc_dtype is not cutlass.Float32 or pv_acc_dtype is not cutlass.Float32:
            raise ValueError("累加必须 fp32")

    #: 侧别路标(防止照 decode 移植时把交换几何搬过来;见离线测试)
    OPERAND_SIDE = {
        "q": "A", "k": "B", "v": "B", "p": "A", "sfq": "A", "sfk": "B", "sfv": "B",
    }

    #: sm100 每 CTA 的动态 SMEM 上限(227KB),留 2KB 余量
    SMEM_LIMIT_BYTES = 227 * 1024
    SMEM_HEADROOM_BYTES = 2 * 1024

    def make_sf_smem_layouts(self, qk_tiled_mma, qk_sfb_tiled_mma, pv_tiled_mma,
                             kv_stages: int):
        """四个 SF 的 SMEM 布局。侧别跟数据操作数走(OPERAND_SIDE)。

        - SFQ:A 侧,随 Q stage(=2,Q0/Q1);
        - SFK:B 侧,随 K stage;**按 max(128,N) 垫**(decode 坑2);
        - SFV:B 侧,随 V stage(与 K 共 stage 数);N=head_dim=128 天然对齐;
        - SFP:A 侧,核内产生,2 个缓冲(P0/P1)。
        """
        n_sfb = max(128, self.qk_mma_tiler[1])
        return {
            "sfq": blockscaled_utils.make_smem_layout_sfa(
                qk_tiled_mma, self.qk_mma_tiler, self.sf_vec_size, self.q_stage
            ),
            "sfk": blockscaled_utils.make_smem_layout_sfb(
                qk_sfb_tiled_mma,
                (self.qk_mma_tiler[0], n_sfb, self.qk_mma_tiler[2]),
                self.sf_vec_size,
                kv_stages,
            ),
            "sfv": blockscaled_utils.make_smem_layout_sfb(
                pv_tiled_mma, self.pv_mma_tiler, self.sf_vec_size, kv_stages
            ),
            "sfp": blockscaled_utils.make_smem_layout_sfa(
                pv_tiled_mma, self.pv_mma_tiler, self.sf_vec_size, 2
            ),
        }

    def sf_smem_bytes(self, layouts, kv_stages: int) -> int:
        """SF 占的 SMEM 字节 —— 必须计入预算(坑21)。"""
        return sum(
            cute.size_in_bytes(self.sf_dtype, lay) for lay in layouts.values()
        )

    def assert_smem_budget(self, data_bytes: int, sf_layouts, kv_stages: int) -> int:
        sf_bytes = self.sf_smem_bytes(sf_layouts, kv_stages)
        total = data_bytes + sf_bytes
        limit = self.SMEM_LIMIT_BYTES - self.SMEM_HEADROOM_BYTES
        if total > limit:
            raise ValueError(
                f"SMEM 超容:数据 {data_bytes}B + SF {sf_bytes}B = {total}B "
                f"> {limit}B(kv_stages={kv_stages})。降 kv_stages 或缩 tile;"
                f"不要靠不给 SF 记账来腾空间(坑21)。"
            )
        return total

    # ------------------------------------------------------------------
    # __call__ fork
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,        # fp4 (s_q, d, h_q, batch)
        k: cute.Tensor,        # fp4 (page=128, d, h_kv, P)
        v: cute.Tensor,        # fp4 (d, page=128, h_kv, P)
        o: cute.Tensor,        # bf16 (s_q, d, h_q, batch)
        sfq: cute.Tensor,      # e4m3 (batch, h_q, s_q*d/16)
        sfk: cute.Tensor,      # e4m3 (P, h_kv, 1024),保留池的真实 stride
        sfv: cute.Tensor,      # e4m3 (P, h_kv, 1024),保留池的真实 stride
        kv_pt: cute.Tensor,    # Int32 (n_logical_pages,) 页表
        kv_indptr: cute.Tensor,# Int32 (batch,) 每请求页表起点
        kv_len: cute.Tensor,   # Int32 (batch,) 有效 kv 长度(含 ctx)
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        stream: cuda.CUstream,
    ):
        """把池格式的 fp4 张量铺成 vendor 形状并 launch。

        head/page 是两个显式 mode,不压成单一 L 维。vLLM 的
        ``[data | scale]`` 分侧池中,page stride 包含 K/V 另一侧和 SF,
        因此并不等于 ``h_kv * head_stride``;显式两维才能零拷贝
        表达生产池。稠密门仍用恒等页表走同一条 kernel 路径。
        """
        h_q = cutlass.const_expr(self.h_q)
        h_kv = cutlass.const_expr(self.h_kv)
        h_r = cutlass.const_expr(h_q // h_kv)
        s_q = q.shape[0]
        d = q.shape[1]
        batches = q.shape[3]

        self.q_dtype = q.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.check_supported_dtypes(
            self.q_dtype, self.v_dtype, self.o_dtype,
            self.qk_acc_dtype, self.pv_acc_dtype,
        )
        # V 必须 K-major(token 最内)—— 与基类的检查**方向相反**(坑28)
        self.q_major_mode = OperandMajorMode.K
        self.k_major_mode = OperandMajorMode.K
        self.v_major_mode = OperandMajorMode.K
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        # ---- vendor 形状的 gmem 视图 -------------------------------------
        # Q/O:(s, d, ((h_r, h_kv), batch));K/V:页当 batch 用,
        # (tok, d, ((h_r0, h_kv), P)),h_r stride 0 = GQA 广播。
        mQ = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (s_q, d, ((h_r, h_kv), batches)),
                stride=(q.stride[0], q.stride[1],
                        ((q.stride[2], q.stride[2] * h_r), q.stride[3])),
            ),
        )
        mO = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (s_q, d, ((h_r, h_kv), batches)),
                stride=(o.stride[0], o.stride[1],
                        ((o.stride[2], o.stride[2] * h_r), o.stride[3])),
            ),
        )
        # KV 侧 (h_kv, pages) 保留运行期真实 stride。kv_head 由
        # kernel 手动算,不需要 vendor 的 h_r 广播模(stride-0 会被
        # filter_zeros 吃掉,反而破坏坐标 arity)。
        pages = k.shape[3]
        mK = cute.make_tensor(
            k.iterator,
            cute.make_layout(
                (self.page_size, d, (h_kv, pages)),
                stride=(k.stride[0], k.stride[1],
                        (k.stride[2], k.stride[3])),
            ),
        )
        mV = cute.make_tensor(
            v.iterator,
            cute.make_layout(
                (d, self.page_size, (h_kv, pages)),
                stride=(v.stride[0], v.stride[1],
                        (v.stride[2], v.stride[3])),
            ),
        )

        # ---- SF 的 gmem 视图(坑30:base 建 (m,k,1),l 模手工补)------------
        # SFQ:逐头块;SFK:逐 (page,head) 满页块;
        # SFV:半页 = group 模干净切(kg//4 stride 512B)。
        # 坑30 习语(decode `_sf_layout_hb` 同款):base 建 (m,k,1),
        # l 模连同嵌套 stride 手工 make_layout —— tile_atom_to_shape_SF 自己
        # 处理嵌套 l 会折叠成 stride-0。
        sfq_base = blockscaled_utils.tile_atom_to_shape_SF((s_q, d, 1), SF_VEC)
        mSFQ = cute.make_tensor(
            sfq.iterator,
            cute.make_layout(
                (sfq_base.shape[0], sfq_base.shape[1],
                 ((h_r, h_kv), batches)),
                stride=(
                    sfq_base.stride[0], sfq_base.stride[1],
                    ((sfq.stride[1], sfq.stride[1] * h_r), sfq.stride[0]),
                ),
            ),
        )
        n_sfb = cutlass.const_expr(max(128, self.qk_mma_tiler[1]))
        sfk_base = blockscaled_utils.tile_atom_to_shape_SF((n_sfb, d, 1), SF_VEC)
        sfk_blk_bytes = cute.cosize(sfk_base)  
        
        
        
        
        
        
        mSFK = cute.make_tensor(
            sfk.iterator,
            cute.make_layout(
                (sfk_base.shape[0], sfk_base.shape[1], (h_kv, pages)),
                stride=(
                    sfk_base.stride[0], sfk_base.stride[1],
                    (sfk.stride[1], sfk.stride[0]),
                ),
            ),
        )
        # SFV:半页 = group 模干净切(kg//4 stride 512B),无垫读
        sfv_base = blockscaled_utils.tile_atom_to_shape_SF(
            (d, self.qk_mma_tiler[1], 1), SF_VEC
        )
        sfv_half_bytes = cute.cosize(sfv_base)  # = 512B(4 组)
        mSFV = cute.make_tensor(
            sfv.iterator,
            cute.make_layout(
                (sfv_base.shape[0], sfv_base.shape[1],
                 ((2, h_kv), pages)),
                stride=(
                    sfv_base.stride[0], sfv_base.stride[1],
                    ((sfv_half_bytes, sfv.stride[1]), sfv.stride[0]),
                ),
            ),
        )

        self._setup_attributes()
        cta_group = tcgen05.CtaGroup.ONE
        p_source = tcgen05.OperandSource.TMEM
        qk_tiled_mma = self._make_qk_tiled_mma(cta_group)
        qk_sfb_tiled_mma = self._make_qk_sfb_tiled_mma(cta_group)
        pv_tiled_mma = self._make_pv_tiled_mma(
            cta_group, OperandMajorMode.K, p_source
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.pv_mma_tiler[:2]

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            cute.shape((s_q, d, ((h_r, h_kv), batches))),
            self.cta_tiler,
            self.is_persistent,
        )

        # ---- SMEM 布局 ----------------------------------------------------
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.q_stage
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, self.qk_mma_tiler, self.k_dtype, self.kv_stage
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma, self.pv_mma_tiler, self.p_dtype, self.acc_stage
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma, self.pv_mma_tiler, self.v_dtype, self.kv_stage
        )
        # K/V chunk 同为 4KB(64tok×128d×4bit)⇒ 不共享缓冲也便宜;
        # v1 各自独立分配,省掉 vendor 的 stage stride 体操。
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.epi_stage
        )
        sf_layouts = self.make_sf_smem_layouts(
            qk_tiled_mma, qk_sfb_tiled_mma, pv_tiled_mma, self.kv_stage
        )
        sfq_smem_layout_staged = sf_layouts["sfq"]
        sfk_smem_layout_staged = sf_layouts["sfk"]
        sfv_smem_layout_staged = sf_layouts["sfv"]
        sfp_smem_layout_staged = sf_layouts["sfp"]

        # ---- TMA atoms ----------------------------------------------------
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        mma_modes = [0, 1, 2]

        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mQ,
            cute.select(q_smem_layout_staged, mode=mma_modes),
            self.qk_mma_tiler, qk_tiled_mma, self.cluster_layout_vmnk.shape,
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mK,
            cute.select(k_smem_layout_staged, mode=mma_modes),
            self.qk_mma_tiler, qk_tiled_mma, self.cluster_layout_vmnk.shape,
        )
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mV,
            cute.select(v_smem_layout_staged, mode=mma_modes),
            self.pv_mma_tiler, pv_tiled_mma, self.cluster_layout_vmnk.shape,
        )
        # A 侧 SF 用普通 op;B 侧 SF 用 SFB op + Int16(交换翻转时两侧对调,
        # decode 里 K/V 在 A 侧、Q 在 B 侧 —— 我们全反过来)
        tma_atom_sfq, tma_tensor_sfq = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mSFQ,
            cute.select(sfq_smem_layout_staged, mode=mma_modes),
            self.qk_mma_tiler, qk_tiled_mma,
        )
        sfb_op_qk = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, qk_sfb_tiled_mma.thr_id
        )
        tma_atom_sfk, tma_tensor_sfk = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op_qk, mSFK,
            cute.select(sfk_smem_layout_staged, mode=mma_modes),
            (self.qk_mma_tiler[0], n_sfb, self.qk_mma_tiler[2]),
            qk_sfb_tiled_mma, internal_type=cutlass.Int16,
        )
        sfb_op_pv = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, pv_tiled_mma.thr_id
        )
        tma_atom_sfv, tma_tensor_sfv = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op_pv, mSFV,
            cute.select(sfv_smem_layout_staged, mode=mma_modes),
            self.pv_mma_tiler, pv_tiled_mma, internal_type=cutlass.Int16,
        )
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op, mO,
            cute.select(o_smem_layout_staged, mode=[0, 1]),
            self.epi_tile,
        )

        # ---- tx 账目(SF 与数据同 barrier ⇒ tx 相加)----------------------
        q_tile_layout = cute.select(q_smem_layout_staged, mode=mma_modes)
        k_tile_layout = cute.select(k_smem_layout_staged, mode=mma_modes)
        v_tile_layout = cute.select(v_smem_layout_staged, mode=mma_modes)
        sfq_tile = cute.select(sfq_smem_layout_staged, mode=mma_modes)
        sfv_tile = cute.select(sfv_smem_layout_staged, mode=mma_modes)
        # SFK 半页箱只搬 64 行真数据?不 —— 箱按垫后 128 行铺,64 行垫是
        # 邻块真数据(§9.3),TMA 实搬 = 布局字节数。tx 必须照实记。
        sfk_tile = cute.select(sfk_smem_layout_staged, mode=mma_modes)
        self.tma_copy_q_bytes = (
            cute.size_in_bytes(self.q_dtype, q_tile_layout)
            + cute.size_in_bytes(self.sf_dtype, sfq_tile)
        )
        self.tma_copy_k_bytes = (
            cute.size_in_bytes(self.k_dtype, k_tile_layout)
            + cute.size_in_bytes(self.sf_dtype, sfk_tile)
        )
        self.tma_copy_v_bytes = (
            cute.size_in_bytes(self.v_dtype, v_tile_layout)
            + cute.size_in_bytes(self.sf_dtype, sfv_tile)
        )

        data_bytes = (
            cute.size_in_bytes(self.q_dtype, q_smem_layout_staged)
            + cute.size_in_bytes(self.k_dtype, k_smem_layout_staged)
            + cute.size_in_bytes(self.v_dtype, v_smem_layout_staged)
            + cute.size_in_bytes(self.o_dtype, o_smem_layout_staged)
        )
        self.assert_smem_budget(data_bytes, sf_layouts, self.kv_stage)

        # ---- SharedStorage(vendor 减 inplace/skip,加 SF)-----------------
        q_stage = self.q_stage
        kv_stage = self.kv_stage

        @cute.struct
        class SharedStorage:
            load_q_mbar_ptr: cute.struct.MemRange[Int64, q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, kv_stage * 2]
            mma_s0_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            mma_s1_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            p0_mma_mbar_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2]
            p1_mma_mbar_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2]
            s0_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            corr_epi_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            tmem_holding_buf: Int32
            sO: cute.struct.Align[
                cute.struct.MemRange[
                    self.o_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[
                    self.v_dtype, cute.cosize(v_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(sfq_smem_layout_staged)], 128,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(sfk_smem_layout_staged)], 128,
            ]
            sSFV: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(sfv_smem_layout_staged)], 128,
            ]
            sSFP: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(sfp_smem_layout_staged)], 128,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            qk_tiled_mma,
            qk_sfb_tiled_mma,
            pv_tiled_mma,
            tma_atom_q, tma_tensor_q,
            tma_atom_k, tma_tensor_k,
            tma_atom_v, tma_tensor_v,
            tma_atom_sfq, tma_tensor_sfq,
            tma_atom_sfk, tma_tensor_sfk,
            tma_atom_sfv, tma_tensor_sfv,
            tma_atom_o, tma_tensor_o,
            kv_pt, kv_indptr, kv_len,
            scale_softmax_log2, scale_softmax, scale_output,
            q_smem_layout_staged, k_smem_layout_staged,
            p_tmem_layout_staged, v_smem_layout_staged,
            sfq_smem_layout_staged, sfk_smem_layout_staged,
            sfv_smem_layout_staged, sfp_smem_layout_staged,
            o_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------
    # device kernel fork
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        qk_sfb_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom, mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom, mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom, mV_dkl: cute.Tensor,
        tma_atom_sfq: cute.CopyAtom, mSFQ_qdl: cute.Tensor,
        tma_atom_sfk: cute.CopyAtom, mSFK_kdl: cute.Tensor,
        tma_atom_sfv: cute.CopyAtom, mSFV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom, mO_qdl: cute.Tensor,
        kv_pt: cute.Tensor,
        kv_indptr: cute.Tensor,
        kv_len: cute.Tensor,
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        sfq_smem_layout_staged, sfk_smem_layout_staged,
        sfv_smem_layout_staged, sfp_smem_layout_staged,
        o_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
    ):
        'vendor kernel 的 fork:六种 warp 分工与四 gemm 轮转原样,改动:'
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfq)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfk)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfv)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_kv_full_mbar_ptr = storage.load_kv_mbar_ptr.data_ptr()
        load_kv_empty_mbar_ptr = load_kv_full_mbar_ptr + self.kv_stage
        mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            barrier_storage=storage.mma_s0_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_s1_producer, mma_s1_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.mma_s1_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        p0_mma_producer, p0_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(1),
            barrier_storage=storage.p0_mma_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        p1_mma_producer, p1_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(1),
            barrier_storage=storage.p1_mma_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s0_corr_producer, s0_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp
                * len((*self.softmax0_warp_ids, self.mma_warp_id))
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s0_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s1_corr_producer, s1_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp
                * len((*self.softmax1_warp_ids, self.mma_warp_id))
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s1_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        corr_epi_producer, corr_epi_consumer = pipeline.PipelineAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * 1
            ),
            barrier_storage=storage.corr_epi_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            arch=self.arch_str,
        )
        pipeline_init_arrive(is_relaxed=True)

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )
        sSFQ = storage.sSFQ.get_tensor(sfq_smem_layout_staged)
        sSFK = storage.sSFK.get_tensor(sfk_smem_layout_staged)
        sSFV = storage.sSFV.get_tensor(sfv_smem_layout_staged)
        sSFP = storage.sSFP.get_tensor(sfp_smem_layout_staged)

        qk_thr_mma = qk_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)

        def make_tmem_tensors(
            self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
        ):
            qk_acc_shape = qk_thr_mma.partition_shape_C(
                (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
            )
            tStS_fake = qk_thr_mma.make_fragment_C(qk_acc_shape)
            tStS = cute.make_tensor(
                tmem_ptr + self.tmem_s0_offset, tStS_fake.layout
            )
            pv_acc_shape = pv_thr_mma.partition_shape_C(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
            )
            tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
            tStS0 = cute.make_tensor(tmem_ptr + self.tmem_s0_offset, tStS.layout)
            tStS1 = cute.make_tensor(tmem_ptr + self.tmem_s1_offset, tStS.layout)
            tOtO0 = cute.make_tensor(tmem_ptr + self.tmem_o0_offset, tOtO.layout)
            tOtO1 = cute.make_tensor(tmem_ptr + self.tmem_o1_offset, tOtO.layout)
            tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
            tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
            tOrP0 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.tmem_p0_offset, dtype=tOrP.dtype),
                tOrP.layout,
            )
            tOrP1 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.tmem_p1_offset, dtype=tOrP.dtype),
                tOrP.layout,
            )
            return tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1

        def make_sf_tmem_tensors(self, tmem_ptr, qk_tiled_mma,
                                 qk_sfb_tiled_mma, pv_tiled_mma,
                                 sfq_lay, sfk_lay, sfv_lay, sfp_lay):
            """SF 的 TMEM 张量:专用区(tmem_sf_base 起)链式排,断言 ≤512。

            签名照 decode:第 4 参是 ``cute.slice_`` 出的单 stage SMEM 布局;
            SFK 用 sfb 专用 mma + 垫后 tiler(坑2)。全部依赖显式传参
            (staged if 里不许闭包捕获)。
            """
            n_sfb = max(128, self.qk_mma_tiler[1])
            off = self.tmem_sf_base
            lay_sfq = blockscaled_utils.make_tmem_layout_sfa(
                qk_tiled_mma, self.qk_mma_tiler, self.sf_vec_size,
                cute.slice_(sfq_lay, (None, None, None, 0)),
            )
            lay_sfk = blockscaled_utils.make_tmem_layout_sfb(
                qk_sfb_tiled_mma,
                (self.qk_mma_tiler[0], n_sfb, self.qk_mma_tiler[2]),
                self.sf_vec_size,
                cute.slice_(sfk_lay, (None, None, None, 0)),
            )
            lay_sfv = blockscaled_utils.make_tmem_layout_sfb(
                pv_tiled_mma, self.pv_mma_tiler, self.sf_vec_size,
                cute.slice_(sfv_lay, (None, None, None, 0)),
            )
            lay_sfp = blockscaled_utils.make_tmem_layout_sfa(
                pv_tiled_mma, self.pv_mma_tiler, self.sf_vec_size,
                cute.slice_(sfp_lay, (None, None, None, 0)),
            )
            # 手工展开(staged 代码里 Python list 累积会类型不稳)
            t_sfq0 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + off, dtype=self.sf_dtype), lay_sfq
            )
            off += tcgen05.find_tmem_tensor_col_offset(t_sfq0)
            t_sfq1 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + off, dtype=self.sf_dtype), lay_sfq
            )
            off += tcgen05.find_tmem_tensor_col_offset(t_sfq1)
            t_sfk = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + off, dtype=self.sf_dtype), lay_sfk
            )
            off += tcgen05.find_tmem_tensor_col_offset(t_sfk)
            t_sfv = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + off, dtype=self.sf_dtype), lay_sfv
            )
            off += tcgen05.find_tmem_tensor_col_offset(t_sfv)
            t_sfp0 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + off, dtype=self.sf_dtype), lay_sfp
            )
            off += tcgen05.find_tmem_tensor_col_offset(t_sfp0)
            t_sfp1 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + off, dtype=self.sf_dtype), lay_sfp
            )
            off += tcgen05.find_tmem_tensor_col_offset(t_sfp1)
            assert off <= 512, f"TMEM 超容:SF 链到 {off} 列(>512)"
            return t_sfq0, t_sfq1, t_sfk, t_sfv, t_sfp0, t_sfp1

        tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        pipeline_init_wait()

        seqlen_q = mQ_qdl.shape[0]
        # v1 无 sliding window。WINDOW_MASK_INFERENCE 下:
        # causal ⇒ (None, 0):可见条件 index_k <= ctx + i(因果+前缀);
        # 非因果 ⇒ (None, None):全可见,只剩 residual 边界掩码。
        window_size_left = None
        # self.causal 是 trace 期 Python bool,三元在 trace 期即定型
        window_size_right = Int32(0) if self.causal else None

        softmax_fn = partial(
            self.softmax,
            qk_thr_mma=qk_thr_mma,
            value_args=(kv_len, seqlen_q, scale_softmax_log2),
            sched_args=(tile_sched, work_tile),
            mask_args=(window_size_left, window_size_right),
        )

        # ///////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                # 调度器的 coord[2] = (flat_head, batch),头是线性坐标
                kv_head = curr_block_coord[2][0] // self.h_r_const
                batch = curr_block_coord[2][1]
                seqlen_k = kv_len[batch]
                pt_base = kv_indptr[batch]

                # ---- Q / SFQ 分区(逐 work tile;q0/q1 两个 128 行 tile)----
                gQ_qdl = cute.flat_divide(
                    mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2])
                )
                tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_q, 0, cute.make_layout(1),
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(tSgQ_qdl, 0, 3),
                )
                tQgQ = tQgQ_qdl[None, None, 0, curr_block_coord[2]]
                gSFQ_qdl = cute.flat_divide(
                    mSFQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2])
                )
                tSgSFQ_qdl = qk_thr_mma.partition_A(gSFQ_qdl)
                tQsSFQ, tQgSFQ_qdl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_sfq, 0, cute.make_layout(1),
                    cute.group_modes(sSFQ, 0, 3),
                    cute.group_modes(tSgSFQ_qdl, 0, 3),
                )
                # 坑13:SF 布局含 stride-0 广播模,不滤则 TMA 零发射
                tQsSFQ = cute.filter_zeros(tQsSFQ)
                tQgSFQ = cute.filter_zeros(
                    tQgSFQ_qdl[None, None, 0, curr_block_coord[2]]
                )

                # ---- K / V / SF 分区(页当 L 用)---------------------------
                gK_kdl = cute.flat_divide(
                    mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2])
                )
                tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k, 0, cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK_kdl, 0, 3),
                )
                # (TMA, chunk_in_page=2, page)
                tKgK = tKgK_kdl[None, None, 0, (kv_head, None)]
                gV_dkl = cute.flat_divide(
                    mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2])
                )
                tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v, 0, cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tSgV_dkl, 0, 3),
                )
                tVgV = tVgV_dkl[None, 0, None, (kv_head, None)]
                # SFK:**垫后 tiler**(max(128,N)) + sfb 专用 mma 分区
                # (decode 坑2/坑13 配方);半页模在 L 里,copy 时按 (h, p) 选。
                n_sfb = max(128, self.qk_mma_tiler[1])
                qk_sfb_thr_mma = qk_sfb_tiled_mma.get_slice(0)
                gSFK_kdl = cute.flat_divide(
                    mSFK_kdl, (n_sfb, self.qk_mma_tiler[2])
                )
                tSgSFK_kdl = qk_sfb_thr_mma.partition_B(gSFK_kdl)
                tKsSFK, tKgSFK_kdl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_sfk, 0, cute.make_layout(1),
                    cute.group_modes(sSFK, 0, 3),
                    cute.group_modes(tSgSFK_kdl, 0, 3),
                )
                tKsSFK = cute.filter_zeros(tKsSFK)
                tKgSFK = cute.filter_zeros(
                    tKgSFK_kdl[None, 0, 0, (kv_head, None)]
                )
                gSFV_dkl = cute.flat_divide(
                    mSFV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2])
                )
                tSgSFV_dkl = pv_thr_mma.partition_B(gSFV_dkl)
                tVsSFV, tVgSFV_dkl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_sfv, 0, cute.make_layout(1),
                    cute.group_modes(sSFV, 0, 3),
                    cute.group_modes(tSgSFV_dkl, 0, 3),
                )
                tVsSFV = cute.filter_zeros(tVsSFV)
                tVgSFV = cute.filter_zeros(
                    tVgSFV_dkl[None, 0, 0, ((None, kv_head), None)]
                )

                chunk_start = fmha_utils.FusedMask.get_trip_start(
                    self.mask_type, curr_block_coord, self.cta_tiler,
                    seqlen_q, seqlen_k, window_size_left,
                )
                chunk_steps = fmha_utils.FusedMask.get_trip_count(
                    self.mask_type, curr_block_coord, self.cta_tiler,
                    seqlen_q, seqlen_k, window_size_left, window_size_right,
                )

                # Q0(+SFQ0,同 barrier)
                q0_coord = 2 * curr_block_coord[0]
                q0_handle = load_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q, tQgQ[None, q0_coord],
                    tQsQ[None, q0_handle.index],
                    tma_bar_ptr=q0_handle.barrier,
                )
                cute.copy(
                    tma_atom_sfq, tQgSFQ[None, q0_coord],
                    tQsSFQ[None, q0_handle.index],
                    tma_bar_ptr=q0_handle.barrier,
                )
                # K0 + SFK0
                c = chunk_start
                k_handle = load_kv_producer.acquire_and_advance()
                p_phys = kv_pt[pt_base + c // 2]
                cute.copy(
                    tma_atom_k, tKgK[None, c % 2, p_phys],
                    tKsK[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )
                cute.copy(
                    tma_atom_sfk,
                    tKgSFK[None, p_phys],
                    tKsSFK[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )
                # Q1(+SFQ1)
                q1_coord = q0_coord + 1
                q1_handle = load_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q, tQgQ[None, q1_coord],
                    tQsQ[None, q1_handle.index],
                    tma_bar_ptr=q1_handle.barrier,
                )
                cute.copy(
                    tma_atom_sfq, tQgSFQ[None, q1_coord],
                    tQsSFQ[None, q1_handle.index],
                    tma_bar_ptr=q1_handle.barrier,
                )
                c += 1

                for _i in cutlass.range(1, chunk_steps, 1, unroll=1):
                    # K_c + SFK_c
                    k_handle = load_kv_producer.acquire_and_advance()
                    p_phys = kv_pt[pt_base + c // 2]
                    cute.copy(
                        tma_atom_k, tKgK[None, c % 2, p_phys],
                        tKsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    cute.copy(
                        tma_atom_sfk,
                        tKgSFK[None, p_phys],
                        tKsSFK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    # V_{c-1} + SFV_{c-1}(自定义 tx)
                    v_handle, load_kv_producer = (
                        self.kv_producer_update_tx_acquire_and_advance(
                            load_kv_producer,
                            load_kv_empty_mbar_ptr,
                            load_kv_full_mbar_ptr,
                            self.tma_copy_v_bytes,
                        )
                    )
                    pv_phys = kv_pt[pt_base + (c - 1) // 2]
                    cute.copy(
                        tma_atom_v, tVgV[None, (c - 1) % 2, pv_phys],
                        tVsV[None, v_handle.index],
                        tma_bar_ptr=load_kv_full_mbar_ptr + v_handle.index,
                    )
                    cute.copy(
                        tma_atom_sfv,
                        tVgSFV[None, (c - 1) % 2, pv_phys],
                        tVsSFV[None, v_handle.index],
                        tma_bar_ptr=load_kv_full_mbar_ptr + v_handle.index,
                    )
                    c += 1
                # V_end + SFV_end
                v_handle, load_kv_producer = (
                    self.kv_producer_update_tx_acquire_and_advance(
                        load_kv_producer,
                        load_kv_empty_mbar_ptr,
                        load_kv_full_mbar_ptr,
                        self.tma_copy_v_bytes,
                    )
                )
                pv_phys = kv_pt[pt_base + (c - 1) // 2]
                cute.copy(
                    tma_atom_v, tVgV[None, (c - 1) % 2, pv_phys],
                    tVsV[None, v_handle.index],
                    tma_bar_ptr=load_kv_full_mbar_ptr + v_handle.index,
                )
                cute.copy(
                    tma_atom_sfv,
                    tVgSFV[None, (c - 1) % 2, pv_phys],
                    tVsSFV[None, v_handle.index],
                    tma_bar_ptr=load_kv_full_mbar_ptr + v_handle.index,
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            tSFQ0, tSFQ1, tSFK, tSFV, tSFP0, tSFP1 = make_sf_tmem_tensors(
                self, tmem_ptr, qk_tiled_mma, qk_sfb_tiled_mma, pv_tiled_mma,
                sfq_smem_layout_staged, sfk_smem_layout_staged,
                sfv_smem_layout_staged, sfp_smem_layout_staged,
            )
            copy_sfq, tCsSFQ_s2t, tCtSFQ0_s2t = _sf_s2t_partition(
                self.sf_dtype, sSFQ, tSFQ0
            )
            _, _, tCtSFQ1_s2t = _sf_s2t_partition(
                self.sf_dtype, sSFQ, tSFQ1
            )
            copy_sfk, tCsSFK_s2t, tCtSFK_s2t = _sf_s2t_partition(
                self.sf_dtype, sSFK, tSFK
            )
            copy_sfv, tCsSFV_s2t, tCtSFV_s2t = _sf_s2t_partition(
                self.sf_dtype, sSFV, tSFV
            )
            copy_sfp, tCsSFP_s2t, tCtSFP0_s2t = _sf_s2t_partition(
                self.sf_dtype, sSFP, tSFP0
            )
            _, _, tCtSFP1_s2t = _sf_s2t_partition(
                self.sf_dtype, sSFP, tSFP1
            )

            # 坑41:SFK 满页常驻;h=1 的 SFB 基址 = 第 64 行的布局偏移
            # (行模 (32,4) 的 rb-stride ×2)。从布局取,不硬编码。
            _rb_stride = tSFK.layout.stride[0][0][0][1]
            tSFK_h1 = cute.make_tensor(
                tSFK.iterator + 2 * _rb_stride, tSFK.layout
            )

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                seqlen_k = kv_len[curr_block_coord[2][1]]
                chunk_start = fmha_utils.FusedMask.get_trip_start(
                    self.mask_type, curr_block_coord, self.cta_tiler,
                    seqlen_q, seqlen_k, window_size_left,
                )
                # Q0 就绪 → SFQ0 UTCCP(专用区,整个 work tile 常驻)
                q0_handle = load_q_consumer.wait_and_advance()
                tSrQ0 = tSrQ[None, None, None, q0_handle.index]
                cute.copy(
                    copy_sfq,
                    tCsSFQ_s2t[None, None, None, None, q0_handle.index],
                    tCtSFQ0_s2t,
                )
                k_handle = load_kv_consumer.wait_and_advance()
                tSrK0 = tSrK[None, None, None, k_handle.index]
                kv_par = chunk_start % 2
                mma_s0_producer, s0_corr_producer = self.mma_qk_bs(
                    qk_tiled_mma,
                    (tSrQ0, tSrK0, tStS0),
                    (copy_sfk, tCsSFK_s2t, tCtSFK_s2t,
                     tSFK, tSFK_h1, tSFQ0, k_handle.index, kv_par),
                    (mma_s0_producer, s0_corr_producer),
                )
                q1_handle = load_q_consumer.wait_and_advance()
                tSrQ1 = tSrQ[None, None, None, q1_handle.index]
                cute.copy(
                    copy_sfq,
                    tCsSFQ_s2t[None, None, None, None, q1_handle.index],
                    tCtSFQ1_s2t,
                )
                mma_s1_producer, s1_corr_producer = self.mma_qk_bs(
                    qk_tiled_mma,
                    (tSrQ1, tSrK0, tStS1),
                    (None, None, None, tSFK, tSFK_h1, tSFQ1, 0, kv_par),
                    (mma_s1_producer, s1_corr_producer),
                )
                k_handle.release()
                chunk_steps = fmha_utils.FusedMask.get_trip_count(
                    self.mask_type, curr_block_coord, self.cta_tiler,
                    seqlen_q, seqlen_k, window_size_left, window_size_right,
                )
                pv_whether_acc = False
                for _i in cutlass.range(1, chunk_steps, 1, unroll=1):
                    k_handle = load_kv_consumer.wait_and_advance()
                    tSrKi = tSrK[None, None, None, k_handle.index]
                    kv_par = (chunk_start + _i) % 2
                    mma_s0_producer, s0_corr_producer = self.mma_qk_bs(
                        qk_tiled_mma,
                        (tSrQ0, tSrKi, tStS0),
                        (copy_sfk, tCsSFK_s2t, tCtSFK_s2t,
                         tSFK, tSFK_h1, tSFQ0, k_handle.index, kv_par),
                        (mma_s0_producer, s0_corr_producer),
                    )
                    v_handle = load_kv_consumer.wait_and_advance()
                    tOrVi = tOrV[None, None, None, v_handle.index]
                    mma_corr_producer, p0_mma_consumer = self.mma_pv_bs(
                        pv_tiled_mma, pv_whether_acc,
                        (tOrP0, tOrVi, tOtO0),
                        (copy_sfp, tCsSFP_s2t, tCtSFP0_s2t, tSFP0, 0,
                         copy_sfv, tCsSFV_s2t, tCtSFV_s2t, tSFV,
                         v_handle.index, True),
                        (mma_corr_producer, p0_mma_consumer),
                    )
                    mma_s1_producer, s1_corr_producer = self.mma_qk_bs(
                        qk_tiled_mma,
                        (tSrQ1, tSrKi, tStS1),
                        (None, None, None, tSFK, tSFK_h1, tSFQ1, 0, kv_par),
                        (mma_s1_producer, s1_corr_producer),
                    )
                    k_handle.release()
                    mma_corr_producer, p1_mma_consumer = self.mma_pv_bs(
                        pv_tiled_mma, pv_whether_acc,
                        (tOrP1, tOrVi, tOtO1),
                        (copy_sfp, tCsSFP_s2t, tCtSFP1_s2t, tSFP1, 1,
                         copy_sfv, tCsSFV_s2t, tCtSFV_s2t, tSFV,
                         v_handle.index, False),
                        (mma_corr_producer, p1_mma_consumer),
                    )
                    pv_whether_acc = True
                    v_handle.release()
                q0_handle.release()
                q1_handle.release()
                v_handle = load_kv_consumer.wait_and_advance()
                tOrVi = tOrV[None, None, None, v_handle.index]
                mma_corr_producer, p0_mma_consumer = self.mma_pv_bs(
                    pv_tiled_mma, pv_whether_acc,
                    (tOrP0, tOrVi, tOtO0),
                    (copy_sfp, tCsSFP_s2t, tCtSFP0_s2t, tSFP0, 0,
                     copy_sfv, tCsSFV_s2t, tCtSFV_s2t, tSFV,
                     v_handle.index, True),
                    (mma_corr_producer, p0_mma_consumer),
                )
                mma_corr_producer, p1_mma_consumer = self.mma_pv_bs(
                    pv_tiled_mma, pv_whether_acc,
                    (tOrP1, tOrVi, tOtO1),
                    (copy_sfp, tCsSFP_s2t, tCtSFP1_s2t, tSFP1, 1,
                     copy_sfv, tCsSFV_s2t, tCtSFV_s2t, tSFV,
                     v_handle.index, False),
                    (mma_corr_producer, p1_mma_consumer),
                )
                v_handle.release()
                vec0_handle = s0_corr_producer.acquire_and_advance()
                vec0_handle.commit()
                vec1_handle = s1_corr_producer.acquire_and_advance()
                vec1_handle.commit()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////
        #  Epilogue(TMA store)
        # ///////////////////////////////////////////////////////////////////
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                o0_coord = 2 * curr_block_coord[0]
                o1_coord = o0_coord + 1
                gO_qdl = cute.flat_divide(
                    mO_qdl, cute.select(self.pv_mma_tiler, mode=[0, 1])
                )
                gO = gO_qdl[None, None, None, 0, curr_block_coord[2]]
                tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_o, 0, cute.make_layout(1),
                    cute.group_modes(sO, 0, 2),
                    cute.group_modes(gO, 0, 2),
                )
                o0_handle = corr_epi_consumer.wait_and_advance()
                cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])
                cute.arch.cp_async_bulk_commit_group()
                o1_handle = corr_epi_consumer.wait_and_advance()
                cute.copy(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(1, read=True)
                o0_handle.release()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                o1_handle.release()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////
        #  Softmax0 / Softmax1
        # ///////////////////////////////////////////////////////////////////
        if warp_idx < self.softmax1_warp_ids[0]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            softmax_fn(
                stage=0,
                tensor_args=(tStS, tStS0, sSFP),
                pipeline_args=(mma_s0_consumer, s0_corr_producer, p0_mma_producer),
            )
        if (
            warp_idx < self.correction_warp_ids[0]
            and warp_idx >= self.softmax1_warp_ids[0]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            softmax_fn(
                stage=1,
                tensor_args=(tStS, tStS1, sSFP),
                pipeline_args=(mma_s1_consumer, s1_corr_producer, p1_mma_producer),
            )

        # ///////////////////////////////////////////////////////////////////
        #  Correction(vendor 原样,vec 偏移已随属性搬进专用区)
        # ///////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            cS = cute.make_identity_tensor(
                (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
            )
            tScS = qk_thr_mma.partition_C(cS)
            tStS_vec_layout = cute.composition(
                tStS.layout, cute.make_layout((128, 2))
            )
            tStS_vec0 = cute.make_tensor(
                tStS.iterator + self.tmem_vec0_offset, tStS_vec_layout
            )
            tStS_vec1 = cute.make_tensor(
                tStS.iterator + self.tmem_vec1_offset, tStS_vec_layout
            )
            tScS_vec_layout = cute.composition(
                tScS.layout, cute.make_layout((128, 2))
            )
            tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
            tmem_load_v_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
                self.qk_acc_dtype,
            )
            tiled_tmem_load_vec = tcgen05.make_tmem_copy(
                tmem_load_v_atom, tStS_vec0
            )
            thread_idx = tidx % (
                self.threads_per_warp * len(self.correction_warp_ids)
            )
            thr_tmem_load_vec = tiled_tmem_load_vec.get_slice(thread_idx)
            tTMEM_LOAD_VECtS0 = thr_tmem_load_vec.partition_S(tStS_vec0)
            tTMEM_LOAD_VECtS1 = thr_tmem_load_vec.partition_S(tStS_vec1)
            tTMEM_LOAD_VECcS = thr_tmem_load_vec.partition_D(tScS_vec)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                seqlen_k = kv_len[curr_block_coord[2][1]]
                row_idx = (
                    curr_block_coord[0] * self.cta_tiler[0]
                    + tTMEM_LOAD_VECcS[0][0]
                )
                vec0_handle = s0_corr_consumer.wait_and_advance()
                vec0_handle.release()
                vec1_handle = s1_corr_consumer.wait_and_advance()
                vec1_handle.release()
                oi_peek_status = cutlass.Boolean(False)
                chunk_steps = fmha_utils.FusedMask.get_trip_count(
                    self.mask_type, curr_block_coord, self.cta_tiler,
                    seqlen_q, seqlen_k, window_size_left, window_size_right,
                )
                for _i in cutlass.range(1, chunk_steps, 1, unroll=1):
                    ((s0_corr_consumer, mma_corr_consumer), oi_peek_status) = (
                        self.correction_rescale(
                            pv_thr_mma, tiled_tmem_load_vec, scale_softmax_log2,
                            (tOtO0, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECcS),
                            (s0_corr_consumer, mma_corr_consumer),
                            oi_peek_status,
                        )
                    )
                    ((s1_corr_consumer, mma_corr_consumer), oi_peek_status) = (
                        self.correction_rescale(
                            pv_thr_mma, tiled_tmem_load_vec, scale_softmax_log2,
                            (tOtO1, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECcS),
                            (s1_corr_consumer, mma_corr_consumer),
                            oi_peek_status,
                        )
                    )
                value_args = (
                    Int32(0), seqlen_q, curr_block_coord,
                    scale_softmax, scale_output,
                )
                s0_corr_consumer, mma_corr_consumer, corr_epi_producer = (
                    self.correction_epilog(
                        pv_thr_mma, tiled_tmem_load_vec,
                        (tOtO0, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECcS,
                         sO[None, None, 0], None, None),
                        (s0_corr_consumer, mma_corr_consumer, corr_epi_producer),
                        (row_idx, *value_args),
                    )
                )
                row_idx += self.qk_mma_tiler[0]
                s1_corr_consumer, mma_corr_consumer, corr_epi_producer = (
                    self.correction_epilog(
                        pv_thr_mma, tiled_tmem_load_vec,
                        (tOtO1, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECcS,
                         sO[None, None, 1], None, None),
                        (s1_corr_consumer, mma_corr_consumer, corr_epi_producer),
                        (row_idx, *value_args),
                    )
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tmem.relinquish_alloc_permit()
            self.tmem_dealloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
        return

    # ------------------------------------------------------------------
    # blockscaled MMA 步(fork:UTCCP + set SFA/SFB)
    # ------------------------------------------------------------------

    @cute.jit
    def mma_qk_bs(self, tiled_mma, tensor_args, sf_args, pipeline_args):
        """KQ 一步。SFK 随 K stage UTCCP(仅 Q0 腿做,Q1 腿复用同一 TMEM);
        SFA=SFQ、SFB=SFK 逐 mma_k 相位 set(接反编译不报、数值整片乱)。

        坑41:SFK 是**满页**(128 token 行)常驻 TMEM;当前 chunk 取哪半页
        由 SFB **基址**选择 —— ``tCtSFK_h1`` 的迭代器带第 64 行的布局偏移
        (行模 (32,4):(·,rb) ⇒ +2·rb,TMEM 无 TMA 的 16B stride 约束)。
        gmem 侧因此永远搬整页,半页选择完全离开 TMA(那里 8B stride 不合法)。

        同 warp 的 tcgen05.cp 与 tcgen05.mma 按发射序执行 —— SFK 单缓冲
        逐 chunk 覆写的安全性依据(decode 成例,B200 数值门背书)。
        """
        tSrQi, tSrK, tStSi = tensor_args
        (copy_sfk, tCsSFK_s2t, tCtSFK_s2t, tCtSFK_h0, tCtSFK_h1,
         tCtSFQi, k_stage, kv_parity) = sf_args
        mma_si_producer, si_corr_producer = pipeline_args
        si_handle = mma_si_producer.acquire_and_advance()
        veci_handle = si_corr_producer.acquire_and_advance()
        veci_handle.commit()
        if cutlass.const_expr(copy_sfk is not None):
            cute.copy(
                copy_sfk,
                tCsSFK_s2t[None, None, None, None, k_stage],
                tCtSFK_s2t,
            )
        # UTCCP → gemm 之间 fence(simple kernel B200 同款)
        cute.arch.fence_view_async_tmem_store()
        num_kphases = cute.size(tSrQi, mode=[2])
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            kphase_coord = (None, None, kphase_idx)
            tiled_mma.set(tcgen05.Field.SFA, tCtSFQi[None, None, kphase_idx].iterator)
            if kv_parity == 0:
                tiled_mma.set(
                    tcgen05.Field.SFB, tCtSFK_h0[None, None, kphase_idx].iterator
                )
            else:
                tiled_mma.set(
                    tcgen05.Field.SFB, tCtSFK_h1[None, None, kphase_idx].iterator
                )
            tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma, tStSi, tSrQi[kphase_coord], tSrK[kphase_coord], tStSi
            )
        si_handle.commit()
        return mma_si_producer, si_corr_producer

    @cute.jit
    def mma_pv_bs(self, tiled_mma, whether_acc, tensor_args, sf_args,
                  pipeline_args):
        """PV 一步。SFP 在 pi wait 之后 UTCCP(softmax 已写 SMEM 并 fence,
        搭 P 的既有握手 —— 零新增同步);SFV 仅在 P0 腿 UTCCP(P1 复用)。
        """
        tOrPi, tOrVi, tOtOi = tensor_args
        (copy_sfp, tCsSFP_s2t, tCtSFPi_s2t, tCtSFPi, p_stage,
         copy_sfv, tCsSFV_s2t, tCtSFV_s2t, tCtSFV, v_stage, do_sfv) = sf_args
        mma_corr_producer, pi_mma_consumer = pipeline_args
        oi_handle = mma_corr_producer.acquire_and_advance()
        pi_handle = pi_mma_consumer.wait_and_advance()
        cute.copy(
            copy_sfp,
            tCsSFP_s2t[None, None, None, None, p_stage],
            tCtSFPi_s2t,
        )
        if cutlass.const_expr(do_sfv):
            cute.copy(
                copy_sfv,
                tCsSFV_s2t[None, None, None, None, v_stage],
                tCtSFV_s2t,
            )
        # 同 mma_qk_bs:UTCCP → gemm 必须 fence
        cute.arch.fence_view_async_tmem_store()
        num_kphases = cute.size(tOrPi, mode=[2])
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            kphase_coord = (None, None, kphase_idx)
            tiled_mma.set(tcgen05.Field.SFA, tCtSFPi[None, None, kphase_idx].iterator)
            tiled_mma.set(tcgen05.Field.SFB, tCtSFV[None, None, kphase_idx].iterator)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, whether_acc or kphase_idx != 0)
            cute.gemm(
                tiled_mma, tOtOi, tOrPi[kphase_coord], tOrVi[kphase_coord], tOtOi
            )
        pi_handle.release()
        oi_handle.commit()
        return mma_corr_producer, pi_mma_consumer

    # ------------------------------------------------------------------
    # softmax fork(fp4 P 宽度、动态 SF、π、无 inplace)
    # ------------------------------------------------------------------

    @cute.jit
    def softmax(self, stage, tensor_args, pipeline_args, qk_thr_mma,
                value_args, mask_args, sched_args):
        (tStS, tStSi, sSFP) = tensor_args
        mma_si_consumer, si_corr_producer, pi_mma_producer = pipeline_args
        kv_lens, seqlen_q, scale_softmax_log2 = value_args
        window_size_left, window_size_right = mask_args
        tile_sched, work_tile = sched_args

        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax0_warp_ids))

        cS_base = cute.make_identity_tensor(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        # 坑(vendor 原文用 o_dtype —— fp4 P 必须用 p_dtype 宽度):
        # P 的 TMEM 足迹 = N/32 × p_width 个 fp32 列(N=64、fp4 ⇒ 8 列)
        tilePlikeFP32 = self.qk_mma_tiler[1] // 32 * self.p_dtype.width
        tScS = qk_thr_mma.partition_C(cS_base)
        tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))
        tmem_vec_offset = (
            self.tmem_vec0_offset if stage == 0 else self.tmem_vec1_offset
        )
        tStS_vec = cute.make_tensor(
            tStS.iterator + tmem_vec_offset, tStS_vec_layout
        )
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
        tStS_P_layout = cute.composition(
            tStS.layout, cute.make_layout((128, tilePlikeFP32))
        )
        tmem_p_offset = self.tmem_p0_offset if stage == 0 else self.tmem_p1_offset
        tStS_P = cute.make_tensor(tStS.iterator + tmem_p_offset, tStS_P_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_tmem_load.partition_S(tStSi)
        tmem_store_vec_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_vec = tcgen05.make_tmem_copy(
            tmem_store_vec_atom, tStS_vec
        )
        thr_tmem_store_vec = tiled_tmem_store_vec.get_slice(thread_idx)
        tTMEM_STORE_VECtS = thr_tmem_store_vec.partition_D(tStS_vec)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(tilePlikeFP32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P)

        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            seqlen_k = kv_lens[curr_block_coord[2][1]]
            logical_offset = (
                curr_block_coord[0] * self.cta_tiler[0]
                + stage * self.qk_mma_tiler[0],
                0,
            )
            cS = cute.domain_offset(logical_offset, cS_base)
            value_args_ = (
                seqlen_k, seqlen_q, scale_softmax_log2,
                window_size_left, window_size_right,
            )
            atom_args = (
                qk_thr_mma, tiled_tmem_load, tiled_tmem_store,
                tiled_tmem_store_vec, thr_tmem_load, thr_tmem_store,
                thr_tmem_store_vec,
            )
            tensor_args_ = (
                tTMEM_LOADtS, tTMEM_STORE_VECtS, tTMEM_STOREtS_x4, sSFP,
            )
            st_cnt, end_cnt, ld_mask_cnt, unmask_cnt, tl_mask_cnt = (
                fmha_utils.FusedMask.get_masked_info(
                    self.mask_type, curr_block_coord, self.cta_tiler,
                    seqlen_q, seqlen_k, window_size_left, window_size_right,
                )
            )
            row_max = -Float32.inf
            row_sum = Float32(0.0)
            stats_args = (row_sum, row_max)
            si_peek_status = mma_si_consumer.try_wait()
            pipeline_args_ = (
                si_peek_status, mma_si_consumer, si_corr_producer,
                pi_mma_producer,
            )

            def softmax_loop(whether_apply_mask, loop_args, stats_args,
                             pipeline_args, cS):
                start_index, iter_num, upper_bound = loop_args
                for i in cutlass.range(
                    start_index, start_index + iter_num, 1, unroll=1
                ):
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    iter_args = (cS_iter, i == upper_bound - 1)
                    stats_args, pipeline_args = self.softmax_step(
                        stage, whether_apply_mask, iter_args, stats_args,
                        pipeline_args, value_args_, atom_args, tensor_args_,
                    )
                return stats_args, pipeline_args

            loop_args = (st_cnt, ld_mask_cnt, end_cnt)
            stats_args, pipeline_args_ = softmax_loop(
                True, loop_args, stats_args, pipeline_args_, cS
            )
            loop_args = (st_cnt + ld_mask_cnt, unmask_cnt, end_cnt)
            stats_args, pipeline_args_ = softmax_loop(
                False, loop_args, stats_args, pipeline_args_, cS
            )
            loop_args = (st_cnt + ld_mask_cnt + unmask_cnt, tl_mask_cnt, end_cnt)
            stats_args, pipeline_args_ = softmax_loop(
                True, loop_args, stats_args, pipeline_args_, cS
            )
            (_, mma_si_consumer, si_corr_producer, pi_mma_producer) = (
                pipeline_args_
            )
            tTMEM_STORE_VECrS = cute.make_rmem_tensor(
                tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
            )
            tTMEM_STORE_VECrS[0] = stats_args[0]
            tTMEM_STORE_VECrS[1] = stats_args[1]
            vec_i_handle = si_corr_producer.acquire_and_advance()
            cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
            cute.arch.fence_view_async_tmem_store()
            vec_i_handle.commit()
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    @cute.jit
    def softmax_step(self, stage, whether_apply_mask, iter_args, stats_args,
                     pipeline_args, value_args, atom_args, tensor_args):
        '一个 kv chunk 的 softmax:S 读回 → (掩码) → 在线 max/sum →'
        row_sum, row_max = stats_args
        cS, is_last_iter = iter_args
        (seqlen_k, seqlen_q, scale_softmax_log2,
         window_size_left, window_size_right) = value_args
        (si_peek_status, mma_si_consumer, si_corr_producer,
         pi_mma_producer) = pipeline_args
        (qk_thr_mma, tiled_tmem_load, tiled_tmem_store, tiled_tmem_store_vec,
         thr_tmem_load, thr_tmem_store, thr_tmem_store_vec) = atom_args
        (tTMEM_LOADtS, tTMEM_STORE_VECtS, tTMEM_STOREtS_x4, sSFP) = tensor_args

        tidx, _, _ = cute.arch.thread_idx()
        thread_row = tidx % (
            self.threads_per_warp * len(self.softmax0_warp_ids)
        )

        tScS = qk_thr_mma.partition_C(cS)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(
            cute.make_tensor(
                tScS.iterator,
                cute.composition(tScS.layout, cute.make_layout((128, 2))),
            )
        )
        tScS_P = cute.make_tensor(
            tScS.iterator,
            cute.composition(
                tScS.layout,
                cute.make_layout(
                    (128, self.qk_mma_tiler[1] // 32 * self.p_dtype.width)
                ),
            ),
        )
        tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P)

        si_handle = mma_si_consumer.wait_and_advance(si_peek_status)
        tTMEM_LOADrS = cute.make_rmem_tensor(
            tTMEM_LOADcS.shape, self.qk_acc_dtype
        )
        old_row_max = row_max
        # ---- S → 寄存器 + (掩码) + tile max ----
        cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)
        if cutlass.const_expr(whether_apply_mask):
            fmha_utils.FusedMask.apply_mask(
                self.mask_type, tTMEM_LOADrS, tTMEM_LOADcS,
                seqlen_q, seqlen_k, window_size_left, window_size_right,
            )
        tile_row_max = tTMEM_LOADrS.load().reduce(
            cute.ReductionOp.MAX, -Float32.inf, 0
        )
        row_max = cute.arch.fmax(row_max, tile_row_max)
        si_handle.release()

        row_max_safe = row_max
        if row_max == -Float32.inf:
            row_max_safe = 0.0
        tTMEM_STORE_VECrS = cute.make_rmem_tensor(
            tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
        )
        tTMEM_STORE_VECrS[0] = old_row_max
        tTMEM_STORE_VECrS[1] = row_max_safe
        vec_i_peek_status = si_corr_producer.try_acquire()

        tTMEM_STORErS_x4 = cute.make_rmem_tensor(
            tTMEM_STOREcS.shape, self.qk_acc_dtype
        )
        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale

        n_elts = cutlass.const_expr(cute.size(tTMEM_LOADrS))
        assert n_elts % SF_VEC == 0

        # ---- FMA + exp2(全部元素,一处做完)----
        for i in cutlass.range_constexpr(0, n_elts, 2):
            tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1] = cute.arch.fma_packed_f32x2(
                (tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1]),
                (scale, scale),
                (minus_row_max_scale, minus_row_max_scale),
            )
            tTMEM_LOADrS[i] = cute.math.exp2(tTMEM_LOADrS[i], fastmath=True)
            tTMEM_LOADrS[i + 1] = cute.math.exp2(
                tTMEM_LOADrS[i + 1], fastmath=True
            )

        # row_max/old_max 交给 correction(先送,重缩放与 exp 并行)
        vec_i_handle = si_corr_producer.acquire_and_advance(vec_i_peek_status)
        cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
        cute.arch.fence_view_async_tmem_store()
        vec_i_handle.commit()

        # ---- 分母:未量化口径 ----
        # 旧实现是一条 32-deep packed-add 依赖链。四个命名累加器与四个
        # SF group 对齐，关键路径降到 8+3；不用动态 rmem tensor，避免 CuTe
        # 的静态展开/布局分析在远端编译时出现资源爆炸。
        sum0 = (Float32(0.0), Float32(0.0))
        sum1 = (Float32(0.0), Float32(0.0))
        sum2 = (Float32(0.0), Float32(0.0))
        sum3 = (Float32(0.0), Float32(0.0))
        for j in cutlass.range_constexpr(0, SF_VEC, 2):
            sum0 = cute.arch.add_packed_f32x2(
                sum0, (tTMEM_LOADrS[j], tTMEM_LOADrS[j + 1])
            )
            sum1 = cute.arch.add_packed_f32x2(
                sum1,
                (tTMEM_LOADrS[SF_VEC + j], tTMEM_LOADrS[SF_VEC + j + 1]),
            )
            sum2 = cute.arch.add_packed_f32x2(
                sum2,
                (tTMEM_LOADrS[2 * SF_VEC + j],
                 tTMEM_LOADrS[2 * SF_VEC + j + 1]),
            )
            sum3 = cute.arch.add_packed_f32x2(
                sum3,
                (tTMEM_LOADrS[3 * SF_VEC + j],
                 tTMEM_LOADrS[3 * SF_VEC + j + 1]),
            )
        acc_scale = cute.math.exp2(
            scale * (old_row_max - row_max_safe), fastmath=True
        )
        row_sum = (
            row_sum * acc_scale
            + sum0[0] + sum0[1]
            + sum1[0] + sum1[1]
            + sum2[0] + sum2[1]
            + sum3[0] + sum3[1]
        )

        # ---- 逐 16 组动态 SF(线程内 amax)+ 就地归一化 ----
        # TMEM load fragment 的寄存器存储顺序就是连续 N(简单 kernel 已用
        # 同一 alias)。避免复制 64 个 fp32 到第二个临时 fragment。
        rS_flat = cute.make_tensor(
            tTMEM_LOADrS.iterator, cute.make_layout(n_elts)
        )
        sf_frag = cute.make_rmem_tensor(
            cute.make_layout(n_elts // SF_VEC), self.sf_dtype
        )
        PDynamicSf.normalize_and_emit_sf(rS_flat, sf_frag)

        # ---- SFP → SMEM(UTCCP 的源;坐标索引交给 CuTe,与冻结的
        #      sf_atom_offsets 公式逐项对上 —— simple kernel B200 同款)----
        for g in cutlass.range_constexpr(cute.size(sf_frag)):
            sSFP[(((thread_row, 0), (0, g % 4)), 0, g // 4, stage)] = sf_frag[g]
        cute.arch.fence_view_async_shared()

        # ---- π 重排 + e2m1 + P → TMEM ----
        pS_pi = cute.make_rmem_tensor(
            cute.make_layout(n_elts), self.qk_acc_dtype
        )
        for s in cutlass.range_constexpr(n_elts):
            pS_pi[s] = rS_flat[(s // SF_VEC) * SF_VEC + PI16[s % SF_VEC]]
        tTMEM_STORErS_x4_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS_x4.iterator, dtype=self.p_dtype),
            cute.make_layout(n_elts),
        )
        tTMEM_STORErS_x4_e.store(pS_pi.load().to(self.p_dtype))
        pi_handle = pi_mma_producer.acquire_and_advance()
        cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4)
        cute.arch.fence_view_async_tmem_store()
        pi_handle.commit()

        if not is_last_iter:
            si_peek_status = mma_si_consumer.try_wait()

        stats_args = (row_sum, row_max_safe)
        pipeline_args = (
            si_peek_status, mma_si_consumer, si_corr_producer, pi_mma_producer,
        )
        return stats_args, pipeline_args

    def check_invalid_shape(self, *args, **kwargs):
        raise NotImplementedError("用 __call__ 的断言;vendor 形状检查不适用")
