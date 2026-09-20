# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'DEPRECATED FOR PRODUCTION: historical prefill correctness prototype.'

from __future__ import annotations

from typing import Optional, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.typing import Float32, Int32

from .kernel_paged_prefill import FP4_DTYPE, SF_DTYPE, PDynamicSf
from .kernel_paged_decode import _sf_s2t_partition
from .kv_layout import PI16, SF_VEC

#: 单 warpgroup:softmax 的 M=128 行正好一线程一行(见 PDynamicSf 的 atom 分析)
THREADS = 128
Q_TILE = 128
KV_TILE = 128
#: SMEM 张量基址对齐(坑27:≥ swizzle 周期)。decode:917 用的同一个值。
STENSOR_ALIGN = 1024


class PrefillSimpleFp4:
    """单 warpgroup、无流水的 nvfp4 prefill。Q 在 M、KV 在 N(不交换)。

    形状约束(**故意收紧**,免得在没验证的组合上浪费上机时间):
      - head_dim == 128;q_tile == kv_tile == 128;
      - 每个 CTA 负责一个 (batch, head, q_tile)。
    """

    def __init__(self, head_dim: int = 128, causal: bool = True,
                 n_kv_tiles: int = 1, h_q: int = 1, h_kv: int = 1,
                 paged: bool = False, residue_levels: int = 1,
                 debug_trace: bool = False, stop_after: int = 4):
        '``debug_trace=True`` 在每个同步点打一行 checkpoint。'
        assert head_dim == 128, f"简化版只支持 head_dim=128,得到 {head_dim}"
        self.head_dim = head_dim
        self.causal = causal
        self.debug_trace = debug_trace
        
        
        
        
        
        
        
        
        
        
        assert 0 <= stop_after <= 4, stop_after
        self.stop_after = stop_after
        # S3:kv tile 数是**编译期**常量(gmem 形状、expected_tx、causal 的
        # ctx 偏移都从它导出)。每个 n 编一份,host 的 lru_cache 按 n 键控 ——
        # 正确性优先;动态 s_kv 是之后的优化。
        assert n_kv_tiles >= 1, n_kv_tiles
        self.n_kv_tiles = n_kv_tiles
        # S4:多头/GQA。grid = (h_q, 1, 1),一个 CTA 一个 q 头;kv 头 =
        # bidx // (h_q // h_kv)。所有 gmem 张量的第三模(原先恒为 1 的 L)
        # 就是 head 模 —— 当初留 L 模就是为了这一步。头数同样是编译期常量
        # (坑38:编译期已定的常量绝不再走运行期参数)。
        assert h_q >= 1 and h_kv >= 1 and h_q % h_kv == 0, (h_q, h_kv)
        self.h_q = h_q
        self.h_kv = h_kv
        # S5:分页取址。K/V/SF 的 gmem 统一成 (tile, D, L) 单 rest 视图,
        # L = 页×头 扁平;稠密与池只差 l 的编号公式(kernel 内编译期分支):
        #   稠密 l = kv_head * n + pt[t](恒等表 ⇒ = kv_head*n + t)
        #   池   l = pt[t] * h_kv + kv_head(pool 物理 [P, H, ...] 页最外)
        # kv_tile == page_size == 128 ⇒ 每 tile 恰好一页,页表一次查表
        # (kv_layout.kv_tile_page 的"红利")。
        self.paged = paged
        # S6:residue。Q 量化成 base + residual 两级(q₁=Q(x)、q₂=Q(x-q₁)),
        # S = Q₁·Kᵀ + Q₂·Kᵀ —— 第二条 KQ MMA 带累加,其余全不变。
        # Q 的 gmem L 模扩成 h_q × levels(lev 外、head 内:l = lev*h_q + h);
        # SMEM 多一块 sQ2/sSFQ2,两级都常驻,一次 q_bar 载入、两条 gemm
        # 同一个 commit(误差主项在 KV 侧,故 KV 不做 residue)。
        assert residue_levels in (1, 2), residue_levels
        self.residue_levels = residue_levels
        self.q_tile = Q_TILE
        self.kv_tile = KV_TILE
        self.sf_vec_size = SF_VEC
        self.sf_dtype = SF_DTYPE
        self.acc_dtype = Float32
        # 测试用的 trace 期探针:设成 {} 后,cute.compile 过程中 kernel 的 TMA
        # 段会把**分区后张量**的元素数记进来(纯 Python 记录,不进 IR)。
        # 为什么值得留:expected_tx 之前只对过 SMEM **布局**的字节数,而硬件
        # 实搬多少由 tma_partition 之后的分区张量决定 —— 两者不相等就是
        # "barrier 永远等不满"式死锁,且编译期无痕。这是唯一能离线抓它的口。
        self.trace_sizes: Optional[dict] = None

    def _mmas(self, cta_group):
        """两个 blockscaled MMA。**A/B 不交换**:A=Q/P(M 轴)、B=K/V(N 轴)。

        对照 ``kernel_paged_decode.py:392`` 那里参数顺序是 (K, Q) —— 交换的
        直接体现。这里是 (Q, K)。
        """
        kq = sm100_utils.make_blockscaled_trivial_tiled_mma(
            FP4_DTYPE, FP4_DTYPE,
            OperandMajorMode.K,  # A = Q,沿 head_dim 收缩
            OperandMajorMode.K,  # B = K,同上
            self.sf_dtype, self.sf_vec_size, cta_group, (self.q_tile, self.kv_tile),
        )
        vp = sm100_utils.make_blockscaled_trivial_tiled_mma(
            FP4_DTYPE, FP4_DTYPE,
            OperandMajorMode.K,  # A = P,沿 kv token 收缩
            OperandMajorMode.K,  # B = V,同上(坑28:token 必须 gmem 连续)
            self.sf_dtype, self.sf_vec_size, cta_group,
            (self.q_tile, self.head_dim),
        )
        return kq, vp

    def smem_layouts(self, kq, vp):
        """Q/K/V 数据 + 四个 SF 的 SMEM 布局(单 stage —— 无流水)。"""
        q = sm100_utils.make_smem_layout_a(kq, (self.q_tile, self.kv_tile,
                                                self.head_dim), FP4_DTYPE, 1)
        k = sm100_utils.make_smem_layout_b(kq, (self.q_tile, self.kv_tile,
                                                self.head_dim), FP4_DTYPE, 1)
        v = sm100_utils.make_smem_layout_b(vp, (self.q_tile, self.head_dim,
                                                self.kv_tile), FP4_DTYPE, 1)
        sf = {
            "sfq": blockscaled_utils.make_smem_layout_sfa(
                kq, (self.q_tile, self.kv_tile, self.head_dim), SF_VEC, 1),
            "sfk": blockscaled_utils.make_smem_layout_sfb(
                kq, (self.q_tile, self.kv_tile, self.head_dim), SF_VEC, 1),
            "sfv": blockscaled_utils.make_smem_layout_sfb(
                vp, (self.q_tile, self.head_dim, self.kv_tile), SF_VEC, 1),
            # SFP 由 softmax **核内产生**(不是 TMA 进来的)⇒ 1 stage
            "sfp": blockscaled_utils.make_smem_layout_sfa(
                vp, (self.q_tile, self.head_dim, self.kv_tile), SF_VEC, 1),
        }
        return q, k, v, sf

    def tmem_layouts(self, kq, vp, sf_smem):
        """TMEM 里要放的四块:S(KQ 累加器)、O(VP 累加器)、SFA/SFB。

        blockscaled MMA 的 SF 必须**在 TMEM 里**,由 UTCCP 从 SMEM 搬进来
        (``kernel_paged_decode.py:60`` 的 ``_sf_s2t_partition``,移植自
        cutlass ``mainloop_s2t_copy_and_partition``)。数据操作数可以直接走
        SMEM descriptor,SF 不行 —— 这是最容易漏的一条。

        侧别跟数据走(见 kernel_paged_prefill.OPERAND_SIDE):
        KQ 的 A=Q ⇒ SFA=SFQ、B=K ⇒ SFB=SFK;VP 的 A=P ⇒ SFA=SFP、B=V ⇒ SFB=SFV。
        """
        one = lambda lay: cute.slice_(lay, (None, None, None, 0))
        # residue:SFQ 的第二级独占一块 TMEM(布局同 sfq)。
        extra = {}
        if self.residue_levels == 2:
            extra["sfq2"] = blockscaled_utils.make_tmem_layout_sfa(
                kq, (self.q_tile, self.kv_tile, self.head_dim), SF_VEC,
                one(sf_smem["sfq"]))
        return {
            **extra,
            "sfq": blockscaled_utils.make_tmem_layout_sfa(
                kq, (self.q_tile, self.kv_tile, self.head_dim), SF_VEC,
                one(sf_smem["sfq"])),
            "sfk": blockscaled_utils.make_tmem_layout_sfb(
                kq, (self.q_tile, self.kv_tile, self.head_dim), SF_VEC,
                one(sf_smem["sfk"])),
            "sfp": blockscaled_utils.make_tmem_layout_sfa(
                vp, (self.q_tile, self.head_dim, self.kv_tile), SF_VEC,
                one(sf_smem["sfp"])),
            "sfv": blockscaled_utils.make_tmem_layout_sfb(
                vp, (self.q_tile, self.head_dim, self.kv_tile), SF_VEC,
                one(sf_smem["sfv"])),
        }

    @staticmethod
    @cute.jit
    def online_softmax_step(s_frag, row_max, row_sum, sf_frag):
        '一个 kv tile 的在线 softmax + P 的动态量化。**语义由参考定义**:'
        n = cutlass.const_expr(cute.size(s_frag))
        tile_max = row_max
        for i in cutlass.range_constexpr(n):
            tile_max = cute.arch.fmax(tile_max, s_frag[i])
        # 旧累加值的重缩放因子(调用方拿它去缩 O)
        corr = cute.math.exp2(row_max - tile_max, fastmath=True)
        for i in cutlass.range_constexpr(n):
            s_frag[i] = cute.math.exp2(s_frag[i] - tile_max, fastmath=True)
        # 口径 2:分母用**未量化**的值
        new_sum = row_sum * corr
        for i in cutlass.range_constexpr(n):
            new_sum = new_sum + s_frag[i]
        # 逐 16 组动态 SF + 就地归一化(见 PDynamicSf:纯线程内,无 SHFL)
        PDynamicSf.normalize_and_emit_sf(s_frag, sf_frag)
        return tile_max, new_sum, corr

    @cute.jit
    def issue_kq_gemm(self, kq, tCtS, sQ_desc, sK_desc, tCtSFQ, tCtSFK):
        """KQ:``S = Q·Kᵀ``。**A=Q、B=K —— 不交换**。

        blockscaled MMA 的 SF 不能当参数传,要先用 ``set_field`` 绑到 tiled_mma
        上(``kernel_paged_decode.py:1739`` 同款)。SFA↔A、SFB↔B **必须对应**:
        接反了编译期不报,数值上整片乱掉。
        """
        n_k = cutlass.const_expr(cute.size(sQ_desc, mode=[2]))
        for mma_k in cutlass.range_constexpr(n_k):
            kq.set(tcgen05.Field.SFA, tCtSFQ[None, None, mma_k].iterator)
            kq.set(tcgen05.Field.SFB, tCtSFK[None, None, mma_k].iterator)
            cute.gemm(
                kq, tCtS,
                sQ_desc[None, None, mma_k, 0],   # A = Q
                sK_desc[None, None, mma_k, 0],   # B = K
                tCtS,
            )
            # 第 1 条之后累加(而不是覆盖)—— residue 也是靠这个累加语义:
            # S = Q_base·Kᵀ + Q_res·Kᵀ,再发一条即可,不需要任何折叠
            # (见 plans §6.1)。
            kq.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def issue_vp_gemm(self, vp, tCtO, sP_desc, sV_desc, tCtSFP, tCtSFV, first: bool):
        """VP:``O += P·V``。A=P(M 轴)、B=V(N=head_dim)。

        ``first`` 决定首条是否清累加器 —— 每个 q_tile 的第一个 kv tile 要清,
        其后累加。这与 online softmax 的 O 重缩放配合:重缩放作用在**读回**的
        累加值上,而不是靠 MMA 的 ACCUMULATE 位。
        """
        vp.set(tcgen05.Field.ACCUMULATE, not first)
        n_k = cutlass.const_expr(cute.size(sP_desc, mode=[2]))
        for mma_k in cutlass.range_constexpr(n_k):
            vp.set(tcgen05.Field.SFA, tCtSFP[None, None, mma_k].iterator)
            vp.set(tcgen05.Field.SFB, tCtSFV[None, None, mma_k].iterator)
            cute.gemm(
                vp, tCtO,
                sP_desc[None, None, mma_k, 0],   # A = P
                sV_desc[None, None, mma_k, 0],   # B = V
                tCtO,
            )
            vp.set(tcgen05.Field.ACCUMULATE, True)

    #: SMEM 用量(字节):fp4 数据块(Q×levels/K/V/P)+ SF 块(同数)+ 标量。
    #: 显式算而不是拍一个大数 —— 拍大数会在 sm_100 的 227KB 上限附近静默失败。
    def smem_bytes(self) -> int:
        data = self.q_tile * self.head_dim // 2
        sf = self.q_tile * self.head_dim // SF_VEC
        blocks = 3 + self.residue_levels                 # K/V/P + Q×levels
        return blocks * data + blocks * sf + 1024        # 1KB 给 mbar/对齐

    #: Q 批(Q×levels + SFQ×levels,启动时一次)与 KV 批(K/V/SFK/SFV,
    #: 每 tile 一次)的 expected_tx。分开算 —— 两个 mbarrier 各等各的。
    def q_tma_bytes(self) -> int:
        return self.residue_levels * (
            self.q_tile * self.head_dim // 2
            + self.q_tile * self.head_dim // SF_VEC)

    def kv_tma_bytes(self) -> int:
        data = self.kv_tile * self.head_dim // 2
        sf = self.kv_tile * self.head_dim // SF_VEC
        return 2 * data + 2 * sf                 # K+V 数据 + SFK+SFV

    #: 整个问题的 TMA 总搬运量(旧口径,离线门用它对总账)。
    def tma_bytes(self) -> int:
        return self.q_tma_bytes() + self.n_kv_tiles * self.kv_tma_bytes()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,     # (S_q, D, h_q) e2m1  —— A 侧
        mK: cute.Tensor,     # (128, D, L_kv) e2m1 —— B 侧,L = 页×头 扁平
        mV: cute.Tensor,     # (D, 128, L_kv) e2m1 —— token 最内(坑28)
        sfq_flat: cute.Tensor,   # 扁平 e4m3;布局在 trace 内用 tile_atom_to_shape_SF 算
        sfk_flat: cute.Tensor,
        sfv_flat: cute.Tensor,
        mO: cute.Tensor,     # (S_q, D, h_q) bf16
        kv_pt: cute.Tensor,  # (n_kv_tiles,) Int32 页表;稠密传恒等表
        sm_scale: Float32,
        stream,
    ):
        'host 入口。SF 必须**扁平**传进来 —— ``tile_atom_to_shape_SF`` 只能在'
        kq, vp = self._mmas(tcgen05.CtaGroup.ONE)
        sQ_l, sK_l, sV_l, sf_l = self.smem_layouts(kq, vp)
        tmem_l = self.tmem_layouts(kq, vp, sf_l)
        mk_sf = lambda f, shape: cute.make_tensor(
            f.iterator, blockscaled_utils.tile_atom_to_shape_SF(shape, SF_VEC))
        # SF 的 gmem 视图覆盖**全部** s_kv;每 tile 的 TMA box 由 kernel 里
        # local_tile 的 rest 模选出。SF atom 序的 K_rest-先-M_rest 展开保证
        # 每个 128-token tile 在 atom 序里是**连续块**(decode:1452 的"定理",
        # CPU 门 test_sf_atom_matches_cutlass_blockscaled_chunk 盯着展开序)。
        # KV 的第三模 = L(页×头 扁平,S5);SF 的 atom 序按 L 最外分块,
        # 每块 = 一页一头的 1024 字节 —— 与池的页局部 SF、与稠密 host 逐头
        # concat 的扁平序都对得上(l 编号见 __init__ 的公式)。
        l_kv = self.n_kv_tiles * self.h_kv
        gSFQ = mk_sf(sfq_flat, (self.q_tile, self.head_dim,
                                self.h_q * self.residue_levels))
        gSFK = mk_sf(sfk_flat, (self.kv_tile, self.head_dim, l_kv))
        gSFV = mk_sf(sfv_flat, (self.head_dim, self.kv_tile, l_kv))
        a = self.make_tma(kq, vp, sQ_l, sK_l, sV_l, sf_l,
                          mQ, mK, mV, gSFQ, gSFK, gSFV)
        # log2 底:参考用 exp2,kernel 也用 exp2 ⇒ scale 先吸收 log2(e)。
        scale_log2 = sm_scale * 1.4426950408889634
        self.kernel(
            kq, vp,
            a["q"][0], a["q"][1], a["k"][0], a["k"][1], a["v"][0], a["v"][1],
            a["sfq"][0], a["sfq"][1], a["sfk"][0], a["sfk"][1],
            a["sfv"][0], a["sfv"][1],
            mO, kv_pt, sQ_l, sK_l, sV_l, sf_l, tmem_l,
            Float32(scale_log2),
        ).launch(
            grid=(self.h_q, 1, 1),           # 一个 CTA 一个 q 头(S4)
            block=[THREADS, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        kq: cute.TiledMma,
        vp: cute.TiledMma,
        tma_q, gQ, tma_k, gK, tma_v, gV,
        tma_sfq, gSFQ, tma_sfk, gSFK, tma_sfv, gSFV,
        mO: cute.Tensor,
        kv_pt: cute.Tensor,
        sQ_l, sK_l, sV_l, sf_l, tmem_l,
        scale_log2: Float32,
    ):
        """一个 CTA = 一个 (batch, head, q_tile)。单 warpgroup、无流水。

        平铺直叙:Q 一次载入 → 逐 kv tile(该 tile 的 K/V TMA → KQ gemm →
        softmax+动态SF → UTCCP → VP gemm)→ epilogue。**不追求性能**,
        用途是给 warp 特化版当逐 tile 对照物。expected_tx 是编译期常量
        (q_tma_bytes / kv_tma_bytes),不再走运行期参数。
        """
        tidx, _, _ = cute.arch.thread_idx()
        # S4:CTA x = q 头;kv 头 = q 头 // GQA 组大小(h_q/h_kv 编译期常量)。
        bidx, _, _ = cute.arch.block_idx()
        kv_head = bidx // cutlass.const_expr(self.h_q // self.h_kv)
        dbg = cutlass.const_expr(self.debug_trace)
        smem = utils.SmemAllocator()
        one = lambda l: cute.slice_(l, (None, None, None, 0))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        alloc = lambda dt, l: smem.allocate_tensor(dt, l.outer, STENSOR_ALIGN,
                                                  l.inner)
        sQ = alloc(FP4_DTYPE, sQ_l)
        sK = alloc(FP4_DTYPE, sK_l)
        sV = alloc(FP4_DTYPE, sV_l)
        
        
        
        
        
        sP = alloc(FP4_DTYPE, sQ_l)
        sf_alloc = lambda n: smem.allocate_tensor(
            self.sf_dtype, one(sf_l[n]), STENSOR_ALIGN)
        sSFQ = sf_alloc("sfq")
        sSFK = sf_alloc("sfk")
        sSFV = sf_alloc("sfv")
        sSFP = sf_alloc("sfp")
        # S6:Q 的第二级(residual)独占一块 SMEM,与 base 同布局、同驻留。
        # 注意:分支必须**内联** `cutlass.const_expr(...)` —— 预处理器只认
        # 这个字面模式;存进变量再 if 会被当 staged 分支,两侧都展开。
        if cutlass.const_expr(self.residue_levels == 2):
            sQ2 = alloc(FP4_DTYPE, sQ_l)
            sSFQ2 = sf_alloc("sfq")
        tmem_holder = smem.allocate_array(Int32)
        q_bar = smem.allocate_array(cutlass.Int64, 1)     # Q+SFQ 载入完成(一次)
        kv_bar = smem.allocate_array(cutlass.Int64, 1)    # 本 tile K/V 载入完成
        # MMA 完成。tcgen05 的 MMA 是**异步**的:`cute.gemm` 发射后累加器什么时候
        # 在 TMEM 里可读,只有 mbarrier 说了算。少了这一步,softmax 会读到上一轮
        # 的 S(或半写状态)—— 编译期无痕、数值上时对时错。
        mma_bar = smem.allocate_array(cutlass.Int64, 1)

        # mbarrier init **必须** warp 内全员进入 + elect_one 选一个执行
        # (cpasync/copy.py 文档原文 "Requires elect_one")。之前写 `if tidx == 0`
        # —— 单线程进入含 elect.sync 的路径 = 部分参与的 warp 汇聚 = UB。
        if cute.arch.warp_idx() == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_bar, 1)
                cute.arch.mbarrier_init(kv_bar, 1)
                cute.arch.mbarrier_init(mma_bar, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        
        
        
        
        
        if dbg and tidx == 0:
            cute.printf("CK1 mbar-init done\n")
        if cute.arch.warp_idx() == 0:
            cute.arch.alloc_tmem(cols, tmem_holder)
        cute.arch.barrier()
        tptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_holder)
        if dbg and tidx == 0:
            cute.printf("CK2 tmem alloc done, cols=%d\n", cols)

        # 每线程一行(M=128,单 warpgroup)—— 见 PDynamicSf 的 atom 分析
        row_max = Float32(-cutlass.Float32.inf)
        row_sum = Float32(0.0)

        
        
        
        
        
        
        
        
        
        
        
        
        
        thr_kq = kq.get_slice(0)
        thr_vp = vp.get_slice(0)
        
        
        
        
        
        
        sQ_desc = thr_kq.make_fragment_A(sQ)
        sK_desc = thr_kq.make_fragment_B(sK)
        sP_desc = thr_vp.make_fragment_A(sP)
        sV_desc = thr_vp.make_fragment_B(sV)
        if cutlass.const_expr(self.residue_levels == 2):
            sQ2_desc = thr_kq.make_fragment_A(sQ2)

        
        
        
        
        
        
        
        off = 0
        tCtS = cute.make_tensor(
            cute.recast_ptr(tptr + off, dtype=self.acc_dtype),
            thr_kq.make_fragment_C(
                thr_kq.partition_shape_C((self.q_tile, self.kv_tile))).layout,
        )
        off += tcgen05.find_tmem_tensor_col_offset(tCtS)
        tCtO = cute.make_tensor(
            cute.recast_ptr(tptr + off, dtype=self.acc_dtype),
            thr_vp.make_fragment_C(
                thr_vp.partition_shape_C((self.q_tile, self.head_dim))).layout,
        )
        off += tcgen05.find_tmem_tensor_col_offset(tCtO)
        tSF = {}
        sf_names = ("sfq", "sfk", "sfp", "sfv") + (
            ("sfq2",) if self.residue_levels == 2 else ())
        for name in sf_names:
            tSF[name] = cute.make_tensor(
                cute.recast_ptr(tptr + off, dtype=self.sf_dtype), tmem_l[name]
            )
            off += tcgen05.find_tmem_tensor_col_offset(tSF[name])
        # TMEM 预算:超了不会报错,只会绕回去踩别人(和上面的叠列同一类事故)。
        assert off <= 512, f"TMEM 各块共 {off} 列 > 512"

        # ---- SF 的 SMEM→TMEM(UTCCP)分区。数据操作数走 SMEM descriptor,
        # SF **必须**先进 TMEM —— 最容易漏的一条。 ----
        s2t = {}
        sf_smem_pairs = [("sfq", sSFQ), ("sfk", sSFK), ("sfv", sSFV),
                         ("sfp", sSFP)]
        if cutlass.const_expr(self.residue_levels == 2):
            sf_smem_pairs.append(("sfq2", sSFQ2))
        for name, sSF in sf_smem_pairs:
            s2t[name] = _sf_s2t_partition(self.sf_dtype, sSF, tSF[name])

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        cta_layout = cute.make_layout(1)
        # 分区是**编译期**的纯值计算,放在线程分支外(所有线程都算,只有
        # warp 0 发 copy)—— 也让 trace_sizes 的记录不落进运行期分支
        # (DSL 不允许 staged `if` 里出现 Python 侧赋值)。
        # ---- Q/SFQ:head 模留作 rest,发 copy 时按 bidx 选头(S4;h_q=1 时
        # rest 恒为 1,与 S1–S3 的"group 全模"语义一致)----
        # residue:第二级 Q/SFQ 复用同一个 atom + gmem 分区,L 偏移 h_q
        # (l = lev*h_q + head,lev 外、head 内),SMEM 落到 sQ2/sSFQ2。
        q_specs = [
            ("q", tma_q, gQ, sQ,
             (Q_TILE, self.head_dim, 1), (0, 0, None), 0, False),
            ("sfq", tma_sfq, gSFQ, sSFQ,
             (Q_TILE, self.head_dim), (0, 0, None), 0, True),
        ]
        if cutlass.const_expr(self.residue_levels == 2):
            q_specs += [
                ("q2", tma_q, gQ, sQ2,
                 (Q_TILE, self.head_dim, 1), (0, 0, None), self.h_q, False),
                ("sfq2", tma_sfq, gSFQ, sSFQ2,
                 (Q_TILE, self.head_dim), (0, 0, None), self.h_q, True),
            ]
        q_parts = []
        for name, atom, g, s, tiler, coord, l_off, is_sf in q_specs:
            g_t = cute.local_tile(g, tiler, coord)   # ((tile...), h_q*levels)
            s_part, g_part = cute.nvgpu.cpasync.tma_partition(
                atom, 0, cta_layout,
                cute.group_modes(s, 0, cute.rank(s)),
                cute.group_modes(g_t, 0, cute.rank(g_t) - 1),
            )
            # 坑13:SF 布局带 stride-0 广播模式,不滤掉 TMA 会**零发射** ——
            # 编译通过、不报错、SF 全是垃圾(decode:1467 同样处理)。
            if cutlass.const_expr(is_sf):
                s_part = cute.filter_zeros(s_part)
                g_part = cute.filter_zeros(g_part)
            if cutlass.const_expr(self.trace_sizes is not None):
                # trace 期(cute.compile 中)记录实搬元素数;布局全静态,
                # cute.size 在这里就是 Python int。见 __init__ 的说明。
                self.trace_sizes[name] = dict(
                    s_elems=cute.size(s_part), g_elems=cute.size(g_part),
                    bits=(self.sf_dtype if is_sf else FP4_DTYPE).width,
                )
            q_parts.append((atom, g_part, s_part, l_off))
        # ---- K/V/SFK/SFV:gmem 的 tile 模已在 L(页×头 扁平)里,tile 全覆盖
        # ⇒ 唯一 rest = L;逐 tile 用 l 公式选 box(S5 统一稠密与池)----
        kv_parts = []
        for name, atom, g, s, tiler, coord, is_sf in (
            ("k", tma_k, gK, sK,
             (KV_TILE, self.head_dim, 1), (0, 0, None), False),
            ("v", tma_v, gV, sV,
             (self.head_dim, KV_TILE, 1), (0, 0, None), False),
            ("sfk", tma_sfk, gSFK, sSFK,
             (KV_TILE, self.head_dim), (0, 0, None), True),
            ("sfv", tma_sfv, gSFV, sSFV,
             (self.head_dim, KV_TILE), (0, 0, None), True),
        ):
            g_t = cute.local_tile(g, tiler, coord)   # ((tile...), L_kv)
            s_part, g_part = cute.nvgpu.cpasync.tma_partition(
                atom, 0, cta_layout,
                cute.group_modes(s, 0, cute.rank(s)),
                cute.group_modes(g_t, 0, cute.rank(g_t) - 1),
            )
            if cutlass.const_expr(is_sf):
                s_part = cute.filter_zeros(s_part)
                g_part = cute.filter_zeros(g_part)
            if cutlass.const_expr(self.trace_sizes is not None):
                self.trace_sizes[name] = dict(
                    s_elems=cute.size(s_part), g_elems=cute.size(g_part),
                    bits=(self.sf_dtype if is_sf else FP4_DTYPE).width,
                )
            kv_parts.append((atom, g_part, s_part))
        
        
        
        
        
        
        
        
        
        
        
        
        if cute.arch.warp_idx() == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    q_bar, cutlass.const_expr(self.q_tma_bytes()))
            for _qa, _qg, _qs, _ql in q_parts:
                cute.copy(_qa, _qg[(None, bidx + _ql)], _qs,
                          tma_bar_ptr=q_bar)
        if dbg and tidx == 0:
            cute.printf("CK3 Q TMA issued\n")
        cute.arch.mbarrier_wait(q_bar, 0)
        if dbg and tidx == 0:
            cute.printf("CK4 Q TMA barrier passed\n")

        
        
        
        
        
        sP_b = cute.recast_tensor(sP, cutlass.Int8)
        sP_flat = cute.make_tensor(
            cute.recast_ptr(sP_b.iterator, sQ_l.inner, dtype=cutlass.Int8),
            cute.make_layout(cute.size(sP_b.layout)),
        )
        # 线程 tidx 负责 query 行 tidx 的整行 P:
        #   元素偏移 = 128*row + token(布局 ((128,64),1,2,1):((128,1),0,64,0) 里
        #   token 的两段 64 拼起来正好连续)⇒ 字节偏移 = 64*row + token/2。
        # **这就是不交换的回报**:相邻 token 落在同一字节的两个 nibble,而且都归
        # 同一个线程 —— 打包是纯线程内的。decode 交换后 token 摊在 N 上,同样的
        # 事要靠 π 槽位重排 + 手推偏移(坑17/坑31 那一大段)。
        p_byte_base = tidx * cutlass.const_expr(KV_TILE // 2)

        
        
        
        
        
        
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(
                tcgen05.copy.Repetition(PDynamicSf.ELTS_PER_THREAD)
            ),
            self.acc_dtype,
        )
        tiled_ld = tcgen05.make_tmem_copy(ld_atom, tCtS)
        thr_ld = tiled_ld.get_slice(tidx)
        tS_src = thr_ld.partition_S(tCtS)
        rS = cute.make_rmem_tensor(thr_ld.partition_D(tCtS).shape, self.acc_dtype)
        rS_flat = cute.make_tensor(rS.iterator, cute.make_layout(cute.size(rS)))

        # ---- TMEM 的 O 读写通路 + 清零 ----
        # 为什么必须显式清零:每个 tile 都要把已累加的 O 乘上 corr,而首个 tile
        # 的 corr = exp2(-inf - m) = **0**。若 O 是未初始化的 TMEM,0×nan = nan
        # 会毒掉整行。清零后"乘 0"正好表达"还没有累加值",与参考的
        # `o * scale` 在首轮的行为逐字一致。
        o_atom_l = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
        o_atom_s = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
        tiled_o = tcgen05.make_tmem_copy(o_atom_l, tCtO)
        thr_o = tiled_o.get_slice(tidx)
        tO_tmem = thr_o.partition_S(tCtO)
        rO = cute.make_rmem_tensor(thr_o.partition_D(tCtO).shape, self.acc_dtype)
        rO_flat = cute.make_tensor(rO.iterator, cute.make_layout(cute.size(rO)))
        rO.fill(Float32(0.0))
        cute.copy(o_atom_s, rO, tO_tmem)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.barrier()
        if dbg and tidx == 0:
            cute.printf("CK5 O zeroed in tmem\n")

        # ---- 逐 kv tile ----
        # S3:每个 tile 先把**本 tile** 的 K/V/SFK/SFV TMA 进单 stage 的 SMEM,
        # 再算。SMEM 复用的安全性由**顺序结构**保证:上一 tile 的 VP 在其
        # mma_bar 等待 + 循环尾 barrier 处完成,之后才会发下一 tile 的 TMA ——
        # 没有流水,故不需要多 stage(warp 特化版才做交错)。
        # 循环顶的 barrier 同时挡住"部分线程还在读上一轮 rS/rO"的越轮竞争。
        mma_phase = Int32(0)
        tma_phase = Int32(0)
        for t_i in cutlass.range(self.n_kv_tiles):
            cute.arch.barrier()
            # l 公式(见 __init__):稠密 = kv_head*n + 表[t](恒等表);
            # 池 = 表[t]*h_kv + kv_head(pool 物理页最外、头次外)。
            # 两条路径都过页表 —— 稠密的恒等表同时验证查表通路本身。
            if cutlass.const_expr(self.paged):
                l_idx = kv_pt[t_i] * self.h_kv + kv_head
            else:
                l_idx = kv_head * self.n_kv_tiles + kv_pt[t_i]
            if cute.arch.warp_idx() == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        kv_bar, cutlass.const_expr(self.kv_tma_bytes()))
                for _ka, _kg, _ks in kv_parts:
                    cute.copy(_ka, _kg[(None, l_idx)], _ks,
                              tma_bar_ptr=kv_bar)
            cute.arch.mbarrier_wait(kv_bar, tma_phase)
            tma_phase = tma_phase ^ 1
            if dbg and tidx == 0:
                
                
                
                
                
                _skb = cute.recast_tensor(sK, cutlass.Int8)
                _svb = cute.recast_tensor(sV, cutlass.Int8)
                _sfkb = cute.recast_tensor(sSFK, cutlass.Int8)
                _sfvb = cute.recast_tensor(sSFV, cutlass.Int8)
                cute.printf("CK5b t=%d sK0=%d sV0=%d sfk0=%d sfv0=%d\n",
                            t_i, Int32(_skb[0]), Int32(_svb[0]),
                            Int32(_sfkb[0]), Int32(_sfvb[0]))
            if cutlass.const_expr(self.stop_after >= 1):
                # SFQ/SFK 搬进 TMEM,再发 KQ。UTCCP 与 MMA 都是 **warp 级**指令,
                # 由 warp 0 发射(decode 同样把它们放在 mma warp 里)。128 个线程
                # 一起发会得到 4 份重复的 MMA。
                if cute.arch.warp_idx() == 0:
                    for name in (("sfq", "sfk", "sfq2")
                                 if self.residue_levels == 2
                                 else ("sfq", "sfk")):
                        cp, src, dst = s2t[name]
                        cute.copy(cp, src, dst)
                    cute.arch.fence_view_async_tmem_store()
                    kq.set(tcgen05.Field.ACCUMULATE, False)
                    self.issue_kq_gemm(kq, tCtS, sQ_desc, sK_desc, tSF["sfq"], tSF["sfk"])
                    if cutlass.const_expr(self.residue_levels == 2):
                        # S6:第二条 KQ 累加 residual —— issue_kq_gemm 首条
                        # 之后 ACCUMULATE 已置 True,这里**不再**清零,
                        # S = Q₁·Kᵀ + Q₂·Kᵀ 就靠这个累加语义(plans §6.1)。
                        self.issue_kq_gemm(kq, tCtS, sQ2_desc, sK_desc,
                                           tSF["sfq2"], tSF["sfk"])
                    with cute.arch.elect_one():
                        tcgen05.commit(mma_bar)
                if dbg and tidx == 0:
                    cute.printf("CK6 KQ gemm issued+committed\n")
                cute.arch.mbarrier_wait(mma_bar, mma_phase)
                mma_phase = mma_phase ^ 1
                if dbg and tidx == 0:
                    cute.printf("CK7 KQ mma barrier passed\n")

            if cutlass.const_expr(self.stop_after >= 2):
                # ---- softmax + 动态 SF:每线程一行(M=128 ↔ 128 线程)----
                cute.copy(tiled_ld, tS_src, rS)
                cute.arch.fence_view_async_tmem_load()

                # scale 用 log2 底(与参考的 exp2 口径一致)
                for i in cutlass.range_constexpr(cute.size(rS)):
                    rS_flat[i] = rS_flat[i] * scale_log2

                # ---- S2/S3:因果掩码(带 ctx 偏移)----
                # 参考口径 vis = col <= ctx_len + row(reference.py:221),
                # 而 harness 恒有 ctx = s_kv - s_q ⇒ 偏移是**编译期**常量
                # (= (n_kv_tiles-1)*KV_TILE,即 vendor 的 end-aligned 公式
                # offset = seqlen_k - seqlen_q,fmha_utils.py:1110)。
                # 线程 tidx = query 行、寄存器下标 i = 本 tile 内 kv 列
                # (S1 探针实证),全局列 = t_i*KV_TILE + i。掩码是纯线程内的
                # 逐元素动态 if,与 FusedMask.apply_mask 同惯用法。
                # 掩成 -inf 经 exp2(s-max) 自然变 0;全掩的 16 组 SF:amax=0
                # → sf=0、inv=1/1e-12、0*inv=0 —— 无 nan
                # (normalize_and_emit_sf:197 的 fmax 兜底),参考侧同为 0。
                # 整 tile 全掩的情形(不可能出现在 ctx=s_kv-s_q + s_q=128:
                # tile 0 恒全可见 ⇒ row_max 有限,后续全掩 tile 走 corr=1、
                # 累加 0)也安全 —— 但当前形状根本不会产生全掩 tile。
                if cutlass.const_expr(self.causal):
                    ctx_off = cutlass.const_expr(
                        (self.n_kv_tiles * self.kv_tile) - self.q_tile)
                    for i in cutlass.range_constexpr(cute.size(rS)):
                        if ctx_off + tidx < t_i * KV_TILE + i:
                            rS_flat[i] = Float32(-cutlass.Float32.inf)

                sf_frag = cute.make_rmem_tensor(
                    cute.make_layout(cute.size(rS) // SF_VEC), self.sf_dtype
                )
                row_max, row_sum, corr = self.online_softmax_step(
                    rS_flat, row_max, row_sum, sf_frag
                )

                # ---- P(已按 SF 归一)→ e2m1 → SMEM ----
                # 坑20:窄精度寄存器向量必须 ≥32 bit ⇒ 每 8 个值转一次(= 4 字节)。
                # π(S5):池里 V 的 16-token 组内是 PI16 预排(物理槽 s 放
                # 逻辑 token PI16[s],append.py:132 的 nib[:, pi] 同向)。
                # P 的物理槽必须同序才能与 V 对位收缩 ⇒ 打包时按 π 取值。
                # 稠密路径 host 侧同样按 π 打包 V(_quantize_v_pool_order)——
                # 两侧同 π,点积不变,kernel 恒用 π、零分支。SF 组的 amax 是
                # π 不变集(π 只在组内置换),normalize_and_emit_sf 不用改。
                for j in cutlass.range_constexpr(cute.size(rS) // 8):
                    _pr = cute.make_rmem_tensor(cute.make_layout(8), self.acc_dtype)
                    for u in cutlass.range_constexpr(8):
                        _slot = j * 8 + u
                        _pr[u] = rS_flat[(_slot // SF_VEC) * SF_VEC
                                         + PI16[_slot % SF_VEC]]
                    _p4 = cute.make_rmem_tensor(cute.make_layout(8), FP4_DTYPE)
                    _p4.store(_pr.load().to(FP4_DTYPE))
                    _pb = cute.recast_tensor(_p4, cutlass.Int8)
                    for b in cutlass.range_constexpr(4):
                        sP_flat[p_byte_base + j * 4 + b] = _pb[b]

                # ---- SFP → SMEM(UTCCP 的源)----
                # 地址不手推:布局 ((((32,4),1),(16,4)),1,2):((((16,4),0),(0,1)),0,512)
                # 按坐标索引即可,行号 tidx 交给 CuTe 按 colex 展开成 (m%32, m//32) ——
                # 与冻结的 kv_layout.sf_atom_offsets 公式
                # (mn%32)*16 + (mn//32)*4 + kg%4 + (kg//4)*512 逐项对上。
                for g in cutlass.range_constexpr(cute.size(sf_frag)):
                    sSFP[(((tidx, 0), (0, g % 4)), 0, g // 4)] = sf_frag[g]
                cute.arch.fence_view_async_shared()

            if cutlass.const_expr(self.stop_after >= 3):
                # ---- 旧累加值按 corr 重缩放(参考里的 `o = o * scale + p·v`)----
                cute.copy(tiled_o, tO_tmem, rO)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(cute.size(rO)):
                    rO_flat[i] = rO_flat[i] * corr
                cute.copy(o_atom_s, rO, tO_tmem)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.barrier()

                # SFP/SFV 进 TMEM,再发 VP。累加位恒为 True —— 清零 + corr 已经
                # 把"第一个 tile"表达掉了,不需要再判 t_i == 0(那是个**动态** bool,
                # `not first` 在 DSL 里也不成立)。
                if cute.arch.warp_idx() == 0:
                    for name in ("sfp", "sfv"):
                        cp, src, dst = s2t[name]
                        cute.copy(cp, src, dst)
                    cute.arch.fence_view_async_tmem_store()
                    self.issue_vp_gemm(
                        vp, tCtO, sP_desc, sV_desc, tSF["sfp"], tSF["sfv"], False)
                    with cute.arch.elect_one():
                        tcgen05.commit(mma_bar)
                if dbg and tidx == 0:
                    cute.printf("CK8 VP gemm issued+committed\n")
                cute.arch.mbarrier_wait(mma_bar, mma_phase)
                mma_phase = mma_phase ^ 1
                cute.arch.barrier()
                if dbg and tidx == 0:
                    cute.printf("CK9 VP mma barrier passed (tile done)\n")

        # ---- epilogue:O 从 TMEM 读回 → 除以 row_sum → 写 gmem ----
        # 归一化放在最后一次做(而不是每 tile 做),与参考的
        # `(o / l).permute(...)` 一致。分母是**未量化** probs 的和 —— 这条
        # 口径在 online_softmax_step 里已经守住,这里只做除法。
        cute.copy(tiled_o, tO_tmem, rO)
        cute.arch.fence_view_async_tmem_load()

        inv_l = 1.0 / cute.arch.fmax(row_sum, Float32(1e-30))
        for i in cutlass.range_constexpr(cute.size(rO)):
            rO_flat[i] = rO_flat[i] * inv_l

        # 线程 tidx ↔ query 行 tidx(与读 S 时同一条 datapath 约定)。
        rO_out = cute.make_rmem_tensor(
            cute.make_layout(cute.size(rO)), mO.element_type)
        rO_out.store(rO_flat.load().to(mO.element_type))
        for d in cutlass.range_constexpr(cute.size(rO)):
            mO[tidx, d, bidx] = rO_out[d]

        if dbg and tidx == 0:
            cute.printf("CK10 epilogue written, all done\n")

        cute.arch.barrier()
        if cute.arch.warp_idx() == 0:
            # relinquish 要在 dealloc 之前(decode:1874 同序)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tptr, cols)

    def make_tma(self, kq, vp, sQ_l, sK_l, sV_l, sf_l, mQ, mK, mV, mSFQ, mSFK, mSFV):
        """Q/K/V + 三个 SF 的 TMA atom。**A/B 两侧用的 op 不同。**

        - A 侧(Q、SFQ):普通 ``CopyBulkTensorTileG2SOp`` + ``_A``;
        - B 侧(K、V、SFK、SFV):数据用 ``_B``;**SF 还要换 op** ——
          ``cluster_shape_to_tma_atom_SFB`` 且 ``internal_type=Int16``。

        decode 里 K/V 在 A 侧、Q 在 B 侧,**我们全反过来**。照抄会两边都接反,
        而这类错**编译期不报**,只在数值上整片乱掉(见 kernel_paged_prefill
        的 OPERAND_SIDE)。
        """
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        sel = lambda l: cute.select(l, mode=[0, 1, 2])
        qk_tile = (self.q_tile, self.kv_tile, self.head_dim)
        vp_tile = (self.q_tile, self.head_dim, self.kv_tile)

        a_q = cute.nvgpu.make_tiled_tma_atom_A(op, mQ, sel(sQ_l), qk_tile, kq)
        a_k = cute.nvgpu.make_tiled_tma_atom_B(op, mK, sel(sK_l), qk_tile, kq)
        a_v = cute.nvgpu.make_tiled_tma_atom_B(op, mV, sel(sV_l), vp_tile, vp)
        # SFQ 在 A 侧 —— 普通 op
        a_sfq = cute.nvgpu.make_tiled_tma_atom_A(
            op, mSFQ, sel(sf_l["sfq"]), qk_tile, kq)
        # SFK/SFV 在 B 侧 —— SFB op + Int16
        sfb_kq = sm100_utils.cluster_shape_to_tma_atom_SFB((1, 1), kq.thr_id)
        sfb_vp = sm100_utils.cluster_shape_to_tma_atom_SFB((1, 1), vp.thr_id)
        a_sfk = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_kq, mSFK, sel(sf_l["sfk"]), qk_tile, kq, internal_type=cutlass.Int16)
        a_sfv = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_vp, mSFV, sel(sf_l["sfv"]), vp_tile, vp, internal_type=cutlass.Int16)
        # 每项是 `(atom, tma_tensor)` —— `make_tiled_tma_atom_*` 的返回形式。
        return {"q": a_q, "k": a_k, "v": a_v,
                "sfq": a_sfq, "sfk": a_sfk, "sfv": a_sfv}
