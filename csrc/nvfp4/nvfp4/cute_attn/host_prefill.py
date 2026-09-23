# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'DEPRECATED FOR PRODUCTION: historical in-house prefill host.'

from __future__ import annotations

import functools
import math
import os
from typing import Optional, Sequence

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from .append import (_nibbles_decode, _pack_low_high, _quant_groups,
                     _unpack_low_high, bulk_quantize_paged)
from .kernel_paged_prefill import FP4_DTYPE
from .kernel_prefill_simple import PrefillSimpleFp4
from .kv_layout import (PAGE_SIZE, PI16, SF_VEC, NvFp4PagedKVPool,
                        sf_atom_gather, sf_atom_scatter)

#: kv tile 的 token 数(= page_size;kernel 侧的 KV_TILE 同值)
KV_TILE_TOKENS = PAGE_SIZE

#: torch 侧的 fp4 打包 dtype(一字节两元素)
FP4_TORCH = torch.float4_e2m1fn_x2


@functools.lru_cache(maxsize=8)
def _single_request_indptr(device_str: str) -> torch.Tensor:
    """Device-resident ``[0]`` reused by every single-request launch."""
    return torch.zeros(1, dtype=torch.int32, device=device_str)


# ---------------------------------------------------------------------------
# 量化:把参考口径的"反量化值"打回 fp4 + atom 序 SF
# ---------------------------------------------------------------------------


def _quantize_k_major(x: torch.Tensor):
    '``(rows, K)`` → ``(packed (rows, K/2) uint8, sf_atom (rows*K/16) uint8)``。'
    rows, k = x.shape
    assert k % SF_VEC == 0, (rows, k)
    nib, sf = _quant_groups(x.float().reshape(rows, k // SF_VEC, SF_VEC))
    packed = _pack_low_high(nib.reshape(rows, k))
    return packed, sf_atom_scatter(sf.reshape(rows, k // SF_VEC))


def _quantize_context_q_cuda(
    q_source: torch.Tensor,
    query_indptr: torch.Tensor,
    s_q_kernel: int,
):
    """Fused ragged-Q padding + NVFP4 quantization, or ``None``.

    ``query_indptr`` indexes the original contiguous ``q_source [T,H,D]`` and
    may contain non-zero global offsets.  The CUDA op writes the common
    left-padded ``[B,H,S_pad,D/2]`` pack and atom-order SF directly, avoiding
    both a padded bf16 allocation and the old B*H torch quantization launches.
    """
    if not (
        q_source.is_cuda
        and q_source.dtype is torch.bfloat16
        and q_source.ndim == 3
        and q_source.is_contiguous()
        and query_indptr.is_cuda
        and query_indptr.dtype is torch.int32
        and query_indptr.ndim == 1
        and query_indptr.is_contiguous()
        and query_indptr.device == q_source.device
        and hasattr(torch.ops, "nvfp4")
        and hasattr(torch.ops.nvfp4, "nvfp4_quantize_context_q")
    ):
        return None
    batch = query_indptr.numel() - 1
    _, h_q, d = q_source.shape
    q_pack = torch.empty(
        batch, h_q, s_q_kernel, d // 2,
        dtype=torch.uint8,
        device=q_source.device,
    )
    q_sf = torch.empty(
        batch, h_q, s_q_kernel * (d // SF_VEC),
        dtype=torch.uint8,
        device=q_source.device,
    )
    torch.ops.nvfp4.nvfp4_quantize_context_q(
        q_source, query_indptr, q_pack, q_sf
    )
    return q_pack, q_sf.view(torch.float8_e4m3fn)


def _dequantize_k_major(packed: torch.Tensor, sf_atom: torch.Tensor, k: int):
    """``_quantize_k_major`` 的逆 —— 只给幂等性自检用。"""
    rows = packed.shape[0]
    nib = _unpack_low_high(packed)[:, :k]
    sf = sf_atom_gather(sf_atom, rows, k // SF_VEC).view(torch.float8_e4m3fn).float()
    return (_nibbles_decode(nib).reshape(rows, k // SF_VEC, SF_VEC)
            * sf.unsqueeze(-1)).reshape(rows, k)


def _quantize_v_pool_order(x: torch.Tensor):
    """V 专用:量化后 16-token 组内按 π(PI16)重排 nibble 再打包。

    与池的写侧(append.py:132 ``nib[:, pi]``)同向:物理槽 s 放逻辑
    token PI16[s]。kernel 的 P 打包恒用 π ⇒ 稠密路径的 V 也必须 π,
    两侧同序点积才对位。SF 不动(π 只在组内置换,amax 集不变)。
    """
    rows, k = x.shape
    assert k % SF_VEC == 0, (rows, k)
    nib, sf = _quant_groups(x.float().reshape(rows, k // SF_VEC, SF_VEC))
    pi = torch.tensor(PI16, device=x.device)
    nib = nib[:, :, pi]                      # 组内 π 重排(与池写侧同一句)
    packed = _pack_low_high(nib.reshape(rows, k))
    return packed, sf_atom_scatter(sf.reshape(rows, k // SF_VEC))


def _assert_on_fp4_grid(x: torch.Tensor, name: str):
    """``x`` 必须**已经在 fp4 网格上**(即 quantize_dequantize 的输出)。

    为什么卡这条:契约说 ``k_deq`` 是"kernel 真正读到的值",参考直接拿它算,
    kernel 却要把它重新量化一遍。只有当重量化**幂等**,两边算的才是同一个
    张量。不幂等时 S1 会挂,但挂的样子像是 kernel 错 —— 那是最贵的一类误导。
    宁可在入口处当场报出来。
    """
    packed, sf = _quantize_k_major(x)
    back = _dequantize_k_major(packed, sf, x.shape[1])
    if not torch.equal(back, x.float()):
        bad = (back - x.float()).abs().max().item()
        raise ValueError(
            f"{name} 不在 fp4 网格上(重量化后最大偏差 {bad:.3e})。"
            "请传 reference.quantize_dequantize 的输出 —— 否则参考与 kernel "
            "算的不是同一个张量,S1 的差值无法归因。"
        )
    return packed, sf


# ---------------------------------------------------------------------------
# 张量视图:fp4 的 L 模步长必须是偶数
# ---------------------------------------------------------------------------


def _fp4_mnl(packs: list) -> torch.Tensor:
    """每头一个 ``(rows, K/2)`` uint8 → CuTe 眼里的 ``(rows, K, H)`` e2m1。

    head/L 模**不能**拿默认的 stride 1:那样张量就有两个 stride-1 维,
    tvm-ffi 的 f4x2→f4 转换会拒绝 ``Dimension 2 with stride != 1 must be
    even``(一字节两元素 ⇒ 非最内维的步长必须能整除 2)。物理上块最外
    (H, rows, half 连续),as_strided 转 (rows, half, H) 视图 —— 与
    ``_fake_args`` 的 stride=(b, 1, a*b) 逐项对应。Q/O 用(K/V 走 L-flat
    的 ``_fp4_kv_dense`` / 池视图)。
    """
    t = torch.stack([p.contiguous() for p in packs], dim=0).contiguous()
    h, rows, half = t.shape
    tt = t.view(FP4_TORCH)
    return torch.as_strided(tt, (rows, half, h), (half, 1, rows * half))


def _fp4_kv_dense(packs: list, kind: str) -> torch.Tensor:
    """稠密 K/V → L-flat 视图(L = h*n + t,头外、tile 内),与池视图同构。

    - ``kind='k'``:每头 ``(s_kv, D/2)``,tile 块 = 连续 128 行 ⇒ 纯视图。
      返回 (128, D, L) e2m1。
    - ``kind='v'``:每头 ``(D, s_kv/2)``,tile 块沿 token 轴**不连续** ⇒
      先重排成块连续(小拷贝,正确性优先)。返回 (D, 128, L) e2m1。

    最终物理形态都是"L 个连续块",L 模单一 stride —— 与 ``_fake_args`` 的
    stride=(b, 1, a*b) 及池的 [P,H,…] 页最外布局同构(kernel 单代码路径)。
    """
    ht = KV_TILE_TOKENS // 2                        # 每 tile 的 token 字节数
    if kind == "k":
        t = torch.stack([p.contiguous() for p in packs], 0)   # (H, s_kv, D/2)
        h, s, half = t.shape
        blk = t.view(h * (s // KV_TILE_TOKENS), KV_TILE_TOKENS, half).contiguous()
    else:
        per = []
        for p in packs:                                        # (D, s_kv/2)
            d, halftok = p.shape
            per.append(p.view(d, halftok // ht, ht).permute(1, 0, 2))
        blk = torch.stack(per, 0).reshape(-1, per[0].shape[1], ht).contiguous()
    L, r, c = blk.shape
    tt = blk.view(FP4_TORCH)
    return torch.as_strided(tt, (r, c, L), (c, 1, r * c))


def _sf_flat(sf_atoms: list) -> torch.Tensor:
    """每头的 atom 序 SF 依次 concat —— tile_atom_to_shape_SF 的 L(head)模
    在 atom 序里最外,逐头块状拼接正好对上。"""
    return torch.cat([s.reshape(-1) for s in sf_atoms]).contiguous().view(
        torch.float8_e4m3fn)


def _fp4_mnl_batch1(packs: list) -> torch.Tensor:
    """``_fp4_mnl`` plus an explicit batch mode for the context kernel."""
    t = _fp4_mnl(packs)
    rows, half, heads = t.shape
    return torch.as_strided(
        t,
        (rows, half, heads, 1),
        (*t.stride(), rows * half * heads),
    )


def _sfq_batch1(sf_atoms: list, heads: int) -> torch.Tensor:
    """Q SF as ``(batch=1, head, atom_bytes_per_head)``."""
    return _sf_flat(sf_atoms).view(1, heads, -1)


def _fake_args(kern):
    """编译期假张量。**与 ``_fp4_mnl`` 造出来的真张量必须同规格** ——
    两边分开写就一定会漂,所以放在同一个文件里挨着,并由离线门一起编。
    K/V 覆盖**全部** ``s_kv = n_kv_tiles * kv_tile``(S3:逐 tile 的 box
    在 kernel 里由 local_tile 的 rest 模选)。"""
    d = kern.head_dim
    l_kv = kern.n_kv_tiles * kern.h_kv
    mkt4 = lambda a, b, h: cute.runtime.make_fake_tensor(
        FP4_DTYPE, (a, b, h), stride=(b, 1, a * b), assumed_align=16)
    flat = lambda: cute.runtime.make_fake_compact_tensor(
        kern.sf_dtype, (cute.sym_int(),), stride_order=(0,), assumed_align=16)
    return (
        mkt4(kern.q_tile, d, kern.h_q * kern.residue_levels),
        # K/V 是 L-flat(S5):每块一个 (tile, D) box,L = 页×头。
        mkt4(kern.kv_tile, d, l_kv), mkt4(d, kern.kv_tile, l_kv),
        flat(), flat(), flat(),
        # O 同样 head 最外:(S, D, H) 视图,物理 (H, S, D) 连续。bf16 没有
        # fp4 的偶步长约束,但显式 stride 免得依赖 stride_order 的推断。
        cute.runtime.make_fake_tensor(
            cutlass.BFloat16, (kern.q_tile, d, kern.h_q),
            stride=(d, 1, kern.q_tile * d), assumed_align=16),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (kern.n_kv_tiles,), stride_order=(0,),
            assumed_align=4),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )


#: 打开后 kernel 在每个同步点打一行 CK<N>。三个 mbarrier 等待点里任何一处
#: 挂住,从外面都只看到"零输出 + 跑到 timeout",没有打点就只能靠猜 + 反复上机。
#: 环境变量控制,免得为了诊断改代码(改代码会让缓存的镜像层失效,又多等一分钟)。
RES_PREFILL_DEBUG = os.environ.get("RES_PREFILL_DEBUG", "0") == "1"


#: 分阶段早退(0..4),用来并发二分定位死锁。见 PrefillSimpleFp4.__init__ 的说明。
#: 生产路径是 4;设成别的值会**算错**,只用于诊断。
RES_PREFILL_STOP_AFTER = int(os.environ.get("RES_PREFILL_STOP_AFTER", "4"))


@functools.lru_cache(maxsize=8)
def _compiled(head_dim: int, causal: bool = False, n_tiles: int = 1,
              h_q: int = 1, h_kv: int = 1, paged: bool = False,
              residue_levels: int = 1,
              debug: bool = RES_PREFILL_DEBUG,
              stop_after: int = RES_PREFILL_STOP_AFTER):
    # causal / n_tiles / 头数 / paged 全是**编译期**参数(gmem 形状、
    # expected_tx、掩码 ctx 偏移、grid、l 公式由它们导出),都进缓存键;
    # 默认值保持 S1/探针的既有调用形式(`_compiled(D)`)语义不变。
    # 坑38:绝不再走运行期参数。
    kern = PrefillSimpleFp4(head_dim=head_dim, causal=causal,
                            n_kv_tiles=n_tiles, h_q=h_q, h_kv=h_kv,
                            paged=paged, residue_levels=residue_levels,
                            debug_trace=debug, stop_after=stop_after)
    return kern, cute.compile(
        kern, *_fake_args(kern), options="--enable-tvm-ffi --opt-level 3"
    )


_UNSUPPORTED = (
    "简化版 prefill kernel 当前只支持:单 batch、多头/GQA(h_q%h_kv==0)、"
    "s_q=128、s_kv=n*128、head_dim=128、p_scale='dynamic'、"
    "residue_levels∈{{1,2}}、分页/稠密均可(causal 任意,"
    "ctx 隐含 = s_kv - s_q)。收到:{got}\n"
    "(短序列要把垫零列掩成 -inf —— 唯一未接的轴。)"
)


def prepare_paged_fp4_prefill(
    q,
    k_deq,
    v_deq,
    sm_scale: float,
    *,
    kv_tile: Optional[int] = 128,
    causal: bool = True,
    p_scale: str = "dynamic",
    residue_levels: int = 1,
    ctx_len: Optional[int] = None,
    paged: bool = False,
    pool=None,
    kv_indices=None,
    kv_indptr=None,
    seq_lens=None,
):
    """host 准备(量化/入池/建视图)与 kernel 调用**分离**的入口。

    返回 ``(compiled, call_args, out)``:``compiled(*call_args)`` 即一次
    kernel 发射,``out`` 是 (h_q, S_q, D) 的输出缓冲。测速用它把 host 侧
    准备(量化、幂等自检、池构建 —— 生产里都是预先/融合做的)排除在
    计时区外;``run_paged_fp4_prefill`` 是"准备 + 调一次"的便捷包装。
    """
    s_q, h_q, d = q.shape
    s_kv, h_kv = k_deq.shape[0], k_deq.shape[1]
    got = dict(s_q=s_q, h_q=h_q, d=d, s_kv=s_kv, h_kv=h_kv, causal=causal,
               p_scale=p_scale, residue_levels=residue_levels, paged=paged,
               kv_tile=kv_tile, ctx_len=ctx_len)
    if not (
        h_q % h_kv == 0
        and d == 128
        and s_q == 128
        and s_kv % 128 == 0
        and s_kv >= 128
        and kv_tile == 128
        and p_scale == "dynamic"
        and residue_levels in (1, 2)
        # kernel 的掩码偏移固定为 s_kv - s_q(end-aligned,harness 的
        # make_data 恒如此);显式传了别的 ctx_len 就拒绝,免得静默算错。
        and ctx_len in (None, 0, s_kv - s_q)
    ):
        raise NotImplementedError(_UNSUPPORTED.format(got=got))

    n = s_kv // 128
    kern, compiled = _compiled(d, causal, n, h_q, h_kv, bool(paged),
                               residue_levels)

    # Q:逐头逐级量化(S6:q₁=Q(x)、q₂=Q(x-q₁),与
    # reference.quantize_dequantize_residue 同链;lev 外、head 内排 L)。
    per_head = []
    for h in range(h_q):
        r = q[:, h, :].float()
        levels = []
        for _ in range(residue_levels):
            pk, sf = _quantize_k_major(r)
            levels.append((pk, sf))
            r = r - _dequantize_k_major(pk, sf, d)
        per_head.append(levels)
    q_pk = [per_head[h][lev][0]
            for lev in range(residue_levels) for h in range(h_q)]
    q_sf = [per_head[h][lev][1]
            for lev in range(residue_levels) for h in range(h_q)]
    # K/V 先逐头过 fp4 网格幂等自检(两条路径都查 —— 池路径的
    # bulk_quantize_paged 也要求输入已在网格上才与参考同值)。
    k_pk, k_sf = zip(*(_assert_on_fp4_grid(k_deq[:, h, :].float(), f"k_deq[{h}]")
                       for h in range(h_kv)))
    for h in range(h_kv):
        _assert_on_fp4_grid(v_deq[:, h, :].float().t().contiguous(),
                            f"v_deq[{h}]")

    if paged:
        # ---- S5:量化入池(乱序页表)+ 池的 L-flat 视图 ----
        # 页数 = tile 数(页表是置换 ⇒ 物理乱序、逻辑齐全);页表确定性
        # shuffle,复跑可对拍。V 的 π 由池写侧(append)完成。
        pool = NvFp4PagedKVPool.allocate(n, h_kv, d, 1, device=q.device)
        gen = torch.Generator(device="cpu").manual_seed(7)
        pages = torch.randperm(n, generator=gen).to(torch.int32).to(q.device)
        kv_indptr_t = torch.tensor([0, n], dtype=torch.int32, device=q.device)
        bulk_quantize_paged(
            k_deq.unsqueeze(0).float(), v_deq.unsqueeze(0).float(),
            torch.tensor([s_kv], device=q.device), pool, pages, kv_indptr_t)
        P = pool.k_data.shape[0]
        mk = torch.as_strided(pool.k_data.view(FP4_TORCH),
                              (128, d // 2, P * h_kv), (d // 2, 1, 64 * d))
        mv = torch.as_strided(pool.v_data.view(FP4_TORCH),
                              (d, 64, P * h_kv), (64, 1, 64 * d))
        sfk_t = pool.k_sf.reshape(-1).view(torch.float8_e4m3fn)
        sfv_t = pool.v_sf.reshape(-1).view(torch.float8_e4m3fn)
        kv_pt = pages
    else:
        # ---- 稠密:K 纯视图切块;V 走 π 打包(与池写侧同序)----
        v_pk, v_sf = zip(*(_quantize_v_pool_order(
            v_deq[:, h, :].float().t().contiguous()) for h in range(h_kv)))
        mk = _fp4_kv_dense(list(k_pk), "k")
        mv = _fp4_kv_dense(list(v_pk), "v")
        sfk_t = _sf_flat(list(k_sf))
        sfv_t = _sf_flat(list(v_sf))
        kv_pt = torch.arange(n, dtype=torch.int32, device=q.device)

    # 物理 (H, S, D) 连续、逻辑 (S, D, H) 视图 —— 与 _fake_args 逐项对应。
    out = torch.zeros(h_q, kern.q_tile, d, device=q.device,
                      dtype=torch.bfloat16)
    call_args = (
        _fp4_mnl(list(q_pk)), mk, mv,
        _sf_flat(list(q_sf)), sfk_t, sfv_t,
        out.as_strided((kern.q_tile, d, h_q), (d, 1, kern.q_tile * d)),
        kv_pt,
        # tile 数与头数**不走运行期参数**(编译期常量,见 kernel __call__ 的
        # 说明;S3 首跑栽在"编译 n=3、运行期还传 1"—— 单一来源杜绝再犯)。
        Float32(sm_scale),
        
        
        
        
    )
    return compiled, call_args, out


def run_paged_fp4_prefill(q, k_deq, v_deq, sm_scale: float, **kw):
    """契约与 ``reference.prefill_reference`` 一致;返回 ``[S_q, H_q, D]``。

    ``q`` 是**未量化**的(与参考一样在本函数内量化);``k_deq``/``v_deq`` 是
    池中反量化的值,必须已在 fp4 网格上(见 ``_assert_on_fp4_grid``)。
    """
    compiled, call_args, out = prepare_paged_fp4_prefill(
        q, k_deq, v_deq, sm_scale, **kw)
    compiled(*call_args)
    return out.permute(1, 0, 2).float()                   # [S_q, H_q, D]


# ---------------------------------------------------------------------------
# warp 特化性能版(kernel_paged_prefill.ResAttnPagedPrefillFp4)的入口
# ---------------------------------------------------------------------------

def _fake_args_perf(kern, s_q: int, pages: int):
    """性能版的编译期假张量。**与 prepare_paged_fp4_prefill_perf 造出来的
    真张量必须同规格**(两边挨着放,离线门一起编)。"""
    d = kern.head_dim
    # Batch is runtime-symbolic: grouped context traffic has highly variable
    # batch size, and compiling one cubin per exact B would thrash the cache.
    batches = cute.sym_int()
    mkt4 = lambda a, b, h: cute.runtime.make_fake_tensor(
        FP4_DTYPE,
        (a, b, h, batches),
        stride=(b, 1, a * b, a * b * h),
        assumed_align=16,
    )
    # 生产 vLLM 池是 [data | scale]-per-side:同页 head stride 紧密,
    # 但 page stride 还包含另一侧数据与 SF。因此 K/V 的 head/page
    # stride 必须用符号步长编译。divisibility=2 是 fp4 打包类型的
    # 硬契约:任意字节 stride 换算成 fp4 元素都是偶数。
    sd2 = lambda: cute.sym_int(divisibility=2)
    pool_t4 = lambda a, b: cute.runtime.make_fake_tensor(
        FP4_DTYPE,
        (a, b, kern.h_kv, pages),
        stride=(sd2(), 1, sd2(), sd2()),
        assumed_align=16,
    )
    sd16 = lambda: cute.sym_int(divisibility=16)
    pool_sf = lambda: cute.runtime.make_fake_tensor(
        kern.sf_dtype,
        (pages, kern.h_kv, PAGE_SIZE * d // SF_VEC),
        stride=(sd16(), sd16(), 1),
        assumed_align=16,
    )
    i32 = lambda n_: cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (n_,), stride_order=(0,), assumed_align=4)
    return (
        mkt4(s_q, d, kern.h_q),
        pool_t4(PAGE_SIZE, d), pool_t4(d, PAGE_SIZE),
        cute.runtime.make_fake_tensor(
            cutlass.BFloat16, (s_q, d, kern.h_q, batches),
            stride=(d, 1, s_q * d, s_q * d * kern.h_q),
            assumed_align=16,
        ),
        cute.runtime.make_fake_compact_tensor(
            kern.sf_dtype,
            (batches, kern.h_q, s_q * d // SF_VEC),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        pool_sf(), pool_sf(),
        i32(cute.sym_int()), i32(batches), i32(batches),
        Float32(1.0), Float32(1.0), Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )


@functools.lru_cache(maxsize=8)
def _compiled_perf(head_dim: int, causal: bool, s_q: int, pages: int,
                   h_q: int = 1, h_kv: int = 1):
    """形状/头数/因果全是编译期参数(坑38 口径,与 simple 的 _compiled 同)。"""
    from .kernel_paged_prefill import ResAttnPagedPrefillFp4
    kern = ResAttnPagedPrefillFp4(head_dim=head_dim, h_q=h_q, h_kv=h_kv,
                                  causal=causal)
    return kern, cute.compile(
        kern, *_fake_args_perf(kern, s_q, pages),
        options="--enable-tvm-ffi --opt-level 3",
    )


def prepare_paged_fp4_prefill_perf(
    q,
    k_deq,
    v_deq,
    sm_scale: float,
    *,
    kv_tile: Optional[int] = 64,
    causal: bool = True,
    p_scale: str = "dynamic",
    residue_levels: int = 1,
    ctx_len: Optional[int] = None,
    paged: bool = False,
):
    """性能版 host 准备。契约与 ``prepare_paged_fp4_prefill`` 一致,差异:

    - ``kv_tile=64``(Design E 的 chunk 宽;参考侧必须同值 —— 在线 softmax
      的重结合粒度跟着它走);
    - **一律走池**(稠密门 = 恒等页表;kernel 单代码路径,S5 教训);
    - ``s_q ∈ {128, 256 的倍数}``(CTA 铺 2×128 行;s_q=128 时第二个
      q tile 由 residual 掩码 + TMA OOB-drop 兜住,vendor 同款);
    - K/V/SF 保留显式 ``(head,page)`` mode;同一个编译物既能吃
      benchmark 的紧密池,也能吃 vLLM ``[data | scale]`` 的跨页大 stride。
    """
    import math as _math

    s_q, h_q, d = q.shape
    s_kv, h_kv = k_deq.shape[0], k_deq.shape[1]
    got = dict(s_q=s_q, h_q=h_q, d=d, s_kv=s_kv, h_kv=h_kv, causal=causal,
               p_scale=p_scale, residue_levels=residue_levels, paged=paged,
               kv_tile=kv_tile, ctx_len=ctx_len)
    if not (
        h_q % h_kv == 0
        and d == 128
        and s_q % 128 == 0
        and s_kv % PAGE_SIZE == 0
        and s_kv >= PAGE_SIZE
        and kv_tile == 64
        and p_scale == "dynamic"
        and residue_levels == 1          # 性能版 residue 后续接
        and ctx_len in (None, 0, s_kv - s_q)
    ):
        raise NotImplementedError(_UNSUPPORTED.format(got=got))

    n = s_kv // PAGE_SIZE
    kern, compiled = _compiled_perf(d, causal, s_q, n, h_q, h_kv)

    # Q 逐头量化(levels=1)
    q_pk, q_sf = zip(*(_quantize_k_major(q[:, h, :].float())
                       for h in range(h_q)))
    for h in range(h_kv):
        _assert_on_fp4_grid(k_deq[:, h, :].float(), f"k_deq[{h}]")
        _assert_on_fp4_grid(v_deq[:, h, :].float().t().contiguous(),
                            f"v_deq[{h}]")

    # 一律入池;dense 门 = 恒等页表(乱序表只在 paged=True)
    pool = NvFp4PagedKVPool.allocate(n, h_kv, d, 1, device=q.device)
    if paged:
        gen = torch.Generator(device="cpu").manual_seed(7)
        pages = torch.randperm(n, generator=gen).to(torch.int32).to(q.device)
    else:
        pages = torch.arange(n, dtype=torch.int32, device=q.device)
    kv_indptr_t = torch.tensor([0, n], dtype=torch.int32, device=q.device)
    bulk_quantize_paged(
        k_deq.unsqueeze(0).float(), v_deq.unsqueeze(0).float(),
        torch.tensor([s_kv], device=q.device), pool, pages, kv_indptr_t)
    mk, mv, sfk_t, sfv_t = _perf_pool_views(pool)

    out = torch.zeros(h_q, s_q, d, device=q.device, dtype=torch.bfloat16)
    kv_len_t = torch.tensor([s_kv], dtype=torch.int32, device=q.device)
    kv_indptr0 = _single_request_indptr(str(q.device))
    call_args = (
        _fp4_mnl_batch1(list(q_pk)), mk, mv,
        out.as_strided(
            (s_q, d, h_q, 1), (d, 1, s_q * d, s_q * d * h_q)
        ),
        _sfq_batch1(list(q_sf), h_q), sfk_t, sfv_t,
        pages, kv_indptr0, kv_len_t,
        Float32(sm_scale * _math.log2(_math.e)),   # scale_softmax_log2
        Float32(sm_scale),                          # scale_softmax(LSE 口径)
        Float32(1.0),                               # scale_output
        # 不传 stream(tvm-ffi 环境流,同 simple 的说明)
    )
    return compiled, call_args, out


def _perf_pool_views(pool: NvFp4PagedKVPool):
    """Build zero-copy CuTe argument views over an MMA-native KV pool.

    The explicit head/page modes are intentional.  Flattening them is only
    valid for ``NvFp4PagedKVPool.allocate``'s compact test pool; vLLM's real
    byte pool has a larger page stride because K data, K SF, V data and V SF
    share one physical page allocation.
    """
    k = pool.k_data.view(FP4_TORCH).permute(2, 3, 1, 0)
    v = pool.v_data.view(FP4_TORCH).permute(2, 3, 1, 0)
    sfk = pool.k_sf.view(torch.float8_e4m3fn)
    sfv = pool.v_sf.view(torch.float8_e4m3fn)
    return k, v, sfk, sfv


def prepare_paged_fp4_prefill_pool(
    q: torch.Tensor,
    pool: NvFp4PagedKVPool,
    kv_indices: torch.Tensor,
    kv_len: int,
    sm_scale: float,
    *,
    causal: bool = True,
    kv_len_tensor: Optional[torch.Tensor] = None,
):
    """Prepare one context-attention launch over an existing paged FP4 pool.

    Unlike ``prepare_paged_fp4_prefill_perf`` this is a production adapter:
    K/V are neither gathered nor requantized.  ``kv_indices`` is the logical
    page table for this one request, while page ids index the existing global
    pool directly.

    ``D=128`` and single-request causal attention remain fixed.  KV tails are
    handled by the kernel's ``kv_len`` mask.  A query tail is *left* padded to
    a 128-row boundary: for a real row ``i`` at padded row ``pad+i``,
    ``kv_len-padded_q+pad+i == kv_len-real_q+i``, so end-aligned causal
    positions are unchanged.  Right padding would be incorrect.
    """
    if q.ndim != 3:
        raise ValueError(f"q must be [S_q,H_q,D], got {tuple(q.shape)}")
    s_q, h_q, d = q.shape
    h_kv = pool.num_kv_heads
    got = dict(
        s_q=s_q,
        h_q=h_q,
        d=d,
        h_kv=h_kv,
        kv_len=kv_len,
        causal=causal,
        page_size=pool.page_size,
    )
    if not (
        h_q % h_kv == 0
        and d == pool.head_dim == 128
        and pool.page_size == PAGE_SIZE
        and s_q > 0
        and kv_len >= s_q
        and causal
    ):
        raise NotImplementedError(
            "native FP4 context kernel v1 requires single-request causal "
            "D=128, positive S_q, and kv_len>=S_q; "
            f"got {got}"
        )
    logical_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    if kv_indices.ndim != 1 or kv_indices.numel() < logical_pages:
        raise ValueError(
            f"kv_indices needs at least {logical_pages} entries, got "
            f"shape={tuple(kv_indices.shape)}"
        )

    # Left padding preserves the end-aligned causal coordinate of every real
    # query row.  The output view below drops the dummy prefix again.
    s_q_kernel = ((s_q + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    q_pad = s_q_kernel - s_q
    if q_pad:
        q_kernel = torch.zeros(
            s_q_kernel, h_q, d, dtype=q.dtype, device=q.device
        )
        q_kernel[q_pad:] = q
    else:
        q_kernel = q

    physical_pages = pool.k_data.shape[0]
    _kern, compiled = _compiled_perf(
        d, causal, s_q_kernel, physical_pages, h_q, h_kv
    )
    q_pk, q_sf = zip(
        *(_quantize_k_major(q_kernel[:, h, :].float()) for h in range(h_q))
    )
    mk, mv, sfk_t, sfv_t = _perf_pool_views(pool)
    out = torch.zeros(h_q, s_q_kernel, d, device=q.device, dtype=torch.bfloat16)
    if kv_len_tensor is None:
        kv_len_t = torch.tensor([kv_len], dtype=torch.int32, device=q.device)
    else:
        if (
            kv_len_tensor.numel() != 1
            or kv_len_tensor.dtype != torch.int32
            or kv_len_tensor.device != q.device
        ):
            raise ValueError(
                "kv_len_tensor must be one int32 value on q.device, got "
                f"shape={tuple(kv_len_tensor.shape)}, "
                f"dtype={kv_len_tensor.dtype}, device={kv_len_tensor.device}"
            )
        # The integer ``kv_len`` is the already-synchronized routing value;
        # this tensor is its device-side alias. Avoid a per-layer allocation
        # and do not .item() it merely to recheck the caller.
        kv_len_t = kv_len_tensor.reshape(1).contiguous()
    kv_indptr0 = _single_request_indptr(str(q.device))
    call_args = (
        _fp4_mnl_batch1(list(q_pk)),
        mk,
        mv,
        out.as_strided(
            (s_q_kernel, d, h_q, 1),
            (d, 1, s_q_kernel * d, s_q_kernel * d * h_q),
        ),
        _sfq_batch1(list(q_sf), h_q),
        sfk_t,
        sfv_t,
        kv_indices[:logical_pages].to(dtype=torch.int32).contiguous(),
        kv_indptr0,
        kv_len_t,
        Float32(sm_scale * math.log2(math.e)),
        Float32(sm_scale),
        Float32(1.0),
    )
    return compiled, call_args, out[:, q_pad:]


def run_paged_fp4_prefill_pool(
    q: torch.Tensor,
    pool: NvFp4PagedKVPool,
    kv_indices: torch.Tensor,
    kv_len: int,
    sm_scale: float,
    *,
    causal: bool = True,
    out: Optional[torch.Tensor] = None,
    kv_len_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run native FP4 context attention and return ``[S_q,H_q,D]`` bf16."""
    compiled, call_args, kernel_out = prepare_paged_fp4_prefill_pool(
        q,
        pool,
        kv_indices,
        kv_len,
        sm_scale,
        causal=causal,
        kv_len_tensor=kv_len_tensor,
    )
    compiled(*call_args)
    result = kernel_out.permute(1, 0, 2)
    if out is None:
        return result
    if out.shape != result.shape:
        raise ValueError(f"out shape {tuple(out.shape)} != {tuple(result.shape)}")
    out.copy_(result.to(out.dtype))
    return out


def prepare_paged_fp4_prefill_pool_batch(
    queries: Sequence[torch.Tensor],
    pool: NvFp4PagedKVPool,
    kv_indices: torch.Tensor,
    kv_offsets: torch.Tensor,
    kv_lens: Sequence[int],
    sm_scale: float,
    *,
    causal: bool = True,
    kv_lens_tensor: Optional[torch.Tensor] = None,
    query_source: Optional[torch.Tensor] = None,
    query_indptr: Optional[torch.Tensor] = None,
):
    """Prepare one grouped launch for several context requests.

    Queries are left-padded to the largest 128-row bucket in the group.  The
    page table stays global: ``kv_offsets[b]`` points at request ``b``'s first
    entry in ``kv_indices``.  This matches vLLM metadata slices and avoids
    concatenating/copying page tables per layer.  Production callers can also
    pass the original contiguous ``query_source`` plus global
    ``query_indptr``; the fused CUDA op then performs padding and quantization
    without materializing padded bf16 queries.
    """
    if not queries or len(queries) != len(kv_lens):
        raise ValueError(
            f"queries/kv_lens must have the same positive length, got "
            f"{len(queries)} and {len(kv_lens)}"
        )
    batch = len(queries)
    q0 = queries[0]
    if q0.ndim != 3:
        raise ValueError(f"query must be [S,H,D], got {tuple(q0.shape)}")
    h_q, d = q0.shape[1:]
    h_kv = pool.num_kv_heads
    q_lens = [q.shape[0] for q in queries]
    if not (
        causal
        and h_q % h_kv == 0
        and d == pool.head_dim == 128
        and pool.page_size == PAGE_SIZE
        and all(
            q.ndim == 3
            and q.shape[1:] == (h_q, d)
            and q.device == q0.device
            and q.dtype == q0.dtype
            and q_len > 0
            and kv_len >= q_len
            for q, q_len, kv_len in zip(queries, q_lens, kv_lens)
        )
    ):
        raise NotImplementedError(
            "grouped native FP4 context requires causal D=128, common "
            "dtype/device/head shape, positive queries, and kv_len>=q_len"
        )
    if (
        kv_indices.ndim != 1
        or kv_indices.dtype != torch.int32
        or kv_indices.device != q0.device
        or kv_offsets.numel() != batch
        or kv_offsets.dtype != torch.int32
        or kv_offsets.device != q0.device
    ):
        raise ValueError(
            "kv_indices must be int32[pages] and kv_offsets int32[batch] "
            "on the query device"
        )
    if (query_source is None) != (query_indptr is None):
        raise ValueError("query_source and query_indptr must be provided together")
    if query_source is not None and (
        query_source.ndim != 3
        or query_source.shape[1:] != (h_q, d)
        or query_source.dtype != q0.dtype
        or query_source.device != q0.device
        or query_indptr.ndim != 1
        or query_indptr.numel() != batch + 1
        or query_indptr.dtype != torch.int32
        or query_indptr.device != q0.device
    ):
        raise ValueError(
            "query_source must be [T,H,D] matching queries and query_indptr "
            "must be int32[batch+1] on the same device"
        )

    s_q_kernel = (
        (max(q_lens) + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
    )
    q_pads = [s_q_kernel - q_len for q_len in q_lens]
    fused_q = None
    if query_source is not None:
        fused_q = _quantize_context_q_cuda(
            query_source, query_indptr.contiguous(), s_q_kernel
        )
    if fused_q is None:
        q_kernel = torch.zeros(
            batch, s_q_kernel, h_q, d, dtype=q0.dtype, device=q0.device
        )
        for b, (q, pad) in enumerate(zip(queries, q_pads)):
            q_kernel[b, pad:] = q

        packed_batches = []
        sf_batches = []
        for b in range(batch):
            per_head = [
                _quantize_k_major(q_kernel[b, :, h, :].float())
                for h in range(h_q)
            ]
            packed_batches.append(torch.stack([x[0] for x in per_head], dim=0))
            sf_batches.append(
                torch.stack([x[1].reshape(-1) for x in per_head], dim=0)
            )
        q_pack = torch.stack(packed_batches, dim=0).contiguous()
        sfq_arg = torch.stack(sf_batches, dim=0).contiguous().view(
            torch.float8_e4m3fn
        )
    else:
        q_pack, sfq_arg = fused_q
    q_arg = q_pack.view(FP4_TORCH).permute(2, 3, 1, 0)

    physical_pages = pool.k_data.shape[0]
    _kern, compiled = _compiled_perf(
        d, causal, s_q_kernel, physical_pages, h_q, h_kv
    )
    mk, mv, sfk_t, sfv_t = _perf_pool_views(pool)
    # Every padded row/head is written by the kernel before any result view is
    # exposed.  Zeroing this potentially multi-megabyte grouped buffer added a
    # pure memset to every context layer without providing initialization
    # semantics.
    out = torch.empty(
        batch, h_q, s_q_kernel, d,
        device=q0.device,
        dtype=torch.bfloat16,
    )
    if kv_lens_tensor is None:
        kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device=q0.device)
    else:
        if (
            kv_lens_tensor.numel() != batch
            or kv_lens_tensor.dtype != torch.int32
            or kv_lens_tensor.device != q0.device
        ):
            raise ValueError(
                "kv_lens_tensor must be int32[batch] on the query device"
            )
        kv_lens_t = kv_lens_tensor.reshape(batch).contiguous()
    call_args = (
        q_arg,
        mk,
        mv,
        out.permute(2, 3, 1, 0),
        sfq_arg,
        sfk_t,
        sfv_t,
        kv_indices.contiguous(),
        kv_offsets.reshape(batch).contiguous(),
        kv_lens_t,
        Float32(sm_scale * math.log2(math.e)),
        Float32(sm_scale),
        Float32(1.0),
    )
    return compiled, call_args, out, q_pads


def run_paged_fp4_prefill_pool_batch(
    queries: Sequence[torch.Tensor],
    pool: NvFp4PagedKVPool,
    kv_indices: torch.Tensor,
    kv_offsets: torch.Tensor,
    kv_lens: Sequence[int],
    sm_scale: float,
    *,
    causal: bool = True,
    outs: Optional[Sequence[torch.Tensor]] = None,
    kv_lens_tensor: Optional[torch.Tensor] = None,
    query_source: Optional[torch.Tensor] = None,
    query_indptr: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    """Run a grouped context launch; results follow the input query order."""
    compiled, call_args, kernel_out, q_pads = (
        prepare_paged_fp4_prefill_pool_batch(
            queries,
            pool,
            kv_indices,
            kv_offsets,
            kv_lens,
            sm_scale,
            causal=causal,
            kv_lens_tensor=kv_lens_tensor,
            query_source=query_source,
            query_indptr=query_indptr,
        )
    )
    compiled(*call_args)
    results = [
        kernel_out[b, :, pad:].permute(1, 0, 2)
        for b, pad in enumerate(q_pads)
    ]
    if outs is None:
        return results
    if len(outs) != len(results):
        raise ValueError(f"outs length {len(outs)} != batch {len(results)}")
    returned = []
    for dst, src in zip(outs, results):
        if dst.shape != src.shape:
            raise ValueError(f"out shape {tuple(dst.shape)} != {tuple(src.shape)}")
        dst.copy_(src.to(dst.dtype))
        returned.append(dst)
    return returned


def run_paged_fp4_prefill_perf(q, k_deq, v_deq, sm_scale: float, **kw):
    """性能版便捷包装;返回 ``[S_q, H_q, D]``(与参考同形)。"""
    compiled, call_args, out = prepare_paged_fp4_prefill_perf(
        q, k_deq, v_deq, sm_scale, **kw)
    compiled(*call_args)
    return out.permute(1, 0, 2).float()
