# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'MMA-native paged NVFP4 KV append:torch 参考实现(语义锁定层)。'

from __future__ import annotations

import torch

from .kv_layout import (
    PAGE_SIZE,
    E2M1_LUT,
    E2M1_MAX,
    PI16,
    SF_VEC,
    NvFp4PagedKVPool,
    sf_atom_gather,
    sf_atom_offsets,
)

_LUT_EDGES_CACHE: dict = {}


def _lut_and_edges(device):
    key = str(device)
    if key not in _LUT_EDGES_CACHE:
        lut = torch.tensor(E2M1_LUT, device=device)
        _LUT_EDGES_CACHE[key] = (lut, (lut[1:] + lut[:-1]) / 2)
    return _LUT_EDGES_CACHE[key]


def _e2m1_nibbles(y: torch.Tensor) -> torch.Tensor:
    """y 已除 SF 并 clamp ±6 → e2m1 nibble(uint8,低 3 位幅值序号+符号位)。"""
    lut, edges = _lut_and_edges(y.device)
    idx = torch.bucketize(y.abs(), edges)
    return (idx + torch.where(y < 0, 8, 0)).to(torch.uint8)


def _nibbles_decode(nib: torch.Tensor) -> torch.Tensor:
    """e2m1 nibble → float 值(gather_dequant 用)。"""
    lut, _ = _lut_and_edges(nib.device)
    mag = lut[(nib & 7).long()]
    return torch.where(nib >= 8, -mag, mag)


def _quant_groups(x: torch.Tensor):
    """沿最后轴按 16 一组量化。x (..., G, 16) → (nib (..., G, 16), sf_bytes (..., G))。

    SF = amax/6 → e4m3(无全局缩放);nib 未打包、未 π。
    """
    sf = (x.abs().amax(-1) / E2M1_MAX).to(torch.float8_e4m3fn)
    sf_f = sf.float().clamp_min(1e-12)
    y = (x / sf_f.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
    return _e2m1_nibbles(y), sf.view(torch.uint8)


def _pack_low_high(nib: torch.Tensor) -> torch.Tensor:
    """相邻 nibble 打包:偶下标进低位。(..., 2n) uint8 → (..., n) uint8。"""
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()


def _unpack_low_high(packed: torch.Tensor) -> torch.Tensor:
    """_pack_low_high 的逆。(..., n) → (..., 2n)。"""
    out = torch.stack([packed & 0xF, packed >> 4], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _token_page_row(positions, kv_indptr, kv_indices, batch_indices, page_size):
    """token → (物理页号, 页内行号)。"""
    logical_page = positions // page_size
    pages = kv_indices[(kv_indptr[batch_indices] + logical_page).long()]
    return pages.long(), (positions % page_size).long()


# ---------------------------------------------------------------------------
# K:无状态追加(SF 沿 d,单 token 自含)
# ---------------------------------------------------------------------------


def _append_k(pool, append_key, batch_indices, positions, kv_indices, kv_indptr):
    nnz, H, D = append_key.shape
    pages, rows = _token_page_row(
        positions, kv_indptr, kv_indices, batch_indices, pool.page_size
    )
    nib, sf_bytes = _quant_groups(append_key.float().reshape(nnz, H, D // SF_VEC, SF_VEC))
    packed = _pack_low_high(nib.reshape(nnz, H, D))  # (nnz, H, D/2)
    pool.k_data[pages, :, rows, :] = packed
    # SF:页局部 atom 序;单行的 D/16 个组散写
    offs = sf_atom_offsets(pool.page_size, D // SF_VEC, device=append_key.device)
    row_offs = offs[rows]  # (nnz, D/16)
    head_idx = torch.arange(H, device=append_key.device)
    pool.k_sf[
        pages[:, None, None],
        head_idx[None, :, None],
        row_offs[:, None, :],
    ] = sf_bytes  # (nnz, H, D/16)


# ---------------------------------------------------------------------------
# V:暂存 + 组内重写
# ---------------------------------------------------------------------------


def _rewrite_v_group(pool, slot: int, page: int, group_local: int, n_valid: int):
    """用 ``v_staging[slot]`` 的前 n_valid 个 token 重算并重写一个 16-token 组。

    组内 token 排布:π 预排(PI16)后沿 token 打包;SF 组序不变。
    无效 token 位置写 0 nibble(读侧被掩码,零值最干净)。
    """
    H, D = pool.num_kv_heads, pool.head_dim
    dev = pool.v_data.device
    xt = pool.v_staging[slot, :, :n_valid, :].float().permute(0, 2, 1)  # (H, D, n)
    sf = (xt.abs().amax(-1) / E2M1_MAX).to(torch.float8_e4m3fn)  # (H, D)
    sf_f = sf.float().clamp_min(1e-12)
    y = (xt / sf_f.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
    nib = torch.zeros(H, D, SF_VEC, dtype=torch.uint8, device=dev)
    nib[:, :, :n_valid] = _e2m1_nibbles(y)
    pi = torch.tensor(PI16, device=dev)
    packed = _pack_low_high(nib[:, :, pi])  # (H, D, 8)
    byte0 = group_local * (SF_VEC // 2)
    pool.v_data[page, :, :, byte0 : byte0 + SF_VEC // 2] = packed
    offs = sf_atom_offsets(D, pool.page_size // SF_VEC, device=dev)[:, group_local]
    head_idx = torch.arange(H, device=dev)
    pool.v_sf[page, head_idx[:, None], offs[None, :]] = sf.view(torch.uint8)


def _append_v(pool, append_value, batch_indices, positions, kv_indices, kv_indptr,
              staging_slots=None):
    '增量 V 追加。要求每序列内 positions 连续递增(serving 追加的自然形态)。'
    ps = pool.page_size
    for b in batch_indices.unique().tolist():
        sel = batch_indices == b
        pos_b = positions[sel]
        val_b = append_value[sel]
        order = pos_b.argsort()
        pos_b, val_b = pos_b[order], val_b[order]
        # 逐组处理(一次 decode 追加只有 1 token/序列;prefill 请走 bulk)
        groups = pos_b // SF_VEC
        for g in groups.unique().tolist():
            gsel = groups == g
            js = (pos_b[gsel] % SF_VEC).long()
            slot = b if staging_slots is None else int(staging_slots[b])
            pool.v_staging[slot, :, js, :] = (
                val_b[gsel].to(torch.bfloat16).permute(1, 0, 2)
            )
            n_valid = int(js.max().item()) + 1
            page = int(
                kv_indices[int(kv_indptr[b].item()) + g * SF_VEC // ps].item()
            )
            _rewrite_v_group(pool, slot, page, (g * SF_VEC % ps) // SF_VEC, n_valid)


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------



def build_v_groups(
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    staging_slots: torch.Tensor | None,
    page_size: int = 128,
) -> torch.Tensor:
    """把 token 列表折成「组出现」列表 [n_groups, 6],供 CUDA V kernel 使用。

    列:``(page, g_local, tok0, n_tok, j0, slot)``。

    **全程向量化、零 GPU 同步** —— 这是整件事的要点:旧路径按组用 python
    循环驱动 kernel,每组 3 次 `.item()`,prefill 时 ≈123 万次同步,占掉了
    GSM8K 那轮 63× 慢里的绝大部分。

    契约:同一 (请求, 组) 的 token 在输入里**连续**(上层按请求、按位置递增
    排布即满足);``unique_consecutive`` 依赖这一点。
    """
    b = batch_indices.long()
    pos = positions.long()
    # 同一请求同一组 → 同一个 key;连续性由契约保证
    key = b * (1 << 32) + pos // SF_VEC
    _, counts = torch.unique_consecutive(key, return_counts=True)
    tok0 = torch.cumsum(counts, 0) - counts
    b0, p0 = b[tok0], pos[tok0]
    g = p0 // SF_VEC
    page = kv_indices.long()[kv_indptr.long()[b0] + (g * SF_VEC) // page_size]
    g_local = ((g * SF_VEC) % page_size) // SF_VEC
    slot = b0 if staging_slots is None else staging_slots.long()[b0]
    return torch.stack(
        [page, g_local, tok0, counts, p0 % SF_VEC, slot], dim=1
    ).to(torch.int32).contiguous()


def bulk_quantize_paged(
    k: torch.Tensor,  # [B, S_max, H, D] bf16(padded)
    v: torch.Tensor,
    seqlens: torch.Tensor,  # [B]
    pool: NvFp4PagedKVPool,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
) -> None:
    """整序列量化入池(prefill / 测试)。与逐 token 追加字节等价
    (等价性由 tests 强制)。同时把每序列的尾部未满组写进 v_staging,
    使后续增量追加可无缝衔接。
    """
    B = k.shape[0]
    dev = k.device
    ps = pool.page_size
    for b in range(B):
        S = int(seqlens[b].item())
        n_pages = (S + ps - 1) // ps
        idx = torch.arange(S, device=dev)
        _append_k(
            pool,
            k[b, :S],
            torch.full((S,), b, device=dev, dtype=torch.long),
            idx,
            kv_indices,
            kv_indptr,
        )
        # V:逐页整组向量化 + 尾组走重写路径
        pages = kv_indices[kv_indptr[b].long() : kv_indptr[b].long() + n_pages].long()
        full_groups_total = S // SF_VEC
        pi = torch.tensor(PI16, device=dev)
        for p in range(n_pages):
            tok0 = p * ps
            n_tok = min(ps, S - tok0)
            n_full = min(n_tok, (full_groups_total * SF_VEC) - tok0)
            n_full_groups = max(0, n_full) // SF_VEC
            if n_full_groups > 0:
                xt = (
                    v[b, tok0 : tok0 + n_full_groups * SF_VEC]
                    .float()
                    .permute(1, 2, 0)  # (H, D, n)
                    .reshape(pool.num_kv_heads, pool.head_dim, n_full_groups, SF_VEC)
                )
                nib, sf_bytes = _quant_groups(xt)
                packed = _pack_low_high(nib[..., pi])  # (H, D, G, 8)
                packed = packed.permute(0, 1, 2, 3).reshape(
                    pool.num_kv_heads, pool.head_dim, n_full_groups * (SF_VEC // 2)
                )
                pool.v_data[pages[p], :, :, : n_full_groups * (SF_VEC // 2)] = packed
                offs = sf_atom_offsets(pool.head_dim, ps // SF_VEC, device=dev)
                head_idx = torch.arange(pool.num_kv_heads, device=dev)
                pool.v_sf[
                    pages[p],
                    head_idx[:, None, None],
                    offs[None, :, :n_full_groups],
                ] = sf_bytes
        # 尾组(未满 16):进暂存 + 组内重写。S%16==0 时无需动暂存:
        # 尾组已完整,后续增量追加必然从新组 j=0 开始,旧暂存内容不会被读到。
        tail = S % SF_VEC
        if tail:
            g = full_groups_total
            pool.v_staging[b, :, :tail, :] = (
                v[b, g * SF_VEC : S].to(torch.bfloat16).permute(1, 0, 2)
            )
            _rewrite_v_group(
                pool, b, int(pages[g * SF_VEC // ps].item()),
                (g * SF_VEC % ps) // SF_VEC, tail
            )


def gather_dequant(
    pool: NvFp4PagedKVPool,
    seqlen: int,
    b: int,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
):
    """按页表取回序列 b 的前 seqlen 个 token 并反量化(校验用)。

    返回 (k_deq, v_deq),形状均为 [seqlen, H, D] float32。
    """
    ps, H, D = pool.page_size, pool.num_kv_heads, pool.head_dim
    n_pages = (seqlen + ps - 1) // ps
    pages = kv_indices[kv_indptr[b].long() : kv_indptr[b].long() + n_pages].long()
    # K
    k_nib = _unpack_low_high(pool.k_data[pages])  # (n_pages, H, ps, D)
    k_val = _nibbles_decode(k_nib)
    k_sf = sf_atom_gather(pool.k_sf[pages], ps, D // SF_VEC)  # (n_pages, H, ps, D/16)
    k_sf_f = k_sf.view(torch.float8_e4m3fn).float().clamp_min(1e-12)
    k_deq = (
        (k_val.reshape(n_pages, H, ps, D // SF_VEC, SF_VEC) * k_sf_f.unsqueeze(-1))
        .reshape(n_pages, H, ps, D)
        .permute(0, 2, 1, 3)
        .reshape(n_pages * ps, H, D)[:seqlen]
    )
    # V:反打包 → 逆 π → 乘 SF → 转回 (S, H, D)
    v_nib_pi = _unpack_low_high(pool.v_data[pages])  # (n_pages, H, D, ps)
    pi = torch.tensor(PI16, device=pool.v_data.device)
    inv_pi = torch.empty_like(pi)
    inv_pi[pi] = torch.arange(SF_VEC, device=pi.device)
    v_nib = (
        v_nib_pi.reshape(n_pages, H, D, ps // SF_VEC, SF_VEC)
        .index_select(-1, inv_pi)
    )
    v_val = _nibbles_decode(v_nib)  # (n_pages, H, D, ps/16, 16)
    v_sf = sf_atom_gather(pool.v_sf[pages], D, ps // SF_VEC)  # (n_pages, H, D, ps/16)
    v_sf_f = v_sf.view(torch.float8_e4m3fn).float().clamp_min(1e-12)
    v_deq = (
        (v_val * v_sf_f.unsqueeze(-1))
        .reshape(n_pages, H, D, ps)
        .permute(0, 3, 1, 2)
        .reshape(n_pages * ps, H, D)[:seqlen]
    )
    return k_deq.contiguous(), v_deq.contiguous()


def gather_dequant_trtllm(
    pool: NvFp4PagedKVPool,
    seqlen: int,
    b: int,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
):
    """Read back vLLM/FlashInfer's TRT-LLM Gen NVFP4 cache contract.

    Both K and V data are ordinary HND pages with D contiguous. K scales are
    linear ``[token, D/16]``; V scales use TRT-LLM's four-token interleave.
    This is a correctness/oracle path for prefix reuse while the native TRT
    context wrapper is being wired, not a performance implementation.
    """
    if pool.layout_profile != "trtllm":
        raise ValueError(
            "gather_dequant_trtllm requires layout_profile='trtllm', got "
            f"{pool.layout_profile!r}"
        )
    ps, H, D = pool.page_size, pool.num_kv_heads, pool.head_dim
    groups = D // SF_VEC
    n_pages = (seqlen + ps - 1) // ps
    pages = kv_indices[
        kv_indptr[b].long() : kv_indptr[b].long() + n_pages
    ].long()

    def _data_values(data):
        nib = _unpack_low_high(data[pages])
        return _nibbles_decode(nib).reshape(n_pages, H, ps, groups, SF_VEC)

    k_values = _data_values(pool.k_data)
    v_values = _data_values(pool.v_data)
    k_sf = pool.k_sf[pages].float().clamp_min(1e-12)

    # The physical V-scale tile is contiguous but its logical [t, group]
    # coordinates are permuted in four-token chunks.
    t = torch.arange(ps, device=pool.v_sf.device).view(ps, 1)
    s = torch.arange(groups, device=pool.v_sf.device).view(1, groups)
    inter = (t // 4) * 4 * groups + s * 4 + (t % 4)
    v_sf_flat = pool.v_sf[pages].reshape(n_pages, H, ps * groups)
    v_sf = v_sf_flat[:, :, inter.reshape(-1)].reshape(
        n_pages, H, ps, groups
    ).float().clamp_min(1e-12)

    def _finish(values, scales, global_scale):
        return (
            (values * scales.unsqueeze(-1))
            .reshape(n_pages, H, ps, D)
            .permute(0, 2, 1, 3)
            .reshape(n_pages * ps, H, D)[:seqlen]
            .mul(float(global_scale))
            .contiguous()
        )

    return (
        _finish(k_values, k_sf, pool.k_global_scale),
        _finish(v_values, v_sf, pool.v_global_scale),
    )


# ---------------------------------------------------------------------------
# trtllm-gen 契约的量化写出 —— prefill 用 trtllm context kernel、decode 用
# 我们的 kernel,两边共享同一份 cache 的前提就是这个格式。
# ---------------------------------------------------------------------------
#
# 三个张量的契约(与 flashinfer ``nvfp4_quantize_paged_kv_cache`` 的 HND 输出
# 逐字节一致;K/V 数据本来就同构,差别只在 SF):
#
#   k_fp4 / v_fp4 : uint8 [P, H, ps, D/2]     D 连续,相邻两值打包
#   k_sf          : e4m3  [P, H, ps, D/16]    **线性**(trtllm 明写 K 不 swizzle)
#   v_sf          : e4m3  [P, H, ps, D/16]    **4-token interleave**
#
# V-SF 的 interleave:每 (page, head) 的 [ps, D/16] 块内,
#
#     offset(token t, block s) = (t // 4) * 4 * S + s * 4 + (t % 4),  S = D/16
#
# 即"同一 block 的 4 个连续 token 相邻"。这条公式与 flashinfer 的
# reshape/permute 配方逐元素等价(离线验证,见 tests)。它同时可以纯用 CuTe
# layout 表达 —— ``(S, (4, ps//4)) : (4, (1, 4*S))`` —— 所以 kernel 侧只换
# 布局、索引代码一个字不改,且每 (page,head) 的 SF 块仍是连续 1024B,
# TMA 整块搬,不引入新的 stride 模(坑41 的 16B 对齐禁区因此完全避开)。

def quantize_paged_trtllm_format(
    k: torch.Tensor,          # [B, S_max, H, D] float/bf16
    v: torch.Tensor,
    seqlens: torch.Tensor,    # [B]
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    n_pages_total: int,
    page_size: int = PAGE_SIZE,
):
    """按 trtllm-gen 契约量化入页,返回 ``(k_fp4, k_sf, v_fp4, v_sf, k_deq, v_deq)``。

    ``k_deq`` / ``v_deq`` 是**反量化后**的值,给参考实现用 —— 对拍时必须用它们
    而不是原始 ``k``/``v``,否则门测的是量化误差而不是 kernel 算得对不对。
    """
    B, _S_max, H, D = k.shape
    dev = k.device
    S = D // SF_VEC
    assert page_size % 4 == 0 and D % 64 == 0, (page_size, D)

    k_fp4 = torch.zeros(n_pages_total, H, page_size, D // 2,
                        dtype=torch.uint8, device=dev)
    v_fp4 = torch.zeros_like(k_fp4)
    k_sf = torch.zeros(n_pages_total, H, page_size, S,
                       dtype=torch.uint8, device=dev)
    v_sf = torch.zeros_like(k_sf)
    k_deq = torch.zeros(k.shape, dtype=torch.float32, device=dev)
    v_deq = torch.zeros_like(k_deq)

    # interleave 的目标下标:一次算好,后面按 (t, s) 散写
    t_idx = torch.arange(page_size, device=dev).view(-1, 1)
    s_idx = torch.arange(S, device=dev).view(1, -1)
    inter = (t_idx // 4) * 4 * S + s_idx * 4 + (t_idx % 4)   # [ps, S]

    for b in range(B):
        s_len = int(seqlens[b])
        n_pg = (s_len + page_size - 1) // page_size
        pages = kv_indices[int(kv_indptr[b]): int(kv_indptr[b]) + n_pg].long()
        for lp in range(n_pg):
            t0 = lp * page_size
            n = min(t0 + page_size, s_len) - t0
            p = pages[lp]
            for src, dst_fp4, dst_sf, deq, interleaved in (
                (k, k_fp4, k_sf, k_deq, False),
                (v, v_fp4, v_sf, v_deq, True),
            ):
                x = src[b, t0:t0 + n].float()                    # (n, H, D)
                grp = x.reshape(n, H, S, SF_VEC)
                sf = (grp.abs().amax(-1) / E2M1_MAX).to(torch.float8_e4m3fn)
                sf_f = sf.float().clamp_min(1e-12)
                y = (grp / sf_f.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
                nib = _e2m1_nibbles(y).reshape(n, H, D)
                dst_fp4[p, :, :n] = _pack_low_high(nib.permute(1, 0, 2))
                sf_hnd = sf.view(torch.uint8).permute(1, 0, 2)   # (H, n, S)
                if interleaved:
                    flat = dst_sf[p].reshape(H, page_size * S)
                    flat[:, inter[:n].reshape(-1)] = sf_hnd.reshape(H, n * S)
                else:
                    dst_sf[p, :, :n] = sf_hnd
                deq[b, t0:t0 + n] = (
                    _nibbles_decode(nib.reshape(n, H, S, SF_VEC))
                    * sf_f.unsqueeze(-1)
                ).reshape(n, H, D)

    e4m3 = torch.float8_e4m3fn
    return (k_fp4, k_sf.view(e4m3), v_fp4, v_sf.view(e4m3),
            k_deq, v_deq)


def quantize_paged_dual(
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: torch.Tensor,
    pool: NvFp4PagedKVPool,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
):
    '一次量化,写出**两套 K-SF 表示**,让 prefill 与 decode 共享一份 cache。'
    bulk_quantize_paged(k, v, seqlens, pool, kv_indices, kv_indptr)
    return quantize_paged_trtllm_format(
        k, v, seqlens, kv_indices, kv_indptr, pool.k_data.shape[0],
        pool.page_size,
    )

def gather_ksf_linear_for_trtllm(
    k_sf_atom: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    b: int,
    n_pages: int,
    num_kv_heads: int,
    head_dim: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    'atom 序的 K-SF → trtllm 要的**线性**序,按页表 gather 到临时缓冲。'
    ps, sfg = PAGE_SIZE, head_dim // SF_VEC
    pages = kv_indices[int(kv_indptr[b]): int(kv_indptr[b]) + n_pages].long()
    # (n_pages, H, ps*sfg) —— 池里每 (page, head) 一块 atom 序扁平 SF
    atom = k_sf_atom[pages].reshape(n_pages, num_kv_heads, ps * sfg)
    lin = sf_atom_gather(atom, ps, sfg)          # (n_pages, H, ps, sfg)
    if out is None:
        return lin.contiguous()
    out[:n_pages].copy_(lin)
    return out[:n_pages]
