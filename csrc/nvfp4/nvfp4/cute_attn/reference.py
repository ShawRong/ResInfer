# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'ResAttn 的规范参考实现(torch),用于给 kernel 定数值门。'

from __future__ import annotations

import math

import torch

from .kv_layout import E2M1_LUT, E2M1_MAX, SF_VEC

LOG2E = math.log2(math.e)

#: P 的常数 scale factor(e4m3 精确可表示);与 host._psf_buffer 一致。
SFP = 0.171875


def _lut_edges(device):
    lut = torch.tensor(E2M1_LUT, device=device)
    return lut, (lut[1:] + lut[:-1]) / 2


def quantize_dequantize(x: torch.Tensor) -> torch.Tensor:
    """沿最后一轴按 SF_VEC 分组做 nvfp4 量化再还原(与 append/host 同源数学)。"""
    lut, edges = _lut_edges(x.device)
    g = x.float().reshape(*x.shape[:-1], x.shape[-1] // SF_VEC, SF_VEC)
    sf = (g.abs().amax(-1) / E2M1_MAX).to(torch.float8_e4m3fn).float().clamp_min(1e-12)
    y = (g / sf.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
    idx = torch.bucketize(y.abs().contiguous(), edges)
    return (lut[idx] * torch.sign(y) * sf.unsqueeze(-1)).reshape(x.shape)


def quantize_dequantize_v(v: torch.Tensor) -> torch.Tensor:
    'V 的量化 —— **SF 组沿 token,不是沿 head_dim**。``v`` 是 ``[S_kv, H, D]``。'
    return quantize_dequantize(v.permute(1, 2, 0)).permute(2, 0, 1).contiguous()


def quantize_p(probs: torch.Tensor) -> torch.Tensor:
    """把 softmax 概率按**常数** SF=SFP 量化到 e2m1 再还原。

    probs ∈ [0,1](在线 softmax 保证 ≤1)⇒ probs/SFP ∈ [0, 5.82] < E2M1_MAX,
    clamp 基本不触发;这正是常数 SF 取 0.171875 的原因(6*SFP = 1.03125)。
    """
    lut, edges = _lut_edges(probs.device)
    y = (probs * (1.0 / SFP)).clamp(0, E2M1_MAX)
    return lut[torch.bucketize(y.contiguous(), edges)] * SFP


def quantize_p_fp8(probs: torch.Tensor) -> torch.Tensor:
    '把 softmax 概率量化到 **e4m3** 再还原(Phase 1 / ``qmul4_d`` 的 P 格式)。'
    return probs.to(torch.float8_e4m3fn).float()


def decode_reference_row(
    q_row: torch.Tensor,  # [D] 已量化-还原的单个 query 头
    k_deq: torch.Tensor,  # [S, D] 从池中反量化的 K
    v_deq: torch.Tensor,  # [S, D]
    sm_scale: float,
    p_fmt: str = "e2m1",
) -> torch.Tensor:
    """单个 (batch, head) 的参考:S → exp2 → P 量化 → VP → 归一化。

    ``p_fmt`` 必须与 kernel 实际用的 P 格式一致:``e2m1`` = 原生 blockscaled
    路径(常数 SF);``fp8`` = Phase 1 / ``qmul4_d``。
    """
    s = (q_row @ k_deq.T) * sm_scale
    probs = torch.exp2((s - s.max()) * LOG2E)
    if p_fmt == "exact":
        qp = probs                      # 不量化 P —— 两个 kernel 对比时的共同基准
    elif p_fmt == "fp8":
        qp = quantize_p_fp8(probs)
    else:
        qp = quantize_p(probs)
    
    
    return (qp @ v_deq) / probs.sum()


def decode_reference_row_split(
    q_row: torch.Tensor, k_deq: torch.Tensor, v_deq: torch.Tensor,
    sm_scale: float, p_fmt: str, splits: int,
) -> torch.Tensor:
    'split-K 版参考:每个 split 用**自己的局部 max** 归一化后再量化 P。'
    s_all = (q_row @ k_deq.T) * sm_scale
    m_global = s_all.max()
    n = k_deq.shape[0]
    step = (n + splits - 1) // splits
    num = torch.zeros(v_deq.shape[1], dtype=torch.float32, device=q_row.device)
    den = torch.zeros((), dtype=torch.float32, device=q_row.device)
    for a in range(0, n, step):
        b = min(a + step, n)
        sv = s_all[a:b]
        m_s = sv.max()
        probs = torch.exp2((sv - m_s) * LOG2E)
        qp = quantize_p_fp8(probs) if p_fmt == "fp8" else quantize_p(probs)
        rescale = torch.exp2((m_s - m_global) * LOG2E)
        num = num + (qp @ v_deq[a:b]) * rescale
        den = den + probs.sum() * rescale
    return num / den


def decode_reference(
    q: torch.Tensor,  # [H_q, D] bf16/fp32,**未**量化(本函数内部量化)
    k_deq: torch.Tensor,  # [S, H_kv, D]
    v_deq: torch.Tensor,  # [S, H_kv, D]
    sm_scale: float,
    p_fmt: str = "e2m1",
    splits: int = 1,
) -> torch.Tensor:
    """整个 batch 行的 decode 参考,返回 [H_q, D](float32)。

    ``k_deq`` / ``v_deq`` 应来自 :func:`~nvfp4.cute_attn.gather_dequant`,
    即 kernel 真正读到的那些 fp4 字节还原出的值。
    """
    h_q = q.shape[0]
    g = h_q // k_deq.shape[1]
    q_dq = quantize_dequantize(q)
    return torch.stack(
        [
            decode_reference_row(q_dq[h].float(), k_deq[:, h // g],
                                 v_deq[:, h // g], sm_scale, p_fmt)
            if splits <= 1 else
            decode_reference_row_split(q_dq[h].float(), k_deq[:, h // g],
                                       v_deq[:, h // g], sm_scale, p_fmt,
                                       splits)
            for h in range(h_q)
        ]
    )


# ---------------------------------------------------------------- prefill --


def _quantize_probs(p: torch.Tensor, p_scale: str) -> torch.Tensor:
    """按 ``p_scale`` 口径量化 P;dynamic 走逐 16-token 组的标准 nvfp4 SF。

    最后一个 KV tile 可能不是 16 的整数倍(kernel 侧是满 tile + 掩码,参考侧
    切成了不等长片)。补零到 16 的倍数再量化:全零组的 amax=0,量化回来仍是 0,
    不污染真实组的 SF。
    """
    if p_scale == "const":
        return quantize_p(p)
    n = p.shape[-1]
    pad = (-n) % SF_VEC
    if pad:
        p = torch.nn.functional.pad(p, (0, pad))
    return quantize_dequantize(p)[..., :n]


def quantize_dequantize_residue(x: torch.Tensor, levels: int = 1) -> torch.Tensor:
    """多级 nvfp4 量化-还原:``x ≈ Σᵢ qᵢ``,``q₁=Q(x)``、``q₂=Q(x-q₁)``…

    ``levels=1`` 即普通 nvfp4;``levels=2`` 就是本项目的 residue —— kernel 侧把
    base/res 两块**拼成两组行**送进同一个 MMA,再在 softmax 之前把 S 的两半相加
    (``kernel_paged_decode.py`` 的 ``residue_fold``)。数学上等价于本函数,
    所以参考实现只需要这一行,不必复刻拼接几何。
    """
    out = torch.zeros_like(x, dtype=torch.float32)
    r = x.float()
    for _ in range(levels):
        step = quantize_dequantize(r)
        out = out + step
        r = r - step
    return out


def prefill_reference(
    q: torch.Tensor,  # [S_q, H_q, D] bf16/fp32,**未**量化(本函数内部量化)
    k_deq: torch.Tensor,  # [S_kv, H_kv, D] 池中反量化的 K(kernel 真正读到的值)
    v_deq: torch.Tensor,  # [S_kv, H_kv, D]
    sm_scale: float,
    *,
    kv_tile: int | None = 128,
    causal: bool = True,
    quantize_probs: bool = True,
    p_scale: str = "dynamic",
    residue_levels: int = 1,
    ctx_len: int | None = None,
) -> torch.Tensor:
    'prefill(含前缀)的规范参考,返回 [S_q, H_q, D] float32。'
    s_q, h_q, d = q.shape
    s_kv, h_kv = k_deq.shape[0], k_deq.shape[1]
    if ctx_len is None:
        ctx_len = s_kv - s_q
    assert ctx_len >= 0 and s_kv >= s_q, (s_kv, s_q, ctx_len)
    assert h_q % h_kv == 0, (h_q, h_kv)
    assert p_scale in ("dynamic", "const"), p_scale
    g = h_q // h_kv
    dev = q.device
    tile = s_kv if kv_tile is None else kv_tile

    q_dq = quantize_dequantize_residue(q, residue_levels)  # [S_q, H_q, D]
    qh = q_dq.permute(1, 0, 2).contiguous()  # [H_q, S_q, D]
    kh = k_deq.float().permute(1, 0, 2).repeat_interleave(g, dim=0)  # [H_q, S_kv, D]
    vh = v_deq.float().permute(1, 0, 2).repeat_interleave(g, dim=0)

    qpos = ctx_len + torch.arange(s_q, device=dev).view(s_q, 1)  # [S_q, 1]
    neg_inf = torch.tensor(float("-inf"), device=dev)
    m = torch.full((h_q, s_q), float("-inf"), device=dev)
    l = torch.zeros((h_q, s_q), device=dev)
    o = torch.zeros((h_q, s_q, d), device=dev)

    for t0 in range(0, s_kv, tile):
        t1 = min(t0 + tile, s_kv)
        s = torch.bmm(qh, kh[:, t0:t1].transpose(1, 2)) * sm_scale  # [H_q, S_q, T]
        if causal:
            vis = torch.arange(t0, t1, device=dev).view(1, t1 - t0) <= qpos
            if not bool(vis.any()):
                continue  # 整块在对角线以上 —— kernel 侧直接跳过该 tile
            s = torch.where(vis.unsqueeze(0), s, neg_inf)

        m_new = torch.maximum(m, s.amax(-1))
        ok = torch.isfinite(m_new)  # 该行到此为止仍无任何可见 KV → 保持全零
        m_safe = torch.where(ok, m_new, torch.zeros_like(m_new))
        # m=-inf 且 m_new 有限 → exp2(-inf)=0(o/l 本就是 0,正确);
        # 两者都是 -inf → 会算出 nan,用 ok 挡掉。
        scale = torch.where(ok, torch.exp2((m - m_safe) * LOG2E), torch.ones_like(m))

        p = torch.exp2((s - m_safe.unsqueeze(-1)) * LOG2E)  # 掩掉的位置 → 0
        l = l * scale + p.sum(-1)
        p_use = _quantize_probs(p, p_scale) if quantize_probs else p
        o = o * scale.unsqueeze(-1) + torch.bmm(p_use, vh[:, t0:t1])
        m = m_new

    # causal 下每行至少可见自身位置 ⇒ l>0;clamp 只是防御,不该被触发。
    return (o / l.clamp_min(1e-30).unsqueeze(-1)).permute(1, 0, 2).contiguous()


def prefill_reference_paged(
    pool,
    q: torch.Tensor,  # [total_q, H_q, D] varlen 打包
    qo_indptr: torch.Tensor,  # [B+1]
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    seq_lens: torch.Tensor,  # [B] 每请求 KV 总长(含本次新增)
    sm_scale: float,
    **kw,
) -> torch.Tensor:
    """从分页池取回 KV 后逐请求调 :func:`prefill_reference`,返回 [total_q, H_q, D]。

    这是 backend 侧要对拍的那条链:池字节 → 反量化 → 参考注意力。
    """
    from .append import gather_dequant

    out = torch.zeros(*q.shape, dtype=torch.float32, device=q.device)
    for b in range(len(qo_indptr) - 1):
        q0, q1 = int(qo_indptr[b]), int(qo_indptr[b + 1])
        if q1 <= q0:
            continue
        seqlen = int(seq_lens[b])
        k_deq, v_deq = gather_dequant(pool, seqlen, b, kv_indices, kv_indptr)
        out[q0:q1] = prefill_reference(
            q[q0:q1], k_deq, v_deq, sm_scale, ctx_len=seqlen - (q1 - q0), **kw
        )
    return out
