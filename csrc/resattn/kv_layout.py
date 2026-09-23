# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 pool views and scale-factor atom addressing.

Production uses stock vLLM cache storage (trtllm profile). Historical
native-token allocation and writers live in resattn.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SF_VEC = 16  # nvfp4 block-scale 组大小(mxf4nvf4 sf_vec_size=16,硬件定死)
PAGE_SIZE = 128  # MMA-native profile 的页大小(= SF atom 行粒度 = KQ mma_tile_m)
E2M1_MAX = 6.0  # e2m1 可表示的最大幅值
# V 的 16-token 组内 nibble 预排(写侧吸收 placement permutation;SF 组序不变)
PI16 = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
# e2m1 幅值表(idx 0..7);bucketize 的桶边界取相邻中点
E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def sf_atom_offsets(rows: int, groups: int, device=None) -> torch.Tensor:
    """(rows, groups) 线性 SF 网格 → atom 序偏移(int64,形状 (rows, groups))。

    公式(tcgen05 SF atom ``((32,4),(sfv,4)):((16,4),(0,1))``,
    展开序 atom→K_rest→M_rest):

        addr(mn,kg) = (mn%32)*16 + ((mn//32)%4)*4 + (kg%4)
                    + (kg//4)*512 + (mn//128)*512*(groups//4)

    K 池:rows=page_size, groups=D/16;V 池:rows=D, groups=page_size/16。
    ``rows % 128 == 0`` 且 ``groups % 4 == 0`` 时该映射是 (rows*groups) 上的双射。
    """
    assert rows % 128 == 0 and groups % 4 == 0, (rows, groups)
    mn = torch.arange(rows, device=device).view(rows, 1)
    kg = torch.arange(groups, device=device).view(1, groups)
    inner = (mn % 32) * 16 + ((mn // 32) % 4) * 4 + (kg % 4)
    return inner + (kg // 4) * 512 + (mn // 128) * 512 * (groups // 4)


def sf_atom_scatter(sf_linear: torch.Tensor) -> torch.Tensor:
    """线性 SF (…, rows, groups) → atom 序扁平 (…, rows*groups)。逐元素散写。"""
    rows, groups = sf_linear.shape[-2:]
    off = sf_atom_offsets(rows, groups, device=sf_linear.device)
    out = torch.zeros(
        (*sf_linear.shape[:-2], rows * groups),
        dtype=sf_linear.dtype,
        device=sf_linear.device,
    )
    out.scatter_(
        -1,
        off.reshape(-1).expand(*sf_linear.shape[:-2], -1),
        sf_linear.reshape(*sf_linear.shape[:-2], -1),
    )
    return out


def sf_atom_gather(sf_atom: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    """atom 序扁平 (…, rows*groups) → 线性 SF (…, rows, groups)。scatter 的逆。"""
    off = sf_atom_offsets(rows, groups, device=sf_atom.device)
    flat = torch.gather(
        sf_atom, -1, off.reshape(-1).expand(*sf_atom.shape[:-1], -1)
    )
    return flat.reshape(*sf_atom.shape[:-1], rows, groups)


def kv_tile_page(kv_indices, kv_indptr, b: int, tile: int) -> int:
    """prefill 的第 ``tile`` 个 KV tile(每个 128 token)所在的**全局页号**。

    这是 ``kv_tile == page_size == 128`` 换来的性质:**每个 KV tile 恰好落在
    一页内**,于是取址退化成一次页表查表 —— 没有跨页拼接、没有页内偏移、
    每 tile 一次 TMA。prefill kernel 的分页取址就建立在这一条上。

    (decode 那边 ``sequence_tile=256`` 跨两页,所以它要处理页内分段;prefill
    不需要。设计文档 §3 记的"page=128 的又一处红利"就是指这个。)

    ``tile`` 越界(超过该请求的页数)时返回 ``-1`` —— 与 kernel 侧的 OOB
    哨兵约定一致。
    """
    lo, hi = int(kv_indptr[b]), int(kv_indptr[b + 1])
    return int(kv_indices[lo + tile]) if 0 <= tile < hi - lo else -1


@dataclass
class NvFp4PagedKVPool:
    """Zero-copy descriptor consumed by the decode kernels.

    Production uses layout_profile="trtllm": K/V data have shape
    [pages, heads, 128, D/2], scales [pages, heads, 128, D/16].
    vLLM owns allocation and writes; data/scales retain its storage strides.
    The native_token profile is retained for research kernels. Its allocator
    and V staging writer live in resattn.kv_reference / append.
    """

    num_kv_heads: int
    head_dim: int
    k_data: torch.Tensor
    v_data: torch.Tensor
    k_sf: torch.Tensor
    v_sf: torch.Tensor
    v_staging: torch.Tensor
    page_size: int = PAGE_SIZE
    # Explicit because native-token and TRT-LLM profiles carry the same dtype
    # and byte count but assign different meanings to V and scale offsets.
    layout_profile: str = "native_token"
    k_global_scale: float = 1.0
    v_global_scale: float = 1.0
