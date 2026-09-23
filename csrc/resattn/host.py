# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'ResAttn conventional-V decode host.'

from __future__ import annotations

import functools
import math
import os
from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from .kernel_paged_decode import GroupedQueryAttentionDecodePagedFp4
from .kv_layout import (
    PAGE_SIZE,
    SF_VEC,
    NvFp4PagedKVPool,
    sf_atom_scatter,
)

from .plan import QMUL4_KERNEL_REDUCTION_MIN_SEQ, select_decode_plan

E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SFP = 0.171875  # P 的常数 SF(e4m3 精确)
N_PAD = 128  # Q-SF 缓冲的 N 垫高(UTCCP 按 N=128 粒度)

Qmul4ReductionWorkspace = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Qmul4GmemReductionWorkspace = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]














GQA_TILE_N = 16
N_SUPPORTED = (16, 32)
RESIDUE_GQA_GROUPS = (4, 6, 8)


def _npo2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _padded_group_size(num_qo_heads: int, num_kv_heads: int) -> int:
    """每 KV 头实际送进 kernel 的 Q 行数(≥ 真实 GQA 比,取 2 的幂)。"""
    return max(GQA_TILE_N, min(32, _npo2(num_qo_heads // num_kv_heads)))


def _residue_padded_group_size(
    num_qo_heads: int, num_kv_heads: int
) -> int:
    """Physical base+residue rows per KV head for the residue tactic."""
    if num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    g = num_qo_heads // num_kv_heads
    if g == 4:
        return 8
    if g in (6, 8):
        return 16
    raise NotImplementedError(
        f"residue Q supports logical GQA ratios {RESIDUE_GQA_GROUPS}, got {g}"
    )


def _pick_tile_shape(group_size: int, prediction: int) -> Tuple[int, int]:
    """group_size 已是垫过的 2 的幂;返回 (grouped_head_tile, prediction_tile)。"""
    grouped_head_tile = min(32, group_size)
    # N 上限 32:colmax 协议是"一 lane 一列",warp 只有 32 lane(dev LOG 迭代 12)
    prediction_tile = min(32 // grouped_head_tile, _npo2(prediction))
    return grouped_head_tile, prediction_tile


def check_shape_supported(num_qo_heads: int, num_kv_heads: int, head_dim: int,
                          prediction: int = 1) -> None:
    """编译前把不支持的形状变成一条可操作的错误(而非底层 enum/index 崩溃)。"""
    if num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError(
            f"num_qo_heads({num_qo_heads}) 必须是 num_kv_heads({num_kv_heads}) 的倍数"
        )
    # The linear K-scale TMA path stages one 128-D slice. D=256 needs a
    # tiled scale transfer and the direct gmem reducer also needs two output
    # blocks per correction thread. Neither has landed yet, so reject it at
    # the public boundary instead of failing deep inside CuTe compilation.
    if head_dim != 128:
        raise NotImplementedError(
            f"ResAttn fp4 decode currently supports head_dim=128 only; "
            f"got {head_dim}. D=256 requires tiled linear K-scale loads and "
            "a wider direct-output reducer."
        )
    ght, pt = _pick_tile_shape(
        _padded_group_size(num_qo_heads, num_kv_heads), prediction
    )
    n = ght * pt
    if n not in N_SUPPORTED:
        raise NotImplementedError(
            f"ResAttn fp4 decode 的 v2 几何只验证过 N∈{N_SUPPORTED},当前 N={n} "
            f"(GQA 比 {num_qo_heads // num_kv_heads} → 垫到 {ght},q_len {prediction} "
            f"→ {pt})。"
        )


@functools.lru_cache(maxsize=8)
def _sm_count(device_str: str) -> int:
    return torch.cuda.get_device_properties(device_str).multi_processor_count


def plan_qmul4_decode(batch, num_kv_heads, max_kv_len, device):
    """BF16 output wrapper over the shared split/reduction policy."""
    return select_decode_plan(
        batch, num_kv_heads, max_kv_len, _sm_count(str(device))
    )


def plan_qmul4_decode_nvfp4(batch, num_qo_heads, num_kv_heads, max_kv_len,
                           device, *, residue_fold=False):
    """Packed output wrapper supplying the physical Q tile to the policy."""
    grouped_head_tile = (
        _residue_padded_group_size(num_qo_heads, num_kv_heads)
        if residue_fold else _padded_group_size(num_qo_heads, num_kv_heads)
    )
    return select_decode_plan(
        batch, num_kv_heads, max_kv_len, _sm_count(str(device)),
        output_nvfp4=True, grouped_head_tile=grouped_head_tile,
        residue_fold=residue_fold,
    )


def _use_single_softmax_scheduler(
    grouped_head_tile: int,
    reduction: str,
    v_mode: str,
    kv_splits: int,
) -> bool:
    """Select the one-softmax-WG scheduler for short Q8/Q16 CTAs.

    Paired B200 A/B shows that one softmax WG preserves the direct-NVFP4 byte
    contract and lowers single-call latency for no-split, split4, and split8.
    Long split16/32 CTAs retain the second softmax WG: their longer per-CTA
    loop can use it to overlap score consumption with the KQ/VP pipeline.
    """
    return (
        grouped_head_tile in (8, 16)
        and v_mode == "qmul4_tmem"
        and (
            (reduction == "none" and kv_splits == 1)
            or (reduction == "gmem_fused" and kv_splits in (4, 8))
        )
    )


def allocate_qmul4_reduction_workspace(
    splits: int,
    batch: int,
    num_qo_heads: int,
    head_dim: int,
    device,
    *,
    partial_o_dtype: torch.dtype = torch.float32,
) -> Qmul4ReductionWorkspace:
    """Allocate persistent partials for the two-kernel split-K path.

    ``partial_o_dtype=torch.bfloat16`` mirrors TRT-LLM Gen's multi-CTA
    contract: the much larger partial-O tensor is 16-bit while the online
    softmax max/sum statistics remain FP32.  FP32 stays the default until the
    paired B200 numerical and latency gates promote the narrower contract.
    """
    if partial_o_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "partial_o_dtype must be torch.float32 or torch.bfloat16, "
            f"got {partial_o_dtype}"
        )
    o_partial = torch.empty(
        splits,
        batch,
        1,
        num_qo_heads,
        head_dim,
        dtype=partial_o_dtype,
        device=device,
    )
    stats_shape = (splits, batch, 1, num_qo_heads)
    l_partial = torch.empty(stats_shape, dtype=torch.float32, device=device)
    m_partial = torch.empty_like(l_partial)
    return o_partial, l_partial, m_partial


def allocate_qmul4_gmem_reduction_workspace(
    splits: int,
    batch: int,
    num_qo_heads: int,
    head_dim: int,
    device,
    *,
    partial_o_dtype: torch.dtype = torch.bfloat16,
) -> Qmul4GmemReductionWorkspace:
    """Allocate TRT mode-1 partials plus a persistent completion slab.

    The final CTA resets each used counter to zero before the selected reducer
    CTAs read their output slices, so this allocation is reusable across decode
    steps without a memset launch.  ``batch * num_qo_heads`` intentionally
    over-allocates the current Q16 tile count while keeping indexing stable
    for padded GQA groups.
    """
    partials = allocate_qmul4_reduction_workspace(
        splits,
        batch,
        num_qo_heads,
        head_dim,
        device,
        partial_o_dtype=partial_o_dtype,
    )
    completion = torch.zeros(
        batch * num_qo_heads,
        dtype=torch.int32,
        device=device,
    )
    return (*partials, completion)


def _reduction_workspace_partial_o_dtype(
    reduction_workspace: (
        Qmul4ReductionWorkspace | Qmul4GmemReductionWorkspace | None
    ),
) -> str:
    """Return the CuTe specialization key for a caller-owned workspace."""
    if reduction_workspace is None:
        return "fp32"
    dtype = reduction_workspace[0].dtype
    if dtype is torch.float32:
        return "fp32"
    if dtype is torch.bfloat16:
        return "bf16"
    raise ValueError(
        "reduction workspace partial O must be float32 or bfloat16, "
        f"got {dtype}"
    )


def allocate_nvfp4_attention_output(
    batch: int,
    num_qo_heads: int,
    head_dim: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the exact packed-A and CUTLASS 128x4 SF buffers for O-proj."""
    hidden = num_qo_heads * head_dim
    if hidden % SF_VEC:
        raise ValueError(f"attention hidden size must be divisible by 16, got {hidden}")
    output = torch.empty(
        batch, hidden // 2, dtype=torch.uint8, device=device
    )
    rounded_m = ((batch + 127) // 128) * 128
    rounded_scale_n = ((hidden // SF_VEC + 3) // 4) * 4
    # Match vLLM create_fp4_scale_tensor: four E4M3 scale bytes are carried
    # by each int32, then exposed to attention as an FP8 dtype view.
    # The downstream NVFP4 GEMM reads a complete 128x4 scale tile, including
    # padding cells the attention epilogue does not write.  Keep those bytes
    # zero, matching vLLM's create_fp4_scale_tensor contract.
    output_scale_i32 = torch.zeros(
        rounded_m,
        rounded_scale_n // 4,
        dtype=torch.int32,
        device=device,
    )
    return output, output_scale_i32.view(torch.float8_e4m3fn)


@functools.cache
def _compiled(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    prediction: int,
    reduction: str,
    v_mode: str = "native_token",
    qmul4_skip: str = "none",
    v_fp8: bool = False,
    k_sf_mode: str = "atom",
    output_nvfp4: bool = False,
    output_num_qo_heads: int | None = None,
    reg_reconfig_q16: bool = False,
    single_softmax_wg: bool = False,
    partial_o_dtype: str = "fp32",
    smem_correction_broadcast: bool = False,
    blocks_per_sm: int = 1,
    residue_fold: bool = False,
    native_qmul4_ptxas: str | None = None,
    qmul4_interleave: int = 0,
):
    """num_qo_heads 此处是**垫后**的头数(= num_kv_heads × padded group size)。"""
    # Keep the SASS-only marker contract local to this specialization.  The
    # process-wide loader hook is idempotent, but leaving N6420 enabled would
    # reinterpret unrelated RZ instructions in kernels compiled afterwards.
    _n6420_had_value = "RESATTN_QMUL4_N6420_PATCH" in os.environ
    _n6420_old_value = os.environ.get("RESATTN_QMUL4_N6420_PATCH")
    if v_mode == "qmul4_tmem":
        from .qmul4_cubin_patch import install_qmul4_loader_patch

        install_qmul4_loader_patch()
    grouped_head_tile, prediction_tile = _pick_tile_shape(
        num_qo_heads // num_kv_heads, prediction
    )
    if output_num_qo_heads is None:
        output_num_qo_heads = num_qo_heads
    if (
        output_num_qo_heads <= 0
        or output_num_qo_heads > num_qo_heads
        or output_num_qo_heads % num_kv_heads
    ):
        raise ValueError(
            "output_num_qo_heads must be a positive multiple of H_kv no "
            f"larger than padded H_q; got {output_num_qo_heads}, "
            f"H_q={num_qo_heads}, H_kv={num_kv_heads}"
        )
    fmha = GroupedQueryAttentionDecodePagedFp4(
        head_dim,
        grouped_head_tile,
        prediction_tile=prediction_tile,
        sequence_tile=256,
        reduction_mode=reduction,  # type: ignore[arg-type]
        page_size=128,
        v_mode=v_mode,  # type: ignore[arg-type]
        k_sf_mode=k_sf_mode,  # type: ignore[arg-type]
        logical_grouped_heads=(
            output_num_qo_heads // num_kv_heads
            if output_nvfp4 or output_num_qo_heads != num_qo_heads
            else None
        ),
        reg_reconfig_q16=reg_reconfig_q16,
        single_softmax_wg=single_softmax_wg,
        smem_correction_broadcast=smem_correction_broadcast,
        qmul4_interleave=qmul4_interleave,
        residue_fold=residue_fold,
        qmul4_skip=qmul4_skip,  # type: ignore[arg-type]
        
        
        
        
        blocks_per_sm=blocks_per_sm,
    )
    in_dtype = cutlass.Float4E2M1FN
    if partial_o_dtype not in ("fp32", "bf16"):
        raise ValueError(
            "partial_o_dtype must be 'fp32' or 'bf16', "
            f"got {partial_o_dtype!r}"
        )
    sym_batch = cute.sym_int()
    sym_pages = cute.sym_int()
    sym_total_logical = cute.sym_int()
    sym_prediction = cute.sym_int()
    q_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype, (sym_batch, sym_prediction, num_qo_heads, head_dim),
        stride_order=(3, 2, 1, 0), assumed_align=16)
    # 池张量用**符号步长**编译,不能用 compact:vLLM 把 KV 池切成
    # [data|scale] 的分侧视图后,页步长是 2*H*ps*full_dim 而非 H*ps*(D/2),
    # compact 编出来的 kernel 在调用时报 "Mismatched k_bshd.strides[0]"。
    # divisibility=2:fp4 一字节两元素,任何字节步长换算成元素步长必为偶数 ——
    # CuTeDSL 需要这条可证性质才能为打包类型生成寻址。
    _sd2 = lambda: cute.sym_int(divisibility=2)
    # K:逻辑 (P, ps, H, D),head_dim 最内(stride 1)
    k_fake = cute.runtime.make_fake_tensor(
        in_dtype, (sym_pages, 128, num_kv_heads, head_dim),
        stride=(_sd2(), _sd2(), _sd2(), 1), assumed_align=16)
    sf_flat = lambda: cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (cute.sym_int(),), stride_order=(0,), assumed_align=16)
    if k_sf_mode == "linear":
        # vLLM's split view aliases a [data | scale] side inside a larger
        # per-page byte pool.  Each (page, head) scale tile is contiguous, but
        # the page stride includes K/V data and the other side, so it must stay
        # symbolic rather than being flattened/copied.
        k_sf_fake = cute.runtime.make_fake_tensor(
            cutlass.Float8E4M3FN,
            (sym_pages, num_kv_heads, 128, head_dim // SF_VEC),
            stride=(
                cute.sym_int(divisibility=16),
                128 * (head_dim // SF_VEC),
                head_dim // SF_VEC,
                1,
            ),
            assumed_align=16,
        )
    else:
        k_sf_fake = sf_flat()
    if v_mode in ("qmul4_d", "qmul4_tmem"):
        # Conventional HND V: D is contiguous. The separate standard scale
        # tensor has one e4m3 byte per adjacent 16-D block.
        v_fake = cute.runtime.make_fake_tensor(
            in_dtype, (sym_pages, 128, num_kv_heads, head_dim),
            stride=(_sd2(), head_dim, _sd2(), 1), assumed_align=16)
        # 坑44:trtllm 契约 [P, H, ps, S] —— 每 (page,head) 的 SF tile 连续,
        # 才能开出 ≥16B 的合法 TMA box。线性契约 [P, ps, H, S] 在 H>1 时
        # tile 跨步、内跑只有 8B,描述符非法(真机 illegal instruction)。
        v_sf_fake = cute.runtime.make_fake_tensor(
            cutlass.Float8E4M3FN,
            (sym_pages, num_kv_heads, 128, head_dim // SF_VEC),
            stride=(
                cute.sym_int(divisibility=16),
                128 * (head_dim // SF_VEC),
                head_dim // SF_VEC,
                1,
            ),
            assumed_align=16,
        )
    else:
        # Native V: token is contiguous for blockscaled VP.
        # v_fp8=True 走 **Phase 1**(普通 fp8 VP MMA)—— 生产路径从不走它
        # (池里的 V 恒为 e2m1 ⇒ use_nvfp4_vp 恒 True),所以它长期无人使用、
        # 无门覆盖。qmul4_d 把 vp_dtype 设成 fp8 等于复活了这条路径,
        # 这个开关让它在**不带任何 qmul4 机器**的情况下单独受检。
        v_dt = cutlass.Float8E4M3FN if v_fp8 else in_dtype
        v_fake = cute.runtime.make_fake_tensor(
            v_dt, (sym_pages, 128, num_kv_heads, head_dim),
            stride=(_sd2(), 1, _sd2(), _sd2()), assumed_align=16)
        v_sf_fake = sf_flat()
    if output_nvfp4 and (
        reduction not in ("none", "kernel", "gmem_fused")
        or v_mode != "qmul4_tmem"
    ):
        raise ValueError(
            "fused NVFP4 output requires qmul4_tmem and no reduction, the "
            "deterministic kernel reducer, or the fused last-CTA reducer"
        )
    o_dtype = cutlass.Float4E2M1FN if output_nvfp4 else cutlass.BFloat16
    o_fake = cute.runtime.make_fake_compact_tensor(
        o_dtype,
        (sym_batch, sym_prediction, output_num_qo_heads, head_dim),
        stride_order=(3, 2, 1, 0), assumed_align=16)
    o_sf_fake = o_global_scale_fake = None
    if output_nvfp4:
        o_sf_fake = sf_flat()
        o_global_scale_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), stride_order=(0,), assumed_align=4
        )
    # The two-kernel split-K path writes configurable FP32/BF16 partial
    # numerators and FP32 statistics, then launches the small deterministic
    # combine kernel via PDL. Keep its
    # workspace dimensions symbolic so one compiled specialization can sweep
    # split counts above the 16-CTA cluster limit.  The normal none/atomic/CGA
    # paths retain the historical null workspace ABI.
    o_partial_fake = l_partial_fake = m_partial_fake = None
    completion_fake = None
    if reduction in ("kernel", "gmem_fused"):
        sym_splits = cute.sym_int()
        o_partial_fake = cute.runtime.make_fake_compact_tensor(
            (
                cutlass.Float32
                if partial_o_dtype == "fp32"
                else cutlass.BFloat16
            ),
            (
                sym_splits,
                sym_batch,
                sym_prediction,
                num_qo_heads,
                head_dim,
            ),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        partial_shape = (
            sym_splits,
            sym_batch,
            sym_prediction,
            num_qo_heads,
        )
        l_partial_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            partial_shape,
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        m_partial_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            partial_shape,
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        if reduction == "gmem_fused":
            completion_fake = cute.runtime.make_fake_compact_tensor(
                Int32,
                (cute.sym_int(),),
                stride_order=(0,),
                assumed_align=16,
            )
    i32 = lambda n: cute.runtime.make_fake_compact_tensor(Int32, (n,), assumed_align=16)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_options = "--enable-tvm-ffi --opt-level 3"
    from contextlib import nullcontext
    from .native_compile import native_qmul4_compilation
    if native_qmul4_ptxas is not None and v_mode != "qmul4_tmem":
        raise ValueError("native ptxas specialization requires qmul4_tmem")
    compile_scope = (
        native_qmul4_compilation(native_qmul4_ptxas)
        if native_qmul4_ptxas is not None else nullcontext()
    )
    if v_mode == "qmul4_tmem":
        os.environ["RESATTN_QMUL4_N6420_PATCH"] = "1"
    try:
        with compile_scope:
            return cute.compile(
                fmha, Int32(1), q_fake, k_fake, v_fake,
                sf_flat(), k_sf_fake, v_sf_fake, sf_flat(),
                o_fake, o_sf_fake, o_global_scale_fake,
                None, None, o_partial_fake, l_partial_fake, m_partial_fake,
                completion_fake,
                Float32(1.0), Float32(1.0), stream_fake, True, Int32(0),
                i32(sym_batch), i32(sym_batch), i32(sym_total_logical),
                options=compile_options,
            )
    finally:
        if v_mode == "qmul4_tmem":
            if _n6420_had_value:
                assert _n6420_old_value is not None
                os.environ["RESATTN_QMUL4_N6420_PATCH"] = _n6420_old_value
            else:
                os.environ.pop("RESATTN_QMUL4_N6420_PATCH", None)


@functools.lru_cache(maxsize=4)
def _psf_buffer(device_str: str):
    sfp_byte = int(torch.tensor(SFP).to(torch.float8_e4m3fn).view(torch.uint8).item())
    return torch.full((65536,), sfp_byte, dtype=torch.uint8, device=device_str).view(
        torch.float8_e4m3fn
    )













_Q_SCRATCH: dict = {}
_Q_RESIDUE_SCRATCH: dict = {}


def _q_scratch(b: int, h_kv: int, g: int, d: int, device):
    key = (b, h_kv, g, d, str(device))
    hit = _Q_SCRATCH.get(key)
    if hit is None:
        q_pack = torch.empty(b, h_kv, g, d // 2, dtype=torch.uint8, device=device)
        # 唯一一次清零:此后垫行不再被触碰
        q_sf = torch.zeros(
            b * h_kv * N_PAD * (d // SF_VEC), dtype=torch.uint8, device=device
        )
        hit = (q_pack, q_sf)
        _Q_SCRATCH[key] = hit
    return hit


def _q_residue_scratch(b: int, h_kv: int, g: int, d: int, device):
    """Persistent outputs for the grouped residue-Q producer.

    GQA4 uses one physical Q8 tile; GQA6/8 use Q16. The first half contains
    base rows and the second half residue rows, with per-half tail padding. Scale
    factors use one independently padded 128-row atom per group, matching the
    decode kernel's (batch, kv_head) outer mode. ``g`` is part of the cache key
    so untouched padding can never retain data from another geometry.
    """
    physical_g = _residue_padded_group_size(h_kv * g, h_kv)
    key = (b, h_kv, g, physical_g, d, str(device))
    hit = _Q_RESIDUE_SCRATCH.get(key)
    if hit is None:
        q_pack = torch.zeros(
            b, h_kv, physical_g, d // 2, dtype=torch.uint8, device=device
        )
        q_sf = torch.zeros(
            b * h_kv * N_PAD * (d // SF_VEC),
            dtype=torch.uint8,
            device=device,
        )
        hit = (q_pack, q_sf)
        _Q_RESIDUE_SCRATCH[key] = hit
    return hit


def _quantize_q_cuda(
    q: torch.Tensor,
    num_kv_heads: int,
    out_to_zero: torch.Tensor | None = None,
):
    """融合 CUDA 版(torch.ops.nvfp4.nvfp4_quantize_q);不可用时返回 None。

    与 ``_quantize_q`` 逐字节等价(同一比较树 / e4m3 舍入),但省掉 torch 版
    的 ~6 次 kernel launch —— decode 时 q 很小,launch 开销与注意力本身同量级。
    """
    if not (
        q.is_cuda
        and hasattr(torch.ops, "nvfp4")
        and hasattr(torch.ops.nvfp4, "nvfp4_quantize_q")
    ):
        return None
    B, HQ, D = q.shape
    g = HQ // num_kv_heads
    q4 = q.reshape(B, num_kv_heads, g, D).contiguous()
    q_pack, q_sf = _q_scratch(B, num_kv_heads, g, D, q.device)
    if (
        out_to_zero is not None
        and hasattr(torch.ops.nvfp4, "nvfp4_quantize_q_and_zero")
    ):
        torch.ops.nvfp4.nvfp4_quantize_q_and_zero(
            q4, q_pack, q_sf, out_to_zero
        )
    else:
        torch.ops.nvfp4.nvfp4_quantize_q(q4, q_pack, q_sf)
        if out_to_zero is not None:
            # Compatibility fallback for an older extension binary.
            out_to_zero.zero_()
    return q_pack.reshape(B, 1, HQ, D // 2), q_sf.view(torch.float8_e4m3fn)


def _quantize_q_residue_cuda(q: torch.Tensor, num_kv_heads: int):
    """GQA4/6/8 Q -> Q8/Q16 `[base; residue]` in one CUDA launch.

    GQA6 deliberately uses the same physical Q16 tile as GQA8: rows 0..5 are
    base, 6..7 are zero, 8..13 are residue, and 14..15 are zero. GQA4 follows
    the analogous 4+4 live-row mapping. The dedicated op reuses MExt pack16
    quantization math but reads only logical Q, avoiding a BF16 pad/copy.
    """
    if not (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and hasattr(torch.ops, "nvfp4")
    ):
        return None
    B, HQ, D = q.shape
    if HQ % num_kv_heads:
        raise ValueError(f"H_q={HQ} must be divisible by H_kv={num_kv_heads}")
    g = HQ // num_kv_heads
    if g not in RESIDUE_GQA_GROUPS:
        raise NotImplementedError(
            "decode residue Q8/Q16 supports logical GQA ratios "
            f"{RESIDUE_GQA_GROUPS}, got {g}"
        )
    q4 = q.reshape(B, num_kv_heads, g, D).contiguous()
    q_pack, q_sf = _q_residue_scratch(
        B, num_kv_heads, g, D, q.device
    )
    physical_g = q_pack.shape[2]
    if hasattr(torch.ops.nvfp4, "nvfp4_quantize_q_residue"):
        torch.ops.nvfp4.nvfp4_quantize_q_residue(q4, q_pack, q_sf)
    elif (
        g == 8
        and physical_g == 16
        and hasattr(torch.ops.nvfp4, "scaled_fp4_quant_mext_r1")
    ):
        # Compatibility with an older extension. Flattened MExt attn_tile8 is
        # correct only when every logical group already contains eight rows.
        input_scale = torch.ones(1, dtype=torch.float32, device=q.device)
        torch.ops.nvfp4.scaled_fp4_quant_mext_r1(
            q_pack, q4.reshape(B * num_kv_heads * g, D), q_sf,
            input_scale, 0, 4
        )
    else:
        return None
    return (
        q_pack.reshape(B, 1, num_kv_heads * physical_g, D // 2),
        q_sf.view(torch.float8_e4m3fn),
    )


def _quantize_q(q: torch.Tensor, num_kv_heads: int):
    """bf16 [B, H_q, D] → (q_pack uint8 [B,1,H_q,D/2], q_sf e4m3 扁平 atom 序)。

    torch 参考实现,也是 CUDA 版的语义基准(amax/6 → e4m3,无全局缩放;
    每 (b, h_kv) 一个 N_PAD=128 行的 SF tile,实际行数 = GQA 组内头数)。
    """
    B, HQ, D = q.shape
    g = HQ // num_kv_heads
    dev = q.device
    lut = torch.tensor(E2M1_LUT, device=dev)
    edges = (lut[1:] + lut[:-1]) / 2
    x = q.float().reshape(B, num_kv_heads, g, D // SF_VEC, SF_VEC)
    sf = (x.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn)
    sf_f = sf.float().clamp_min(1e-12)
    y = (x / sf_f.unsqueeze(-1)).clamp(-6, 6)
    idx = torch.bucketize(y.abs().contiguous(), edges)
    nib = (idx + torch.where(y < 0, 8, 0)).to(torch.uint8).reshape(B, num_kv_heads, g, D)
    q_pack = (nib[..., 0::2] | (nib[..., 1::2] << 4)).reshape(B, 1, HQ, D // 2).contiguous()
    # SF → N_PAD 行的 atom 序 tile(垫零)
    sf_pad = torch.zeros(B, num_kv_heads, N_PAD, D // SF_VEC, dtype=torch.uint8, device=dev)
    sf_pad[:, :, :g] = sf.view(torch.uint8)
    q_sf = sf_atom_scatter(sf_pad).reshape(-1)
    return q_pack, q_sf.view(torch.float8_e4m3fn)


def run_paged_fp4_decode_qmul4_v(
    q: torch.Tensor,
    pool: NvFp4PagedKVPool,
    v_data_hnd: torch.Tensor,
    v_sf_bthd: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    out: torch.Tensor,
    max_seq_len: int = 0,
    qmul4_skip: str = "none",
    force_single_split: bool = False,
    v_mode: str = "qmul4_d",
    k_sf_mode: str = "atom",
    value_scale: float = 1.0,
    reduction_workspace: Qmul4ReductionWorkspace | None = None,
    residue_fold: bool = False,
    direct_output: bool = False,
    plan: tuple[int, str] | None = None,
) -> torch.Tensor:
    """Experimental conventional-V decode path.

    K and K scales still come from ``pool`` (the native blockscaled KQ
    contract). V uses the context/TRT-friendly contract:

    - ``v_data_hnd``: packed uint8 ``[page, head, token, D/2]``;
    - ``v_sf_bthd``: e4m3 ``[page, head, token, D/16]``,tile 内按 TRT-LLM 的
      4-token interleave 排布(``append.quantize_paged_trtllm_format`` 产出)。

    The decode kernel expands each V page to E4M3 in shared memory with
    QMUL4, then executes the ordinary FP8 VP MMA. This wrapper is kept
    separate from the production native-token entry until the B200 numerical
    and performance gates are complete.
    """
    B, HQ, D = q.shape
    H = pool.num_kv_heads
    g = HQ // H
    check_shape_supported(HQ, H, D, prediction=1)
    if v_data_hnd.shape != (pool.k_data.shape[0], H, PAGE_SIZE, D // 2):
        raise ValueError(
            "v_data_hnd must be [page, head, 128, D/2], got "
            f"{tuple(v_data_hnd.shape)}"
        )
    if v_sf_bthd.shape != (pool.k_data.shape[0], H, PAGE_SIZE, D // SF_VEC):
        raise ValueError(
            "v_sf_bthd must be [page, head, 128, D/16] (trtllm contract), got "
            f"{tuple(v_sf_bthd.shape)}"
        )

    g_pad = (
        _residue_padded_group_size(HQ, H)
        if residue_fold
        else _padded_group_size(HQ, H)
    )
    if residue_fold and (
        g not in RESIDUE_GQA_GROUPS or g_pad not in (8, 16)
    ):
        raise NotImplementedError(
            "decode QK residue requires logical GQA ratio in "
            f"{RESIDUE_GQA_GROUPS} and physical tile 8/16, got "
            f"g={g}, g_pad={g_pad}"
        )
    if direct_output:
        if not residue_fold:
            raise ValueError("direct_output requires QK residue")
        if (tuple(out.shape) != (B, HQ, D) or out.dtype != torch.bfloat16
                or out.device != q.device or not out.is_contiguous()):
            raise ValueError("direct_output requires contiguous BF16 [B, H_q, D]")
        q_eff, out_eff = q, out
    elif residue_fold:
        q_eff = q
        out_eff = torch.zeros(
            B, H * g_pad, D, dtype=torch.bfloat16, device=q.device
        )
    elif g_pad != g:
        q_eff = q.new_zeros(B, H * g_pad, D)
        q_eff.view(B, H, g_pad, D)[:, :, :g, :] = q.view(B, H, g, D)
        out_eff = torch.zeros(B, H * g_pad, D, dtype=torch.bfloat16, device=q.device)
    else:
        q_eff, out_eff = q, out

    # The caller may already have selected this exact plan to size its
    # reduction workspace. Standalone calls retain the original selector.
    if plan is None:
        max_len = max_seq_len or int(seq_lens.max().item())
        splits, reduction = plan_qmul4_decode(B, H, max_len, q.device)
    else:
        splits, reduction = plan
    if force_single_split:
        # 诊断用:关掉 split-K。split 下 kernel 按**每 split 局部 max** 归一化
        # 后量化 P,参考用全局 max —— 量化标度不同,残差自然不同。
        splits, reduction = 1, "none"
    if direct_output and reduction not in ("none", "atomic", "kernel"):
        # CGA has unpredicated scalar stores to all physical heads; the
        # gmem-fused reducer currently has an NVFP4-specific output contract.
        raise NotImplementedError(f"direct BF16 output: unsupported {reduction=}")
    if v_mode not in ("qmul4_d", "qmul4_tmem"):
        raise ValueError(f"unsupported conventional-V mode: {v_mode}")
    if k_sf_mode not in ("atom", "linear"):
        raise ValueError(f"unsupported K-scale mode: {k_sf_mode}")
    single_softmax_wg = _use_single_softmax_scheduler(
        g_pad, reduction, v_mode, splits
    )
    f = _compiled(
        H * g_pad,
        H,
        D,
        1,
        reduction,
        v_mode,
        qmul4_skip,
        False,
        k_sf_mode,
        single_softmax_wg=single_softmax_wg,
        partial_o_dtype=_reduction_workspace_partial_o_dtype(
            reduction_workspace
        ),
        smem_correction_broadcast=single_softmax_wg,
        residue_fold=residue_fold,
        output_num_qo_heads=HQ if direct_output else None,
    )
    if residue_fold:
        fused = _quantize_q_residue_cuda(q_eff, H)
        if fused is None:
            raise RuntimeError(
                "decode QK residue requires the nvfp4 MExt CUDA extension"
            )
        q_pack, q_sf = fused
    else:
        fused = _quantize_q_cuda(
            q_eff, H, out_eff if reduction == "atomic" else None
        )
        q_pack, q_sf = fused if fused is not None else _quantize_q(q_eff, H)
    o_partial = l_partial = m_partial = None
    if direct_output and reduction == "none":
        # Empty/inactive rows skip the main kernel's output store. Preserve
        # the zero-row contract of the old padded torch.zeros allocation.
        # Atomic already clears below; the separate reducer writes zero.
        out_eff.zero_()
    if reduction == "atomic" and (fused is None or residue_fold):
        out_eff.zero_()
    elif reduction == "kernel":
        if reduction_workspace is None:
            reduction_workspace = allocate_qmul4_reduction_workspace(
                splits, B, H * g_pad, D, q.device
            )
        o_partial, l_partial, m_partial = reduction_workspace
        expected_shapes = (
            (splits, B, 1, H * g_pad, D),
            (splits, B, 1, H * g_pad),
            (splits, B, 1, H * g_pad),
        )
        for index, (name, tensor, shape) in enumerate(zip(
            ("o_partial", "l_partial", "m_partial"),
            reduction_workspace,
            expected_shapes,
        )):
            expected_dtypes = (
                (torch.float32, torch.bfloat16)
                if index == 0
                else (torch.float32,)
            )
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype not in expected_dtypes
                or tensor.device != q.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous with dtype in "
                    f"{expected_dtypes}, shape {shape}, on "
                    f"{q.device}, got shape={tuple(tensor.shape)} "
                    f"dtype={tensor.dtype} device={tensor.device}"
                )
    k_arg = pool.k_data.view(torch.float4_e2m1fn_x2).permute(0, 2, 1, 3)
    v_arg = v_data_hnd.view(torch.float4_e2m1fn_x2).permute(0, 2, 1, 3)
    f(
        Int32(splits),
        q_pack.view(torch.float4_e2m1fn_x2),
        k_arg,
        v_arg,
        q_sf,
        (
            pool.k_sf
            if k_sf_mode == "linear"
            else pool.k_sf.reshape(-1).view(torch.float8_e4m3fn)
        ),
        v_sf_bthd,
        _psf_buffer(str(q.device)),
        out_eff.view(B, 1, HQ if direct_output else H * g_pad, D),
        None, None, None, None, o_partial, l_partial, m_partial, None,
        Float32(sm_scale),
        Float32(value_scale),
        v_mode != "qmul4_tmem",
        Int32(0),
        seq_lens.int().contiguous(),
        kv_indptr[:B].int().contiguous(),
        kv_indices.int().contiguous(),
    )
    if out_eff is not out:
        out.view(B, H, g, D).copy_(
            out_eff.view(B, H, g_pad, D)[:, :, :g, :]
        )
    return out


def plan_decode_compile_variants(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_model_len: int,
    max_num_seqs: int,
    device,
    residue_fold: bool = False,
) -> list[dict]:
    """Every distinct `_compiled` key `run_paged_fp4_decode_qmul4_tmem` can
    reach for this geometry, for warmup.

    Built with the SAME helpers the production entry uses -- `_padded_group_
    size`, `plan_qmul4_decode`, `_use_single_softmax_scheduler` -- so warmup
    compiles exactly what live requests will hit. A hand-assembled key that
    drifts anywhere compiles a variant nothing uses, and that failure is
    silent: warmup "succeeds" and the first request still pays the JIT.

    Candidate (batch, seq) points are derived from the ENGINE's limits, not
    hardcoded: `max_model_len` bounds the sequence axis (a 4K-context model
    must not probe 65536 and then print a failure banner every boot), and the
    kernel-reduction variant needs `batch * num_kv_heads == 1`
    (`plan_qmul4_decode`), so batch=1 is always probed. Points that select
    the same key deduplicate; a geometry where a variant is unreachable (e.g.
    "kernel" with num_kv_heads > 1) simply never yields it -- which is
    correct, because serving cannot reach it either.

    `partial_o_dtype` is resolved through `_reduction_workspace_partial_o_
    dtype`: `None` and the fp32 workspace the backend allocates for the
    "kernel" reduction both map to "fp32", so the key matches production for
    every surviving reduction.
    """
    check_shape_supported(num_qo_heads, num_kv_heads, head_dim)
    if (
        residue_fold
        and num_qo_heads // num_kv_heads not in RESIDUE_GQA_GROUPS
    ):
        raise NotImplementedError(
            "decode QK residue warmup requires logical GQA ratio in "
            f"{RESIDUE_GQA_GROUPS}"
        )
    g_pad = (
        _residue_padded_group_size(num_qo_heads, num_kv_heads)
        if residue_fold
        else _padded_group_size(num_qo_heads, num_kv_heads)
    )
    seqs = sorted(
        {min(s, max_model_len)
         for s in (256, 4096, QMUL4_KERNEL_REDUCTION_MIN_SEQ, 65536)}
        | {max_model_len}
    )
    batches = sorted({1, max(1, min(8, max_num_seqs)),
                      max(1, min(64, max_num_seqs))})
    plans: list[dict] = []
    seen: set[tuple] = set()
    for batch in batches:
        for seq in seqs:
            splits, reduction = plan_qmul4_decode(
                batch, num_kv_heads, seq, device
            )
            ssw = _use_single_softmax_scheduler(
                g_pad, reduction, "qmul4_tmem", splits
            )
            args = (num_kv_heads * g_pad, num_kv_heads, head_dim, 1,
                    reduction, "qmul4_tmem", "none", False, "linear")
            kwargs = dict(
                single_softmax_wg=ssw,
                partial_o_dtype=_reduction_workspace_partial_o_dtype(None),
                smem_correction_broadcast=ssw,
                residue_fold=residue_fold,
            )
            key = args + tuple(sorted(kwargs.items()))
            if key in seen:
                continue
            seen.add(key)
            plans.append({
                "label": f"reduction={reduction} ssw={ssw} "
                         f"(batch={batch}, seq={seq})",
                "args": args,
                "kwargs": kwargs,
            })
    return plans


def plan_decode_nvfp4_compile_variants(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_model_len: int,
    max_num_seqs: int,
    device,
    residue_fold: bool = False,
) -> list[dict]:
    """Direct-NVFP4 ``_compiled`` keys reachable by the decode planner."""
    check_shape_supported(num_qo_heads, num_kv_heads, head_dim)
    if (
        residue_fold
        and num_qo_heads // num_kv_heads not in RESIDUE_GQA_GROUPS
    ):
        raise NotImplementedError(
            "decode QK residue warmup requires logical GQA ratio in "
            f"{RESIDUE_GQA_GROUPS}"
        )
    g_pad = (
        _residue_padded_group_size(num_qo_heads, num_kv_heads)
        if residue_fold
        else _padded_group_size(num_qo_heads, num_kv_heads)
    )
    seqs = sorted(
        {
            min(s, max_model_len)
            for s in (256, 1024, 2048, 4096, 8192, 65536)
        }
        | {max_model_len}
    )
    batches = sorted(
        {1, max(1, min(8, max_num_seqs)), max(1, min(64, max_num_seqs))}
    )
    plans: list[dict] = []
    seen: set[tuple] = set()
    for batch in batches:
        for seq in seqs:
            splits, reduction = plan_qmul4_decode_nvfp4(
                batch,
                num_qo_heads,
                num_kv_heads,
                seq,
                device,
                residue_fold=residue_fold,
            )
            if reduction not in ("none", "kernel", "gmem_fused"):
                continue
            ssw = _use_single_softmax_scheduler(
                g_pad, reduction, "qmul4_tmem", splits
            )
            args = (
                num_kv_heads * g_pad,
                num_kv_heads,
                head_dim,
                1,
                reduction,
                "qmul4_tmem",
                "none",
                False,
                "linear",
                True,
                num_qo_heads,
            )
            kwargs = dict(
                single_softmax_wg=ssw,
                partial_o_dtype=(
                    "bf16" if reduction == "gmem_fused" else "fp32"
                ),
                smem_correction_broadcast=ssw,
                residue_fold=residue_fold,
            )
            key = args + tuple(sorted(kwargs.items()))
            if key in seen:
                continue
            seen.add(key)
            plans.append(
                {
                    "label": (
                        f"nvfp4 reduction={reduction} ssw={ssw} "
                        f"(batch={batch}, seq={seq})"
                    ),
                    "args": args,
                    "kwargs": kwargs,
                }
            )
    return plans


def compile_decode_variant(plan: dict) -> None:
    """Compile one plan from `plan_decode_compile_variants` (cached)."""
    _compiled(*plan["args"], **plan["kwargs"])


def run_paged_fp4_decode_qmul4_tmem(
    q: torch.Tensor,
    pool: NvFp4PagedKVPool,
    v_data_hnd: torch.Tensor,
    v_sf_bthd: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    out: torch.Tensor,
    max_seq_len: int = 0,
    qmul4_skip: str = "none",
    force_single_split: bool = False,
    k_sf_mode: str = "atom",
    value_scale: float = 1.0,
    reduction_workspace: Qmul4ReductionWorkspace | None = None,
    residue_fold: bool = False,
    direct_output: bool = False,
    plan: tuple[int, str] | None = None,
) -> torch.Tensor:
    """TRT-style conventional-V decode with converted V resident in TMEM."""
    return run_paged_fp4_decode_qmul4_v(
        q,
        pool,
        v_data_hnd,
        v_sf_bthd,
        kv_indices,
        kv_indptr,
        seq_lens,
        sm_scale,
        out,
        max_seq_len=max_seq_len,
        qmul4_skip=qmul4_skip,
        force_single_split=force_single_split,
        v_mode="qmul4_tmem",
        k_sf_mode=k_sf_mode,
        value_scale=value_scale,
        reduction_workspace=reduction_workspace,
        residue_fold=residue_fold,
        direct_output=direct_output,
        plan=plan,
    )


def run_paged_fp4_decode_qmul4_tmem_nvfp4(
    q: torch.Tensor,
    pool: NvFp4PagedKVPool,
    v_data_hnd: torch.Tensor,
    v_sf_bthd: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    output_global_scale_inv: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    max_seq_len: int = 0,
    k_sf_mode: str = "linear",
    value_scale: float = 1.0,
    reduction_workspace: (
        Qmul4ReductionWorkspace | Qmul4GmemReductionWorkspace | None
    ) = None,
    plan: tuple[int, str] | None = None,
    residue_fold: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode directly into the NVFP4 activation contract consumed by O-proj.

    No-split quantizes in the attention CTA.  The deterministic split-K path
    publishes FP32 partial numerators and quantizes only in its unique final
    reducer, after max/sum/O have been merged.  Atomic/CGA reduction is kept
    on the BF16 compatibility path because it has no single global reducer.

    Returns
    -------
    out
        Packed uint8 E2M1 tensor shaped ``[B, H_q * D / 2]``.
    output_block_scale
        E4M3 scale tensor in vLLM/CUTLASS's padded 128x4 physical layout.
    """
    if q.ndim != 3 or q.dtype != torch.bfloat16 or not q.is_cuda:
        raise ValueError("q must be CUDA BF16 [B, H_q, D]")
    B, HQ, D = q.shape
    H = pool.num_kv_heads
    if HQ % H:
        raise ValueError(f"H_q={HQ} must be divisible by H_kv={H}")
    check_shape_supported(HQ, H, D, prediction=1)
    g = HQ // H
    g_pad = (
        _residue_padded_group_size(HQ, H)
        if residue_fold
        else _padded_group_size(HQ, H)
    )
    if residue_fold and (
        g not in RESIDUE_GQA_GROUPS or g_pad not in (8, 16)
    ):
        raise NotImplementedError(
            "decode QK residue requires logical GQA ratio in "
            f"{RESIDUE_GQA_GROUPS} and physical tile 8/16, got "
            f"g={g}, g_pad={g_pad}"
        )
    if k_sf_mode not in ("atom", "linear"):
        raise ValueError(f"unsupported K-scale mode: {k_sf_mode}")
    if v_data_hnd.shape != (pool.k_data.shape[0], H, PAGE_SIZE, D // 2):
        raise ValueError(
            "v_data_hnd must be [page, head, 128, D/2], got "
            f"{tuple(v_data_hnd.shape)}"
        )
    if v_sf_bthd.shape != (pool.k_data.shape[0], H, PAGE_SIZE, D // SF_VEC):
        raise ValueError(
            "v_sf_bthd must be [page, head, 128, D/16], got "
            f"{tuple(v_sf_bthd.shape)}"
        )
    if (
        output_global_scale_inv.device != q.device
        or output_global_scale_inv.dtype != torch.float32
        or output_global_scale_inv.numel() != 1
    ):
        raise ValueError(
            "output_global_scale_inv must be a one-element CUDA float32 tensor"
        )

    if out is None or output_block_scale is None:
        if out is not None or output_block_scale is not None:
            raise ValueError("out and output_block_scale must be supplied together")
        out, output_block_scale = allocate_nvfp4_attention_output(
            B, HQ, D, q.device
        )
    expected_out = (B, HQ * D // 2)
    expected_scale = (
        ((B + 127) // 128) * 128,
        ((HQ * D // SF_VEC + 3) // 4) * 4,
    )
    if (
        tuple(out.shape) != expected_out
        or out.dtype != torch.uint8
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous uint8 {expected_out} on {q.device}")
    if (
        tuple(output_block_scale.shape) != expected_scale
        or output_block_scale.dtype != torch.float8_e4m3fn
        or output_block_scale.device != q.device
        or not output_block_scale.is_contiguous()
    ):
        raise ValueError(
            "output_block_scale must be contiguous E4M3 "
            f"{expected_scale} on {q.device}"
        )

    max_len = max_seq_len or int(seq_lens.max().item())
    if plan is None:
        splits, reduction = plan_qmul4_decode_nvfp4(
            B, HQ, H, max_len, q.device, residue_fold=residue_fold
        )
    else:
        splits, reduction = plan
        if splits < 1:
            raise ValueError(f"decode plan splits must be positive, got {splits}")
    if reduction not in ("none", "kernel", "gmem_fused"):
        raise NotImplementedError(
            "direct NVFP4 output supports split=1, deterministic kernel, "
            "or fused gmem "
            f"reduction; planner selected splits={splits}, reduction={reduction!r}"
        )

    # Allocate before selecting the compiled specialization so the kernel's
    # partial-O dtype always matches the workspace passed at launch.  In
    # particular, gmem_fused defaults to BF16 partials while a missing
    # workspace otherwise looks like FP32 to the specialization helper.
    if reduction_workspace is None and reduction in ("kernel", "gmem_fused"):
        if reduction == "gmem_fused":
            reduction_workspace = allocate_qmul4_gmem_reduction_workspace(
                splits, B, H * g_pad, D, q.device
            )
        else:
            reduction_workspace = allocate_qmul4_reduction_workspace(
                splits, B, H * g_pad, D, q.device
            )

    single_softmax_wg = _use_single_softmax_scheduler(
        g_pad, reduction, "qmul4_tmem", splits
    )
    f = _compiled(
        H * g_pad,
        H,
        D,
        1,
        reduction,
        "qmul4_tmem",
        "none",
        False,
        k_sf_mode,
        True,
        HQ,
        single_softmax_wg=single_softmax_wg,
        partial_o_dtype=_reduction_workspace_partial_o_dtype(
            reduction_workspace
        ),
        smem_correction_broadcast=single_softmax_wg,
        residue_fold=residue_fold,
    )
    if residue_fold:
        q_eff = q
    elif g_pad != g:
        q_eff = q.new_zeros(B, H * g_pad, D)
        q_eff.view(B, H, g_pad, D)[:, :, :g, :] = q.view(B, H, g, D)
    else:
        q_eff = q
    if residue_fold:
        fused_q = _quantize_q_residue_cuda(q_eff, H)
        if fused_q is None:
            raise RuntimeError(
                "decode QK residue requires the nvfp4 MExt CUDA extension"
            )
        q_pack, q_sf = fused_q
    else:
        fused_q = _quantize_q_cuda(q_eff, H)
        q_pack, q_sf = (
            fused_q if fused_q is not None else _quantize_q(q_eff, H)
        )
    k_arg = pool.k_data.view(torch.float4_e2m1fn_x2).permute(0, 2, 1, 3)
    v_arg = v_data_hnd.view(torch.float4_e2m1fn_x2).permute(0, 2, 1, 3)
    o_partial = l_partial = m_partial = completion_counter = None
    if reduction in ("kernel", "gmem_fused"):
        assert reduction_workspace is not None
        expected_workspace_len = 4 if reduction == "gmem_fused" else 3
        if len(reduction_workspace) != expected_workspace_len:
            raise ValueError(
                f"{reduction} workspace must contain "
                f"{expected_workspace_len} tensors, got "
                f"{len(reduction_workspace)}"
            )
        o_partial, l_partial, m_partial = reduction_workspace[:3]
        if reduction == "gmem_fused":
            completion_counter = reduction_workspace[3]
        expected_shapes = (
            (splits, B, 1, H * g_pad, D),
            (splits, B, 1, H * g_pad),
            (splits, B, 1, H * g_pad),
        )
        for index, (name, tensor, shape) in enumerate(zip(
            ("o_partial", "l_partial", "m_partial"),
            reduction_workspace,
            expected_shapes,
        )):
            expected_dtypes = (
                (torch.float32, torch.bfloat16)
                if index == 0
                else (torch.float32,)
            )
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype not in expected_dtypes
                or tensor.device != q.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous with dtype in "
                    f"{expected_dtypes}, shape {shape}, on "
                    f"{q.device}, got shape={tuple(tensor.shape)} "
                    f"dtype={tensor.dtype} device={tensor.device}"
                )
        if reduction == "gmem_fused" and (
            tuple(completion_counter.shape) != (B * H * g_pad,)
            or completion_counter.dtype != torch.int32
            or completion_counter.device != q.device
            or not completion_counter.is_contiguous()
        ):
            raise ValueError(
                "gmem_fused completion counter must be contiguous int32 "
                f"{(B * H * g_pad,)} on {q.device}"
            )
    f(
        Int32(splits),
        q_pack.view(torch.float4_e2m1fn_x2),
        k_arg,
        v_arg,
        q_sf,
        (
            pool.k_sf
            if k_sf_mode == "linear"
            else pool.k_sf.reshape(-1).view(torch.float8_e4m3fn)
        ),
        v_sf_bthd,
        _psf_buffer(str(q.device)),
        out.view(B, 1, HQ, D // 2).view(torch.float4_e2m1fn_x2),
        output_block_scale.reshape(-1),
        output_global_scale_inv.reshape(1),
        None,
        None,
        o_partial,
        l_partial,
        m_partial,
        completion_counter,
        Float32(sm_scale),
        Float32(value_scale),
        False,
        Int32(0),
        seq_lens.int().contiguous(),
        kv_indptr[:B].int().contiguous(),
        kv_indices.int().contiguous(),
    )
    return out, output_block_scale
