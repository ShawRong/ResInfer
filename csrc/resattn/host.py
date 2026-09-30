# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'ResAttn conventional-V decode host.'

from __future__ import annotations

import functools
import math
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
)


from .plan import select_decode_plan, static_compile_plans

N_PAD = 128  # Q-SF 缓冲的 N 垫高(UTCCP 按 N=128 粒度)

Qmul4ReductionWorkspace = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Qmul4GmemReductionWorkspace = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


# Physical Q tiles store base+residue rows; GQA4 uses Q8, GQA6/8 use Q16.
RESIDUE_GQA_GROUPS = (4, 6, 8)






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
    if group_size not in (8, 16) or prediction != 1:
        raise NotImplementedError("production ResAttn requires single-token Q8/Q16")
    return group_size, 1


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
    _residue_padded_group_size(num_qo_heads, num_kv_heads)
    if prediction != 1:
        raise NotImplementedError("production ResAttn supports single-token decode only")


@functools.lru_cache(maxsize=8)
def _sm_count(device_str: str) -> int:
    return torch.cuda.get_device_properties(device_str).multi_processor_count


def plan_qmul4_decode(batch, num_kv_heads, max_kv_len, device, *,
                       grouped_head_tile=16, residue_fold=True):
    """BF16 output wrapper over the shared split/reduction policy."""
    if not residue_fold:
        raise NotImplementedError("ordinary QK moved to .dev/tools/resattn/experimental")
    return select_decode_plan(
        batch, num_kv_heads, max_kv_len, _sm_count(str(device)),
        grouped_head_tile=grouped_head_tile, residue_fold=residue_fold,
    )


def plan_qmul4_decode_nvfp4(batch, num_qo_heads, num_kv_heads, max_kv_len,
                           device, *, residue_fold=True):
    """Packed output wrapper supplying the physical Q tile to the policy."""
    if not residue_fold:
        raise NotImplementedError("ordinary QK moved to .dev/tools/resattn/experimental")
    grouped_head_tile = (
        _residue_padded_group_size(num_qo_heads, num_kv_heads)
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
    v_mode: str = "qmul4_tmem",
    k_sf_mode: str = "linear",
    output_nvfp4: bool = False,
    output_num_qo_heads: int | None = None,
    single_softmax_wg: bool = False,
    partial_o_dtype: str = "fp32",
    smem_correction_broadcast: bool = False,
    residue_fold: bool = True,
):
    """num_qo_heads 此处是**垫后**的头数(= num_kv_heads × padded group size)。"""
    if not residue_fold:
        raise NotImplementedError("ordinary QK moved to .dev/tools/resattn/experimental")
    if k_sf_mode != "linear":
        raise NotImplementedError("production ResAttn uses linear K scales")
    if reduction not in ("none", "kernel", "gmem_fused", "cga_sharded"):
        raise ValueError(f"unsupported production reducer: {reduction}")
    if v_mode != "qmul4_tmem":
        raise ValueError("production decode supports qmul4_tmem only")
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
        logical_grouped_heads=(
            output_num_qo_heads // num_kv_heads
            if output_nvfp4 or output_num_qo_heads != num_qo_heads
            else None
        ),
        single_softmax_wg=single_softmax_wg,
        smem_correction_broadcast=smem_correction_broadcast,
    )
    in_dtype = cutlass.Float4E2M1FN
    if partial_o_dtype not in ("fp32", "bf16"):
        raise ValueError(
            "partial_o_dtype must be 'fp32' or 'bf16', "
            f"got {partial_o_dtype!r}"
        )
    sym_batch = cute.sym_int()
    sym_pages = cute.sym_int()
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
    # Conventional HND V: D is contiguous. Each scale byte covers 16 D.
    v_fake = cute.runtime.make_fake_tensor(
        in_dtype, (sym_pages, 128, num_kv_heads, head_dim),
        stride=(_sd2(), head_dim, _sd2(), 1), assumed_align=16)
    # TRT-LLM contract [P, H, ps, S] keeps each scale tile contiguous for TMA.
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
    if output_nvfp4 and reduction not in ("none", "kernel", "gmem_fused"):
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
    # split counts above the 16-CTA cluster limit.  The normal none/CGA
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
    seq_lens_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_batch,), assumed_align=4,
    )
    block_tables_fake = cute.runtime.make_fake_tensor(
        Int32, (sym_batch, cute.sym_int()),
        stride=(cute.sym_int(), 1), assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_options = "--enable-tvm-ffi --opt-level 3"
    return cute.compile(
        fmha, Int32(1), q_fake, k_fake, v_fake,
        sf_flat(), k_sf_fake, v_sf_fake,
        o_fake, o_sf_fake, o_global_scale_fake,
        None, o_partial_fake, l_partial_fake, m_partial_fake,
        completion_fake,
        Float32(1.0), Float32(1.0), seq_lens_fake, block_tables_fake,
        stream_fake, True,
        options=compile_options,
    )












_Q_RESIDUE_SCRATCH: dict = {}


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
    if not hasattr(torch.ops.nvfp4, "scaled_fp4_quant_mext_r1_grouped_q"):
        return None
    torch.ops.nvfp4.scaled_fp4_quant_mext_r1_grouped_q(q4, q_pack, q_sf)
    return (
        q_pack.reshape(B, 1, num_kv_heads * physical_g, D // 2),
        q_sf.view(torch.float8_e4m3fn),
    )






def _check_decode_metadata(block_tables, seq_lens, batch, device):
    """Validate borrowed page metadata without casting, copying or GPU reads."""
    if (
        block_tables.dtype != torch.int32
        or block_tables.device != device
        or block_tables.ndim != 2
        or block_tables.shape[0] != batch
        or block_tables.shape[1] < 1
        or block_tables.stride(1) != 1
        or block_tables.stride(0) < block_tables.shape[1]
    ):
        raise ValueError(
            "block_tables must be int32 [batch, max_pages_per_seq] on the Q "
            "device, with unit column stride and non-overlapping rows"
        )
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.device != device
        or tuple(seq_lens.shape) != (batch,)
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("seq_lens must be contiguous int32 [batch] on the Q device")


def run_paged_fp4_decode_qmul4_tmem(
    q: torch.Tensor,
    pool: NvFp4PagedKVPool,
    v_data_hnd: torch.Tensor,
    v_sf_bthd: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    out: torch.Tensor,
    max_seq_len: int = 0,
    k_sf_mode: str = "linear",
    value_scale: float = 1.0,
    reduction_workspace: Qmul4ReductionWorkspace | None = None,
    residue_fold: bool = True,
    direct_output: bool = True,
    plan: tuple[int, str] | None = None,
) -> torch.Tensor:
    """Decode with conventional V and the LDSM/QMUL4/TMEM conversion.

    K and K scales still come from ``pool`` (the native blockscaled KQ
    contract). V uses the context/TRT-friendly contract:

    - ``v_data_hnd``: packed uint8 ``[page, head, token, D/2]``;
    - ``v_sf_bthd``: e4m3 ``[page, head, token, D/16]``,tile 内按 TRT-LLM 的
      4-token interleave 排布(``append.quantize_paged_trtllm_format`` 产出)。

    The decode kernel converts V to E4M3 in TMEM using QMUL4, then runs the
    ordinary FP8 VP MMA.
    """
    if not residue_fold:
        raise NotImplementedError("ordinary QK moved to .dev/tools/resattn/experimental")
    if k_sf_mode != "linear":
        raise NotImplementedError("production ResAttn uses linear K scales")
    if direct_output is False:
        raise NotImplementedError("padded output moved to .dev/tools/resattn/experimental")
    B, HQ, D = q.shape
    H = pool.num_kv_heads
    g = HQ // H
    check_shape_supported(HQ, H, D, prediction=1)
    _check_decode_metadata(block_tables, seq_lens, B, q.device)
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

    g_pad = _residue_padded_group_size(HQ, H)
    if (tuple(out.shape) != (B, HQ, D) or out.dtype != torch.bfloat16
            or out.device != q.device or not out.is_contiguous()):
        raise ValueError("direct_output requires contiguous BF16 [B, H_q, D]")
    q_eff, out_eff = q, out

    # The caller may already have selected this exact plan to size its
    # reduction workspace. Standalone calls retain the original selector.
    if plan is None:
        max_len = max_seq_len or int(seq_lens.max().item())
        splits, reduction = plan_qmul4_decode(
            B, H, max_len, q.device, grouped_head_tile=g_pad,
            residue_fold=residue_fold,
        )
    else:
        splits, reduction = plan
    if direct_output and reduction not in ("none", "kernel", "cga_sharded"):
        # The gmem-fused reducer has an NVFP4-specific output contract.
        raise NotImplementedError(f"direct BF16 output: unsupported {reduction=}")
    f = compile_decode_variant(decode_compile_spec(
        HQ, H, D, (splits, reduction),
        output_nvfp4=False, residue_fold=residue_fold,
        k_sf_mode=k_sf_mode, direct_output=direct_output,
        partial_o_dtype=_reduction_workspace_partial_o_dtype(reduction_workspace),
    ))
    fused = _quantize_q_residue_cuda(q_eff, H)
    if fused is None:
        raise RuntimeError(
            "decode QK residue requires the nvfp4 MExt CUDA extension"
        )
    q_pack, q_sf = fused
    o_partial = l_partial = m_partial = None
    if direct_output and reduction == "none":
        # Empty/inactive rows skip the main kernel's output store. Preserve
        # the zero-row contract of the old padded torch.zeros allocation.
        # Atomic already clears below; the separate reducer writes zero.
        out_eff.zero_()
    if reduction == "kernel":
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
        pool.k_sf,
        v_sf_bthd,
        out_eff.view(B, 1, HQ if direct_output else H * g_pad, D),
        None, None, None, o_partial, l_partial, m_partial, None,
        Float32(sm_scale),
        Float32(value_scale),
        seq_lens,
        block_tables,
        False,
    )
    if out_eff is not out:
        out.view(B, H, g, D).copy_(
            out_eff.view(B, H, g_pad, D)[:, :, :g, :]
        )
    return out


def decode_compile_spec(num_qo_heads, num_kv_heads, head_dim, plan, *,
                        output_nvfp4=False, residue_fold=True,
                        k_sf_mode="linear", direct_output=True,
                        partial_o_dtype=None):
    """The single compilation contract for launches and all warmup callers.

    Batch, sequence length and split count are runtime arguments. Only the
    reducer and the scheduler selected from the split count specialize code.
    """
    if not residue_fold:
        raise NotImplementedError("ordinary QK moved to .dev/tools/resattn/experimental")
    if k_sf_mode != "linear":
        raise NotImplementedError("production ResAttn uses linear K scales")
    if direct_output is False:
        raise NotImplementedError("padded output moved to .dev/tools/resattn/experimental")
    splits, reduction = plan
    tile = _residue_padded_group_size(num_qo_heads, num_kv_heads)
    ssw = _use_single_softmax_scheduler(tile, reduction, "qmul4_tmem", splits)
    return dict(
        args=(num_kv_heads * tile, num_kv_heads, head_dim, 1, reduction,
              "qmul4_tmem", k_sf_mode, output_nvfp4,
              num_qo_heads),
        kwargs=dict(single_softmax_wg=ssw, smem_correction_broadcast=ssw,
                    residue_fold=residue_fold,
                    partial_o_dtype=partial_o_dtype or (
                        "bf16" if reduction == "gmem_fused" else "fp32")),
    )


def decode_compile_variants(num_qo_heads, num_kv_heads, head_dim, *,
                             residue_fold, output_nvfp4):
    if not residue_fold:
        raise NotImplementedError("ordinary QK moved to .dev/tools/resattn/experimental")
    check_shape_supported(num_qo_heads, num_kv_heads, head_dim)
    tile = _residue_padded_group_size(num_qo_heads, num_kv_heads)
    specs, seen = [], set()
    for plan in static_compile_plans(output_nvfp4=output_nvfp4,
                                     grouped_head_tile=tile, residue_fold=residue_fold):
        spec = decode_compile_spec(num_qo_heads, num_kv_heads, head_dim, plan,
                                    output_nvfp4=output_nvfp4, residue_fold=residue_fold)
        key = (spec["args"], tuple(sorted(spec["kwargs"].items())))
        if key not in seen:
            seen.add(key)
            specs.append(dict(spec, label=f"{plan[1]} ssw={spec['kwargs']['single_softmax_wg']} "
                                         f"nvfp4={output_nvfp4}"))
    return specs


def compile_decode_variant(plan: dict):
    """Compile one specialization from `decode_compile_variants` (cached)."""
    return _compiled(*plan["args"], **plan["kwargs"])


def run_paged_fp4_decode_qmul4_tmem_nvfp4(
    q: torch.Tensor,
    pool: NvFp4PagedKVPool,
    v_data_hnd: torch.Tensor,
    v_sf_bthd: torch.Tensor,
    block_tables: torch.Tensor,
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
    residue_fold: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode directly into the NVFP4 activation contract consumed by O-proj.

    No-split quantizes in the attention CTA.  The deterministic split-K path
    publishes FP32 partial numerators and quantizes only in its unique final
    reducer, after max/sum/O have been merged.  CGA reduction is kept
    on the BF16 compatibility path because it has no single global reducer.

    Returns
    -------
    out
        Packed uint8 E2M1 tensor shaped ``[B, H_q * D / 2]``.
    output_block_scale
        E4M3 scale tensor in vLLM/CUTLASS's padded 128x4 physical layout.
    """
    if not residue_fold:
        raise NotImplementedError("ordinary QK moved to .dev/tools/resattn/experimental")
    if k_sf_mode != "linear":
        raise NotImplementedError("production ResAttn uses linear K scales")
    if q.ndim != 3 or q.dtype != torch.bfloat16 or not q.is_cuda:
        raise ValueError("q must be CUDA BF16 [B, H_q, D]")
    B, HQ, D = q.shape
    H = pool.num_kv_heads
    if HQ % H:
        raise ValueError(f"H_q={HQ} must be divisible by H_kv={H}")
    check_shape_supported(HQ, H, D, prediction=1)
    _check_decode_metadata(block_tables, seq_lens, B, q.device)
    g = HQ // H
    g_pad = _residue_padded_group_size(HQ, H)
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

    f = compile_decode_variant(decode_compile_spec(
        HQ, H, D, (splits, reduction),
        output_nvfp4=True, residue_fold=residue_fold,
        k_sf_mode=k_sf_mode, direct_output=True,
        partial_o_dtype=_reduction_workspace_partial_o_dtype(reduction_workspace),
    ))
    q_eff = q
    fused_q = _quantize_q_residue_cuda(q_eff, H)
    if fused_q is None:
        raise RuntimeError(
            "decode QK residue requires the nvfp4 MExt CUDA extension"
        )
    q_pack, q_sf = fused_q
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
        pool.k_sf,
        v_sf_bthd,
        out.view(B, 1, HQ, D // 2).view(torch.float4_e2m1fn_x2),
        output_block_scale.reshape(-1),
        output_global_scale_inv.reshape(1),
        None,
        o_partial,
        l_partial,
        m_partial,
        completion_counter,
        Float32(sm_scale),
        Float32(value_scale),
        seq_lens,
        block_tables,
        False,
    )
    return out, output_block_scale
