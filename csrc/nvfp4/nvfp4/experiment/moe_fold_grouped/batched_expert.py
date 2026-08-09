# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'A batched MoE expert block with the residue fold in the only place it works.'

from __future__ import annotations

import torch


def swiglu(u: torch.Tensor) -> torch.Tensor:
    """[gate | up] -> up * silu(gate). Halves the last dim."""
    gate, up = u.chunk(2, dim=-1)
    return up * torch.nn.functional.silu(gate)


def build_routing(sel: torch.Tensor, scales: torch.Tensor, num_experts: int,
                  m_tok: int, active_only: bool = True):
    """Router output -> the padded per-expert slot table this module wants.

    sel      (tokens, topk) int   -- expert chosen per token per slot
    scales   (tokens, topk) float -- the router weight for that choice

    Returns (expert_ids, token_idx, token_scale, dst_idx):
        expert_ids  (E_act,) int32   real expert index per occupied l slot
        token_idx   (E_act, m_tok)   which token owns the slot, num_tokens
                                     for padding
        token_scale (E_act, m_tok)   the router weight, 0 for padding
        dst_idx     (E_act, m_tok)   row of the (tokens*topk, K) buffer this
                                     slot's OUTPUT belongs in, tokens*topk for
                                     padding

    `token_idx` is the table `swiglu_quant(row_idx=...)` consumes: it is what
    turns the quantiser into the input permute, so nothing has to materialise
    a reordered copy of the activation. That is the shape this has to keep.

    `dst_idx` is the other end. vLLM's reduction wants (tokens, topk, K) --
    `TopKWeightAndReduceContiguous` views it that way and calls the real
    `ops.moe_sum` kernel -- while this block computes (E_act, m_tok, K). The
    two differ by a permutation of whole rows, and dst_idx IS that
    permutation. It is free to build here: flattening (token, choice) gives
    row `t*topk + c` directly, so the destination is just the pre-sort
    position of each entry.

    Padding points at a real, allocated row rather than -1: the consumers have
    no masked form, and a boolean mask would make the shape data-dependent and
    therefore uncapturable in a CUDA graph. One throwaway row is cheaper.

    NOT the reduction itself. vLLM's FusedMoE owns that -- see the module
    docstring on where this block plugs in.
    """
    tokens, topk = sel.shape
    dev = sel.device
    flat_e = sel.reshape(-1)
    flat_t = torch.arange(tokens, device=dev).repeat_interleave(topk)
    order = torch.argsort(flat_e, stable=True)
    e_sorted, t_sorted = flat_e[order], flat_t[order]
    s_sorted = scales.reshape(-1)[order]

    # NOT torch.bincount: on CUDA it syncs on the data's max value, which is
    # illegal during CUDA-graph capture -- and the no-skip path of this
    # function must be capturable (vLLM captures decode graphs, and the
    # bench captures the whole closure). index_add_ is shape-static.
    counts = torch.zeros(num_experts, dtype=torch.long, device=dev)
    counts.index_add_(0, flat_e, torch.ones_like(flat_e))
    starts = torch.cumsum(counts, 0) - counts
    # Rank of each entry within its expert -> its slot.
    slot = torch.arange(e_sorted.numel(), device=dev) - starts[e_sorted]

    ids = (torch.nonzero(counts > 0).flatten() if active_only
           else torch.arange(num_experts, device=dev))
    n_l = int(ids.numel())
    # Dense map real expert -> l slot, so the scatter below is one pass.
    where = torch.full((num_experts,), -1, dtype=torch.long, device=dev)
    where[ids] = torch.arange(n_l, device=dev)

    token_idx = torch.full((n_l, m_tok), tokens, dtype=torch.long, device=dev)
    token_scale = torch.zeros(n_l, m_tok, dtype=scales.dtype, device=dev)
    # Padding aims one past the last real row of the (tokens*topk, K) buffer,
    # the same throwaway convention token_idx uses.
    dst_idx = torch.full((n_l, m_tok), tokens * topk, dtype=torch.long,
                         device=dev)
    if not active_only and m_tok >= tokens:
        # Capture-safe path: every expert owns a slot (where = identity) and
        # an expert receives at most `tokens` entries, so slot < m_tok is
        # PROVABLY true -- no mask needed. Boolean indexing would produce
        # data-dependent shapes, which sync and break graph capture.
        token_idx[e_sorted, slot] = t_sorted
        token_scale[e_sorted, slot] = s_sorted
        # `order` is the pre-sort flat position, i.e. exactly t*topk + c.
        dst_idx[e_sorted, slot] = order
    else:
        keep = (slot < m_tok) & (where[e_sorted] >= 0)
        li, sl = where[e_sorted[keep]], slot[keep]
        token_idx[li, sl] = t_sorted[keep]
        token_scale[li, sl] = s_sorted[keep]
        dst_idx[li, sl] = order[keep]
    return ids.to(torch.int32), token_idx, token_scale, dst_idx


def build_routing_aligned(sel: torch.Tensor, m_tok: int, num_experts: int):
    "The same three tables, from vLLM's one-kernel builder."
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    topk = sel.shape[1]
    sorted_ids, expert_blk, _num_pad = moe_align_block_size(
        sel, m_tok, num_experts, pad_sorted_ids=True)
    n_blocks = sorted_ids.numel() // m_tok
    ids = expert_blk[:n_blocks].clamp_min(0).to(torch.int32)
    token_idx = (sorted_ids.long() // topk).view(n_blocks, m_tok)
    dst_idx = sorted_ids.long().view(n_blocks, m_tok)
    return ids, token_idx, dst_idx


def build_routing_aligned_ref(sel: torch.Tensor, m_tok: int,
                              num_experts: int):
    """vLLM-free structural twin of build_routing_aligned, for probes/tests.

    Reproduces the moe_align_block_size CONTRACT without torch.ops._moe_C:
      * static extent, sized from shapes only (the property that makes the
        real op capture-safe):  round_up(M*topk + E*(m_tok-1), m_tok),
        capped at M*topk*m_tok when M*topk < E (each routed (token, choice)
        owns at most one block) -- vLLM's exact formula, in that order;
      * per-expert regions block-aligned, experts packed ascending;
      * empty slots hold sentinel M*topk (-> the appended zero activation
        row and the throwaway dst row); unowned trailing blocks get expert 0,
        matching build_routing_aligned's clamp of vLLM's -1.

    Within-expert token order may differ from the CUDA kernel's atomic
    arrival order. The real kernels are slot-position independent and
    batch-extent stable (proven bitwise on B200), so any downstream output
    difference vs the real op's tables is a real bug, not table cosmetics.

    Host-synced sizing (fine for a reference); serving uses
    build_routing_aligned. The Modal probe uses THIS to bitwise-test the
    aligned path with the real kernels in an image that has no vLLM.
    """
    M, topk = sel.shape
    dev = sel.device
    flat = sel.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts)
    blocks_per_e = (counts + m_tok - 1) // m_tok
    total = M * topk + num_experts * (m_tok - 1)
    total = (total + m_tok - 1) // m_tok * m_tok
    if M * topk < num_experts:
        total = min(M * topk * m_tok, total)
    n_blocks = total // m_tok

    ids = torch.zeros(n_blocks, dtype=torch.int32, device=dev)
    n_owned = int(blocks_per_e.sum())
    ids[:n_owned] = torch.repeat_interleave(
        torch.arange(num_experts, device=dev), blocks_per_e).to(torch.int32)

    slot_base = torch.cumsum(blocks_per_e * m_tok, 0) - blocks_per_e * m_tok
    start = torch.cumsum(counts, 0) - counts       # first sorted index
    rank = torch.arange(M * topk, device=dev) - start[flat[order]]
    slot = slot_base[flat[order]] + rank

    dst = torch.full((total,), M * topk, dtype=torch.long, device=dev)
    dst[slot] = order                              # order == t*topk + c
    return ids, (dst // topk).view(n_blocks, m_tok), dst.view(n_blocks, m_tok)


def moe_apply(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    quantize_permute,
    expert_chain,
    skip_idle: bool = False,
    _tables=None,
) -> torch.Tensor:
    'Router output + raw activation -> (M, topk, K) expert outputs.'
    assert hidden_states.dim() == 2
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    # Static per shape: an expert receives at most M tokens (a token names
    # an expert at most once across its topk), so no device->host sync --
    # sizing from per_expert.max() would be one, and is illegal mid-capture.
    m_tok = (M + 7) // 8 * 8

    if _tables is not None:
        ids, token_idx, dst_idx = _tables
        # Prebuilt tables carry their own m_tok (possibly TIGHTER than the
        # static ceil8(M) bound -- e.g. built from the actual busiest expert).
        # Using the local bound against them would mis-stride every block.
        m_tok = token_idx.shape[1]
    else:
        ids, token_idx, _scale, dst_idx = build_routing(
            topk_ids, torch.zeros_like(topk_ids, dtype=torch.float32),
            num_experts, m_tok, active_only=skip_idle)
    n_slots = int(ids.numel())
    # Identity (slot == expert, no gather) holds ONLY for the internal
    # no-skip build. Inferring it from n_slots == num_experts was a bug:
    # aligned tables pack blocks by ACTIVE order, so the counts can match
    # while the order does not -- prebuilt tables always go through the
    # gather.
    identity = _tables is None and not skip_idle
    expert_ids = None if identity else ids

    # Padding slots read this appended zero row (row M); the quantiser does
    # not branch on padding. Their outputs land on dst row M*topk, sliced
    # off below.
    hs_padded = torch.cat(
        [hidden_states, hidden_states.new_zeros(1, K)], dim=0).contiguous()

    act, act_sf = quantize_permute(
        hs_padded, token_idx.reshape(-1).to(torch.int32), n_slots, m_tok)
    out_e = expert_chain(act, act_sf, n_slots, m_tok, expert_ids)

    # Rows -> (token, choice) order; one throwaway row absorbs padding.
    # dst_idx is a verified permutation (probe_routing_table_local.py).
    stage = out_e.new_empty(M * topk + 1, K)
    stage[dst_idx.reshape(-1)] = out_e.reshape(n_slots * m_tok, K)
    return stage[: M * topk].view(M, topk, K)










def batched_expert_fold(
    act_rowpair_fp4: torch.Tensor,
    act_sf: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_sf: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_sf: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    *,
    experts: int,
    m_tok: int,
    intermediate_scale: torch.Tensor,
    quantize_intermediate=None,
    intermediate_buffers: tuple | None = None,
    expert_ids: torch.Tensor | None = None,
    num_experts: int | None = None,
    pre_scale: torch.Tensor | None = None,
    post_scale: torch.Tensor | None = None,
    # (128,128)/(1,1) won the tile sweep for both GEMMs of this chain and is
    # interior to the searched space -- see da61f14. The old (128,64) default
    # predates that measurement.
    mma_tiler_mn: tuple[int, int] = (128, 128),
    cluster_shape_mn: tuple[int, int] = (1, 1),
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    'One batched MoE expert block. Returns (E, m_tok, K).'
    from nvfp4.cute_fold.host import mext_fold_gemm_sm103

    from .swiglu_quant import swiglu_quant

    m2 = 2 * m_tok
    # The weight buffers are shaped by the MODEL's expert count; with a gather
    # table `experts` counts only the active slots, so it cannot derive the
    # per-expert row block. Getting this wrong scales n1/n2 by the idle
    # fraction and every shape downstream still lines up.
    if expert_ids is not None:
        assert num_experts is not None, "expert_ids needs num_experts"
        assert expert_ids.numel() == experts, (
            f"expert_ids has {expert_ids.numel()} entries but experts="
            f"{experts}; pass experts=len(active)")
    _e_w = num_experts if expert_ids is not None else experts
    n1 = w1_fp4.shape[0] // _e_w           # 2I
    common = dict(out_dtype=out_dtype, mma_tiler_mn=mma_tiler_mn,
                  cluster_shape_mn=cluster_shape_mn, store_mode="tma_mext",
                  kernel_arch="sm100")
    gather = dict(expert_ids=expert_ids, num_experts=num_experts)

    # GEMM1 with the fold in its epilogue: 2*m_tok rows in, m_tok out, so the
    # SUMMED activation is what SwiGLU sees. This is the whole point -- see the
    # module docstring for what folding later costs.
    u = mext_fold_gemm_sm103(
        weight_fp4=w1_fp4[:n1], act_fp4_rowpair=act_rowpair_fp4[:m2],
        weight_sf=w1_sf, act_sf=act_sf, alpha=alpha1,
        batch=experts, **gather, **common)                # (E, m_tok, 2I)
    u_flat = u.reshape(experts * m_tok, n1)

    if quantize_intermediate is None:
        buf = intermediate_buffers or (None, None)
        hq, h_sf = swiglu_quant(u_flat, intermediate_scale, experts=experts,
                                fuse_swiglu=True, out=buf[0], out_sf=buf[1],
                                pre_scale=pre_scale, post_scale=post_scale)
    else:
        hq, h_sf = quantize_intermediate(u_flat)

    # hq is (E*2*m_tok, I/2), l-outermost: expert e owns rows
    # [e*m2, (e+1)*m2). Hand the kernel expert 0's slice; it strides to the
    # rest. Slicing by m_tok instead of m2 here would feed GEMM2 half of one
    # expert -- the shapes still line up, so nothing would complain.
    n2 = w2_fp4.shape[0] // _e_w           # K
    out = mext_fold_gemm_sm103(
        weight_fp4=w2_fp4[:n2], act_fp4_rowpair=hq[:m2],
        weight_sf=w2_sf, act_sf=h_sf, alpha=alpha2,
        batch=experts, **gather, **common)

    return out
