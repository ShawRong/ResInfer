# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""The residue MoE expert block as a vLLM `FusedMoEExpertsModular`.

This is the shell around the already-verified pieces: the MExt-R1 quantiser
(`swiglu_quant`, doubles rows to base+residue and can gather through
`row_idx`), the fold GEMM chain (`batched_expert_fold`), and the routing
tables (`build_routing`). vLLM supplies everything else -- routing, the
weighted reduction (`ops.moe_sum` via `TopKWeightAndReduceContiguous`), CUDA
graphs, DP/EP scaffolding.

Where it sits in vLLM's dispatch:

    FusedMoE layer -> FusedMoEMethodCls -> Modular
        PrepareAndFinalize:  MoEPrepareAndFinalizeNoDPEPModular
                             (Standard format; `expects_unquantized_inputs`
                             on THIS class makes prepare hand over raw bf16)
        Experts:             ResidueNvFp4Experts  <- this file

Why not the batched PrepareAndFinalize, whose (E, max_tokens, K) layout is
exactly ours: it quantises for us -- plain NVFP4, one row per token, no
residue possible -- and its `defer_input_quant` raises. On the Standard
route the permute is ours, done inside the quantiser via `row_idx`, so no
reordered bf16 copy is ever materialised.

Data flow inside `apply` (M tokens, E experts, topk choices):

    hidden_states (M, K) bf16                                [from prepare]
      build_routing         -> token_idx / dst_idx tables    [torch, here]
      quantise+permute      -> (E*2*m_tok, K/2) FP4 + SF     [swiglu_quant]
      GEMM1(fold) -> SwiGLU+quant -> GEMM2(fold)             [the chain]
                            -> (E, m_tok, K) bf16
      scatter by dst_idx    -> output (M, topk, K)           [torch, here]
    vLLM finalize: output *= topk_weights; ops.moe_sum       [vLLM]

`apply` must NOT weight by the router -- `TopKWeightAndReduceContiguous`
does that. Doing it twice is a silent quality bug, not a crash.

EXPERIMENT-GRADE limits, all deliberate and all checked loudly:

  * per-expert scale handling: input_scale collapses to the max (the quant
    scale is a free choice as long as alpha uses the SAME value -- the
    identity is exact, and it is what vLLM's own NVFP4 MoE paths do);
    weight_scale_2 keeps per-expert RATIOS against the max, applied by
    swiglu_quant's per-slot pre_scale (GEMM1's ratio, before SwiGLU --
    the nonlinearity forbids later) and post_scale (GEMM2's ratio, folded
    into the intermediate quantisation). Both epilogue alphas stay global.
    Equal-scale checkpoints skip the vectors entirely (bit-identical to
    the pre-ratio path). Gate/up halves sharing one w13 alpha is still
    required -- differing halves would need a per-COLUMN prescale.
  * decode-shaped: the padded batched format allocates E * ceil8(M) rows,
    so a long prefill through this path is memory-quadratic-ish. Serving
    would route prefill elsewhere; the probes only drive small M.
  * no expert skipping yet (`active_only=False`): every expert occupies an
    l slot, idle ones all-padding. Static shapes first -- the gather that
    skips idle experts changes l per step and needs the bounded-shape
    treatment before it can sit inside a CUDA graph.
  * no EP/DP: `_supports_parallel_config` accepts single-rank only.
"""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform


def _ceil8(x: int) -> int:
    return (x + 7) // 8 * 8


def _max_input_scale(t: torch.Tensor, name: str) -> torch.Tensor:
    """Per-expert input scales -> one, by max.

    The activation is quantised ONCE for all experts, so a per-expert quant
    scale is not expressible -- but it does not need to be: the quant scale
    is a free parameter of the NVFP4 identity as long as alpha is built from
    the SAME value. Max keeps every expert's calibrated range representable.
    This is the same collapse vLLM's NVFP4 MoE paths apply to input_scale.
    """
    del name
    return t.reshape(-1).float().max().reshape(1).clone()


def _per_expert_ratio(t: torch.Tensor, num_experts: int,
                      name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """weight_scale_2 -> (reference scalar, per-expert ratios or None).

    The fold GEMM's epilogue alpha stays a single scalar built from the max
    scale; each expert's true alpha is recovered by ratio_e = s2_e / s2_max
    applied inside swiglu_quant (pre_scale for GEMM1, post_scale for GEMM2).
    All-equal checkpoints return None so the no-vector path (bit-identical
    to the pre-ratio code) keeps running.

    A fused w13 tensor may carry one scale per half; one alpha covers both
    halves of the [gate | up] GEMM, so differing halves stay a loud refusal
    (that would need a per-column prescale, which nothing motivates yet).
    """
    per = t.reshape(num_experts, -1).float()
    if per.shape[1] > 1 and not bool(torch.all(per == per[:, :1])):
        raise NotImplementedError(
            f"{name}: gate/up halves carry different scales; one alpha "
            "covers both halves of the fused GEMM1.")
    flat = per[:, 0]
    ref = flat.max().reshape(1).clone()
    if bool(torch.all(flat == flat[0])):
        return ref, None
    return ref, (flat / ref).contiguous()


class ResidueNvFp4Experts(mk.FusedMoEExpertsModular):
    """MExt-R1 (base+residue) NVFP4 expert block on the fold GEMM chain."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        # Standard format forbids max_num_tokens/num_dispatchers (base checks).
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.out_dtype = moe_config.in_dtype
        # Filled by process_weights_after_loading.
        self.alpha1: torch.Tensor | None = None
        self.alpha2: torch.Tensor | None = None
        self.res_a1_gscale: torch.Tensor | None = None
        self.res_a2_gscale: torch.Tensor | None = None
        self.w13_sf_swizzled: torch.Tensor | None = None
        self.w2_sf_swizzled: torch.Tensor | None = None
        # Per-expert weight_scale_2 ratios vs the alpha reference; None when
        # all experts share one scale (then no per-slot vectors are built).
        self.pre1_ratio: torch.Tensor | None = None
        self.post2_ratio: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # capability surface                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        # Standard, NOT BatchedExperts: the batched PrepareAndFinalize would
        # quantise for us (no residue fits), and its single-node variant is
        # the naive fallback. We build the batched layout ourselves inside
        # the quantiser's row_idx gather.
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # prepare() hands over raw bf16; the MExt-R1 quantiser is ours and
        # doubles rows, which no vLLM quant path expresses.
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        if not (p.is_cuda() and p.is_device_capability_family(100)):
            return False  # the M-ext fold kernel is sm100-only today
        try:
            from nvfp4.experiment.moe_fold_grouped.swiglu_quant import load
            return load() is not None
        except Exception:
            return False

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False  # the chain fuses gated SwiGLU; non-gated is not wired

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Weight side is a plain static NVFP4 checkpoint; the activation is
        # nominally dynamic NVFP4 -- what actually runs is our MExt-R1
        # (base+residue) variant of it, which vLLM has no key for.
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        # Single rank only for now: expert_map/EP would shift expert ids and
        # nothing here handles that.
        return (moe_parallel_config.ep_size == 1
                and moe_parallel_config.dp_size == 1)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # (M, topk, K) out of apply -> vLLM weights by the router and reduces
        # with the real ops.moe_sum kernel. NOT Delegate: on some routes that
        # falls back to TopKWeightAndReduceNaiveBatched, a Python loop.
        return TopKWeightAndReduceContiguous()

    # ------------------------------------------------------------------ #
    # weights                                                             #
    # ------------------------------------------------------------------ #

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Fold static scales and swizzle the weight SF, once.

        Mirrors what TrtLlmNvFp4ExpertsBase does for its kernel (scale
        algebra only -- the NVFP4 trtllm path does no weight-row remap; that
        infra belongs to MXFP8), plus the 128x4 SF swizzle the fold GEMM
        reads. Everything here is load-time; apply() sees ready tensors.
        """
        num_experts = layer.w13_weight.shape[0]
        w13_s2, self.pre1_ratio = _per_expert_ratio(
            layer.w13_weight_scale_2.data, num_experts, "w13_weight_scale_2")
        w2_s2, self.post2_ratio = _per_expert_ratio(
            layer.w2_weight_scale_2.data, num_experts, "w2_weight_scale_2")
        a1_in = _max_input_scale(layer.w13_input_scale.data,
                                 "w13_input_scale")
        a2_in = _max_input_scale(layer.w2_input_scale.data, "w2_input_scale")

        dev = layer.w13_weight.device
        if self.pre1_ratio is not None:
            self.pre1_ratio = self.pre1_ratio.to(dev)
        if self.post2_ratio is not None:
            self.post2_ratio = self.post2_ratio.to(dev)
        # alpha = weight_scale_2 * input_scale: the dequant product the fold
        # GEMM applies to its accumulator (same identity the linear path and
        # TrtLlmNvFp4ExpertsBase use). Per-expert weight_scale_2 deviations
        # from the reference travel as pre/post ratios through swiglu_quant.
        self.alpha1 = (w13_s2 * a1_in).to(dev)
        self.alpha2 = (w2_s2 * a2_in).to(dev)
        # Global scales for the two activation quantisations (1/input_scale
        # in the shipped recipe's SFScaleVal convention).
        self.res_a1_gscale = (1.0 / a1_in).to(dev)
        self.res_a2_gscale = (1.0 / a2_in).to(dev)

        # Weight SF: checkpoint layout is linear (E, rows, K/16) e4m3; the
        # fold GEMM reads the 128x4 swizzled layout. vLLM's own converter
        # handles the batched 3D case directly (pads rows to 128 per expert,
        # which is exactly the per-expert padded block the GEMM strides by).
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            swizzle_blockscale,
        )
        self.w13_sf_swizzled = swizzle_blockscale(
            layer.w13_weight_scale.data).view(torch.uint8).contiguous()
        self.w2_sf_swizzled = swizzle_blockscale(
            layer.w2_weight_scale.data).view(torch.uint8).contiguous()

    # ------------------------------------------------------------------ #
    # shapes                                                              #
    # ------------------------------------------------------------------ #

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # output feeds TopKWeightAndReduceContiguous: one row per (token,
        # choice), token-major -- dst_idx lands each expert row here.
        # Workspaces are (0,): the chain owns its buffers (make_buffers) and
        # the scatter staging is data-dependent in a way the (tokens-first)
        # workspace contract does not express. Same choice as the in-tree
        # Standard-format CuTeDSL experts.
        return ((0,), (0,), (M, topk, K))

    # ------------------------------------------------------------------ #
    # the seams apply() is tested through                                 #
    # ------------------------------------------------------------------ #
    # Local plumbing tests override these two with faithful fp32 torch
    # stand-ins (the real kernels are sm100-only and bitwise-verified
    # separately on B200). Everything OUTSIDE the seams -- the tables, the
    # padding row, the scatter, the interplay with vLLM's finalize -- runs
    # for real in those tests.

    def _quantize_permute(self, hs_padded: torch.Tensor, row_idx: torch.Tensor,
                          experts: int, m_tok: int):
        from nvfp4.experiment.moe_fold_grouped.swiglu_quant import swiglu_quant
        return swiglu_quant(
            hs_padded, self.res_a1_gscale, experts=experts, fuse_swiglu=False,
            row_idx=row_idx, m_tok=m_tok)

    def _expert_chain(self, act, act_sf, w1, w1_sf, w2, w2_sf,
                      experts: int, m_tok: int, expert_ids, num_experts: int,
                      pre_scale=None, post_scale=None):
        from nvfp4.experiment.moe_fold_grouped.batched_expert import (
            batched_expert_fold,
        )
        return batched_expert_fold(
            act, act_sf, w1, w1_sf, w2, w2_sf, self.alpha1, self.alpha2,
            experts=experts, m_tok=m_tok, intermediate_scale=self.res_a2_gscale,
            expert_ids=expert_ids, num_experts=num_experts,
            pre_scale=pre_scale, post_scale=post_scale)

    # ------------------------------------------------------------------ #
    # apply                                                               #
    # ------------------------------------------------------------------ #

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        from nvfp4.experiment.moe_fold_grouped.batched_expert import (
            build_routing_aligned,
            moe_apply,
        )

        assert expert_map is None, "EP expert_map not supported (declared)"
        assert not apply_router_weight_on_input, (
            "router weights are applied by vLLM's finalize on this path; "
            "applying them on the input too would double-count them")

        
        
        
        
        
        
        
        
        
        
        tables = build_routing_aligned(topk_ids, 8, global_num_experts)

        # Per-expert weight_scale_2 ratios, gathered per SLOT through the
        # same table the weight gather uses (trailing padding blocks read
        # expert 0's ratio; their rows land on the throwaway dst row).
        slot_ids = tables[0].long()
        pre = (self.pre1_ratio[slot_ids]
               if self.pre1_ratio is not None else None)
        post = (self.post2_ratio[slot_ids]
                if self.post2_ratio is not None else None)

        def _chain(act, act_sf, n_slots, m_tok, expert_ids):
            return self._expert_chain(
                act, act_sf,
                w1.reshape(-1, w1.shape[-1]), self.w13_sf_swizzled,
                w2.reshape(-1, w2.shape[-1]), self.w2_sf_swizzled,
                n_slots, m_tok, expert_ids,
                global_num_experts if expert_ids is not None else None,
                pre_scale=pre, post_scale=post)
        output.copy_(moe_apply(
            hidden_states, topk_ids, global_num_experts,
            self._quantize_permute, _chain, _tables=tables))
