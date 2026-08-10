# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Residue MoE experts on the flashinfer CuTeDSL graft.'

from __future__ import annotations

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
import vllm.model_executor.layers.fused_moe.modular_kernel as mk


def deterministic_moe_reduce(
    pair_out: torch.Tensor,
    p2e: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    topk: int,
) -> torch.Tensor:
    """Fixed-order router-weighted reduction for the deterministic path.

    `pair_out[j]` holds the base+residue sum of pair-slot j (two commutative
    atomic adds into a zeroed row -- bitwise order-invariant). `p2e[j]` is
    the token-space id `token*topk + choice` for slot j, or an out-of-range
    value for padding slots. A stable argsort turns the slot order into flat
    (token, choice) order, so the topk reduction below runs in the SAME
    order every step -- no atomics anywhere on this path.

    Pure torch on purpose: unit-testable on CPU, and its determinism is a
    property of the ops (sort + fixed-shape sum), not of scheduling.
    """
    ids = p2e.long()
    valid = (ids >= 0) & (ids < num_tokens * topk)
    ids_v = ids[valid]
    rows_v = pair_out[valid]
    order = torch.argsort(ids_v, stable=True)
    # every (token, choice) is routed exactly once, so the sorted ids are
    # exactly arange(num_tokens * topk) -- anything else means the tables
    # and the routing disagree, and the result would be silently wrong
    rows_sorted = rows_v[order].view(num_tokens, topk, -1)
    w = topk_weights.float().reshape(num_tokens, topk, 1)
    return (rows_sorted.float() * w).sum(dim=1)


class ResidueCuteDSLExperts(FlashInferCuteDSLExperts):
    """MExt-R1 base+residue on the grafted CuTeDSL MoE kernels."""

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self._sf_idx_cache: dict[tuple[int, int], torch.Tensor] = {}

    @property
    def expects_unquantized_inputs(self) -> bool:
        # The MExt-R1 quantiser doubles rows; no vLLM quant path expresses
        # that, so prepare() must hand over raw bf16.
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        if not FlashInferCuteDSLExperts._supports_current_device():
            return False
        try:
            from nvfp4.experiment.moe_fold_grouped.swiglu_quant import load
            return load() is not None
        except Exception:
            return False

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
        # The stock class sees PRE-QUANTIZED x (K = hidden/2, fp4-packed)
        # and asserts hidden_dim == K*2. With expects_unquantized_inputs
        # the input stays bf16, so K IS the hidden dim.
        assert self.hidden_dim == K, (self.hidden_dim, K)
        return ((0,), (0,), (M, self.hidden_dim))

    def _sf_idx(self, rows: int, cols: int, device) -> torch.Tensor:
        key = (rows, cols)
        if key not in self._sf_idx_cache:
            from nvfp4.experiment.moe_cutedsl_graft.residue_glue import (
                sf_unswizzle_indices,
            )
            self._sf_idx_cache[key] = sf_unswizzle_indices(rows, cols,
                                                           device)
        return self._sf_idx_cache[key]

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort

        from nvfp4.experiment.moe_cutedsl_graft.blockscaled_contiguous_grouped_gemm_finalize_fusion import (  # noqa: E501
            blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4 as g2,
        )

        from nvfp4.experiment.moe_cutedsl_graft.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4 as g1,
        )
        from nvfp4.experiment.moe_fold_grouped import swiglu_quant as sq

        assert expert_map is None, "EP not supported on this path"
        assert not apply_router_weight_on_input
        M, K = hidden_states.shape
        E = global_num_experts
        interm = self.intermediate_size_per_partition

        # 1+2. MExt-R1 input quantisation (base at 2t, residue at 2t+1),
        # SF emitted ROW-MAJOR by the kernel itself (linear_sf) -- the
        # unswizzle gather retired with optimization B.
        a1_gs = self.a1_gscale.reshape(-1)[:1].float()
        x_fp4, x_sf_lin = sq.swiglu_quant(
            hidden_states, a1_gs, experts=1, fuse_swiglu=False,
            layout_mode=sq.LAYOUT_ROW_PAIR, linear_sf=True)
        x_sf = x_sf_lin.reshape(2 * M, K // 16).view(torch.float8_e4m3fn)

        
        
        
        
        (tiles, mn_lim, _e2p, p2e, _tot, n_tiles) = moe_sort(
            topk_ids.to(torch.int32), topk_weights.float(),
            num_experts=E, top_k=self.topk,
            local_expert_offset=self.local_expert_offset,
            num_local_experts=self.local_num_experts,
            tile_tokens_dim=64)

        # 5. Vendored gather GEMM1: fold + residue emission in-epilogue.
        m_phys = 2 * p2e.numel()
        g1_out = torch.zeros((m_phys, interm // 2), dtype=torch.uint8,
                             device=hidden_states.device)
        g1_sfc = torch.zeros((32, 4, m_phys // 128, 4, interm // 64, 1),
                             dtype=torch.uint8, device=hidden_states.device)
        norm_const = self.a2_gscale.reshape(-1)[:1].float()
        g1(x_fp4, w1, x_sf, self.w1_scale, self.g1_alphas,
           tiles, mn_lim, p2e, n_tiles,
           out=g1_out, out_scale=g1_sfc, global_scale=norm_const,
           topk=self.topk, sf_vec_size=16, mma_tiler_mn=(128, 128),
           cluster_shape_mn=(1, 1), vectorized_f32=True,
           c_dtype="float4_e2m1fn", residue=True)

        # 6. Finalize. Two paths:
        #    fused (default): g2's atomic scatter-add merges base+residue
        #      and applies router weights in the meta warp -- fastest, but
        #      NOT run-to-run deterministic (add order follows CTA
        #      scheduling; stock cutedsl shares this).
        #    deterministic (RESIDUE_MOE_DETERMINISTIC_FINALIZE=1, eval
        #      runs): same g2 kernel fed an IDENTITY token table, so each
        #      pair's two adds land in a private zeroed row (two
        #      commutative adds -> bitwise order-invariant), then the
        #      router-weighted topk reduction runs in fixed order in torch
        #      (deterministic_moe_reduce). Costs one (n_pairs, K) buffer
        #      and a sort -- an eval knob, not a production default.
        import os as _os
        if _os.environ.get("RESIDUE_MOE_DETERMINISTIC_FINALIZE", "0") == "1":
            n_pairs = p2e.numel()
            pair_out = torch.zeros((n_pairs, K), dtype=output.dtype,
                                   device=hidden_states.device)
            ident = torch.arange(n_pairs, dtype=torch.int32,
                                 device=hidden_states.device)
            unit_w = torch.ones(n_pairs, dtype=torch.float32,
                                device=hidden_states.device)
            g2(g1_out, w2, g1_sfc, self.w2_scale, self.g2_alphas,
               tiles, n_tiles, mn_lim, ident, unit_w,
               out=pair_out, mma_tiler_mn=(128, 128),
               cluster_shape_mn=(1, 1), residue=True)
            output.copy_(deterministic_moe_reduce(
                pair_out, p2e, topk_weights, M, self.topk).to(output.dtype))
            return

        output.zero_()
        g2(g1_out, w2, g1_sfc, self.w2_scale, self.g2_alphas,
           tiles, n_tiles, mn_lim, p2e, topk_weights.float(),
           out=output, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1),
           residue=True)


class ExtendedKCuteDSLExperts(FlashInferCuteDSLExperts):
    """Partial-ratio (extended_k) experts: K-extended weights, stock GEMMs.

    The other end of the expert cost law (RESULTS sec 28): mext_r1 pays 2x
    rows above the absorption point (~M=1024), extended_k pays (1+r) K --
    this class is the prefill-savings rung. Contract (paired with the
    export side, PLAN_moe_per_layer_ratio.md):

      * w13 weights/scales in the checkpoint are PHYSICALLY K-extended to
        K + num_salient, columns [W | W[:, salient]] quantised as one
        tensor. create_weights (residue_moe.py) sizes the params so the
        loader accepts them; the stock FLASHINFER_CUTEDSL weight prep then
        runs unchanged at the bigger K.
      * the activation is quantised ONCE per layer under the layer-shared
        salient mask (P3: one token row is read by all topk experts) into
        [base | quant(x - dq(x_q))_salient] with LINEAR SF (W3 quantiser).
      * both grouped GEMMs run in STOCK mode (residue=False) -- single
        rows, full-size tiles, no pairs; g1 simply has a bigger K. w2 and
        the g2 path are untouched (base-K intermediate; g2-side residue is
        plan item W2, not built).

    `_extk_channel_masks` / `_extk_num_salient` are stashed on the
    instance by the method's process_weights_after_loading, which resolved
    the layer policy.
    """

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self._extk_channel_masks: torch.Tensor | None = None
        self._extk_num_salient: int = 0

    @property
    def expects_unquantized_inputs(self) -> bool:
        # Quantisation happens here, under the layer's channel mask; no
        # stock quant path knows about the K extension.
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        if not FlashInferCuteDSLExperts._supports_current_device():
            return False
        try:
            from nvfp4.quant import scaled_fp4_quant_with_mask_linear_sf  # noqa: F401
            return True
        except Exception:
            return False

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
        # With expects_unquantized_inputs the input stays bf16 at base K.
        assert self.hidden_dim == K, (self.hidden_dim, K)
        return ((0,), (0,), (M, self.hidden_dim))

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort

        from nvfp4.experiment.moe_cutedsl_graft.blockscaled_contiguous_grouped_gemm_finalize_fusion import (  # noqa: E501
            blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4 as g2,
        )
        from nvfp4.experiment.moe_cutedsl_graft.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4 as g1,
        )
        from nvfp4.quant import scaled_fp4_quant_with_mask_linear_sf

        assert expert_map is None, "EP not supported on this path"
        assert not apply_router_weight_on_input
        masks = self._extk_channel_masks
        num_salient = self._extk_num_salient
        assert masks is not None and num_salient > 0, (
            "extended_k experts served without a salient mask -- the "
            "method's process_weights_after_loading must stash it")
        M, K = hidden_states.shape
        E = global_num_experts
        interm = self.intermediate_size_per_partition
        k_ext = K + num_salient
        # the K-extended weight is the layer's ground truth; a mismatch
        # here means policy and checkpoint disagree -- refuse loudly
        assert w1.shape[-1] * 2 == k_ext, (w1.shape, K, num_salient)

        # 1. Masked quantisation at base K -> [base | residue] at K_ext,
        # LINEAR SF (the W3 quantiser). One buffer, shared by all experts.
        a1_gs = self.a1_gscale.reshape(-1)[:1].float()
        x_fp4, x_sf = scaled_fp4_quant_with_mask_linear_sf(
            hidden_states, a1_gs, masks, num_salient)
        x_sf = x_sf.view(torch.float8_e4m3fn)

        # 2. STOCK tables at full tile size -- single rows, no pairs.
        (tiles, mn_lim, _e2p, p2e, _tot, n_tiles) = moe_sort(
            topk_ids.to(torch.int32), topk_weights.float(),
            num_experts=E, top_k=self.topk,
            local_expert_offset=self.local_expert_offset,
            num_local_experts=self.local_num_experts,
            tile_tokens_dim=128)

        # 3. Stock-mode gather GEMM1 at K_ext.
        m_phys = p2e.numel()
        g1_out = torch.zeros((m_phys, interm // 2), dtype=torch.uint8,
                             device=hidden_states.device)
        g1_sfc = torch.zeros((32, 4, (m_phys + 127) // 128, 4,
                              interm // 64, 1),
                             dtype=torch.uint8,
                             device=hidden_states.device)
        norm_const = self.a2_gscale.reshape(-1)[:1].float()
        g1(x_fp4, w1, x_sf, self.w1_scale, self.g1_alphas,
           tiles, mn_lim, p2e, n_tiles,
           out=g1_out, out_scale=g1_sfc, global_scale=norm_const,
           topk=self.topk, sf_vec_size=16, mma_tiler_mn=(128, 128),
           cluster_shape_mn=(1, 1), vectorized_f32=True,
           c_dtype="float4_e2m1fn", residue=False)

        # 4. Stock-mode finalize GEMM2 (base-K w2, untouched).
        output.zero_()
        g2(g1_out, w2, g1_sfc, self.w2_scale, self.g2_alphas,
           tiles, n_tiles, mn_lim, p2e, topk_weights.float(),
           out=output, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1),
           residue=False)
