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

        # 6. Grafted finalize GEMM2 (residue=True reads the token-space
        # table at row//2): both rows of a pair share (token, choice), so
        # its atomic adds merge base+residue; router weight folded in the
        # meta warp. Output must be pre-zeroed (accumulation target).
        output.zero_()
        g2(g1_out, w2, g1_sfc, self.w2_scale, self.g2_alphas,
           tiles, n_tiles, mn_lim, p2e, topk_weights.float(),
           out=output, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1),
           residue=True)
