# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Phase-separated ResAttn host adapter for the vLLM 0.26 Runtime.

Stock FlashInfer owns metadata planning, cache shape/writing and prefill.
Only decode consumes raw BF16 Q and invokes our existing QK-residue kernel.
Any step containing prefill is delegated intact to stock, so continuous-
batching mixed steps remain correct but are deliberately not optimized.
"""
from dataclasses import dataclass, fields
import os

import torch

from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    nvfp4_split_data_scale,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
    TRTLLMPrefill,
)

from .attn_backend import (
    NvFp4PagedKVPool,
    ResAttnImpl,
    _register_decode_warmup,
)


@dataclass
class ResidueDecodeMetadata:
    num_decodes: int
    num_decode_tokens: int
    max_seq_len: int
    seq_lens: torch.Tensor
    kv_indices: torch.Tensor
    kv_indptr: torch.Tensor


@dataclass
class ResAttnFlashInferMetadata(FlashInferMetadata):
    residue_decode: ResidueDecodeMetadata | None = None
    # Required for pure prefill and the conservative mixed-step fallback.
    # It is the object produced by stock's builder, not a reconstructed view.
    stock: FlashInferMetadata | None = None


class ResAttnFlashInferMetadataBuilder(FlashInferMetadataBuilder):
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        if os.environ.get("RESIDUE_ATTN_CUDAGRAPH", "0") == "1":
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError("ResAttn decode does not support DCP")
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self._residue_offsets = {}
        self._residue_max_reqs = vllm_config.scheduler_config.max_num_seqs
        self._cg_capture_seq_len = int(
            os.environ.get("RESIDUE_ATTN_CG_CAPTURE_LEN", "0")
        ) or vllm_config.model_config.max_model_len

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        stock = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if stock.use_cascade or not stock.causal:
            raise NotImplementedError("ResAttn requires causal, non-cascade attention")
        if stock.num_prefill_tokens and not isinstance(stock.prefill, TRTLLMPrefill):
            raise NotImplementedError("This adapter requires stock TRTLLM prefill")
        decode = None
        n = stock.num_decodes
        # The optimized contract is phase-separated. A mixed step is executed
        # wholly by stock, so do not prepare a decode descriptor that cannot be
        # consumed and do not make its padded page-table views part of capture.
        if n and not stock.num_prefill_tokens:
            if n != stock.num_decode_tokens:
                raise NotImplementedError("ResAttn requires one token per decode")
            cam = common_attn_metadata
            table = cam.block_table_tensor[:n]
            if table.dtype != torch.int32 or not table.is_contiguous():
                raise ValueError("Expected contiguous int32 stock block table")
            width = table.shape[1]
            key = (width, str(table.device))
            offsets = self._residue_offsets.get(key)
            if offsets is None:
                offsets = torch.arange(
                    self._residue_max_reqs + 1,
                    dtype=torch.int32, device=table.device,
                ) * width
                self._residue_offsets[key] = offsets
            # The frozen kernel uses only indptr[b] and seq_lens[b], NOT
            # indptr[b+1]-indptr[b]. Padded rows therefore need no compaction.
            decode = ResidueDecodeMetadata(
                n, n, cam.max_seq_len, cam.seq_lens[:n],
                table.view(-1), offsets[:n + 1],
            )
        return ResAttnFlashInferMetadata(
            **{f.name: getattr(stock, f.name) for f in fields(FlashInferMetadata)},
            residue_decode=decode,
            stock=stock if stock.num_prefill_tokens else None,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        md = self.build(0, common_attn_metadata)
        if md.residue_decode is not None:
            md.residue_decode.max_seq_len = max(
                md.residue_decode.max_seq_len, self._cg_capture_seq_len
            )
        return md


class ResAttnFlashInferImpl(FlashInferImpl):
    can_return_lse_for_decode = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (not self.is_kvcache_nvfp4 or self.head_size != 128
                or self.num_heads != 8 * self.num_kv_heads):
            raise NotImplementedError("ResAttn candidate requires NVFP4, D128, GQA8")
        if (self.alibi_slopes is not None or self.window_left != -1
                or self.logits_soft_cap not in (None, 0.0)
                or self.sinks is not None
                or self.kv_sharing_target_layer_name is not None):
            raise NotImplementedError("Unsupported ResAttn decode attention variant")
        # Attention.forward must preserve raw BF16 Q for pure decode. Stock's
        # forward performs its normal query conversion when we delegate an
        # entire prefill or mixed step below.
        # Set after super(): FlashInfer's constructor sets an instance value.
        self.supports_quant_query_input = False
        self._decode_reduction_workspaces = {}
        self._residue_pool_key = None
        self._residue_pool = None
        _register_decode_warmup(self.num_heads, self.num_kv_heads, self.head_size)

    def fused_output_quant_supported(self, quant_key):
        return False

    _get_decode_reduction_workspace = ResAttnImpl._get_decode_reduction_workspace
    _decode_forward = ResAttnImpl._decode_forward

    def _pool_for(self, kv_cache, layer):
        key = (kv_cache.data_ptr(), tuple(kv_cache.shape), kv_cache.stride(),
               float(layer._k_scale_float), float(layer._v_scale_float))
        if key != self._residue_pool_key:
            k, v = kv_cache.split(self.num_kv_heads, dim=1)
            order = FlashInferBackend.get_kv_cache_stride_order()
            k = canonicalize_singleton_dim_strides(k.permute(*order))
            v = canonicalize_singleton_dim_strides(v.permute(*order))
            kd, ks = nvfp4_split_data_scale(k)
            vd, vs = nvfp4_split_data_scale(v)
            if tuple(kd.shape[1:]) != (self.num_kv_heads, 128, 64):
                raise ValueError("ResAttn requires stock HND, page128, D128 cache")
            self._residue_pool = NvFp4PagedKVPool(
                num_kv_heads=self.num_kv_heads, head_dim=self.head_size,
                k_data=kd, v_data=vd, k_sf=ks, v_sf=vs,
                v_staging=torch.empty(0, dtype=torch.bfloat16, device=kv_cache.device),
                page_size=128, layout_profile="trtllm",
                k_global_scale=key[-2], v_global_scale=key[-1],
            )
            self._residue_pool_key = key
        return self._residue_pool

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Initial stock-aligned adapter uses BF16 output")
        if attn_metadata is None:
            return output.fill_(0)
        if query.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise ValueError("ResAttn needs original BF16 Q and BF16 output, not FP8 Q")
        md = attn_metadata
        if md.num_prefill_tokens:
            if md.stock is None:
                raise RuntimeError("ResAttn prefill fallback requires stock metadata")
            # vLLM already invoked the inherited stock cache writer exactly
            # once. Delegate the complete step: pure prefill stays identical
            # to stock and mixed continuous batching is a correctness fallback.
            return super().forward(layer, query, key, value, kv_cache,
                                   md.stock, output)
        if md.residue_decode is not None:
            self._decode_forward(query, key, value, self._pool_for(kv_cache, layer),
                                 md.residue_decode, output)
        return output


class ResAttnFlashInferBackend(FlashInferBackend):
    """Stock prefill/cache contract with a pure-decode ResAttn dispatch."""
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["nvfp4"]

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return ResAttnFlashInferImpl

    @staticmethod
    def get_builder_cls():
        return ResAttnFlashInferMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls):
        return [128]

    @classmethod
    def supports_sink(cls):
        return False
