# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Phase-separated ResAttn host adapter for the vLLM 0.26 Runtime.

Stock FlashInfer owns metadata planning, cache shape/writing and prefill.
Only decode consumes raw BF16 Q and invokes our existing QK-residue kernel.
Pure prefill is delegated to stock. Mixed prefill/decode steps are rejected:
stock would convert decode Q to FP8 and bypass ResAttn's BF16-Q residue path.
Continuous batching is supported only while each step remains phase-separated;
this backend does not change the scheduler or prevent it from proposing a mix.
"""
import os

import torch

from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    nvfp4_split_data_scale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferDecodeKernel,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
    FlashInferTrtllmAPIDecode,
    TRTLLMPrefill,
)

from nvfp4.cute_attn.kv_layout import NvFp4PagedKVPool

from .decode import ResAttnDecodeMixin
from .warmup import _register_decode_warmup


def _require_separate_phases(metadata):
    """Reject mixed steps instead of silently changing decode numerics."""
    if metadata.num_prefill_tokens and (metadata.num_decodes or metadata.num_decode_tokens):
        # An explicit exception also protects production runs under python -O.
        raise NotImplementedError(
            "ResAttn does not support mixed prefill/decode batches: "
            f"num_prefill_tokens={metadata.num_prefill_tokens}, "
            f"num_decodes={metadata.num_decodes}, "
            f"num_decode_tokens={metadata.num_decode_tokens}. "
            "Run phase-separated prefill and decode steps. Stock fallback is "
            "disabled because it quantizes decode Q to FP8 and bypasses "
            "ResAttn residue attention."
        )


def _require_trtllm_decode(metadata):
    decode = metadata.decode
    if (not isinstance(decode, FlashInferTrtllmAPIDecode)
            or decode.kernel is not FlashInferDecodeKernel.TRTLLM_GEN):
        raise NotImplementedError(
            "ResAttn requires stock FlashInfer TRTLLM_GEN decode metadata "
            "for NVFP4 KV and page128"
        )
    return decode


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
        self._cg_capture_seq_len = int(
            os.environ.get("RESIDUE_ATTN_CG_CAPTURE_LEN", "0")
        ) or vllm_config.model_config.max_model_len

    def build(self, common_prefix_len, common_attn_metadata,
              fast_build=False) -> FlashInferMetadata:
        stock = super().build(common_prefix_len, common_attn_metadata, fast_build)
        _require_separate_phases(stock)
        if stock.use_cascade or not stock.causal:
            raise NotImplementedError("ResAttn requires causal, non-cascade attention")
        if stock.num_prefill_tokens and not isinstance(stock.prefill, TRTLLMPrefill):
            raise NotImplementedError("This adapter requires stock TRTLLM prefill")
        n = stock.num_decodes
        if n and not stock.num_prefill_tokens:
            if n != stock.num_decode_tokens:
                raise NotImplementedError("ResAttn requires one token per decode")
            decode = _require_trtllm_decode(stock)
            table = decode.block_tables
            if (
                table.dtype != torch.int32 or table.ndim != 2
                or table.shape[0] != n or table.shape[1] < 1
                or table.stride(1) != 1 or table.stride(0) < table.shape[1]
            ):
                raise ValueError("Expected int32 block table with unit column stride")
        return stock

    def build_for_cudagraph_capture(self, common_attn_metadata):
        md = self.build(0, common_attn_metadata)
        if md.num_decodes:
            decode = _require_trtllm_decode(md)
            decode.max_seq_len = max(
                decode.max_seq_len, self._cg_capture_seq_len
            )
        return md


class ResAttnFlashInferImpl(ResAttnDecodeMixin, FlashInferImpl):
    can_return_lse_for_decode = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gqa = (
            self.num_heads // self.num_kv_heads
            if self.num_kv_heads > 0
            and self.num_heads % self.num_kv_heads == 0
            else 0
        )
        if (
            not self.is_kvcache_nvfp4
            or self.head_size not in (128, 256)
            or gqa not in (4, 6, 8)
        ):
            raise NotImplementedError(
                "ResAttn candidate requires NVFP4, D128/D256, and GQA4/6/8"
            )
        if (self.alibi_slopes is not None or self.window_left != -1
                or self.logits_soft_cap not in (None, 0.0)
                or self.sinks is not None
                or self.kv_sharing_target_layer_name is not None):
            raise NotImplementedError("Unsupported ResAttn decode attention variant")
        # Attention.forward must preserve raw BF16 Q for pure decode. Stock's
        # forward performs its normal query conversion when we delegate a
        # pure prefill step below; mixed steps are rejected.
        # Set after super(): FlashInfer's constructor sets an instance value.
        self.supports_quant_query_input = False
        self._decode_reduction_workspaces = {}
        self._residue_pool_key = None
        self._residue_pool = None
        _register_decode_warmup(
            self.num_heads, self.num_kv_heads, self.head_size)

    def fused_output_quant_supported(self, quant_key):
        return bool(self.is_kvcache_nvfp4 and self.head_size == 128
                    and quant_key == kNvfp4Dynamic)

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
            if tuple(kd.shape[1:]) != (self.num_kv_heads, 128, self.head_size // 2):
                raise ValueError("ResAttn requires stock HND, page128, matching head-dimension cache")
            self._residue_pool = NvFp4PagedKVPool(
                num_kv_heads=self.num_kv_heads, head_dim=self.head_size,
                k_data=kd, v_data=vd, k_sf=ks, v_sf=vs,
                page_size=128,
                k_global_scale=key[-2], v_global_scale=key[-1],
            )
            self._residue_pool_key = key
        return self._residue_pool

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if output_scale is None and output_block_scale is None:
            scales = None
        elif output_scale is not None and output_block_scale is not None:
            scales = (output_scale, output_block_scale)
        else:
            raise ValueError("NVFP4 attention output requires data and block scales")
        fused_nvfp4 = scales is not None
        if attn_metadata is None:
            return output.fill_(0)
        _require_separate_phases(attn_metadata)
        if query.dtype != torch.bfloat16:
            raise ValueError("ResAttn needs original BF16 Q, not FP8 Q")
        if output.dtype != (torch.uint8 if fused_nvfp4 else torch.bfloat16):
            raise ValueError("ResAttn output dtype does not match fusion mode")
        md = attn_metadata
        if md.num_prefill_tokens:
            # vLLM already invoked the inherited stock cache writer exactly
            # once. Only pure prefill can reach the stock execution path.
            if scales is None:
                return super().forward(layer, query, key, value, kv_cache,
                                       md, output)
            output_scale, output_block_scale = scales
            stock_output = torch.empty_like(query, dtype=torch.bfloat16)
            super().forward(layer, query, key, value, kv_cache,
                            md, stock_output)
            n = md.num_actual_tokens
            torch.ops._C.scaled_fp4_quant.out(
                input=stock_output[:n].reshape(n, self.num_heads * self.head_size),
                input_scale=output_scale,
                is_sf_swizzled_layout=True,
                output=output[:n].reshape(n, self.num_heads * self.head_size // 2),
                output_scale=output_block_scale.view(torch.int32),
            )
            return output
        if md.num_decodes:
            _require_trtllm_decode(md)
            pool = self._pool_for(kv_cache, layer)
            if scales is not None:
                output_scale, output_block_scale = scales
                self._decode_forward_nvfp4(
                    query, pool, md, output,
                    output_scale, output_block_scale,
                )
            else:
                self._decode_forward(query, pool, md, output)
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

    @staticmethod
    def get_supported_kernel_block_sizes():
        # vLLM splits larger hybrid-cache allocation blocks into kernel pages.
        return [128]

    @classmethod
    def get_supported_head_sizes(cls):
        return [128, 256]

    @classmethod
    def supports_sink(cls):
        return False
