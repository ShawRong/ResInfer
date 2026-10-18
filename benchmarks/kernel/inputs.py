# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight preparation and residue masks shared by kernel/precision benchmarks.

Quantization and scale layouts use the existing vLLM/FlashInfer APIs.
Preparation stays outside timing; no GPU packages are imported until called.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def global_scale(t: torch.Tensor) -> torch.Tensor:
    """vLLM/CUTLASS FP4 convention: 448*6/amax, passed into the quantiser."""
    return (448.0 * 6.0 / t.abs().max().float()).reshape(())


def make_weight(n: int, k: int, device: torch.device, dtype: torch.dtype):
    """Quantize a weight with vLLM; retain BF16 for selective K-extension."""
    import torch
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale)

    w_bf16 = torch.randn(n, k, device=device, dtype=dtype) / (k ** 0.5)
    w_gs = global_scale(w_bf16)
    w_fp4, w_sf = scaled_fp4_quant(w_bf16, w_gs, is_sf_swizzled_layout=False)
    return w_bf16, w_fp4, swizzle_blockscale(w_sf), w_gs


def salient_indices(k: int, ratio: float, device: torch.device) -> torch.Tensor:
    """The per-8-channel salient pattern the export format uses.

    Selection is structurally confined to each 8-channel block
    (`residue_per_8`), so a benchmark that picked globally would be measuring
    a layout the exporter cannot produce. Which channels within the block are
    picked does not affect cost, so a deterministic prefix is used.
    """
    import torch
    if k % 8:
        raise ValueError(f"k must be a multiple of 8, got {k}")
    per_block = int(round(ratio * 8))
    if per_block == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    base = torch.arange(0, k, 8, dtype=torch.long, device=device)
    off = torch.arange(per_block, dtype=torch.long, device=device)
    return (base[:, None] + off[None, :]).reshape(-1).contiguous()


def make_extended_weights(w_ref, k, ratios, indices_by_ratio=None):
    import torch
    import nvfp4
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

    extended = {}
    for ratio in ratios:
        idx = (salient_indices(k, ratio, w_ref.device) if indices_by_ratio is None else
               torch.tensor(indices_by_ratio[ratio], device=w_ref.device, dtype=torch.long))
        w_ext = torch.cat([w_ref, w_ref.index_select(-1, idx)], dim=-1)
        gs = global_scale(w_ext)
        packed, scales = scaled_fp4_quant(w_ext, gs, is_sf_swizzled_layout=False)
        extended[ratio] = (packed, swizzle_blockscale(scales),
            swizzle_blockscale(scales[:, :k // 16].contiguous()), gs,
            nvfp4.indices_to_channel_masks(idx, k), idx.numel())
    return extended


def prepare_mext_weights(n, k, device, arms):
    import torch
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16) / (k ** 0.5)
    return quantize_mext_weights(weight, arms)


def quantize_mext_weights(weight, arms):
    """Pack a supplied real or synthetic weight for the shared MEXT runners."""
    import torch
    from flashinfer import SfLayout, mxfp4_quantize, nvfp4_quantize
    from .mext_trtllm_adapter import prepare_weight as prepare_trtllm_weight, require_dense_ops

    wgs = global_scale(weight)
    w_nv, ws_nv = nvfp4_quantize(weight, wgs, sfLayout=SfLayout.layout_128x4)
    ws_mext = ws_nv.view(torch.float8_e4m3fn)
    # FlashInfer quantizers already return SM100-interleaved scale
    # factors. Re-interleaving them silently corrupts W4A8 output.
    w_mx, ws_mx = mxfp4_quantize(weight)
    ws_mx_flat = ws_mx.flatten().contiguous()
    w_mx_batch = w_mx.unsqueeze(0)
    ws_mx_batch = ws_mx.unsqueeze(0)
    if "trtllm_nvfp4_fp8_w4a8" in arms:
        trtllm_fp4_utils = require_dense_ops()
        w_trt, ws_trt, wgs_trt = prepare_trtllm_weight(weight, trtllm_fp4_utils)
    return {"reference": weight, "nvfp4": (w_nv, ws_nv, ws_mext, wgs),
            "mxfp4": (w_mx, ws_mx, ws_mx_flat, w_mx_batch, ws_mx_batch),
            "trtllm": (w_trt, ws_trt, wgs_trt)
            if "trtllm_nvfp4_fp8_w4a8" in arms else None}


def calibrated_indices(metadata, layer_name, k, ratio):
    """Resolve the export's mask, including shared HF constituents of a fused layer."""
    from quant.vllm_integration.metadata import (
        get_fused_hf_candidates, get_residue_layer_candidates)

    layers = metadata["layers"]
    groups = get_fused_hf_candidates(layer_name)
    constituents = {name for group in groups for name in group}
    whole = [name for name in get_residue_layer_candidates(layer_name)
             if name not in constituents and name in layers]
    candidates = [[name] for name in whole] + groups
    records = next(([layers[name] for name in group] for group in candidates
                    if all(name in layers for name in group)), None)
    if not records:
        raise ValueError(f"Missing calibrated mask for {layer_name}")
    indices = records[0].get("salient_indices")
    if indices is None or any(row.get("salient_indices") != indices for row in records):
        raise ValueError(f"Fused projections need identical calibrated masks: {layer_name}")
    per_block = int(ratio * 8)
    if (ratio not in (.125, .25, .5) or k % 8 or len(indices) != k * per_block // 8
            or any(type(i) is not int or i < 0 or i >= k for i in indices)
            or any(a >= b for a, b in zip(indices, indices[1:]))
            or any(i // 8 != j // per_block for j, i in enumerate(indices))):
        raise ValueError(f"Mask does not match K={k}, ratio={ratio}: {layer_name}")
    return indices
