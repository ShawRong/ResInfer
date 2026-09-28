"""Small tensor-layout utilities shared by the vLLM integration."""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

try:
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        swizzle_blockscale,
    )
except ImportError:
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )


def prepare_decode_mext_scale(layer, raw_weight_scale_ext: torch.Tensor) -> None:
    """Prepare the base-K scale used by decode MExt after weight loading.

    Slice the raw checkpoint scale before swizzling: the full extended-K
    layout cannot be sliced after swizzling to obtain the base-K layout.
    The weight itself remains shared; Fold takes its base-K view at runtime.
    """
    k_base = int(layer.original_in_features)
    assert k_base > 0 and k_base % 16 == 0
    assert layer.weight.ndim == 2
    assert raw_weight_scale_ext.ndim == 2
    assert k_base <= layer.weight.shape[1] * 2
    assert raw_weight_scale_ext.shape[1] >= k_base // 16

    base_scale = raw_weight_scale_ext[:, :k_base // 16].contiguous()
    layer.weight_scale_base = Parameter(
        swizzle_blockscale(base_scale), requires_grad=False,
    )
    layer._mext_decode_k_base = k_base
    layer._decode_mext_layout_enabled = True
