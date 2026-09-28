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


def derive_decode_mext_weight_layout(
    weight_ext: torch.Tensor,
    weight_scale_ext: torch.Tensor,
    original_in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the base-K prefix tensors needed by decode-time MExt.

    The layer still owns the legacy extended-K residue weight so prefill can
    use the configured residue ratio. Decode MExt only needs the base-K prefix
    plus the corresponding raw block scales.
    """

    k_base = int(original_in_features)
    assert k_base > 0
    assert k_base % 16 == 0
    assert weight_ext.ndim == 2
    assert weight_scale_ext.ndim == 2

    k_ext = weight_ext.shape[1] * 2
    assert k_base <= k_ext

    k_base_packed = k_base // 2
    k_base_groups = k_base // 16
    assert weight_scale_ext.shape[1] >= k_base_groups

    weight_base_view = weight_ext[:, :k_base_packed]
    weight_scale_base_raw = weight_scale_ext[:, :k_base_groups].contiguous()
    return weight_base_view, weight_scale_base_raw


def prepare_decode_mext_weight_layout_from_raw_scale(
    layer,
    raw_weight_scale_ext: torch.Tensor,
) -> None:
    """Attach decode-only MExt helpers to an already-loaded residue layer.

    `super().process_weights_after_loading(...)` swizzles `layer.weight_scale`.
    The base scale for prefix GEMM must therefore be sliced from the raw
    checkpoint layout captured before that swizzle.
    """

    _, weight_scale_base_raw = derive_decode_mext_weight_layout(
        layer.weight,
        raw_weight_scale_ext,
        layer.original_in_features,
    )

    layer.weight_scale_base = Parameter(
        swizzle_blockscale(weight_scale_base_raw),
        requires_grad=False,
    )
    layer._mext_decode_k_base = int(layer.original_in_features)
    layer._decode_mext_layout_enabled = True
