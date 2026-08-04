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


_FOLD_REGISTER_WARNED = set()


def _warn_once_fold_register(tag, exc):
    if tag in _FOLD_REGISTER_WARNED:
        return
    _FOLD_REGISTER_WARNED.add(tag)
    print(f"[residue] sm120 fold shape registration failed ({tag}): "
          f"{type(exc).__name__}: {exc}", flush=True)


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

    weight_base_view, weight_scale_base_raw = derive_decode_mext_weight_layout(
        layer.weight,
        raw_weight_scale_ext,
        layer.original_in_features,
    )

    layer.weight_base_view = weight_base_view
    layer.weight_scale_base = Parameter(
        swizzle_blockscale(weight_scale_base_raw),
        requires_grad=False,
    )
    layer._mext_decode_k_base = int(layer.original_in_features)
    layer._decode_mext_layout_enabled = True
    # Tell the SM120 CuTeDSL warmup this shape exists. k_ext is the STORED
    # width, which differs from k_base here and selects a different compiled
    # kernel (strided weight read), so both numbers matter.
    try:
        from .residue_linear_op import (_base_k_weight_view,
                                        register_sm120_fold_shape)

        # Derive the warmup key from the SAME tensor the op will hand the
        # kernel. This is the ext-K branch; the plain-NVFP4 and mext_r1
        # branches register from linear_method_v2 (_register_fold_shape_for).
        k_base = int(layer.original_in_features)
        view = _base_k_weight_view(layer.weight, k_base)
        register_sm120_fold_shape(
            int(view.shape[0]), k_base, int(view.stride(0)) * 2,
            is_mext_r1=False)   # ext-K layer: keeps the decode-band threshold
    except Exception as e:                                # noqa: BLE001
        # Never block weight loading -- but do not swallow it either: a silent
        # miss here leaves the warmup with nothing to compile, and the only
        # symptom is a slow cudagraph capture.
        _warn_once_fold_register("extended_k", e)
