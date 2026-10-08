"""Public Python API for the local NVFP4 CUDA extension."""

from .extension import ensure_loaded
from .mext import scaled_fp4_quant_mext_r1
from .quant import (
    indices_to_channel_masks,
    scaled_fp4_quant,
    scaled_fp4_quant_with_mask,
)

__all__ = [
    "ensure_loaded",
    "indices_to_channel_masks",
    "scaled_fp4_quant",
    "scaled_fp4_quant_mext_r1",
    "scaled_fp4_quant_with_mask",
]
