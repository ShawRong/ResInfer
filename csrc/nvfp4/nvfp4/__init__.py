"""Public Python API for the local NVFP4 CUDA extension."""

from .extension import ensure_loaded
from .gemm import (
    cutlass_scaled_fp4_mm_ext_b_prefix,
    cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tactic,
    cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean,
    cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean_tactic,
    cutlass_scaled_fp4_mm_ext_b_prefix_swap_tactic,
    cutlass_scaled_fp4_mm_ext_b_prefix_tactic,
)
from .mext import scaled_fp4_quant_mext_r1
from .quant import (
    indices_to_channel_masks,
    official_scaled_fp4_quant,
    scaled_fp4_quant,
    scaled_fp4_quant_pack8_r0125_no_swap_with_mask,
    scaled_fp4_quant_pack8_r025_no_swap_with_mask,
    scaled_fp4_quant_pack8_r050_no_swap_with_mask,
    scaled_fp4_quant_pack8_r1000_no_swap_with_mask,
    scaled_fp4_quant_with_mask,
)

__all__ = [
    "cutlass_scaled_fp4_mm_ext_b_prefix",
    "cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tactic",
    "cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean",
    "cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean_tactic",
    "cutlass_scaled_fp4_mm_ext_b_prefix_swap_tactic",
    "cutlass_scaled_fp4_mm_ext_b_prefix_tactic",
    "ensure_loaded",
    "indices_to_channel_masks",
    "official_scaled_fp4_quant",
    "scaled_fp4_quant",
    "scaled_fp4_quant_mext_r1",
    "scaled_fp4_quant_pack8_r0125_no_swap_with_mask",
    "scaled_fp4_quant_pack8_r025_no_swap_with_mask",
    "scaled_fp4_quant_pack8_r050_no_swap_with_mask",
    "scaled_fp4_quant_pack8_r1000_no_swap_with_mask",
    "scaled_fp4_quant_with_mask",
]
