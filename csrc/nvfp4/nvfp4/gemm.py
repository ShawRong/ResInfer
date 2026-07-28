"""Python allocation wrappers for local NVFP4 CUTLASS GEMM kernels."""

from __future__ import annotations

import torch

from .extension import ensure_loaded

ensure_loaded()

_MEXT_ROW_PAIR_FOLD_TACTICS = frozenset(range(16, 32))
_MEXT_TILE8_FOLD_TACTICS = frozenset(range(16))
_MEXT_FOLD_TACTICS = _MEXT_TILE8_FOLD_TACTICS | _MEXT_ROW_PAIR_FOLD_TACTICS


def cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean(
    a: torch.Tensor,
    b_ext: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    out_dtype: torch.dtype,
):
    assert a.ndim == 2 and b_ext.ndim == 2
    m2, n = a.shape[0], b_ext.shape[0]
    if m2 % 16 != 0:
        raise ValueError(
            f"tile8 clean vec-store fold requires A rows divisible by 16, got {m2}"
        )
    out = torch.empty((m2 // 2, n), dtype=out_dtype, device=a.device)
    torch.ops.nvfp4.cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean(
        out,
        a,
        b_ext,
        block_scale_a,
        block_scale_b,
        alpha,
        int(k_base),
    )
    return out


def cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tactic(
    a: torch.Tensor,
    b_ext: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    out_dtype: torch.dtype,
    tactic: int,
):
    """Run a tactic-selected MExt fold GEMM.

    Tactics 0..15 use the original tile8 row-wise fold layout. Tactics 16..31
    are encoded row-pair col-swap tactics, where the path-local tactic is
    tactic - 16. The C++ op keeps the historic long name for ABI compatibility.
    """
    assert a.ndim == 2 and b_ext.ndim == 2
    m2, n = a.shape[0], b_ext.shape[0]
    if tactic not in _MEXT_FOLD_TACTICS:
        raise ValueError(
            "mext fold tactic must be one of tile8 row-wise tactics 0..15 "
            "or encoded row-pair col-swap tactics 16..31, "
            f"got {tactic}"
        )
    if tactic in _MEXT_ROW_PAIR_FOLD_TACTICS:
        if m2 % 2 != 0:
            raise ValueError(
                f"row-pair swap fold tactic requires A rows divisible by 2, got {m2}"
            )
    elif m2 % 16 != 0:
        raise ValueError(
            f"tile8 clean vec-store fold tactic requires A rows divisible by 16, got {m2}"
        )
    out = torch.empty((m2 // 2, n), dtype=out_dtype, device=a.device)
    torch.ops.nvfp4.cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tactic(
        out,
        a,
        b_ext,
        block_scale_a,
        block_scale_b,
        alpha,
        int(k_base),
        int(tactic),
    )
    return out


def cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean_tactic(
    *args,
    **kwargs,
):
    """Compatibility alias for the historic tactic-selected MExt fold wrapper."""
    return cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tactic(*args, **kwargs)


def cutlass_scaled_fp4_mm_ext_b_prefix(
    a: torch.Tensor,
    b_ext: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    out_dtype: torch.dtype,
):
    assert a.ndim == 2 and b_ext.ndim == 2
    m, n = a.shape[0], b_ext.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops.nvfp4.cutlass_scaled_fp4_mm_ext_b_prefix(
        out,
        a,
        b_ext,
        block_scale_a,
        block_scale_b,
        alpha,
        int(k_base),
    )
    return out


def cutlass_scaled_fp4_mm_ext_b_prefix_tactic(
    a: torch.Tensor,
    b_ext: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    out_dtype: torch.dtype,
    tactic: int,
):
    assert a.ndim == 2 and b_ext.ndim == 2
    if tactic < 0 or tactic > 15:
        raise ValueError(f"prefix tactic must be in [0, 15], got {tactic}")
    m, n = a.shape[0], b_ext.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops.nvfp4.cutlass_scaled_fp4_mm_ext_b_prefix_tactic(
        out,
        a,
        b_ext,
        block_scale_a,
        block_scale_b,
        alpha,
        int(k_base),
        int(tactic),
    )
    return out


def cutlass_scaled_fp4_mm_ext_b_prefix_swap_tactic(
    a: torch.Tensor,
    b_ext: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    out_dtype: torch.dtype,
    tactic: int,
):
    assert a.ndim == 2 and b_ext.ndim == 2
    if tactic < 0 or tactic > 15:
        raise ValueError(f"prefix swap-ab tactic must be in [0, 15], got {tactic}")
    m, n = a.shape[0], b_ext.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops.nvfp4.cutlass_scaled_fp4_mm_ext_b_prefix_swap_tactic(
        out,
        a,
        b_ext,
        block_scale_a,
        block_scale_b,
        alpha,
        int(k_base),
        int(tactic),
    )
    return out


__all__ = [
    "cutlass_scaled_fp4_mm_ext_b_prefix",
    "cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tactic",
    "cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean",
    "cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean_tactic",
    "cutlass_scaled_fp4_mm_ext_b_prefix_swap_tactic",
    "cutlass_scaled_fp4_mm_ext_b_prefix_tactic",
]
