"""Independent inverse of the CUTLASS 128x4 E8M0 scale permutation."""
import torch


def unswizzle_scales(scales: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    if rows < 0 or k <= 0 or k % 32:
        raise ValueError("rows must be nonnegative and K a positive multiple of 32")
    if scales.dtype != torch.uint8 or not scales.is_cuda or not scales.is_contiguous():
        raise ValueError("expected contiguous CUDA uint8 scales")
    row_tiles = (rows + 127) // 128
    groups = k // 32
    group_tiles = (groups + 3) // 4
    expected = row_tiles * group_tiles * 512
    if scales.ndim != 1 or scales.numel() != expected:
        raise ValueError(f"expected {expected} swizzled scale bytes")
    padded = (
        scales.reshape(row_tiles, group_tiles, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(row_tiles * 128, group_tiles * 4)
    )
    return padded[:rows, :groups].contiguous()
