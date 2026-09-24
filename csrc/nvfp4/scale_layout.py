# SPDX-License-Identifier: Apache-2.0
"""Shared 128x4 byte-scale layout for NVFP4 and MXFP4 operators."""
import torch
import torch.nn.functional as F


def swizzle_scales(scales):
    if (scales.ndim not in (2, 3) or scales.dtype != torch.uint8
            or not scales.is_cuda or not scales.is_contiguous()):
        raise ValueError(
            "expected contiguous CUDA uint8 [M,G] or [B,M,G] scales"
        )
    rows, groups = scales.shape[-2:]
    if groups <= 0:
        raise ValueError("scale K extent must be nonempty")
    padded_rows = ((rows + 127) // 128) * 128
    padded_groups = ((groups + 3) // 4) * 4
    batched = scales.ndim == 3
    batch = scales.shape[0] if batched else 1
    padded = F.pad(
        scales.reshape(batch, rows, groups),
        (0, padded_groups - groups, 0, padded_rows - rows),
    )
    result = (
        padded.reshape(
            batch, padded_rows // 128, 4, 32, padded_groups // 4, 4
        )
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(batch, padded_rows * padded_groups)
    )
    return result if batched else result[0]
