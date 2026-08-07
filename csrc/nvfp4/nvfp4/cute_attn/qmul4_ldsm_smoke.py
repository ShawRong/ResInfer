# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Isolated smoke gate for the TRT-style ``QMUL4.N6420.B3210`` primitive.

The full decode path obtains these byte containers from ``LDSM.U4``.  This
small kernel accepts the same register representation directly (one FP4
nibble in the low half of every byte), so instruction encoding and numerics
can be verified before the LDSM/STTM/TMEM pipeline is integrated.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from .qmul4 import dequantize_nvfp4x8_ldsm


@cute.kernel
def _ldsm_smoke_kernel(
    src: cute.Tensor,
    scales: cute.Tensor,
    dst: cute.Tensor,
    n_blocks: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        for block in cutlass.range_constexpr(n_blocks):
            src_block = cute.local_tile(src, (8,), (block,))
            scale_block = cute.local_tile(scales, (8,), (block,))
            dst_block = cute.local_tile(dst, (8,), (block,))
            dst_block.store(dequantize_nvfp4x8_ldsm(src_block, scale_block))


@cute.jit
def _ldsm_smoke_launch(
    src: cute.Tensor,
    scales: cute.Tensor,
    dst: cute.Tensor,
    n_blocks: cutlass.Constexpr,
):
    _ldsm_smoke_kernel(src, scales, dst, n_blocks).launch(
        grid=(1, 1, 1), block=(32, 1, 1)
    )


def run_smoke(n_blocks: int = 8, seed: int = 11):
    """Run the primitive and return ``(max_abs_err, n_wrong, lines)``."""
    import torch
    from cutlass.cute.runtime import from_dlpack

    from .kv_layout import E2M1_LUT

    n = n_blocks * 8
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lut = torch.tensor(E2M1_LUT)
    index = torch.randint(0, 8, (n,), generator=generator)
    sign = torch.randint(0, 2, (n,), generator=generator)
    values = torch.where(sign.bool(), -lut[index], lut[index])
    nibbles = (index + sign * 8).to(torch.uint8).cuda()
    scale_pattern = torch.tensor(
        [0.5, 1.0, 2.0, 4.0, 0.25, 0.5, 1.0, 2.0],
        dtype=torch.float32,
    )
    scales_f32 = scale_pattern.repeat(n_blocks)
    scales = scales_f32.to(torch.float8_e4m3fn).cuda()
    dst = torch.zeros(n, dtype=torch.float8_e4m3fn, device="cuda")

    src_ct = from_dlpack(nibbles, assumed_align=16).mark_layout_dynamic()
    src_ct.element_type = cutlass.Int8
    scales_ct = from_dlpack(scales, assumed_align=16).mark_layout_dynamic()
    dst_ct = from_dlpack(dst, assumed_align=16).mark_layout_dynamic()
    compiled = cute.compile(
        _ldsm_smoke_launch, src_ct, scales_ct, dst_ct, n_blocks
    )
    compiled(src_ct, scales_ct, dst_ct)
    torch.cuda.synchronize()

    got = dst.float().cpu()
    want = values * scales_f32
    n_wrong = int((got != want).sum())
    max_err = float((got - want).abs().max())
    lines = [f"max|err|={max_err:.4f}  wrong={n_wrong}/{n}"]
    for i in (got != want).nonzero().flatten()[:8].tolist():
        lines.append(
            f"  i={i:3d} got={got[i]:8.3f} want={want[i]:8.3f}"
        )
    return max_err, n_wrong, lines


__all__ = ["run_smoke"]
