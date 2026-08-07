# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'``dequantize_nvfp4x16`` 的**隔离**冒烟验证(一条 QMUL4 链,不含流水线)。'

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from .qmul4 import dequantize_nvfp4x16


@cute.kernel
def _smoke_kernel(src: cute.Tensor, sf: cute.Tensor, dst: cute.Tensor,
                  n_blocks: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        for b in cutlass.range_constexpr(n_blocks):
            blk = cute.local_tile(src, (16,), (b,))
            out = cute.local_tile(dst, (16,), (b,))
            out.store(dequantize_nvfp4x16(blk, sf[b]))


@cute.jit
def _smoke_launch(src: cute.Tensor, sf: cute.Tensor, dst: cute.Tensor,
                  n_blocks: cutlass.Constexpr):
    _smoke_kernel(src, sf, dst, n_blocks).launch(
        grid=(1, 1, 1), block=(32, 1, 1))


def run_smoke(n_blocks: int = 8, seed: int = 3):
    """在当前设备上跑一次原语,返回 ``(max_abs_err, n_wrong, lines)``。

    输入刻意取在 **e2m1 网格上**(直接从 LUT 采样)、scale 取 1.0,这样
    参考值是精确的 —— 任何偏差都是 kernel 的错,不是量化误差。
    """
    import torch
    from cutlass.cute.runtime import from_dlpack

    from .kv_layout import E2M1_LUT

    n = n_blocks * 16
    g = torch.Generator(device="cpu").manual_seed(seed)
    lut = torch.tensor(E2M1_LUT)
    idx = torch.randint(0, 8, (n,), generator=g)
    sign = torch.randint(0, 2, (n,), generator=g)
    vals = torch.where(sign.bool(), -lut[idx], lut[idx])
    nib = (idx + sign * 8).to(torch.uint8)
    packed = (nib[0::2] | (nib[1::2] << 4)).contiguous().cuda()
    sf_f32 = torch.full((n_blocks,), 1.0, dtype=torch.float32)
    sf_e4m3 = sf_f32.to(torch.float8_e4m3fn).cuda()
    dst = torch.zeros(n, dtype=torch.float8_e4m3fn, device="cuda")

    # fp4 视图:uint8 缓冲 .view(float4_e2m1fn_x2) 后逻辑末维翻倍(host.py 同款)
    src_ct = from_dlpack(
        packed.view(torch.float4_e2m1fn_x2), assumed_align=16
    ).mark_layout_dynamic()
    sf_ct = from_dlpack(sf_e4m3, assumed_align=16).mark_layout_dynamic()
    dst_ct = from_dlpack(dst, assumed_align=16).mark_layout_dynamic()

    compiled = cute.compile(_smoke_launch, src_ct, sf_ct, dst_ct, n_blocks)
    compiled(src_ct, sf_ct, dst_ct)
    torch.cuda.synchronize()

    got = dst.float().cpu()
    want = vals * sf_f32.repeat_interleave(16).float()
    n_wrong = int((got != want).sum())
    max_err = (got - want).abs().max().item()
    lines = [f"max|err|={max_err:.4f}  错值={n_wrong}/{n}"]
    shown = 0
    for i in range(n):
        if got[i] != want[i] and shown < 8:
            lines.append(f"  i={i:3d} blk={i // 16} slot={i % 16} "
                         f"got={got[i]:8.3f} want={want[i]:8.3f}")
            shown += 1
    return max_err, n_wrong, lines


__all__ = ["run_smoke"]
