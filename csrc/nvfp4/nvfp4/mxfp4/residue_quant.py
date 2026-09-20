# SPDX-License-Identifier: Apache-2.0
"""Allocation wrappers for the wheel-built SM100 MXFP4 residue operator."""
import torch

from ..extension import ensure_loaded


def _run(x,q,sf,masks,per8,residue,swizzle):
    ensure_loaded()
    torch.ops.nvfp4.mxfp4_residue_quant(
        x,q,sf,masks,per8,residue,swizzle)


def quantize_fused(x,*,masks=None,per8=0,residue=False,layout='linear',kp=None):
    m,k=x.shape
    kp=k if kp is None else kp
    rows=m*(2 if residue and masks is None else 1)
    q=torch.empty((rows,kp//2),device=x.device,dtype=torch.uint8)
    sshape=(rows,kp//32) if layout=='linear' else (((rows+127)//128)*((kp+127)//128)*512,)
    sf=torch.empty(sshape,device=x.device,dtype=torch.uint8)
    if m:
        _run(x,q,sf,masks,per8,residue,layout=='128x4')
    return q,sf
