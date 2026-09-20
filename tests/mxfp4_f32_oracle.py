"""Independent CPU FP32 MXFP4 arithmetic; native BF16 bytes checked separately."""
import numpy as np
from quant.mxfp4_reference import dequantize


def quantize(source,*,recipe='ceil',dtype=None):
    assert recipe=='ceil'
    x=np.asarray(source,dtype=np.float32)
    m,k=x.shape
    blocks=x.reshape(m,k//32,32)
    amax=np.abs(blocks).max(-1)
    normalized=np.multiply(amax,np.float32(1./6.),dtype=np.float32)
    bits=normalized.view(np.uint32)
    exponent=(bits>>23)&255;mantissa=bits&0x7fffff
    bump=(mantissa!=0)&~((exponent==0)&(mantissa<=0x400000))
    code=np.where(normalized<=0,0,np.minimum(exponent+bump,254)).astype(np.uint8)
    inverse_bits=np.where(code==0,0,(254-code.astype(np.uint32))<<23).astype(np.uint32)
    values=np.multiply(blocks,inverse_bits.view(np.float32)[...,None],dtype=np.float32)
    mids=np.array([.25,.75,1.25,1.75,2.5,3.5,5.],np.float32)
    mag=np.abs(values)
    q=np.searchsorted(mids,mag,side='left').astype(np.uint8)
    for lower in (1,3,5):q+=(mag==mids[lower]).astype(np.uint8)
    q|=(np.signbit(values).astype(np.uint8)<<3)
    q=q.reshape(m,k)
    return q[:,::2]|(q[:,1::2]<<4),code


def structured_indices(k,ratio,device='cpu'):
    import torch
    n=int(ratio*8)
    local=torch.rand(k//8,8).argsort(-1)[:,:n].sort(-1).values
    return (local+torch.arange(k//8)[:,None]*8).flatten().to(device)
