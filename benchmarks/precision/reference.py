"""Shared high-precision references for operator benchmarks."""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from .nvfp4_reference import dequant_nvfp4_kv_cache


def metrics(output, reference):
    a, b = output.double(), reference.double()
    assert a.shape == b.shape
    nonfinite = int((~torch.isfinite(a) | ~torch.isfinite(b)).sum())
    if nonfinite:
        raise ValueError(f'nonfinite output/reference: {nonfinite}')
    error = a - b
    sse, energy, n = float(error.square().sum()), float(b.square().sum()), b.numel()
    return dict(sse=sse, reference_energy=energy, n=n, mse=sse/n,
                nmse=sse/energy if energy else None,
                zero_reference=energy == 0, maxabs=float(error.abs().max()), nonfinite=0)


def attention(q, k, v, scale, dtype=torch.float32):
    """One decode query attending to all supplied past/current KV; output [1,Hq,D]."""
    assert q.ndim == 3 and q.shape[0] == 1 and k.shape == v.shape
    # [1,Hq,D] -> [1,Hq,1,D]; GQA is handled by PyTorch's math backend.
    with sdpa_kernel(SDPBackend.MATH):
        output = scaled_dot_product_attention(
            q.to(dtype).unsqueeze(2), k.to(dtype).transpose(0, 1).unsqueeze(0),
            v.to(dtype).transpose(0, 1).unsqueeze(0),
            dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=True,
        )
    return output.squeeze(2)


def unpack_kv(pool, pages, length):
    """Gather valid HND pages and decode stock K-linear/V-swizzled scales to FP32."""
    ps, hk, d = pool.page_size, pool.num_kv_heads, pool.head_dim
    ids = pages[:(length + ps - 1) // ps].long()
    outputs = []
    for data, sf, gs, swizzled in [
        (pool.k_data, pool.k_sf, pool.k_global_scale, False),
        (pool.v_data, pool.v_sf, pool.v_global_scale, True),
    ]:
        values = dequant_nvfp4_kv_cache(data[ids], sf[ids], gs, d, ps, swizzled)
        outputs.append(values.permute(0, 2, 1, 3).reshape(-1, hk, d)[:length])
    return tuple(outputs)
