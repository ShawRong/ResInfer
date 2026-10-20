"""Bounded-memory synthetic TRT-LLM KV and independent attention reference."""
import torch

from benchmarks.precision.nvfp4_reference import dequant_nvfp4_kv_cache


E2M1_LUT = (0., .5, 1., 1.5, 2., 3., 4., 6.)
E2M1_MAX = 6.0

_LUT_EDGES_CACHE: dict = {}


def _lut_and_edges(device):
    key = str(device)
    if key not in _LUT_EDGES_CACHE:
        lut = torch.tensor(E2M1_LUT, device=device)
        _LUT_EDGES_CACHE[key] = (lut, (lut[1:] + lut[:-1]) / 2)
    return _LUT_EDGES_CACHE[key]


def _e2m1_nibbles(y: torch.Tensor) -> torch.Tensor:
    """y 已除 SF 并 clamp ±6 → e2m1 nibble(uint8,低 3 位幅值序号+符号位)。"""
    lut, edges = _lut_and_edges(y.device)
    idx = torch.bucketize(y.abs(), edges)
    return (idx + torch.where(y < 0, 8, 0)).to(torch.uint8)


def _nibbles_decode(nib: torch.Tensor) -> torch.Tensor:
    """e2m1 nibble → float 值(gather_dequant 用)。"""
    lut, _ = _lut_and_edges(nib.device)
    mag = lut[(nib & 7).long()]
    return torch.where(nib >= 8, -mag, mag)


def _quant_groups(x: torch.Tensor):
    """沿最后轴按 16 一组量化。x (..., G, 16) → (nib (..., G, 16), sf_bytes (..., G))。

    SF = amax/6 → e4m3(无全局缩放);nib 未打包、未 π。
    """
    sf = (x.abs().amax(-1) / E2M1_MAX).to(torch.float8_e4m3fn)
    sf_f = sf.float().clamp_min(1e-12)
    y = (x / sf_f.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
    return _e2m1_nibbles(y), sf.view(torch.uint8)


def _pack_low_high(nib: torch.Tensor) -> torch.Tensor:
    """相邻 nibble 打包:偶下标进低位。(..., 2n) uint8 → (..., n) uint8。"""
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()


def make_kv(pages, heads, device, dim=128):
    """Quantize Gaussian pages in chunks; never allocate a full FP32 KV cache."""
    data = [torch.empty(pages, heads, 128, dim // 2, device=device, dtype=torch.uint8)
            for _ in range(2)]
    scales = [torch.empty(pages, heads, 128, dim // 16, device=device,
                          dtype=torch.float8_e4m3fn) for _ in range(2)]
    for start in range(0, pages, 64):
        end = min(start + 64, pages)
        for i in range(2):
            x = torch.randn(end-start, heads, 128, dim // 16, 16, device=device,
                            dtype=torch.bfloat16).float()
            nib, sf = _quant_groups(x)
            data[i][start:end] = _pack_low_high(nib.reshape(end-start,heads,128,dim))
            if i:
                sf = sf.reshape(end-start, heads, 32, 4, dim // 16).transpose(-1,-2).contiguous()
            scales[i][start:end] = sf.reshape(end-start,heads,128,dim // 16).view(torch.float8_e4m3fn)
    return data[0], scales[0], data[1], scales[1]


def attention_reference(q, kd, ks, vd, vs, table, lengths):
    """FP32 streaming softmax over dequantized KV; includes every logical head."""
    # Keep the oracle independent of the runtime TF32 preference.
    torch.backends.cuda.matmul.allow_tf32 = False
    b, hq, d = q.shape
    hk = kd.shape[1]
    g = hq // hk
    result = torch.zeros_like(q, dtype=torch.float32)
    for row in range(b):
        length = int(lengths[row])
        query = q[row].reshape(hk,g,d).float()
        maximum = torch.full((hk,g,1),-torch.inf,device=q.device)
        denom = torch.zeros_like(maximum)
        numerator = torch.zeros((hk,g,d),device=q.device)
        for first in range(0,(length+127)//128,32):
            ids = table[row,first:min(first+32,(length+127)//128)].long()
            count = ids.numel()
            values = []
            for data, scale, interleaved in [(kd,ks,False),(vd,vs,True)]:
                deq = dequant_nvfp4_kv_cache(
                    data[ids], scale[ids], 1.0, d, 128, interleaved)
                values.append(deq.permute(1,0,2,3).reshape(hk,count*128,d)[:,:min(count*128,length-first*128)])
            key, value = values
            scores = torch.bmm(query,key.transpose(1,2)) * d**-.5
            next_max = torch.maximum(maximum,scores.amax(-1,keepdim=True))
            correction = torch.exp(maximum-next_max)
            probs = torch.exp(scores-next_max)
            numerator = numerator*correction + torch.bmm(probs,value)
            denom = denom*correction + probs.sum(-1,keepdim=True)
            maximum = next_max
        if length: result[row] = (numerator/denom).reshape(hq,d)
    return result
