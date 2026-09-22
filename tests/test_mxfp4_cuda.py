"""Run with PYTHONPATH=csrc/nvfp4:. python -m pytest ... on the target GPU."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from nvfp4.mxfp4 import (quantize, quantize_mext,
                         swizzle_scales, gemm, fold_gemm)
from mxfp4_reference import dequantize
from mxfp4_scale_layout import unswizzle_scales
from mxfp4_f32_oracle import quantize as ref_quant, structured_indices


@pytest.mark.parametrize("m,k", [(1, 32), (7, 96), (129, 160), (8, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("recipe", ["ceil"])
def test_quant_bytes_and_layout(m, k, dtype, recipe):
    torch.manual_seed(17)
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    x[0, :8] = torch.tensor([0., -0., .25, .75, 1.25, 1.75, 3.5, 7.], device="cuda")
    q, sf = quantize(x, layout="linear", recipe=recipe)
    expected_q, expected_sf = ref_quant(x.float().cpu().numpy(), recipe=recipe, dtype=dtype)
    np.testing.assert_array_equal(q.cpu(), expected_q)
    np.testing.assert_array_equal(sf.cpu(), expected_sf)
    qs, ss = quantize(x, recipe=recipe)
    torch.testing.assert_close(qs, q, rtol=0, atol=0)
    torch.testing.assert_close(ss, swizzle_scales(sf), rtol=0, atol=0)
    torch.testing.assert_close(unswizzle_scales(ss, m, k), sf, rtol=0, atol=0)
    # Independent index equation, not just a self-consistent round trip.
    linear = sf.cpu().numpy()
    expected = np.zeros(ss.numel(), np.uint8)
    for row in range(m):
        for group in range(k // 32):
            tile = (row // 128) * ((k + 127) // 128) + group // 4
            index = tile * 512 + (row % 32) * 16 + (row % 128 // 32) * 4 + group % 4
            expected[index] = linear[row, group]
    np.testing.assert_array_equal(ss.cpu(), expected)


def test_quant_mext():
    torch.manual_seed(19)
    x = torch.randn((7, 160), device="cuda")
    pq, ps = quantize_mext(x, layout="linear")
    base_q, base_s = ref_quant(x.cpu().numpy())
    err_q, err_s = ref_quant(x.cpu().numpy() - dequantize(base_q, base_s))
    np.testing.assert_array_equal(pq[::2].cpu(), base_q)
    np.testing.assert_array_equal(ps[::2].cpu(), base_s)
    np.testing.assert_array_equal(pq[1::2].cpu(), err_q)
    np.testing.assert_array_equal(ps[1::2].cpu(), err_s)


@pytest.mark.parametrize("m,pad_multiple", [(1, 8), (7, 64), (8, 8), (9, 8), (128, 8)])
def test_quant_mext_tile8_matches_row_pair(m, pad_multiple):
    torch.manual_seed(109 + m)
    k = 4096
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    row_q, row_sf = quantize_mext(x)
    tile_q, tile_sf = quantize_mext(
        x, mext_layout="tile8", m_pad_multiple=pad_multiple)
    output_m = ((m + pad_multiple - 1) // pad_multiple) * pad_multiple
    assert tile_q.shape == (2 * output_m, k // 2)
    blocks_q = tile_q.reshape(output_m // 8, 2, 8, k // 2)
    base_q = blocks_q[:, 0].reshape(output_m, k // 2)
    res_q = blocks_q[:, 1].reshape(output_m, k // 2)
    torch.testing.assert_close(base_q[:m], row_q[::2], rtol=0, atol=0)
    torch.testing.assert_close(res_q[:m], row_q[1::2], rtol=0, atol=0)
    assert not bool(torch.count_nonzero(base_q[m:]))
    assert not bool(torch.count_nonzero(res_q[m:]))

    row_linear = unswizzle_scales(row_sf, 2 * m, k)
    tile_linear = unswizzle_scales(tile_sf, 2 * output_m, k)
    blocks_sf = tile_linear.reshape(output_m // 8, 2, 8, k // 32)
    base_sf = blocks_sf[:, 0].reshape(output_m, k // 32)
    res_sf = blocks_sf[:, 1].reshape(output_m, k // 32)
    torch.testing.assert_close(base_sf[:m], row_linear[::2], rtol=0, atol=0)
    torch.testing.assert_close(res_sf[:m], row_linear[1::2], rtol=0, atol=0)
    assert not bool(torch.count_nonzero(base_sf[m:]))
    assert not bool(torch.count_nonzero(res_sf[m:]))


def test_mxfp4_swap_ab_fold_matches_row_pair():
    torch.manual_seed(223)
    m, n, k = 128, 256, 4096
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    wq, ws = quantize(w)
    row_q, row_sf = quantize_mext(x)
    tile_q, tile_sf = quantize_mext(x, mext_layout="tile8")
    reference = fold_gemm(row_q, wq, row_sf, ws)
    for mode, pi in (("tma_mext", False), ("direct_mext", True)):
        actual = fold_gemm(
            tile_q, wq, tile_sf, ws, store_mode=mode,
            mma_tiler_mn=(128, 64), pi_via_layout=pi)
        torch.testing.assert_close(actual, reference, rtol=.02, atol=.125)


def test_quant_zero_and_scale_extremes():
    maxima = np.array([0., -0., 2.**-127, 2.**-120, 2.**120,
                       np.finfo(np.float32).max], dtype=np.float32)
    source = np.repeat(maxima[:,None],32,axis=1)
    x = torch.from_numpy(source).cuda()
    for recipe in ("ceil",):
        q,s = quantize(x,layout="linear",recipe=recipe)
        rq,rs = ref_quant(source,recipe=recipe)
        np.testing.assert_array_equal(q.cpu(),rq)
        np.testing.assert_array_equal(s.cpu(),rs)


@pytest.mark.parametrize("m,n,k", [(8,128,128), (65,256,384), (128,128,4096)])
def test_dense_and_fold(m,n,k):
    if torch.cuda.get_device_capability() not in ((10,0),(10,3)):
        pytest.skip("SM100/SM103 required for FP4 Tensor Core GEMM")
    torch.manual_seed(23)
    x = torch.randn((m,k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n,k), device="cuda", dtype=torch.bfloat16)
    aq,sa = quantize(x)
    bq,sb = quantize(w)
    def decode(q,s):
        lin = unswizzle_scales(s,q.shape[0],k)
        return torch.from_numpy(dequantize(q.cpu().numpy(),lin.cpu().numpy())).cuda()
    xd,wd = decode(aq,sa), decode(bq,sb)
    expected = xd @ wd.T
    actual = gemm(aq,bq,sa,sb).float()
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)
    pq,sp = quantize_mext(x)
    pd = decode(pq,sp)
    expected_fold = (pd[::2] @ wd.T) + (pd[1::2] @ wd.T)
    actual_fold = fold_gemm(pq,bq,sp,sb).float()
    torch.testing.assert_close(actual_fold, expected_fold, rtol=.02, atol=.125)


def test_nvfp4_fold_cache_isolation():
    if torch.cuda.get_device_capability() not in ((10,0),(10,3)):
        pytest.skip("SM100/SM103 required")
    from nvfp4.cute_fold.host import mext_fold_gemm_sm103
    # All decoded values are exactly one; only SF format/encoding differ.
    aq=torch.full((16,128),0x22,dtype=torch.uint8,device="cuda")
    bq=torch.full((128,128),0x22,dtype=torch.uint8,device="cuda")
    mxsf=torch.full((1024,),127,dtype=torch.uint8,device="cuda")
    nvsf=torch.full((2048,),56,dtype=torch.uint8,device="cuda")
    mx=fold_gemm(aq,bq,mxsf,mxsf)
    nv=mext_fold_gemm_sm103(bq,aq,nvsf,nvsf,torch.ones(1,device="cuda"),
                           kernel_arch="sm100" if torch.cuda.get_device_capability()==(10,0) else "sm103")
    torch.testing.assert_close(mx,torch.full_like(mx,512),rtol=0,atol=0)
    torch.testing.assert_close(nv,torch.full_like(nv,512),rtol=0,atol=0)


@pytest.mark.parametrize("m", [1, 7, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_direct_dense_against_decoded_reference(m, dtype):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("direct output is SM100-only")
    torch.manual_seed(83)
    a, sa = quantize(torch.randn(m, 384, device="cuda", dtype=dtype))
    b, sb = quantize(torch.randn(256, 384, device="cuda", dtype=dtype))
    ad = torch.from_numpy(dequantize(
        a.cpu().numpy(), unswizzle_scales(sa, m, 384).cpu().numpy()
    )).cuda()
    bd = torch.from_numpy(dequantize(
        b.cpu().numpy(), unswizzle_scales(sb, 256, 384).cpu().numpy()
    )).cuda()
    expected = ad @ bd.T
    actual = gemm(a, b, sa, sb, out_dtype=dtype).float()
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)
