import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from nvfp4.mxfp4 import (prepare_indices, quantize_selective,
                         prepare_selective_weight, unswizzle_scales, gemm)
from quant.mxfp4_reference import dequantize
from mxfp4_f32_oracle import quantize as ref_quant, structured_indices


@pytest.mark.parametrize("ratio", [.125, .25, .5])
@pytest.mark.parametrize("k", [256, 768, 4096, 16384])
def test_selective_quant_and_gemm(ratio, k):
    torch.manual_seed(53)
    m, n, s = 7, 128, int(k * ratio)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * .1
    idx = prepare_indices(structured_indices(k,ratio), k, x.device)
    q, sf = quantize_selective(x, idx, layout="linear")
    base_q, base_s = ref_quant(x.float().cpu().numpy())
    error = (x.float().cpu().numpy() - dequantize(base_q, base_s))[:, idx.cpu().numpy()]
    res_q, res_s = ref_quant(error)
    np.testing.assert_array_equal(q[:, :k//2].cpu(), base_q)
    np.testing.assert_array_equal(q[:, k//2:(k+s)//2].cpu(), res_q)
    np.testing.assert_array_equal(sf[:, :k//32].cpu(), base_s)
    np.testing.assert_array_equal(sf[:, k//32:(k+s)//32].cpu(), res_s)
    assert (q[:, (k+s)//2:] == 0).all()
    qs, ss = quantize_selective(x, idx)
    torch.testing.assert_close(qs, q, rtol=0, atol=0)
    kp = q.shape[1] * 2
    torch.testing.assert_close(unswizzle_scales(ss, m, kp), sf, rtol=0, atol=0)
    wq, ws = prepare_selective_weight(w, idx)
    raw = np.concatenate((w.float().cpu().numpy(), w.float().cpu().numpy()[:, idx.cpu().numpy()]), axis=1)
    raw = np.pad(raw, ((0,0), (0,kp-k-s)))
    expected_wq, expected_ws = ref_quant(raw)
    np.testing.assert_array_equal(wq.cpu(), expected_wq)
    np.testing.assert_array_equal(unswizzle_scales(ws,n,kp).cpu(), expected_ws)
    expected = dequantize(q.cpu().numpy(),sf.cpu().numpy()) @ dequantize(expected_wq,expected_ws).T
    actual = gemm(qs,wq,ss,ws).float().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=.02, atol=.125)


def test_selective_rejects_invalid_indices():
    for values in ([0]*32, list(range(1,33))[::-1], list(range(-1,31)), list(range(225,257))):
        with pytest.raises(ValueError):
            prepare_indices(values,256,"cuda")
    with pytest.raises(ValueError):
        prepare_indices(list(range(31)),256,"cuda")
