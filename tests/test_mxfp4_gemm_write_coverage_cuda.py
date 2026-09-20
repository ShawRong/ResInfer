"""Poison output allocation to diagnose GEMM initcheck reports independently."""
import pytest
torch=pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('CUDA required',allow_module_level=True)
from nvfp4.mxfp4 import quantize,gemm


@pytest.mark.parametrize('k',[4608,5120,6144])
def test_direct_gemm_writes_poisoned_output(monkeypatch,k):
    torch.manual_seed(931)
    a,sa=quantize(torch.randn(7,k,device='cuda',dtype=torch.bfloat16))
    b,sb=quantize(torch.randn(128,k,device='cuda',dtype=torch.bfloat16))
    allocate=torch.empty
    def poison(*args,**kwargs):
        t=allocate(*args,**kwargs)
        if t.is_cuda and t.dtype in (torch.float16,torch.bfloat16):t.fill_(float('nan'))
        return t
    monkeypatch.setattr(torch,'empty',poison)
    expected=gemm(a,b,sa,sb)
    assert torch.isfinite(expected).all()
    for _ in range(5):
        actual=gemm(a,b,sa,sb)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
