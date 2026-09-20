"""Upstream producer dispatch, residual source precision and graph contracts."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('CUDA required',allow_module_level=True)
from nvfp4.mxfp4 import quantize,quantize_mext,quantize_selective,prepare_indices
from quant.mxfp4_reference import dequantize
from mxfp4_f32_oracle import quantize as native_quant, structured_indices


def test_reject_legacy_ocp():
    x=torch.zeros(1,256,device='cuda',dtype=torch.bfloat16)
    for fn in (quantize,quantize_mext):
        with pytest.raises(ValueError,match='legacy ocp'):
            fn(x,recipe='ocp')
    with pytest.raises(ValueError,match='legacy ocp'):
        quantize_selective(x,prepare_indices(structured_indices(256,.125),256,'cuda'),recipe='ocp')


def test_calls_upstream_standard_and_fused_residue(monkeypatch):
    import flashinfer
    import flashinfer.quantization.kernels.mxfp4_quantize as native
    # Warm import and compilation before replacing either upstream entry.
    x=torch.randn(3,256,device='cuda',dtype=torch.bfloat16)
    quantize(x);quantize_mext(x)
    public=flashinfer.mxfp4_quantize;linear=native.mxfp4_quantize_cute_dsl
    calls=[]
    def public_spy(*args,**kwargs):
        calls.append(('public',kwargs['backend']))
        return public(*args,**kwargs)
    def linear_spy(*args,**kwargs):
        calls.append(('native',kwargs.get('sf_layout')))
        return linear(*args,**kwargs)
    monkeypatch.setattr(flashinfer,'mxfp4_quantize',public_spy)
    monkeypatch.setattr(native,'mxfp4_quantize_cute_dsl',linear_spy)
    quantize(x)
    assert ('public','cute-dsl') in calls
    import nvfp4.mxfp4.residue_quant as module
    run=module._run;launches=[]
    def fused_spy(*args):
        launches.append(1)
        return run(*args)
    monkeypatch.setattr(module,'_run',fused_spy)
    # Kernel inputs/outputs stay on GPU. No eager arithmetic/layout fallback.
    idx=prepare_indices(structured_indices(256,.25),256,'cuda')
    def forbidden(*args,**kwargs):
        raise AssertionError('Torch residue/layout operation reached hot path')
    for name in ('stack','cat','where','exp','exp2','ldexp'):
        monkeypatch.setattr(torch,name,forbidden)
    calls.clear();quantize_mext(x)
    assert not calls and len(launches)==1
    quantize_selective(x,idx)
    assert not calls and len(launches)==2


def test_fp32_residue_uses_original_source():
    x=torch.full((2,256),1.0019,device='cuda')
    q,s=quantize_mext(x,layout='linear')
    bq,bs=native_quant(x.cpu().numpy())
    rq,rs=native_quant(x.cpu().numpy()-dequantize(bq,bs))
    np.testing.assert_array_equal(q[1::2].cpu(),rq)
    np.testing.assert_array_equal(s[1::2].cpu(),rs)
    assert torch.count_nonzero(q[1::2]).item()>0


def test_quant_graph_on_nondefault_stream():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x=torch.randn(7,256,device='cuda')
        idx=prepare_indices(structured_indices(256,.25),256,'cuda')
        def forward():
            return (*quantize_mext(x),*quantize_selective(x,idx))
        for _ in range(3):expected=forward()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):actual=forward()
        graph.replay()
    stream.synchronize()
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
