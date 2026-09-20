"""Export-environment tests for ModelOpt's MXFP4/residue interfaces."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("modelopt")
from quant.modelopt_integration.tensor_quantizer import ResidueTensorQuantizer
from quant.modelopt_integration.registry import build_residue_cfg
from quant.modelopt_integration.algo import _restore_quantizer_amax
from quant.mxfp4_reference import quantize, dequantize


def test_dynamic_amax_restore_and_format():
    q = ResidueTensorQuantizer({"num_bits": (2, 1), "block_sizes": {-1:32,"type":"dynamic","scale_bits":(8,0)}})
    _restore_quantizer_amax(q,None)
    assert q.amax is None
    with pytest.raises(ValueError, match="Unsupported"):
        build_residue_cfg("fp4_typo",.125)


def test_mxfp4_materialized_width_is_gemm_aligned():
    from types import SimpleNamespace
    from quant.modelopt_integration.algo import _materialized_residue_width
    mx=SimpleNamespace(weight_quantizer=SimpleNamespace(
        block_sizes={-1:32,"type":"dynamic","scale_bits":(8,0)}))
    nv=SimpleNamespace(weight_quantizer=SimpleNamespace(
        block_sizes={-1:16,"scale_bits":(4,3)}))
    assert _materialized_residue_width(mx,256,32)==384
    assert _materialized_residue_width(mx,4096,512)==4608
    assert _materialized_residue_width(nv,256,32)==288


@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16,torch.float16])
def test_mxfp4_activation_residue_matches_rne(dtype):
    if not torch.cuda.is_available():
        pytest.skip("ModelOpt dynamic MX QDQ requires CUDA")
    # With amax=6, scale=1: .75,1.75,3.5 distinguish RNE from ties-down.
    x=torch.tensor([6,.75,1.75,3.5,-.75,-1.75,-3.5,0]*32,device="cuda",dtype=dtype).reshape(1,256)
    idx=torch.arange(32,device="cuda")
    q=ResidueTensorQuantizer({"num_bits":(2,1),"block_sizes":{-1:32,"type":"dynamic","scale_bits":(8,0)}}).cuda()
    actual=q.residue_extension(x,idx)
    packed,scale=quantize(x.float().cpu().numpy(),recipe="ceil",min_scale_exponent=-126)
    base=dequantize(packed,scale)
    expected=np.concatenate((x.float().cpu().numpy(),(x.float().cpu().numpy()-base)[:,:32]),axis=1)
    np.testing.assert_array_equal(actual.float().cpu().numpy(),expected)


def test_nvfp4_residue_signature_preserved():
    from quant.modelopt_integration.util import append_quant_residual_channels
    calls = []
    class FakeNV:
        @classmethod
        def quantize(cls, tensor, block, scale, double_scale, **kwargs):
            calls.append((block, scale, double_scale, kwargs))
            return cls(), scale, double_scale
        def dequantize(self, **kwargs):
            return torch.ones(1,32)
    x = torch.arange(32).float().reshape(1,32)
    indices = torch.tensor([0,7,16])
    scale,double_scale = torch.tensor(2.),torch.tensor(3.)
    actual = append_quant_residual_channels(x,16,scale,double_scale,indices,FakeNV)
    torch.testing.assert_close(actual,torch.cat((x,(x-1)[:,indices]),dim=1))
    assert calls == [(16,scale,double_scale,dict(keep_high_precision=False,try_tensorrt=True))]


def test_mx_residue_respects_calibration_quant_disable(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import modelopt.torch.quantization as mtq
    import modelopt.torch.quantization.plugins.custom as custom_plugins
    from quant.modelopt_integration.algo import inject_residue_salient_map
    # This unit uses a plain torch Linear. Avoid unrelated HF plugin discovery,
    # which is version-coupled to Transformers and is covered by export smokes.
    monkeypatch.setattr(custom_plugins,"register_custom_model_plugins_on_the_fly",
                        lambda model: None)
    monkeypatch.setattr(custom_plugins,"register_custom_post_conversion_plugins",
                        lambda model: None)
    torch.manual_seed(89)
    model = torch.nn.Sequential(torch.nn.Linear(256,128,bias=False)).cuda().bfloat16()
    original = model[0].weight.detach().clone()
    model = mtq.quantize(model, mtq.MXFP4_DEFAULT_CFG)
    inject_residue_salient_map(model,{"0":torch.arange(32,device="cuda")})
    model[0].input_quantizer.disable_quant()
    model[0].weight_quantizer.disable_quant()
    x = torch.zeros(1,256,device="cuda",dtype=torch.bfloat16)
    x[0,0] = 1
    with torch.inference_mode():
        torch.testing.assert_close(model(x),original[:,0].unsqueeze(0),rtol=0,atol=0)
