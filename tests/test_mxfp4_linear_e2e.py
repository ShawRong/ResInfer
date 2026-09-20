"""Actual vLLM ReplicatedLinear lifecycle + dummy MLP, no model downloads."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
pytest.importorskip("vllm")

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import get_quantization_config
from quant.vllm_integration.linear_method_mxfp4 import ResidueMxfp4Config
from quant.mxfp4_reference import dequantize, quantize as weight_ref_quant
from mxfp4_f32_oracle import quantize as ref_quant, structured_indices
from nvfp4.mxfp4 import unswizzle_scales


@pytest.fixture(autouse=True)
def inference_only():
    # Parametrized tests create unrelated layer instances with different
    # static modes/index sizes. Do not share Dynamo's per-code recompile
    # budget across these independent models.
    torch._dynamo.reset()
    try:
        with torch.inference_mode():
            yield
    finally:
        torch._dynamo.reset()


def reference(layer, x, mode, bias=None):
    shape = x.shape
    xf = x.reshape(-1,shape[-1]).float().cpu().numpy()
    recipe = layer.quant_method.activation_recipe
    q,s = ref_quant(xf,recipe=recipe)
    base = dequantize(q,s)
    if mode == "selective":
        idx = layer.salient_indices.cpu().numpy()
        rq,rs = ref_quant((xf-base)[:,idx],recipe=recipe)
        base = np.concatenate((base,dequantize(rq,rs)),axis=-1)
    elif mode == "mext_r1":
        rq,rs = ref_quant(xf-base,recipe=recipe)
        base += dequantize(rq,rs)
    kp = layer.weight.shape[1]*2
    if base.shape[1] < kp:
        base = np.pad(base,((0,0),(0,kp-base.shape[1])))
    ws = unswizzle_scales(layer.weight_scale,layer.weight.shape[0],kp)
    wd = dequantize(layer.weight.cpu().numpy(),ws.cpu().numpy())
    y = torch.from_numpy(base@wd.T).to(device=x.device,dtype=layer._mxfp4_output_dtype)
    if bias is not None:
        y = y + bias
    return y.reshape(*shape[:-1],layer.weight.shape[0])


def make_linear(k,n,mode,ratio=.25,prefix="proj",skip_bias_add=False):
    spec = {"mode":mode}
    if mode == "selective":
        # Uneven, deterministic distribution across source blocks.
        spec["indices"] = structured_indices(k,ratio).tolist()
    config = ResidueMxfp4Config({prefix:spec})
    with torch.device("cuda"):
        layer = ReplicatedLinear(k,n,params_dtype=torch.bfloat16,quant_config=config,
                                 prefix=prefix,disable_tp=True,skip_bias_add=skip_bias_add)
    w = torch.randn(n,k,device="cuda",dtype=torch.bfloat16)*.1
    bias = torch.randn(n,device="cuda",dtype=torch.bfloat16)*.01
    layer.weight.weight_loader(layer.weight,w)
    layer.bias.weight_loader(layer.bias,bias)
    layer.quant_method.process_weights_after_loading(layer)
    before = layer.weight.data_ptr()
    layer.quant_method.process_weights_after_loading(layer)  # idempotent
    assert layer.weight.data_ptr()==before
    assert layer.weight.dtype==torch.uint8
    if mode == "selective":
        # Tensor Python attributes do not survive Module._apply. The runtime
        # must use its separately registered mask buffer after such a move.
        layer.salient_indices = layer.salient_indices.clone()
        assert not hasattr(layer.salient_indices, "_mxfp4_masks")
    return layer


@pytest.mark.parametrize("mode,ratio", [("standard",0),("mext_r1",1),("selective",.125),("selective",.25),("selective",.5)])
def test_linear_lifecycle_compile_graph(mode,ratio):
    torch.manual_seed(61)
    assert get_quantization_config("residue_mxfp4") is ResidueMxfp4Config
    layer = make_linear(256,128,mode,ratio)
    for m in (1,7,65):
        x = torch.randn((m,256),device="cuda",dtype=torch.bfloat16)
        actual,returned_bias = layer(x)
        assert returned_bias is None
        expected = reference(layer,x,mode,layer.bias)
        torch.testing.assert_close(actual,expected,rtol=.03,atol=.0625)
    compiled = torch.compile(layer,backend="inductor",fullgraph=True)
    for m in (1,7):
        x = torch.randn((m,256),device="cuda",dtype=torch.bfloat16)
        actual,_ = compiled(x)
        expected,_ = layer(x)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    x = torch.randn((2,3,256),device="cuda",dtype=torch.bfloat16)
    layer(x)  # warmup before capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out,_ = layer(x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    expected = reference(layer,x,mode,layer.bias)
    torch.testing.assert_close(out,expected,rtol=.03,atol=.0625)
    assert layer(torch.empty((0,256),device="cuda",dtype=torch.bfloat16))[0].shape==(0,128)


def test_dummy_mlp_e2e():
    torch.manual_seed(67)
    first = make_linear(256,256,"selective",.125,prefix="up")
    second = make_linear(256,128,"selective",.5,prefix="down",skip_bias_add=True)
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up,self.down = first,second
        def forward(self,x):
            y,_=self.up(x)
            y,bias=self.down(torch.nn.functional.silu(y))
            return y+bias
    model=MLP()
    x=torch.randn((3,5,256),device="cuda",dtype=torch.bfloat16)
    expected=reference(first,x,"selective",first.bias)
    expected=reference(second,torch.nn.functional.silu(expected),"selective",second.bias)
    actual=model(x)
    torch.testing.assert_close(actual,expected,rtol=.04,atol=.08)
    compiled=torch.compile(model,backend="inductor",fullgraph=True)
    torch.testing.assert_close(compiled(x),actual,rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out=model(x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.testing.assert_close(out,model(x),rtol=0,atol=0)


def test_linear_fp32_bias_dtype_contract():
    layer = make_linear(256,128,"selective",.25)
    x = torch.randn((7,256),device="cuda",dtype=torch.bfloat16)
    bias = torch.randn(128,device="cuda",dtype=torch.float32)*.01
    def forward(x,bias):
        return layer.quant_method.apply(layer,x,bias)
    actual = forward(x,bias)
    expected = (reference(layer,x,"selective").float()+bias).bfloat16()
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual,expected,rtol=.03,atol=.0625)
    compiled = torch.compile(forward,backend="inductor",fullgraph=True)
    torch.testing.assert_close(compiled(x,bias),actual,rtol=0,atol=0)


@pytest.mark.parametrize("mode,ratio", [("standard",0),("mext_r1",1),("selective",.125),("selective",.25),("selective",.5)])
def test_packed_loader(mode, ratio):
    from quant.mxfp4_checkpoint import FORMAT, padded_k
    torch.manual_seed(83)
    spec = {"mode":mode, "input_size":256, "output_size":128}
    if mode == "selective":
        spec["indices"] = structured_indices(256,ratio).tolist()
    config = ResidueMxfp4Config(layers={"proj":spec}, **FORMAT)
    with torch.device("cuda"):
        layer = ReplicatedLinear(256,128,bias=False,params_dtype=torch.bfloat16,
                                 quant_config=config,prefix="proj",disable_tp=True)
    w = torch.randn(128,256).numpy()
    if mode == "selective":
        w = np.concatenate((w,w[:,spec["indices"]]),axis=1)
    w = np.pad(w,((0,0),(0,padded_k(256,len(spec.get("indices",[])))-w.shape[1])))
    q,s = weight_ref_quant(w,recipe="ceil",min_scale_exponent=-126)
    q,s = torch.from_numpy(q).cuda(),torch.from_numpy(s).cuda()
    with pytest.raises(ValueError,match="incomplete"):
        layer.quant_method.process_weights_after_loading(layer)
    with pytest.raises(ValueError,match="uint8"):
        layer.weight.weight_loader(layer.weight,q.float())
    layer.weight.weight_loader(layer.weight,q)
    layer.weight_scale.weight_loader(layer.weight_scale,s)
    layer.quant_method.process_weights_after_loading(layer)
    assert torch.equal(layer.weight,q)
    assert torch.equal(unswizzle_scales(layer.weight_scale,128,w.shape[1]),s)
    for m in (1,7):
        x=torch.randn(m,256,device="cuda",dtype=torch.bfloat16)
        actual=layer(x)[0]
        torch.testing.assert_close(actual,reference(layer,x,mode),rtol=.03,atol=.125)
    compiled=torch.compile(layer,fullgraph=True)
    torch.testing.assert_close(compiled(x)[0],layer(x)[0],rtol=0,atol=0)


def test_qwen_unquantized_and_tuple_shard_loading(monkeypatch):
    # The engine initializes TP; this isolated layer test supplies TP1 rank metadata.
    import vllm.model_executor.parameter as parameter
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    from quant.mxfp4_checkpoint import FORMAT
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, UnquantizedLinearMethod
    config=ResidueMxfp4Config(layers={"gdn":dict(mode="standard",input_size=256,output_size=192)},**FORMAT)
    with torch.device("cuda"):
        small=ReplicatedLinear(256,48,bias=False,params_dtype=torch.bfloat16,
                               quant_config=config,prefix="in_proj_ba",disable_tp=True)
        merged=MergedColumnParallelLinear(256,[32,32,64,64],bias=False,params_dtype=torch.bfloat16,
                                          quant_config=config,prefix="gdn",disable_tp=True)
    assert isinstance(small.quant_method,UnquantizedLinearMethod)
    # Distinct rows/columns expose dropped or reordered GDN constituent shards.
    row=torch.arange(192,device="cuda")[:,None]
    q=((row+torch.arange(128,device="cuda")[None,:])%256).to(torch.uint8)
    s=(124+(row+torch.arange(8,device="cuda")[None,:])%7).to(torch.uint8)
    for name,data in (("weight",q),("weight_scale",s)):
        param=getattr(merged,name)
        param.weight_loader(param,data[:128],(0,1,2))
        param.weight_loader(param,data[128:],3)
    merged.quant_method.process_weights_after_loading(merged)
    assert torch.equal(merged.weight,q)
    assert torch.equal(unswizzle_scales(merged.weight_scale,192,256),s)
