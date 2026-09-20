"""Independent on-disk checks exercise shapes, NaN scale codes, and shard index."""
import json
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file
from quant.mxfp4_checkpoint import FORMAT
from export.stages.inspect_quant_export import InspectQuantExportStage
from quant.modelopt_integration.mxfp4_export import (
    _collect_preserved_state,
    _validate_modelopt_export,
)
from quant.mxfp4_checkpoint import runtime_layer_specs


def test_dense_qwen3_uses_hf_layer_names_and_fusion_contract():
    layers = {
        "model.layers.0.self_attn.q_proj": {
            "mode": "standard", "input_size": 256, "output_size": 256,
        },
        "model.layers.0.self_attn.k_proj": {
            "mode": "standard", "input_size": 256, "output_size": 64,
        },
        "model.layers.0.self_attn.v_proj": {
            "mode": "standard", "input_size": 256, "output_size": 64,
        },
    }
    actual = runtime_layer_specs(layers, "qwen3")
    assert actual == {
        "model.layers.0.self_attn.qkv_proj": {
            "mode": "standard", "input_size": 256, "output_size": 384,
        }
    }


@pytest.mark.parametrize("damage", [None,"width","nan_scale","index","unsharded"])
def test_inspect_opens_actual_mxfp4_tensors(tmp_path,damage):
    prefix="model.layers.0.self_attn.o_proj"
    spec=dict(mode="selective",input_size=256,output_size=256,indices=list(range(0,256,8)))
    layers={prefix:spec}
    contract=dict(FORMAT,quant_method="residue_mxfp4",layers=layers)
    (tmp_path/"config.json").write_text(json.dumps(dict(hidden_size=256,num_attention_heads=4,quantization_config=contract)))
    (tmp_path/"residue_mxfp4_config.json").write_text(json.dumps(contract))
    q=torch.zeros(256,192,dtype=torch.uint8)
    sf=torch.full((256,12),127,dtype=torch.uint8)
    if damage=="width":q=q[:,:128].contiguous()
    if damage=="nan_scale":sf[0,0]=255
    tensors={prefix+".weight":q,prefix+".weight_scale":sf}
    save_file(tensors,tmp_path/"model.safetensors")
    index={name:"model.safetensors" for name in tensors}
    if damage=="index":index.pop(prefix+".weight_scale")
    if damage != "unsharded":
        (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":index}))
    if damage in (None,"unsharded"):
        InspectQuantExportStage._assert_mxfp4_tensors(tmp_path,layers)
    else:
        with pytest.raises(AssertionError):
            InspectQuantExportStage._assert_mxfp4_tensors(tmp_path,layers)


@pytest.mark.parametrize("indexed", [False, True])
def test_modelopt_export_bytes_are_not_repacked(tmp_path,indexed):
    prefix="model.layers.0.self_attn.o_proj"
    spec=dict(mode="selective",input_size=256,output_size=256,
              indices=list(range(0,256,8)))
    native={"quant_method":"modelopt","quant_algo":"MXFP4"}
    (tmp_path/"config.json").write_text(json.dumps({"quantization_config":native}))
    tensors={
        prefix+".weight":torch.arange(256*192,dtype=torch.int64).to(torch.uint8).reshape(256,192),
        prefix+".weight_scale":torch.zeros(256,12,dtype=torch.uint8),
        prefix+".input_quantizer._salient_indices":torch.arange(0,256,8),
    }
    shard=tmp_path/"model.safetensors"
    save_file(tensors,shard,metadata={"format":"pt","producer":"modelopt"})
    if indexed:
        (tmp_path/"model.safetensors.index.json").write_text(json.dumps({
            "metadata":{"total_size":sum(t.numel()*t.element_size() for t in tensors.values())},
            "weight_map":{name:shard.name for name in tensors},
        }))
    before=tensors[prefix+".weight"].clone()
    _,actual_native,hashes=_validate_modelopt_export(tmp_path,{prefix:spec},{})
    assert actual_native==native and set(hashes)=={prefix+".weight",prefix+".weight_scale"}
    from safetensors import safe_open
    with safe_open(shard,framework="pt") as handle:
        assert handle.metadata()=={"format":"pt","producer":"modelopt"}
        assert prefix+".input_quantizer._salient_indices" not in handle.keys()
        torch.testing.assert_close(handle.get_tensor(prefix+".weight"),before,rtol=0,atol=0)
    index_path=tmp_path/"model.safetensors.index.json"
    if indexed:
        index=json.loads(index_path.read_text())
        assert prefix+".input_quantizer._salient_indices" not in index["weight_map"]
    else:
        assert not index_path.exists()


def test_unsharded_source_checkpoint_preservation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.ones(128, 256),
        "model.embed_tokens.weight": torch.arange(512, dtype=torch.float32).reshape(2, 256),
        "mtp.proj.weight": torch.full((128, 256), 2.0),
    }
    save_file(tensors, source / "model.safetensors")

    model = torch.nn.Module()
    model.register_parameter(
        "placeholder", torch.nn.Parameter(torch.zeros(1)),
    )
    model.state_dict = lambda: {
        "model.layers.0.self_attn.q_proj.weight": tensors[
            "model.layers.0.self_attn.q_proj.weight"
        ],
        "model.embed_tokens.weight": tensors["model.embed_tokens.weight"],
    }
    extra, hashes, record = _collect_preserved_state(
        model,
        source,
        {"model.layers.0.self_attn.q_proj": {}},
        ("mtp.",),
    )
    assert set(extra) == {"mtp.proj.weight"}
    assert set(hashes) == {"model.embed_tokens.weight", "mtp.proj.weight"}
    assert record["weight_inventory"] == {
        "kind": "single_safetensors",
        "filename": "model.safetensors",
    }
