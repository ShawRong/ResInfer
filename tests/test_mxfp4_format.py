import pytest
from quant.mxfp4_checkpoint import FORMAT, validate_format, validate_layer, fuse_projection_specs


def test_format_rejects_nvfp4_and_missing_metadata():
    validate_format(FORMAT)
    with pytest.raises(ValueError, match="checkpoint_format"):
        validate_format(dict(FORMAT, checkpoint_format="residue_modelopt_mxfp4_v1"))
    producers = FORMAT["activation_quantizers"]
    assert producers["standard"] == "flashinfer_cute_dsl"
    assert producers["selective"] == producers["mext_r1"]
    assert producers["selective"] == "torch.ops.nvfp4.mxfp4_residue_quant"
    for key in FORMAT:
        bad = dict(FORMAT)
        del bad[key]
        with pytest.raises(ValueError):
            validate_format(bad)
    with pytest.raises(ValueError):
        validate_format(dict(FORMAT, block_size=16))
    with pytest.raises(ValueError, match="activation_min_scale_exponent"):
        validate_format(dict(FORMAT, activation_min_scale_exponent=-126))


def test_fused_indices_and_partial_scope_rejected():
    spec = dict(mode="selective", input_size=256, output_size=128,
                indices=list(range(0,256,8)))
    layers = {f"model.layers.0.self_attn.{p}_proj":dict(spec) for p in ("q","k","v")}
    fused = fuse_projection_specs(layers)
    assert fused["model.layers.0.self_attn.qkv_proj"]["output_size"] == 384
    layers["model.layers.0.self_attn.k_proj"] = dict(
        spec, indices=list(range(1,256,8)))
    with pytest.raises(ValueError,match="share"):
        fuse_projection_specs(layers)
    layers.pop("model.layers.0.self_attn.k_proj")
    with pytest.raises(ValueError,match="partially"):
        fuse_projection_specs(layers)


@pytest.mark.parametrize("indices", [list(range(31)), list(range(32))[::-1], [0]*32, list(range(225,257))])
def test_bad_selective_indices(indices):
    with pytest.raises(ValueError):
        validate_layer(dict(mode="selective",input_size=256,output_size=128,indices=indices))


def test_qwen_vl_prefix_and_gdn_fusion():
    from quant.mxfp4_checkpoint import runtime_layer_specs
    prefix="model.language_model.layers.0.linear_attn."
    spec=dict(mode="selective",input_size=256,output_size=128,
              indices=[i+j for i in range(0,256,8) for j in (0,1)])
    layers={prefix+"in_proj_qkv":spec,prefix+"in_proj_z":dict(spec,output_size=64)}
    result=runtime_layer_specs(layers,"qwen3_5")
    assert result["language_model.model.layers.0.linear_attn.in_proj_qkvz"]["output_size"]==192
    with pytest.raises(ValueError,match="language-model"):
        runtime_layer_specs({"model.visual.proj":spec},"qwen3_5")
