# SPDX-License-Identifier: Apache-2.0
"""Checkpoint contract shared by ModelOpt export and the SM100 vLLM loader."""
FORMAT = {
    "checkpoint_format": "residue_modelopt_mxfp4_v1",
    "block_size": 32,
    "weight_encoding": "e2m1_even_low_nibble",
    "scale_encoding": "e8m0_uint8",
    "scale_layout": "linear",
    "activation_recipe": "ceil",
    "activation_rounding": "nearest_even",
    "activation_min_scale_exponent": -127,
    "activation_quantizer": "flashinfer_cute_dsl",
    # Stable producer contract. The exact ModelOpt package version belongs in
    # export provenance, not in the runtime ABI checked by the loader.
    "weight_quantizer": "modelopt_mxfp4",
}


def validate_format(config):
    for key, value in FORMAT.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported MXFP4 {key}: {config.get(key)!r}; expected {value!r}")


def padded_k(k, selected=0):
    return ((k + selected + 127) // 128) * 128


def validate_layer(spec):
    k, n = spec.get("input_size"), spec.get("output_size")
    if type(k) is not int or k <= 0 or k % 128 or type(n) is not int or n <= 0 or n % 32:
        raise ValueError("MXFP4 requires positive input_size%128==0 and output_size%32==0")
    mode = spec.get("mode")
    if mode not in ("standard", "selective", "mext_r1"):
        raise ValueError(f"unsupported MXFP4 mode {mode}")
    indices = spec.get("indices", [])
    if mode == "selective":
        if (len(indices) not in (k//8, k//4, k//2)
                or any(type(i) is not int or i < 0 or i >= k for i in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))
                or any(value//8 != j//(len(indices)*8//k) for j,value in enumerate(indices))):
            raise ValueError("selective indices must be sorted, unique, in range, with exactly 1/2/4 selections per group of 8")
    elif indices:
        raise ValueError("indices are only stored for selective mode")


def fuse_projection_specs(layers):
    """Fused HF projections must share one activation-residue contract."""
    result = dict(layers)
    for suffixes, target in ((("q_proj", "k_proj", "v_proj"), "qkv_proj"),
                             (("gate_proj", "up_proj"), "gate_up_proj"),
                             (("in_proj_qkv", "in_proj_z"), "in_proj_qkvz")):
        parents = {name.rsplit(".", 1)[0] for name in layers if name.rsplit(".", 1)[-1] in suffixes}
        for parent in sorted(parents):
            names = [f"{parent}.{suffix}" for suffix in suffixes]
            if any(name not in layers for name in names):
                raise ValueError(f"partially quantized fused group is unsupported: {names}")
            specs = [layers[name] for name in names]
            first = {k: v for k, v in specs[0].items() if k != "output_size"}
            if any({k: v for k, v in spec.items() if k != "output_size"} != first for spec in specs):
                raise ValueError(f"fused group must share mode, indices, input size: {names}")
            for name in names:
                del result[name]
            result[f"{parent}.{target}"] = dict(first, output_size=sum(s["output_size"] for s in specs))
    return result


def runtime_layer_specs(layers, model_type):
    fused = fuse_projection_specs(layers)
    if model_type == "qwen3_5":
        source_prefix = "model.language_model."
        if any(not name.startswith(source_prefix) for name in fused):
            raise ValueError("Qwen3.5 MXFP4 scope must be language-model linears")
        return {"language_model.model." + name[len(source_prefix):]: spec
                for name, spec in fused.items()}
    if model_type not in ("llama", "qwen3"):
        raise ValueError(f"unsupported MXFP4 model_type: {model_type}")
    return fused
