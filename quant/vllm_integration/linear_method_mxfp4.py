# SPDX-License-Identifier: Apache-2.0
"""SM100 MXFP4: floating/dummy preparation and explicit packed ModelOpt loading."""
from __future__ import annotations

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

from quant.mxfp4_linear import mxfp4_linear

_MODES = ("standard", "selective", "mext_r1")


@register_quantization_config("residue_mxfp4")
class ResidueMxfp4Config(QuantizationConfig):
    """Per-layer explicit modes and channel indices; default is standard MXFP4."""

    def __init__(self, layers=None, default_mode="standard", checkpoint_format=None,
                 activation_recipe="ceil", quantize_unlisted=True,
                 **format_metadata):
        super().__init__()
        if default_mode not in ("standard", "mext_r1"):
            raise ValueError("default_mode must be standard or mext_r1")
        self.checkpoint_format = checkpoint_format
        self.activation_recipe = activation_recipe
        self.quantize_unlisted = quantize_unlisted
        if checkpoint_format is not None:
            from quant.mxfp4_checkpoint import validate_format
            validate_format(dict(checkpoint_format=checkpoint_format,
                                 activation_recipe=activation_recipe, **format_metadata))
        elif activation_recipe != "ceil":
            raise ValueError("MXFP4 uses vLLM/FlashInfer ceil; legacy ocp is unsupported")
        self.layers = dict(layers or {})
        self.default_mode = default_mode
        if checkpoint_format is not None:
            from quant.mxfp4_checkpoint import validate_layer
            if not self.layers:
                raise ValueError("packed MXFP4 checkpoint requires explicit layers")
            for spec in self.layers.values():
                validate_layer(spec)
        for prefix, spec in self.layers.items():
            if spec.get("mode") not in _MODES:
                raise ValueError(f"unknown MXFP4 mode for {prefix}")
            if spec["mode"] == "selective" and not spec.get("indices"):
                raise ValueError(f"selective layer {prefix} requires explicit indices")
            if spec["mode"] != "selective" and spec.get("indices"):
                raise ValueError(f"indices are only valid for selective layers: {prefix}")

    @classmethod
    def get_name(cls):
        return "residue_mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 100

    @staticmethod
    def get_config_filenames():
        return ["residue_mxfp4_config.json"]

    @classmethod
    def from_config(cls, config):
        return cls(**{k: v for k, v in config.items() if k != "quant_method"})

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            if (self.checkpoint_format is not None or not self.quantize_unlisted) and prefix not in self.layers:
                return UnquantizedLinearMethod()
            return ResidueMxfp4LinearMethod(
                self.layers.get(prefix, {"mode": self.default_mode}),
                packed=self.checkpoint_format is not None, activation_recipe=self.activation_recipe)
        return None


class ResidueMxfp4LinearMethod(LinearMethodBase):
    """Float weight loader -> one-time quantization -> opaque linear forward."""

    def __init__(self, spec, *, packed=False, activation_recipe="ceil"):
        self.packed = packed
        self.activation_recipe = activation_recipe
        self.spec = dict(spec)
        self.mode = spec["mode"]
        self.indices = tuple(spec.get("indices", ()))
        if self.mode not in _MODES:
            raise ValueError(f"unknown mode {self.mode}")

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        n = sum(output_partition_sizes)
        k = input_size_per_partition
        if input_size != k or output_size != n:
            raise ValueError("MXFP4 linear currently supports unsharded/TP1 weights only")
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("load MXFP4 linear from fp16 or bf16 weights")
        if k <= 0 or k % 128 or n <= 0 or n % 32:
            raise ValueError("MXFP4 linear requires positive K%128==0 and N%32==0")
        if self.packed:
            from quant.mxfp4_checkpoint import padded_k, validate_layer
            validate_layer(self.spec)
            if (k, n) != (self.spec["input_size"], self.spec["output_size"]):
                raise ValueError("checkpoint dimensions do not match this runtime linear")
            kp = padded_k(k, len(self.indices))
            loader = extra_weight_attrs.pop("weight_loader")
            layer._mxfp4_loaded = {"weight": set(), "weight_scale": set()}
            layer._mxfp4_shards = len(output_partition_sizes)
            def checked_loader(name):
                def load(param, tensor, *args, **kwargs):
                    if tensor.dtype != torch.uint8:
                        raise ValueError(f"packed MXFP4 {name} must be uint8")
                    loader(param, tensor, *args, **kwargs)
                    shard = args[0] if args else kwargs.get("loaded_shard_id")
                    if isinstance(shard, (tuple, list)):
                        layer._mxfp4_loaded[name].update(shard)
                    else:
                        layer._mxfp4_loaded[name].add(shard)
                return load
            for name, width in (("weight", kp // 2), ("weight_scale", kp // 32)):
                param = Parameter(torch.empty(n, width, dtype=torch.uint8), requires_grad=False)
                set_weight_attrs(param, {"input_dim": 1, "output_dim": 0,
                                       **extra_weight_attrs, "weight_loader": checked_loader(name)})
                layer.register_parameter(name, param)
        else:
            layer.register_parameter("weight", Parameter(torch.empty(n, k, dtype=params_dtype), requires_grad=False))
            set_weight_attrs(layer.weight, {"input_dim": 1, "output_dim": 0, **extra_weight_attrs})
        layer._mxfp4_input_size = k
        layer._mxfp4_output_dtype = params_dtype
        layer._mxfp4_prepared = False

    def process_weights_after_loading(self, layer):
        from nvfp4.mxfp4 import quantize, prepare_indices, prepare_selective_weight, swizzle_scales
        if layer._mxfp4_prepared:
            return
        w = layer.weight.detach()
        if not w.is_cuda or torch.cuda.get_device_capability(w.device) != (10, 0):
            raise ValueError("MXFP4 weight preparation requires SM100")
        with torch.no_grad(), torch.cuda.device(w.device):
            indices = torch.empty(0, dtype=torch.int32, device=w.device)
            masks = torch.empty(0, dtype=torch.uint8, device=w.device)
            if self.mode == "selective":
                indices = prepare_indices(self.indices, layer._mxfp4_input_size, w.device)
                masks = indices._mxfp4_masks
            if self.packed:
                for name, loaded in layer._mxfp4_loaded.items():
                    if None not in loaded and len(loaded) != layer._mxfp4_shards:
                        raise ValueError(f"incomplete packed MXFP4 {name} loading: {loaded}")
                if (layer.weight_scale == 255).any().item():
                    raise ValueError("MXFP4 checkpoint contains NaN E8M0 scale code 255")
                q = w
                sf = swizzle_scales(layer.weight_scale.detach())
                del layer.weight_scale
            elif self.mode == "selective":
                q, sf = prepare_selective_weight(w, indices, recipe=self.activation_recipe)
            else:
                q, sf = quantize(w, recipe=self.activation_recipe)
            layer.weight = Parameter(q, requires_grad=False)
            layer.register_buffer("weight_scale", sf)
            layer.register_buffer("salient_indices", indices)
            layer.register_buffer("salient_masks", masks)
            layer._mxfp4_prepared = True

    def apply(self, layer, x, bias=None):
        if not layer._mxfp4_prepared:
            raise RuntimeError("call process_weights_after_loading before MXFP4 apply")
        if x.shape[-1] != layer._mxfp4_input_size:
            raise ValueError("input feature dimension does not match loaded MXFP4 weight")
        return mxfp4_linear(x, layer.weight, layer.weight_scale, layer.salient_indices,
                           layer.salient_masks,
                           self.mode, layer._mxfp4_output_dtype, bias, self.activation_recipe)
