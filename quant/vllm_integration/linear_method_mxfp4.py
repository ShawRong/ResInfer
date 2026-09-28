# SPDX-License-Identifier: Apache-2.0
"""Native ModelOpt MXFP4 loading and ResInfer Linear execution on SM100."""
from __future__ import annotations

import os

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import ModelOptQuantConfigBase
from vllm.model_executor.utils import set_weight_attrs

from quant.mxfp4_linear import mxfp4_linear

@register_quantization_config("modelopt_mxfp4")
class ModelOptMxfp4Config(ModelOptQuantConfigBase):
    """vLLM adapter for native ModelOpt MXFP4; residue uses the shared sidecar."""

    def __init__(self, exclude_modules=None):
        super().__init__(exclude_modules or [])

    @classmethod
    def get_name(cls):
        return "modelopt_mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 100

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if cls._extract_modelopt_quant_algo(hf_quant_cfg) == "MXFP4":
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config):
        # Upstream's base parser rejects MXFP4 before constructing a method.
        # Keep this adapter local; do not modify the global algorithm whitelist.
        quant = config.get("quantization", config)
        if str(quant.get("quant_algo", "")).upper() != "MXFP4":
            raise ValueError("ModelOpt MXFP4 adapter requires quant_algo=MXFP4")
        if quant.get("group_size", 32) != 32:
            raise ValueError("MXFP4 requires group_size=32")
        if quant.get("kv_cache_quant_algo") or quant.get("kv_cache_scheme"):
            raise ValueError("MXFP4 adapter does not support quantized KV cache")
        excluded = quant.get("exclude_modules", quant.get("ignore", []))
        if not isinstance(excluded, list) or any(not isinstance(v, str) for v in excluded):
            raise ValueError("ModelOpt exclude_modules/ignore must be a list of strings")
        return cls(excluded)

    def get_quant_method(self, layer, prefix):
        from .metadata import (get_residue_metadata, get_residue_layer_candidates,
                               get_fused_hf_candidates)
        from vllm.model_executor.layers.fused_moe import RoutedExperts

        if isinstance(layer, RoutedExperts):
            raise ValueError("ResInfer MXFP4 supports dense Linear layers only")
        if not isinstance(layer, LinearBase):
            return None
        candidates = get_residue_layer_candidates(prefix)
        groups = get_fused_hf_candidates(prefix)
        excluded = [self.is_layer_excluded(name) for name in candidates]
        # An alias of the whole fused layer, or every member of one HF group,
        # may be excluded. A single excluded constituent is incompatible.
        if any(excluded):
            constituents = {name for group in groups for name in group}
            whole = any(self.is_layer_excluded(name) for name in candidates
                        if name not in constituents)
            all_members = any(all(self.is_layer_excluded(name) for name in group)
                              for group in groups)
            if not whole and not all_members:
                raise ValueError(f"partially quantized fused group: {prefix}")
            return UnquantizedLinearMethod()
        data = get_residue_metadata(self, require_local=True)
        layers = data.get("layers", {})
        metas = [layers[name] for name in candidates if name in layers]
        if metas and any(meta != metas[0] for meta in metas[1:]):
            raise ValueError(f"fused group must share residue metadata: {prefix}")
        # Export validates the full group; reject missing constituent metadata
        # for externally authored sidecars as well.
        for names in groups:
            present = [name in layers for name in names]
            if any(present) and not all(present):
                raise ValueError(f"partially specified residue fused group: {prefix}")
        meta = metas[0] if metas else {}
        dims = data.get("extended_linear_dims", {})
        widths = {dims[name] for name in candidates if name in dims}
        if len(widths) > 1:
            raise ValueError(f"fused group must share extended input width: {prefix}")
        return ResidueMxfp4LinearMethod(meta, extended_dim=next(iter(widths), None))


class ResidueMxfp4LinearMethod(LinearMethodBase):
    """Load packed uint8 weights/scales and dispatch the opaque MXFP4 op."""

    def __init__(self, metadata=None, *, extended_dim=None, activation_recipe="ceil"):
        self.activation_recipe = activation_recipe
        self.metadata = metadata or {}
        self.extended_dim = extended_dim
        runtime_mode = self.metadata.get("runtime_mode", "standard")
        if runtime_mode == "mext_r1":
            from ..flashinfer_integration import mxfp4_fold_tuner
            if mxfp4_fold_tuner.tuner_enabled():
                from vllm.config import get_current_vllm_config

                config = get_current_vllm_config()
                limit = int(config.scheduler_config.max_num_batched_tokens)
                sizes = {1, limit, 2, 4, 8, 16, 32, 64, 128, 192, 256,
                         384, 512}
                sizes.update(config.compilation_config.cudagraph_capture_sizes or [])
                self.tuning_buckets = sorted(m for m in sizes if 0 < m <= limit)
            else:
                self.tuning_buckets = None
        else:
            self.tuning_buckets = None

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        from quant.kernel_metadata import padded_k, residue_spec
        from .residue_tp import current_tp_rank, plan_from_partition

        n = sum(output_partition_sizes)
        k = input_size_per_partition
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MXFP4 output dtype must be fp16 or bf16")
        if k <= 0 or k % 128 or n <= 0 or n % 32:
            raise ValueError("MXFP4 rank-local linear requires positive K%128==0 and N%32==0")
        # Sidecar indices and extended dimensions describe the full checkpoint,
        # while vLLM allocates parameters for this rank's partition.
        self.spec = residue_spec(self.metadata, input_size, output_size)
        self.mode = self.spec["mode"]
        self.indices = tuple(self.spec.get("indices", ()))
        checkpoint_k = padded_k(input_size, len(self.indices))
        if self.extended_dim is not None and self.extended_dim != checkpoint_k:
            raise ValueError("metadata extended dimension does not match MXFP4 weights")
        row_parallel = k != input_size
        column_parallel = n != output_size
        if (row_parallel or column_parallel) and os.environ.get(
            "RESIDUE_MXFP4_TP_EXPERIMENTAL", "0"
        ) != "1":
            raise ValueError(
                "MXFP4 TP loading is experimental; set "
                "RESIDUE_MXFP4_TP_EXPERIMENTAL=1 to opt in"
            )
        if row_parallel and column_parallel:
            raise ValueError("MXFP4 cannot shard both input and output axes in one linear")
        if row_parallel and (k > input_size or input_size % k or output_size != n):
            raise ValueError("MXFP4 row-parallel partition does not match global shape")
        if column_parallel and (input_size != k or n > output_size):
            raise ValueError("MXFP4 column-parallel partition does not match global shape")

        shard_plan = None
        if row_parallel and self.mode == "selective":
            shard_plan = plan_from_partition(
                extended_dim=input_size + len(self.indices),
                input_size_per_partition=k,
                input_size=input_size,
                num_salient=len(self.indices),
                tp_rank=current_tp_rank(),
                scale_vec_size=32,
                scale_swizzle_blocks=1,  # checkpoint scales are still linear
                k_alignment=128,
            )

        # vLLM's row-parallel loader slices the parameter's physical width.
        # For packed FP4 that width is K/2 bytes; for raw E8M0 scales it is
        # K/32 blocks. K_local%128 keeps both slices on the GEMM scale grid.
        # The output-parallel loaders slice rows, including fused QKV shards.
        # Swizzle scales only after loading this rank's raw linear slice.
        kp = padded_k(k, shard_plan.salient_shard if shard_plan else len(self.indices))
        loader = extra_weight_attrs.pop("weight_loader")
        layer._mxfp4_shard_plan = shard_plan
        layer._mxfp4_loaded = {"weight": set(), "weight_scale": set()}
        layer._mxfp4_shards = len(output_partition_sizes)

        def checked_loader(name):
            def load(param, tensor, *args, **kwargs):
                if tensor.dtype != torch.uint8:
                    raise ValueError(f"packed MXFP4 {name} must be uint8")
                if shard_plan is not None:
                    from .residue_tp import interleave_extended_for_tp

                    tensor = interleave_extended_for_tp(
                        tensor, shard_plan,
                        channels_per_element=2 if name == "weight" else 32,
                    )
                loader(param, tensor, *args, **kwargs)
                shard = args[0] if args else kwargs.get("loaded_shard_id")
                if isinstance(shard, (tuple, list)):
                    layer._mxfp4_loaded[name].update(shard)
                else:
                    layer._mxfp4_loaded[name].add(shard)

            return load

        for name, width in (("weight", kp // 2), ("weight_scale", kp // 32)):
            param = Parameter(torch.empty(n, width, dtype=torch.uint8),
                              requires_grad=False)
            set_weight_attrs(
                param,
                {
                    "input_dim": 1,
                    "output_dim": 0,
                    **extra_weight_attrs,
                    "weight_loader": checked_loader(name),
                },
            )
            layer.register_parameter(name, param)
        layer._mxfp4_input_size = k
        layer._mxfp4_output_dtype = params_dtype
        layer._mxfp4_prepared = False

    def process_weights_after_loading(self, layer):
        from nvfp4.mxfp4 import prepare_indices, swizzle_scales

        if layer._mxfp4_prepared:
            return
        weight = layer.weight.detach()
        if (not weight.is_cuda
                or torch.cuda.get_device_capability(weight.device) != (10, 0)):
            raise ValueError("MXFP4 weight preparation requires SM100")

        with torch.no_grad(), torch.cuda.device(weight.device):
            indices = torch.empty(0, dtype=torch.int32, device=weight.device)
            masks = torch.empty(0, dtype=torch.uint8, device=weight.device)
            if self.mode == "selective":
                selected = self.indices
                plan = layer._mxfp4_shard_plan
                if plan is not None:
                    selected = plan.local_salient_indices(
                        torch.tensor(self.indices, dtype=torch.int32,
                                     device=weight.device)
                    )
                indices, masks = prepare_indices(
                    selected,
                    layer._mxfp4_input_size,
                    weight.device,
                )
            for name, loaded in layer._mxfp4_loaded.items():
                if None not in loaded and len(loaded) != layer._mxfp4_shards:
                    raise ValueError(
                        f"incomplete packed MXFP4 {name} loading: {loaded}"
                    )
            if (layer.weight_scale == 255).any().item():
                raise ValueError("MXFP4 checkpoint contains NaN E8M0 scale code 255")

            scales = swizzle_scales(layer.weight_scale.detach())
            del layer.weight_scale
            layer.weight = Parameter(weight, requires_grad=False)
            layer.register_buffer("weight_scale", scales)
            layer.register_buffer("salient_indices", indices)
            layer.register_buffer("salient_masks", masks)
            layer._mxfp4_prepared = True

    def apply(self, layer, x, bias=None):
        if not layer._mxfp4_prepared:
            raise RuntimeError("call process_weights_after_loading before MXFP4 apply")
        if x.shape[-1] != layer._mxfp4_input_size:
            raise ValueError("input feature dimension does not match loaded MXFP4 weight")
        return mxfp4_linear(
            x,
            layer.weight,
            layer.weight_scale,
            layer.salient_indices,
            layer.salient_masks,
            self.mode,
            layer._mxfp4_output_dtype,
            bias,
            self.activation_recipe,
            self.tuning_buckets,
        )
