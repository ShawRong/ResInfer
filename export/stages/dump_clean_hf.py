# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
DumpCleanHFStage - Save GPTQ-adjusted weights as a clean Hugging Face model.

This stage is intentionally different from ModelOpt quantized export:
- it does not keep quantizer modules or ModelOpt runtime state
- it does not write quantization metadata/config
- it only preserves the adjusted float weights that match a fresh HF model

Primary use case:
    Treat GPTQ/AWQ calibration as a black-box weight optimizer, then continue
    with downstream residue calibration/export on top of a clean HF checkpoint.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict

import torch
from modelopt.torch.utils.logging import print_rank_0

from .base import Stage, StageContext


class DumpCleanHFStage(Stage):
    """Dump a clean HF model from ModelOpt checkpoint or in-memory quantized model."""

    def __init__(self, name: str = "dump_clean_hf"):
        super().__init__(name)

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        model_path = params.get("model_path") or ctx.get_artifact("model_path")
        if not model_path:
            raise ValueError(
                "DumpCleanHFStage requires:\n"
                "  - params['model_path'], or\n"
                "  - ctx.artifacts['model_path']"
            )

        export_dir = (
            params.get("clean_hf_export_dir")
            or params.get("export_dir")
            or ctx.get_artifact("clean_hf_dir")
            or ctx.get_artifact("export_dir")
        )
        if not export_dir:
            raise ValueError(
                "DumpCleanHFStage requires:\n"
                "  - params['clean_hf_export_dir'], or\n"
                "  - params['export_dir'], or\n"
                "  - ctx.artifacts['clean_hf_dir'], or\n"
                "  - ctx.artifacts['export_dir']"
            )

        checkpoint_path = params.get("checkpoint_path") or ctx.get_artifact("modelopt_checkpoint")
        if checkpoint_path is None and ctx.get_ref("model") is None:
            raise ValueError(
                "DumpCleanHFStage requires one of:\n"
                "  - params['checkpoint_path'] / ctx.artifacts['modelopt_checkpoint'], or\n"
                "  - ctx.refs['model']"
            )

    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_path = params.get("model_path") or ctx.get_artifact("model_path")
        export_dir_param = (
            params.get("clean_hf_export_dir")
            or params.get("export_dir")
            or ctx.get_artifact("clean_hf_dir")
            or ctx.get_artifact("export_dir")
        )
        export_dir = ctx.resolve_path(Path(export_dir_param))
        checkpoint_path = params.get("checkpoint_path") or ctx.get_artifact("modelopt_checkpoint")
        checkpoint_path = ctx.resolve_path(Path(checkpoint_path)) if checkpoint_path else None
        preserve_export_dir = "clean_hf_export_dir" in params

        dtype = self._resolve_dtype(params.get("dtype", "bfloat16"))
        trust_remote_code = bool(params.get("trust_remote_code", False))
        safe_serialization = params.get("safe_serialization", True)
        max_shard_size = params.get("max_shard_size", "10GB")
        free_source_model = params.get("free_source_model", True)
        validate_load = params.get("validate_load", False)
        verbose = params.get("verbose", False)
        apply_awq_scale_fusion = params.get(
            "apply_awq_scale_fusion",
            params.get("apply_export_resmooth", False),
        )
        allow_unfused_pre_quant_scale = params.get("allow_unfused_pre_quant_scale", False)

        if checkpoint_path is not None and free_source_model:
            self._free_source_model(ctx)

        if apply_awq_scale_fusion:
            self._apply_awq_scale_fusion(
                ctx,
                checkpoint_path,
                allow_unfused_pre_quant_scale=allow_unfused_pre_quant_scale,
                verbose=verbose,
            )

        source_state_dict, source_label = self._load_source_state_dict(ctx, checkpoint_path)

        print_rank_0(f"Dumping clean HF model to: {export_dir}")
        print_rank_0(f"  Source: {source_label}")

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        target_model = self._build_clean_model(
            AutoModelForCausalLM,
            config,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        filtered_state_dict, stats = self._filter_state_dict(
            source_state_dict=source_state_dict,
            target_state_dict=target_model.state_dict(),
        )
        if stats["matched_keys"] == 0:
            raise RuntimeError(
                "No regular HF weights matched the source state_dict. "
                "This does not look like a usable clean-HF dump source."
            )

        load_result = target_model.load_state_dict(filtered_state_dict, strict=False)
        stats["missing_keys"] = sorted(load_result.missing_keys)
        stats["unexpected_keys"] = sorted(load_result.unexpected_keys)

        export_dir.mkdir(parents=True, exist_ok=True)
        target_model.save_pretrained(
            export_dir,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
        tokenizer.save_pretrained(export_dir)

        summary_path = export_dir / "clean_hf_dump_summary.json"
        summary = {
            "source": source_label,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "base_model_path": str(model_path),
            "dtype": str(dtype),
            "apply_awq_scale_fusion": bool(apply_awq_scale_fusion),
            **stats,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        if verbose:
            print_rank_0(
                "  "
                f"Matched {stats['matched_keys']} keys "
                f"(extra={stats['extra_source_keys']}, mismatched={stats['shape_mismatch_keys']})"
            )
            if stats["missing_keys"]:
                print_rank_0(f"  Missing target keys: {len(stats['missing_keys'])}")

        reloaded_model = None
        reloaded_tokenizer = None
        if validate_load:
            reloaded_model = AutoModelForCausalLM.from_pretrained(
                export_dir,
                torch_dtype=dtype,
                device_map="cpu",
                trust_remote_code=trust_remote_code,
            ).eval()
            reloaded_tokenizer = AutoTokenizer.from_pretrained(
                export_dir,
                trust_remote_code=trust_remote_code,
            )
            quantizer_modules = sum(
                1
                for module in reloaded_model.modules()
                if hasattr(module, "input_quantizer") or hasattr(module, "weight_quantizer")
            )
            if quantizer_modules != 0:
                raise RuntimeError(
                    f"Reloaded clean HF model still exposes {quantizer_modules} quantizer-bearing modules."
                )
            ctx.set_ref("model", reloaded_model)
            ctx.set_ref("tokenizer", reloaded_tokenizer)
            ctx.metrics["clean_hf_reloaded_quantizer_modules"] = quantizer_modules
        else:
            ctx.set_ref("model", None)
            ctx.set_ref("tokenizer", None)

        ctx.set_artifact("clean_hf_dir", export_dir)
        ctx.set_artifact("model_path", export_dir)
        ctx.set_artifact("clean_hf_dump_summary", summary_path)
        if not preserve_export_dir:
            ctx.set_artifact("export_dir", export_dir)

        ctx.metrics["clean_hf_matched_keys"] = stats["matched_keys"]
        ctx.metrics["clean_hf_extra_source_keys"] = stats["extra_source_keys"]
        ctx.metrics["clean_hf_shape_mismatch_keys"] = stats["shape_mismatch_keys"]
        ctx.metrics["clean_hf_missing_keys"] = len(stats["missing_keys"])
        ctx.metrics["clean_hf_unexpected_keys"] = len(stats["unexpected_keys"])

        del target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ctx

    def _apply_awq_scale_fusion(
        self,
        ctx: StageContext,
        checkpoint_path: Path | None,
        *,
        allow_unfused_pre_quant_scale: bool = False,
        verbose: bool = False,
    ) -> None:
        """Run ModelOpt AWQ scale fusion without quantized packing.

        ModelOpt's normal HF export does two logically separate things:
        1. AWQ pre-quant-scale fusion / shared-input resmoothing:
           - fuse_prequant_to_linear(model, fuse_grouped_heads=True)
           - locate attention/MLP input LayerNorms via export forward hooks
           - fuse_prequant_layernorm(layernorm, shared-input modules)
        2. Conversion of floating weights to packed quantized tensors.

        Clean-HF dumping for this experiment wants (1) but explicitly not (2).
        This helper mutates the in-memory ModelOpt model before its regular HF
        weights are filtered into a fresh clean model.
        """
        if checkpoint_path is not None:
            raise ValueError(
                "apply_awq_scale_fusion=True requires an in-memory model source; "
                "do not set checkpoint_path for this dump_clean_hf stage."
            )

        model = ctx.get_ref("model")
        if model is None:
            raise ValueError(
                "apply_awq_scale_fusion=True requires ctx.refs['model'] from a previous calibration stage"
            )

        if verbose:
            print_rank_0("Applying ModelOpt AWQ scale fusion before clean HF dump")

        import modelopt.torch.export.unified_export_hf as export_hf
        from modelopt.torch.export.quant_utils import fuse_prequant_to_linear

        # Follow the same fusion order as export_hf_checkpoint(), but make the
        # first pre-quant-scale fusion GQA/MQA aware. Upstream
        # requantize_resmooth_fused_llm_layers() calls fuse_prequant_to_linear()
        # without fuse_grouped_heads=True, which leaves Llama GQA o_proj scales
        # unfused (for example 2048 input scale vs 512 v_proj channels).
        before = self._find_pre_quant_scale_modules(model)
        with torch.inference_mode():
            fuse_prequant_to_linear(model, fuse_grouped_heads=True)

            orig_fuse_prequant_to_linear = export_hf.fuse_prequant_to_linear
            export_hf.fuse_prequant_to_linear = lambda *_args, **_kwargs: None
            try:
                export_hf.requantize_resmooth_fused_llm_layers(model)
            finally:
                export_hf.fuse_prequant_to_linear = orig_fuse_prequant_to_linear
        after = self._find_pre_quant_scale_modules(model)

        ctx.metrics["clean_hf_pre_quant_scale_before_awq_fusion"] = len(before)
        ctx.metrics["clean_hf_pre_quant_scale_after_awq_fusion"] = len(after)
        if verbose:
            print_rank_0(
                "  pre_quant_scale modules: "
                f"{len(before)} before AWQ fusion, {len(after)} after AWQ fusion"
            )

        if after and not allow_unfused_pre_quant_scale:
            preview = ", ".join(after[:10])
            raise RuntimeError(
                "Clean-HF AWQ dump would drop unfused pre_quant_scale tensors. "
                f"Unfused modules after AWQ scale fusion: {preview}"
            )

    @staticmethod
    def _find_pre_quant_scale_modules(model) -> list[str]:
        modules = []
        for name, module in model.named_modules():
            input_quantizer = getattr(module, "input_quantizer", None)
            if input_quantizer is not None and hasattr(input_quantizer, "_pre_quant_scale"):
                modules.append(name)
        return modules

    def _free_source_model(self, ctx: StageContext) -> None:
        model = ctx.get_ref("model")
        if model is None:
            return
        ctx.set_ref("model", None)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_source_state_dict(
        self,
        ctx: StageContext,
        checkpoint_path: Path | None,
    ) -> tuple[Dict[str, torch.Tensor], str]:
        if checkpoint_path is not None:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "model_state_dict" not in checkpoint:
                raise ValueError(
                    f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}"
                )
            return checkpoint["model_state_dict"], f"checkpoint:{checkpoint_path}"

        model = ctx.get_ref("model")
        assert model is not None, "model ref must exist when checkpoint_path is absent"
        source_state_dict: Dict[str, torch.Tensor] = {}
        for key, value in model.state_dict().items():
            if isinstance(value, torch.nn.Parameter):
                value = value.data
            if torch.is_tensor(value):
                source_state_dict[key] = value.detach().cpu()
        return source_state_dict, "in_memory_model"

    def _build_clean_model(
        self,
        auto_model_cls,
        config,
        *,
        dtype: torch.dtype,
        trust_remote_code: bool,
    ):
        kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        try:
            return auto_model_cls.from_config(
                config,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        except TypeError:
            return auto_model_cls.from_config(config, **kwargs)

    def _filter_state_dict(
        self,
        *,
        source_state_dict: Dict[str, Any],
        target_state_dict: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        filtered: Dict[str, torch.Tensor] = {}
        extra_source_keys = []
        shape_mismatch_keys = []

        for key, value in source_state_dict.items():
            if isinstance(value, torch.nn.Parameter):
                value = value.data
            if not torch.is_tensor(value):
                extra_source_keys.append(key)
                continue
            if key not in target_state_dict:
                extra_source_keys.append(key)
                continue
            target_value = target_state_dict[key]
            if tuple(value.shape) != tuple(target_value.shape):
                shape_mismatch_keys.append(key)
                continue
            filtered[key] = value.detach().to(device="cpu", dtype=target_value.dtype).clone()

        return filtered, {
            "matched_keys": len(filtered),
            "extra_source_keys": len(extra_source_keys),
            "shape_mismatch_keys": len(shape_mismatch_keys),
            "extra_source_key_names": sorted(extra_source_keys),
            "shape_mismatch_key_names": sorted(shape_mismatch_keys),
        }

    def _resolve_dtype(self, dtype_value: Any) -> torch.dtype:
        if isinstance(dtype_value, torch.dtype):
            return dtype_value
        if dtype_value in (None, "auto"):
            return torch.bfloat16
        if isinstance(dtype_value, str):
            dtype = getattr(torch, dtype_value, None)
            if isinstance(dtype, torch.dtype):
                return dtype
        raise ValueError(f"Unsupported dtype for clean HF dump: {dtype_value}")
