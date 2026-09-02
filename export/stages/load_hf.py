# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
LoadHFStage - Load HuggingFace model and tokenizer.

This stage loads a model and tokenizer from HuggingFace,
providing them to subsequent stages via context.

Input:
    - params["model_path"]: Path to HuggingFace model

Output:
    - ctx.refs["model"]: Loaded model
    - ctx.refs["tokenizer"]: Loaded tokenizer
"""
import os
import torch
from typing import Dict, Any, Optional

from modelopt.torch.utils.logging import print_rank_0

from .base import Stage, StageContext


class LoadHFStage(Stage):
    """
    Load HuggingFace model and tokenizer.

    This stage consolidates model loading logic that was duplicated across
    CalibrateRunner, CompressExportRunner, and other runners. It loads a
    HuggingFace model in FP16 with appropriate device placement.
    """

    def __init__(self, name: str = "load"):
        """Initialize load stage."""
        super().__init__(name)

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        """Validate required inputs."""
        model_path = params.get("model_path") or ctx.get_artifact("model_path")
        if not model_path:
            raise ValueError(
                "LoadHFStage requires model_path in:\n"
                "  - params['model_path'], or\n"
                "  - ctx.artifacts['model_path']"
            )

    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        """
        Load HuggingFace model and tokenizer.

        Args:
            ctx: Stage context
            params: Parameters including:
                - model_path: Path to HuggingFace model
                - device: (optional) Device string (auto, cuda, cuda:0, etc.)
                - dtype: (optional) Torch dtype (default: float16)
                - trust_remote_code: (optional) Trust remote code (default: False)
                - verbose: (optional) Enable debug logging

        Returns:
            Updated context with model and tokenizer in refs
        """
        # Get model path
        model_path = params.get("model_path") or ctx.get_artifact("model_path")

        # Get device
        device = params.get("device")

        # Get dtype
        dtype_str = params.get("dtype", "auto")
        if dtype_str == "auto" or dtype_str is None:
            dtype = None
        else:
            dtype = getattr(torch, dtype_str) if isinstance(dtype_str, str) else dtype_str

        # Get other options
        trust_remote_code = bool(params.get("trust_remote_code", False))
        verbose = params.get("verbose", False)
        gpu_max_mem_percentage = params.get("gpu_max_mem_percentage")

        # Load model
        print_rank_0(f"Loading model: {model_path}")
        model, tokenizer = self._load_model(
            str(model_path),
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            verbose=verbose,
            gpu_max_mem_percentage=gpu_max_mem_percentage,
        )

        # Update context
        ctx.set_ref("model", model)
        ctx.set_ref("tokenizer", tokenizer)
        ctx.set_artifact("model_path", model_path)

        return ctx

    def _load_model(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        verbose: bool = False,
        gpu_max_mem_percentage: Optional[float] = None,
    ):
        'Load HuggingFace model and tokenizer.'
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # Determine device
        if device is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                device = "cuda"
            else:
                device = "auto"
        elif device.isdigit():
            device = f"cuda:{device}"

        load_kwargs: Dict[str, Any] = {
            "device_map": device if device != "cuda" else "auto",
            "trust_remote_code": trust_remote_code,
        }
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        # Apply gpu_max_mem_percentage headroom (only when meaningful: device is
        # GPU and percentage < 1.0). Mirrors example_utils.py::get_model.
        if (
            gpu_max_mem_percentage is not None
            and 0.0 < gpu_max_mem_percentage < 1.0
            and load_kwargs["device_map"] in ("auto", "sequential")
        ):
            from accelerate import init_empty_weights, infer_auto_device_map
            from accelerate.utils import get_max_memory

            cfg_kwargs = {"trust_remote_code": trust_remote_code} if trust_remote_code else {}
            hf_config = AutoConfig.from_pretrained(model_path, **cfg_kwargs)
            torch_dtype = dtype if dtype is not None else getattr(hf_config, "torch_dtype", torch.bfloat16)

            with init_empty_weights(include_buffers=True):
                empty_model = AutoModelForCausalLM.from_config(
                    hf_config, dtype=torch_dtype, trust_remote_code=trust_remote_code
                )

            max_memory = get_max_memory()
            inferred = infer_auto_device_map(empty_model, max_memory=max_memory)
            del empty_model

            on_cpu = "cpu" in inferred.values() or "disk" in inferred.values()
            if on_cpu:
                for _dev in max_memory:
                    if isinstance(_dev, int):
                        max_memory[_dev] = int(max_memory[_dev] * gpu_max_mem_percentage)
                load_kwargs["max_memory"] = max_memory
                print_rank_0(
                    f"  [gpu_max_mem_percentage={gpu_max_mem_percentage}] "
                    f"Model spills to CPU; applying headroom: max_memory={max_memory}"
                )
            else:
                print_rank_0(
                    f"  [gpu_max_mem_percentage={gpu_max_mem_percentage}] "
                    f"Model fits in GPU(s); no max_memory constraint applied."
                )

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        # Fix pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Log info
        model_info = f"{model.config.model_type} on {model.device} ({model.dtype})"
        print_rank_0(f"  Loaded: {model_info}")

        if verbose:
            num_params = sum(p.numel() for p in model.parameters())
            print_rank_0(f"  Parameters: {num_params:,}")

        return model, tokenizer
