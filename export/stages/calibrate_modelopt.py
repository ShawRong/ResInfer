# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
ModelOptCalibrateStage - Calibration using ModelOpt.

This stage runs calibration to collect quantization statistics (amax, scales, salient channels).

Input:
    - ctx.refs["model"], ctx.refs["tokenizer"] (from LoadHFStage)
    - OR ctx.artifacts["model_path"] (fallback for standalone use)
    - params: dataset config, quant config

Output:
    - ctx.artifacts["calib_artifact_path"]: Path to calibration artifact directory
    - ctx.meta["dataset_provenance"]: Dataset provenance for reproducibility
    - ctx.meta["quant_cfg"]: Quantization config used
"""
import os
import gc
import copy
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from modelopt.torch.utils.logging import print_rank_0
from modelopt.torch.utils.perf import clear_cuda_cache, report_memory, AccumulatingTimer

from .base import Stage, StageContext
from export.core.dataset_manager import get_dataset_manager, DatasetManager
from export.core.artifacts import (
    CalibrationArtifact,
    DatasetProvenance,
    QuantConfig,
    ArtifactManager,
    get_environment_provenance,
    compute_indices_hash,
    generate_artifact_id,
)


class ModelOptCalibrateStage(Stage):
    """
    Calibration stage using ModelOpt.

    Runs calibration to collect quantization statistics without compression.
    Saves artifacts that can later be used by RehydrateStage for compress/export.
    """

    def __init__(
        self,
        name: str = "calibrate",
        dataset_manager: Optional[DatasetManager] = None,
    ):
        """
        Initialize calibration stage.

        Args:
            name: Stage name
            dataset_manager: Optional DatasetManager instance
        """
        super().__init__(name)
        self.dm = dataset_manager or get_dataset_manager()

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        """Validate required inputs."""
        # Must have either model in refs or model_path in artifacts
        has_model = ctx.get_ref("model") is not None
        has_model_path = params.get("model_path") or ctx.get_artifact("model_path")

        if not (has_model or has_model_path):
            raise ValueError(
                "ModelOptCalibrateStage requires either:\n"
                "  - ctx.refs['model'] and ctx.refs['tokenizer'] (from LoadHFStage), or\n"
                "  - params['model_path'] or ctx.artifacts['model_path']"
            )

        # Check required params
        required = ["dataset_ref", "calib_samples", "quant_mode", "quant_fmt"]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing required params: {missing}")

    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        """
        Run calibration and save artifacts.

        Args:
            ctx: Stage context
            params: Parameters including:
                - dataset_ref: Dataset identifier (e.g., "c4")
                - calib_samples: Number of calibration samples
                - calib_seq_len: Max sequence length
                - calib_batch_size: Batch size (unused for now, individual samples)
                - seed: Random seed
                - quant_mode: "residue", "awq_lite", "awq_full", "awq_clip",
                  "gptq", "w4a4", "w4a8", "w8a8", "mse", "mse_residue"
                - quant_fmt: "nvfp4", "mxfp4", "mxfp8", "nvfp4_fp8", "fp8", "fp8_pcpt", "fp8_pbwo"
                - ratio: Residue ratio (for residue mode)
                - block_size: Quantization block size
                - calib_block_size: Calibration block size
                - artifact_root: Root directory for artifacts
                - model_path: (optional) Model path if not in ctx.refs
                - device: (optional) Device string
                - verbose: (optional) Enable debug logging

        Returns:
            Updated context with artifacts and metadata
        """
        start_time = time.time()
        report_memory("before calibration")
        AccumulatingTimer.reset()

        with AccumulatingTimer("calibration"):
            # Enable debug logging if requested
            if params.get("verbose", False):
                os.environ["RESIDUE_DEBUG"] = "1"

            # Get or load model/tokenizer
            with AccumulatingTimer("load_model"):
                model, tokenizer = self._get_or_load_model(ctx, params)

            # Skip loading calibration data - ModelOpt will create its own dataloader
            print_rank_0(f"Using ModelOpt's internal dataloader for dataset: {params['dataset_ref']}")
            data_provenance = None  # No provenance tracking when using ModelOpt's dataloader

            # Run calibration
            print_rank_0("Running calibration...")
            with AccumulatingTimer("quantize"):
                model, calib_state, quantizer_state, quant_cfg_dict = self._run_calibration(
                    model, params, None, tokenizer  # Pass None for calib_data
                )

            # Collect state
            with AccumulatingTimer("collect_state"):
                # State already collected in _run_calibration, just cleanup
                gc.collect()
                clear_cuda_cache()
                report_memory("after calibration")

            # Save artifact
            with AccumulatingTimer("save_artifact"):
                duration = time.time() - start_time
                artifact_root = params.get("artifact_root", "artifacts")
                artifact_path = self._save_artifact(
                    params, calib_state, quantizer_state, quant_cfg_dict,
                    data_provenance, duration, artifact_root
                )

        AccumulatingTimer.report()
        print_rank_0(f"Calibration complete. Artifact saved to: {artifact_path}")

        # Update context
        ctx.set_artifact("calib_artifact_path", artifact_path)
        ctx.set_meta("dataset_provenance", data_provenance)
        ctx.set_meta("quant_cfg", quant_cfg_dict)

        # Update refs if model was loaded here
        if ctx.get_ref("model") is None:
            ctx.set_ref("model", model)
            ctx.set_ref("tokenizer", tokenizer)

        return ctx

    def _get_or_load_model(self, ctx: StageContext, params: Dict[str, Any]):
        """Get model from context or load it."""
        model = ctx.get_ref("model")
        tokenizer = ctx.get_ref("tokenizer")

        if model is not None and tokenizer is not None:
            print_rank_0("Using model from context")
            return model, tokenizer

        # Fallback: load model
        model_path = params.get("model_path") or ctx.get_artifact("model_path")
        if not model_path:
            raise ValueError("No model available in context and no model_path provided")

        print_rank_0(f"Loading model: {model_path}")
        return self._load_model(
            str(model_path),
            device=params.get("device"),
            dtype_str=params.get("dtype", "auto"),
            trust_remote_code=bool(params.get("trust_remote_code", False)),
        )

    def _load_model(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype_str: str = "auto",
        trust_remote_code: bool = False,
    ):
        """Load HuggingFace model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine device
        if device is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                device = "cuda"
            else:
                device = "auto"
        elif device.isdigit():
            device = f"cuda:{device}"

        # Use checkpoint dtype by default ("auto").
        dtype = None
        if dtype_str and dtype_str != "auto":
            dtype = getattr(torch, dtype_str, None)

        load_kwargs = {
            "device_map": device if device != "cuda" else "auto",
            "trust_remote_code": trust_remote_code,
        }
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print_rank_0(f"Loaded: {model.config.model_type} on {model.device} ({model.dtype})")
        return model, tokenizer

    def _get_calibration_data(
        self, params: Dict[str, Any], tokenizer
    ) -> tuple[Any, DatasetProvenance]:
        """
        Get calibration data with full provenance.

        Uses streaming + materialize pattern for large datasets (C4, Pile, etc.)
        to avoid downloading the entire dataset.

        Returns:
            Tuple of (tokenized_data, DatasetProvenance)
        """
        dataset_ref = params["dataset_ref"]
        calib_samples = params["calib_samples"]
        calib_seq_len = params.get("calib_seq_len", 512)
        seed = params.get("seed", 42)

        # Get dataset spec
        spec = self.dm.get_spec(dataset_ref)

        # 对大数据集使用 streaming + materialize 模式
        if spec.is_large or spec.streaming_default:
            return self._get_calibration_data_streaming(
                params, tokenizer, spec, calib_samples, calib_seq_len, seed
            )

        # 小数据集使用原来的方式
        return self._get_calibration_data_standard(
            params, tokenizer, spec, calib_samples, calib_seq_len, seed
        )

    def _get_calibration_data_streaming(
        self,
        params: Dict[str, Any],
        tokenizer,
        spec,
        calib_samples: int,
        calib_seq_len: int,
        seed: int,
    ) -> tuple[DataLoader, DatasetProvenance]:
        """
        Return DataLoader that yields batch dicts for calibration.
        Uses streaming + materialize pattern to avoid downloading entire dataset.
        """
        from datasets import Dataset

        dataset_ref = params["dataset_ref"]
        run_dir = params.get("output_dir", "runs/calib_artifacts")

        # Materialize samples from streaming dataset
        samples_path = self.dm.materialize_samples(
            dataset_ref,
            calib_samples,
            seed=seed,
            run_dir=run_dir,
            shuffle_buffer=10000,
        )

        # Read texts from materialized samples
        texts = []
        with open(samples_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                if i == 0 and "__meta__" in obj:
                    continue
                text_field = spec.text_field or "text"
                if text_field in obj:
                    texts.append(obj[text_field])
                else:
                    k = next(iter(obj.keys()))
                    texts.append(obj[k])

        print_rank_0(f"Loaded {len(texts)} calibration samples")

        # Create dataset with text only (tokenization in collate_fn)
        dataset = Dataset.from_dict({"text": texts})

        # Collate function for batch tokenization
        def collate_fn(examples):
            """Tokenize batch of texts on-the-fly."""
            texts_batch = [ex["text"] for ex in examples]
            encoded = tokenizer(
                texts_batch,
                max_length=calib_seq_len,
                truncation=True,
                padding=True,  # Batch padding
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }

        # Create DataLoader
        batch_size = params.get("calib_batch_size", 0)
        if batch_size == 0:
            batch_size = 1  # Will be overridden by create_forward_loop

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

        # Compute provenance
        indices = list(range(calib_samples))
        indices_hash = compute_indices_hash(indices)

        provenance = DatasetProvenance(
            dataset_ref=dataset_ref,
            repo=spec.repo,
            config=spec.config,
            split=spec.split,
            revision=spec.revision,
            seed=seed,
            num_samples=calib_samples,
            seq_len=calib_seq_len,
            indices_hash=indices_hash,
        )

        return dataloader, provenance

    def _get_calibration_data_standard(
        self,
        params: Dict[str, Any],
        tokenizer,
        spec,
        calib_samples: int,
        calib_seq_len: int,
        seed: int,
    ) -> tuple[DataLoader, DatasetProvenance]:
        """
        Return DataLoader that yields batch dicts for calibration.
        Uses standard dataset loading for small datasets.
        """
        import random
        from datasets import Dataset

        dataset_ref = params["dataset_ref"]

        # Load and sample dataset
        ds = self.dm.get(dataset_ref)
        sampled = self.dm.sample(ds, calib_samples, seed)

        # Track indices for reproducibility
        rng = random.Random(seed)
        total = len(ds)
        if calib_samples < total:
            indices = rng.sample(range(total), k=calib_samples)
        else:
            indices = list(range(total))

        indices_hash = compute_indices_hash(indices)

        # Get text column
        text_col = spec.text_field or "text"
        if text_col not in sampled.column_names:
            # Try common alternatives
            for col in ["content", "sentence", "document"]:
                if col in sampled.column_names:
                    text_col = col
                    break

        print_rank_0(f"Loaded {len(sampled)} calibration samples")

        # Extract texts
        texts = [item[text_col] for item in sampled]
        dataset = Dataset.from_dict({"text": texts})

        # Collate function for batch tokenization
        def collate_fn(examples):
            """Tokenize batch of texts on-the-fly."""
            texts_batch = [ex["text"] for ex in examples]
            encoded = tokenizer(
                texts_batch,
                max_length=calib_seq_len,
                truncation=True,
                padding=True,  # Batch padding
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }

        # Create DataLoader
        batch_size = params.get("calib_batch_size", 0)
        if batch_size == 0:
            batch_size = 1  # Will be overridden by create_forward_loop

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

        # Create provenance
        provenance = DatasetProvenance(
            dataset_ref=dataset_ref,
            repo=spec.repo,
            config=spec.config,
            split=spec.split,
            revision=spec.revision,
            seed=seed,
            num_samples=calib_samples,
            seq_len=calib_seq_len,
            indices_hash=indices_hash,
        )

        return dataloader, provenance

    def _run_calibration(
        self, model, params: Dict[str, Any], calib_dataloader: DataLoader, tokenizer=None
    ) -> tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Run calibration using ModelOpt's create_forward_loop.

        Returns:
            Tuple of (model, calib_state, quantizer_state, quant_cfg_dict)
        """
        import modelopt.torch.quantization as mtq
        from modelopt.torch.utils import create_forward_loop

        quant_mode = params["quant_mode"]
        quant_fmt = params["quant_fmt"]
        quant_scope = params.get("quant_scope", "all")

        # Extract calibration parameters from params
        dataset_name = params.get("dataset_ref", "wikitext2")
        calib_samples = params.get("calib_samples", 512)
        calib_seq_len = params.get("calib_seq_len", 512)

        # Build quantization config
        if quant_mode == "residue":
            from quant.modelopt_integration.registry import build_residue_cfg

            quant_cfg = build_residue_cfg(
                fmt=quant_fmt,
                ratio=params.get("ratio", 0.25),
                calib_block_size=params.get("calib_block_size", 8),
                quant_scope=quant_scope,
                use_sequential=params.get("use_sequential", False),
            )
        elif quant_mode == "awq_lite":
            # Standalone AWQ-lite W4A4: import from our package to register custom
            # backend (handles pre_quant_scale that ModelOpt native backend ignores)
            from quant.modelopt_integration.registry import build_awq_lite_cfg

            quant_cfg = build_awq_lite_cfg(
                fmt=quant_fmt,
                alpha_step=params.get("alpha_step", 0.1),
                quant_scope=quant_scope,
            )
        elif quant_mode in ("awq_full", "awq"):
            from quant.modelopt_integration.registry import build_awq_full_cfg

            quant_cfg = build_awq_full_cfg(
                fmt=quant_fmt,
                alpha_step=params.get("alpha_step", 0.1),
                max_co_batch_size=params.get("awq_max_co_batch_size", 1024),
                max_tokens_per_batch=params.get("awq_max_tokens_per_batch", 64),
                min_clip_ratio=params.get("awq_min_clip_ratio", 0.5),
                shrink_step=params.get("awq_shrink_step", 0.05),
                quant_scope=quant_scope,
            )
        elif quant_mode == "smoothquant":
            # SmoothQuant on NVFP4: single fixed-alpha per-channel scale (not
            # a search like AWQ-lite). Composes mtq.NVFP4_DEFAULT_CFG with the
            # SmoothQuant algorithm internally.
            from quant.modelopt_integration.registry import build_smoothquant_cfg

            quant_cfg = build_smoothquant_cfg(
                fmt=quant_fmt,
                alpha=params.get("smoothquant_alpha", 0.5),
                quant_scope=quant_scope,
            )
        elif quant_mode == "awq_clip":
            from quant.modelopt_integration.registry import build_awq_clip_cfg

            quant_cfg = build_awq_clip_cfg(
                fmt=quant_fmt,
                max_co_batch_size=params.get("awq_max_co_batch_size", 1024),
                max_tokens_per_batch=params.get("awq_max_tokens_per_batch", 64),
                min_clip_ratio=params.get("awq_min_clip_ratio", 0.5),
                shrink_step=params.get("awq_shrink_step", 0.05),
                quant_scope=quant_scope,
            )
        elif quant_mode == "gptq":
            from quant.modelopt_integration.registry import build_gptq_cfg

            quant_cfg = build_gptq_cfg(
                fmt=quant_fmt,
                perc_damp=params.get("percdamp", params.get("perc_damp", 0.01)),
                block_size=params.get("gptq_block_size", 128),
                quant_scope=quant_scope,
            )
        elif quant_mode == "w4a4":
            from modelopt.torch.quantization import config as mtq_config
            from quant.modelopt_integration.registry import apply_quant_scope
            if quant_fmt == "nvfp4":
                quant_cfg = copy.deepcopy(mtq_config.NVFP4_DEFAULT_CFG)
            else:
                quant_cfg = copy.deepcopy(mtq_config.MXFP4_DEFAULT_CFG)
            apply_quant_scope(quant_cfg, quant_scope)
        elif quant_mode in ("mse", "mse_residue"):
            # NVFP4/MXFP4 with MSE-based amax search (replaces "max" algorithm).
            # `mse_residue` is the same calibration but with a post-quantize step
            # that injects external salient indices (metadata path).  See
            # _post_quantize_inject below.
            from modelopt.torch.quantization import config as mtq_config
            base_cfg = (
                mtq_config.NVFP4_DEFAULT_CFG if quant_fmt == "nvfp4"
                else mtq_config.MXFP4_DEFAULT_CFG
            )
            quant_cfg = copy.deepcopy(base_cfg)
            # MseCalibConfig fields: step_size (default 0.1) controls candidate density;
            # num_candidates = ceil((stop - start) / step_size) + 1.
            # Defaults: start=0.25, stop=4.0, step=0.4 → ~10 candidates.
            quant_cfg["algorithm"] = {
                "method": "mse",
                "step_size": params.get("mse_step_size", 0.4),
                "start_multiplier": params.get("mse_start_multiplier", 0.25),
                "stop_multiplier": params.get("mse_stop_multiplier", 4.0),
            }
            from quant.modelopt_integration.registry import apply_quant_scope
            apply_quant_scope(quant_cfg, quant_scope)
            if quant_mode == "mse_residue" and not params.get("residue_metadata_path"):
                raise ValueError(
                    "quant_mode=mse_residue requires params['residue_metadata_path']"
                )
        elif quant_mode == "w4a8":
            # W4A8: 4-bit weights + 8-bit activations
            from modelopt.torch.quantization import config as mtq_config
            from quant.modelopt_integration.registry import apply_quant_scope
            if quant_fmt in ("nvfp4_fp8", "nvfp4"):
                quant_cfg = copy.deepcopy(mtq_config.W4A8_NVFP4_FP8_CFG)
            elif quant_fmt in ("mxfp4_fp8", "mxfp4"):
                quant_cfg = copy.deepcopy(mtq_config.W4A8_MXFP4_FP8_CFG)
            else:
                raise ValueError(f"Unknown quant_fmt for w4a8: {quant_fmt}. Use 'nvfp4_fp8' or 'mxfp4_fp8'")
            apply_quant_scope(quant_cfg, quant_scope)
        elif quant_mode == "w8a8":
            # W8A8 quantization (FP8 variants + MXFP8)
            from modelopt.torch.quantization import config as mtq_config
            from quant.modelopt_integration.registry import apply_quant_scope
            FP8_CFG_MAP = {
                "fp8": mtq_config.FP8_DEFAULT_CFG,
                "fp8_pcpt": mtq_config.FP8_PER_CHANNEL_PER_TOKEN_CFG,
                "fp8_pbwo": mtq_config.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
                "mxfp8": mtq_config.MXFP8_DEFAULT_CFG,
            }
            quant_cfg = FP8_CFG_MAP.get(quant_fmt)
            if quant_cfg is None:
                raise ValueError(
                    f"Unknown quant_fmt for w8a8: {quant_fmt}. "
                    "Use 'fp8', 'fp8_pcpt', 'fp8_pbwo', or 'mxfp8'"
                )
            quant_cfg = copy.deepcopy(quant_cfg)
            apply_quant_scope(quant_cfg, quant_scope)
        else:
            raise ValueError(f"Unknown quant_mode: {quant_mode}")

        # Apply NVFP4 KV cache quantization on top of any quant_mode.
        # Safe for all modes: every base config has "default": {"enable": False},
        # so adding "*[kv]_bmm_quantizer" only enables KV quantizers without
        # touching weight/input quantizers.
        if params.get("kv_cache"):
            from modelopt.torch.quantization import config as mtq_config
            quant_cfg = copy.deepcopy(quant_cfg)
            # quant_cfg is now a list of QuantizerCfgEntry dicts (modelopt main
            # schema change). Append the KV entry from NVFP4_KV_CFG's list.
            quant_cfg["quant_cfg"].extend(mtq_config.NVFP4_KV_CFG["quant_cfg"])

        # Save quant config as dict
        quant_cfg_dict = {
            "mode": quant_mode,
            "fmt": quant_fmt,
            "quant_scope": quant_scope,
            "ratio": params.get("ratio", 0.25),
            "block_size": params.get("block_size", 16),
            "calib_block_size": params.get("calib_block_size", 8),
        }
        if quant_mode in ("awq_lite", "awq_full", "awq", "awq_clip"):
            quant_cfg_dict["alpha_step"] = params.get("alpha_step", 0.1)
        if quant_mode in ("awq_full", "awq", "awq_clip"):
            quant_cfg_dict["awq_max_co_batch_size"] = params.get("awq_max_co_batch_size", 1024)
            quant_cfg_dict["awq_max_tokens_per_batch"] = params.get("awq_max_tokens_per_batch", 64)
            quant_cfg_dict["awq_min_clip_ratio"] = params.get("awq_min_clip_ratio", 0.5)
            quant_cfg_dict["awq_shrink_step"] = params.get("awq_shrink_step", 0.05)
        if quant_mode == "gptq":
            quant_cfg_dict["perc_damp"] = params.get("percdamp", params.get("perc_damp", 0.01))
            quant_cfg_dict["gptq_block_size"] = params.get("gptq_block_size", 128)

        # Prepare device parameter
        device = params.get("device")
        if device and device.isdigit():
            device = f"cuda:{device}"

        # If no device specified, use model's device to ensure tensors are moved correctly
        if device is None and hasattr(model, "device"):
            device = str(model.device)
        elif device is None:
            # Fallback: try to get device from first parameter
            try:
                device = str(next(model.parameters()).device)
            except StopIteration:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get batch size (0 = auto-detect)
        batch_size = params.get("calib_batch_size", 0)

        # Set swap mode for residue quantization
        if quant_mode == "residue":
            swap = params.get("swap", 0)
            os.environ["RESIDUE_SWAP"] = "1" if swap else "0"
            if params.get("verbose"):
                print_rank_0(f"[Residue] Using swap mode: {swap} (RESIDUE_SWAP={os.environ['RESIDUE_SWAP']})")

        # Debug: Print device info
        print_rank_0(f"[DEBUG] Device detection:")
        print_rank_0(f"  - Params device: {params.get('device')}")
        print_rank_0(f"  - Detected device: {device}")
        print_rank_0(f"  - Model device: {next(model.parameters()).device}")

        if dataset_name == "synthetic":
            forward_loop = self._create_synthetic_forward_loop(
                tokenizer=tokenizer,
                batch_size=batch_size,
                num_samples=calib_samples,
                max_sample_length=calib_seq_len,
                device=device,
            )
        else:
            # Create forward loop using ModelOpt official API. For normal
            # calibration, let ModelOpt own dataset loading and device placement.
            forward_loop = create_forward_loop(
                model=model,
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                batch_size=batch_size,  # 0 = auto-detect
                num_samples=calib_samples,
                max_sample_length=calib_seq_len,
                device=device,
            )

        # Run quantization (calibration phase)
        print_rank_0(f"Quantizing with mode={quant_mode}, fmt={quant_fmt}, scope={quant_scope}")
        model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

        # mse_residue: inject external salient indices on top of MSE-calibrated model.
        if quant_mode == "mse_residue":
            metadata_path = Path(params["residue_metadata_path"])
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"residue_metadata_path does not exist: {metadata_path}"
                )
            with open(metadata_path) as f:
                metadata = json.load(f)
            layers = metadata.get("layers", {})
            salient_map = {}
            for layer_name, entry in layers.items():
                indices = entry.get("salient_indices") or []
                if indices:
                    salient_map[layer_name] = torch.tensor(
                        indices, dtype=torch.long, device=device
                    )
            if not salient_map:
                raise ValueError(
                    f"No salient_indices found in {metadata_path}"
                )
            residue_ratio = params.get("residue_ratio")
            print_rank_0(
                f"[mse_residue] Injecting salient indices for {len(salient_map)} layers "
                f"(residue_ratio={residue_ratio}, source={metadata_path.name})"
            )
            from quant.modelopt_integration import inject_residue_salient_map
            inject_residue_salient_map(
                model,
                salient_indices_map=salient_map,
                residue_ratio=residue_ratio,
                preserve_existing_amax=True,
            )

        # Collect calibration state (human-readable summary)
        calib_state = self._collect_calibration_state(model, params)

        # Collect quantizer state (machine-restorable)
        print_rank_0("Collecting quantizer state for future restoration...")
        quantizer_state = self._collect_quantizer_state(model)
        print_rank_0(f"  Collected state from {len(quantizer_state['modules'])} modules")

        return model, calib_state, quantizer_state, quant_cfg_dict

    def _create_synthetic_forward_loop(
        self,
        *,
        tokenizer,
        batch_size: int,
        num_samples: int,
        max_sample_length: int,
        device: str,
    ):
        """Create a deterministic local calibration loop for smoke tests.

        This avoids Hugging Face dataset downloads while still exercising the
        real ModelOpt quantization path. It is intentionally only selected by
        ``dataset_ref: synthetic``.
        """

        if tokenizer is None:
            raise ValueError("dataset_ref=synthetic requires a tokenizer")

        batch_size = int(batch_size) if int(batch_size) > 0 else 1
        texts = [
            (
                "Synthetic calibration sample "
                f"{idx}. The quick brown fox jumps over the lazy dog. "
                "FP4 quantization calibration smoke text."
            )
            for idx in range(int(num_samples))
        ]
        encoded = tokenizer(
            texts,
            max_length=int(max_sample_length),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        batches = []
        target_device = torch.device(device)
        for start in range(0, int(num_samples), batch_size):
            end = start + batch_size
            batches.append(
                {
                    key: value[start:end].to(target_device)
                    for key, value in encoded.items()
                }
            )

        def forward_loop(active_model):
            active_model.eval()
            with torch.no_grad():
                for batch in batches:
                    active_model(**batch)

        return forward_loop

    def _collect_calibration_state(
        self, model, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collect calibration state from quantized model.

        For residue mode, this includes:
        - Per-layer amax values
        - Salient channel indices
        - Channel magnitude statistics
        """
        #TODO: make this generalized one, not only for the residue
        state = {
            "quant_mode": params["quant_mode"],
            "quant_fmt": params["quant_fmt"],
            "ratio": params.get("ratio", 0.25),
            "layers": {},
        }

        for name, module in model.named_modules():
            layer_state = {}

            # Check for quantizers
            if hasattr(module, "input_quantizer"):
                iq = module.input_quantizer
                if hasattr(iq, "amax") and iq.amax is not None:
                    layer_state["input_amax"] = iq.amax.cpu().tolist()
                if hasattr(iq, "_salient_indices") and iq._salient_indices is not None:
                    layer_state["salient_indices"] = iq._salient_indices.cpu().tolist()

            if hasattr(module, "weight_quantizer"):
                wq = module.weight_quantizer
                if hasattr(wq, "amax") and wq.amax is not None:
                    layer_state["weight_amax"] = wq.amax.cpu().tolist()

            if layer_state:
                state["layers"][name] = layer_state

        return state

    def _collect_quantizer_state(self, model) -> Dict[str, Any]:
        """
        Collect machine-restorable quantizer state.

        Returns:
            Dict with version and module-level quantizer states
        """
        qstate = {
            "version": 1,
            "modules": {},
        }

        for name, module in model.named_modules():
            entry = {}

            # Collect input quantizer state
            if hasattr(module, "input_quantizer"):
                try:
                    entry["input_quantizer"] = module.input_quantizer.state_dict()
                    entry["input_quantizer_type"] = type(module.input_quantizer).__name__
                except Exception as e:
                    self.logger.warning(f"Failed to collect input_quantizer state for {name}: {e}")

            # Collect weight quantizer state
            if hasattr(module, "weight_quantizer"):
                try:
                    entry["weight_quantizer"] = module.weight_quantizer.state_dict()
                    entry["weight_quantizer_type"] = type(module.weight_quantizer).__name__
                except Exception as e:
                    self.logger.warning(f"Failed to collect weight_quantizer state for {name}: {e}")

            if entry:
                entry["module_type"] = type(module).__name__
                qstate["modules"][name] = entry

        return qstate

    def _save_artifact(
        self,
        params: Dict[str, Any],
        calib_state: Dict[str, Any],
        quantizer_state: Dict[str, Any],
        quant_cfg_dict: Dict[str, Any],
        data_provenance: DatasetProvenance,
        duration: float,
        artifact_root: str,
    ) -> Path:
        """
        Save calibration artifact with full metadata.

        Saves:
        - meta.json: Artifact metadata
        - calib_state.json: Human-readable calibration summary
        - quantizer_state.pt: Machine-restorable quantizer state
        - quant_cfg.json: Quantization config
        """
        import tempfile

        # Initialize artifact manager
        artifact_manager = ArtifactManager(artifact_root)

        # Extract model info
        model_path = params.get("model_path", "")
        model_id = params.get("model_id", "") or Path(model_path).name if model_path else "unknown"

        # Generate artifact ID
        artifact_id = generate_artifact_id(
            model_id=model_id,
            artifact_type="calib",
            dataset_ref=params["dataset_ref"],
            seed=params.get("seed", 42),
            num_samples=params["calib_samples"],
            quant_mode=params["quant_mode"],
            ratio=params.get("ratio") if params["quant_mode"] == "residue" else None,
        )

        # Create artifact metadata
        artifact = CalibrationArtifact(
            artifact_id=artifact_id,
            artifact_type="calibration",
            model_id=model_id,
            model_path=str(model_path) if model_path else "",
            dataset=data_provenance,
            environment=get_environment_provenance(),
            quant_config=QuantConfig(
                mode=params["quant_mode"],
                fmt=params["quant_fmt"],
                ratio=params.get("ratio", 0.25),
                block_size=params.get("block_size", 16),
                calib_block_size=params.get("calib_block_size", 8),
            ),
            created_at=datetime.now().isoformat(),
            duration_seconds=duration,
        )

        # Save all files to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 1. Human-readable calibration state
            calib_state_path = tmpdir_path / "calib_state.json"
            with open(calib_state_path, "w") as f:
                json.dump(calib_state, f, indent=2)

            # 2. Machine-restorable quantizer state
            quantizer_state_path = tmpdir_path / "quantizer_state.pt"
            torch.save(quantizer_state, quantizer_state_path)

            # 3. Quantization config
            quant_cfg_path = tmpdir_path / "quant_cfg.json"
            with open(quant_cfg_path, "w") as f:
                json.dump(quant_cfg_dict, f, indent=2)

            # Save artifact with all files
            artifact_path = artifact_manager.save_calibration_artifact(
                artifact,
                files={
                    "calib_state": calib_state_path,
                    "quantizer_state": quantizer_state_path,
                    "quant_cfg": quant_cfg_path,
                },
            )

        return artifact_path
