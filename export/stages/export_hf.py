# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
ExportHFStage - Export model to HuggingFace format.

This stage exports a model (quantized or compressed) to HuggingFace format.

Input:
    - ctx.refs["model"], ctx.refs["tokenizer"]

Output:
    - ctx.artifacts["export_dir"]: Path to exported model directory
"""
from pathlib import Path
from typing import Dict, Any

from modelopt.torch.utils.logging import print_rank_0

from .base import Stage, StageContext


class ExportHFStage(Stage):
    """
    Export model to HuggingFace format.

    This stage saves a model and tokenizer. Residue vLLM exports use standard
    HF config plus residue_kernel_metadata.json; legacy custom HF model support
    is archived and no longer generated here.
    """

    def __init__(self, name: str = "export"):
        """Initialize export stage."""
        super().__init__(name)

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        """Validate required inputs."""
        if ctx.get_ref("model") is None or ctx.get_ref("tokenizer") is None:
            raise ValueError(
                "ExportHFStage requires:\n"
                "  - ctx.refs['model']\n"
                "  - ctx.refs['tokenizer']"
            )

        # Must have export_dir
        export_dir = ctx.get_artifact("export_dir") or params.get("export_dir")
        if not export_dir:
            raise ValueError(
                "ExportHFStage requires export directory in:\n"
                "  - ctx.artifacts['export_dir'], or\n"
                "  - params['export_dir']"
            )

    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        """
        Export model to HuggingFace format.

        Args:
            ctx: Stage context with model and tokenizer in refs
            params: Parameters including:
                - export_dir: Directory to export to
                - safe_serialization: Use safetensors (default: True)
                - use_custom_config: Deprecated legacy option, ignored
                - export_target: Target runtime ("huggingface" or "vllm")
                - save_kernel_metadata: Save kernel metadata for custom deployment (default: False)
                - residue_ratio: Residue ratio (required if save_kernel_metadata=True)
                - verbose: (optional) Enable debug logging

        Returns:
            Updated context with export_dir in artifacts
        """
        verbose = params.get("verbose", False)
        safe_serialization = params.get("safe_serialization", True)
        use_custom_config = params.get("use_custom_config", False)
        export_target = params.get("export_target", "huggingface")
        save_kernel_metadata = params.get("save_kernel_metadata", False)
        residue_ratio = params.get("residue_ratio")

        if use_custom_config:
            print_rank_0(
                "  Note: use_custom_config is archived and ignored; "
                "export uses standard HF config plus residue metadata."
            )
            use_custom_config = False

        # Get export directory (resolve relative to case_dir)
        export_dir = ctx.get_artifact("export_dir") or params.get("export_dir")
        export_dir = ctx.resolve_path(Path(export_dir))

        # Get model and tokenizer
        model = ctx.get_ref("model")
        tokenizer = ctx.get_ref("tokenizer")

        # Export model
        print_rank_0(f"Exporting model to: {export_dir}")
        print_rank_0(f"  Export target: {export_target}")
        export_path = self._export_model(
            model,
            tokenizer,
            export_dir,
            safe_serialization,
            use_custom_config,
            export_target,
            save_kernel_metadata,
            residue_ratio,
            int(params.get("calib_block_size", params.get("block_size", 8))),
            verbose,
        )

        # Update context
        ctx.set_artifact("export_dir", export_path)

        return ctx

    def _export_model(
        self,
        model,
        tokenizer,
        export_dir: Path,
        safe_serialization: bool = True,
        use_custom_config: bool = False,
        export_target: str = "huggingface",
        save_kernel_metadata: bool = False,
        residue_ratio: float = None,
        kernel_metadata_block_size: int = 8,
        verbose: bool = False,
    ) -> Path:
        """Export model with HF save_pretrained and optional kernel metadata."""
        if model is None:
            raise RuntimeError(
                "Model is None! Cannot export. "
                "This indicates a bug in a previous stage."
            )

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # Collect extended dimensions for vLLM residue metadata.
        from quant.modelopt_integration.kernel_metadata import (
            collect_extended_linear_dims,
        )
        extended_dims = collect_extended_linear_dims(model)

        if verbose:
            print_rank_0(f"  Saving model (safe_serialization={safe_serialization})...")

        # ALWAYS use ModelOpt export to correctly save FP4 packed weights
        # export_hf_checkpoint() calls to_quantized_weight() to properly pack FP16 → FP4
        # Standard save_pretrained() doesn't understand QTensor packing
        import torch

        def _resolve_export_dtype(m: torch.nn.Module) -> torch.dtype:
            # Prefer HF config dtype when present.
            candidate = getattr(m.config, "torch_dtype", None)
            if isinstance(candidate, str):
                candidate = getattr(torch, candidate, None)
            if isinstance(candidate, torch.dtype) and candidate.is_floating_point:
                return candidate

            # Some models expose a top-level dtype.
            candidate = getattr(m, "dtype", None)
            if isinstance(candidate, torch.dtype) and candidate.is_floating_point:
                return candidate

            # Fall back to any floating parameter/buffer (avoid uint8 fp4 packed weights).
            for p in m.parameters():
                if isinstance(p.dtype, torch.dtype) and p.dtype.is_floating_point:
                    return p.dtype
            for b in m.buffers():
                if isinstance(b.dtype, torch.dtype) and b.dtype.is_floating_point:
                    return b.dtype

            # Last resort: BF16 is the safest default for TRTLLM FP4 GEMM.
            return torch.bfloat16

        model_dtype = _resolve_export_dtype(model)
        if verbose:
            print_rank_0(f"  Export dtype: {model_dtype}")

        # Check if model has residue quantization
        from quant.modelopt_integration.export import is_residue_quantized

        if is_residue_quantized(model):
            # Use our wrapper that corrects config.json dimensions
            # This is a universal correctness fix - config.json should reflect actual weight dims
            from quant.modelopt_integration.export import export_residue_checkpoint

            if verbose:
                print_rank_0(f"  Detected residue-quantized model, using export_residue_checkpoint()")

            export_residue_checkpoint(
                model=model,
                export_dir=str(export_path),
                dtype=model_dtype,
                residue_ratio=residue_ratio if residue_ratio is not None else 0.125,
                save_modelopt_state=True,
            )
        else:
            # Standard ModelOpt export for non-residue models
            from modelopt.torch.export import export_hf_checkpoint

            export_hf_checkpoint(
                model=model,
                dtype=model_dtype,
                export_dir=export_path,
                save_modelopt_state=True,
            )

        # Normalize quant_algo for vLLM compatibility (idempotent).
        # Rewrites "NVFP4_AWQ" → "NVFP4" in hf_quant_config.json / config.json
        # so vLLM's ModelOpt whitelist accepts the checkpoint.
        from quant.modelopt_integration.export import _normalize_quant_algo_for_vllm
        _normalize_quant_algo_for_vllm(export_path)

        if verbose:
            print_rank_0(f"  ✓ Exported FP4 packed weights (nibble-packed uint8)")

        # Save tokenizer
        tokenizer.save_pretrained(export_path)

        if verbose:
            print_rank_0(f"  ✓ Model saved to {export_path}")

        # Save kernel metadata if requested
        if save_kernel_metadata:
            if residue_ratio is None:
                self.logger.warning(
                    "  save_kernel_metadata=True but residue_ratio not provided, "
                    "skipping metadata export"
                )
            else:
                if verbose:
                    print_rank_0("  Saving kernel metadata...")

                from quant.modelopt_integration.kernel_metadata import (
                    save_kernel_metadata as save_metadata_fn,
                )

                try:
                    saved_files = save_metadata_fn(
                        model=model,
                        export_dir=export_path,
                        residue_ratio=residue_ratio,
                        block_size=kernel_metadata_block_size,
                        save_channel_masks=True,
                    )

                    if verbose:
                        for key, path in saved_files.items():
                            print_rank_0(f"    ✓ {key}: {path.name}")
                except Exception as e:
                    self.logger.error(f"  Failed to save kernel metadata: {e}")
                    if verbose:
                        import traceback
                        self.logger.error(traceback.format_exc())

        # For vLLM target: save extended_linear_dims to metadata
        # This allows patched residue linear methods to read extended dimensions.
        if export_target == "vllm" and extended_dims:
            import json
            metadata_path = export_path / "residue_kernel_metadata.json"

            # Load existing metadata if present
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Add extended_linear_dims
            metadata["extended_linear_dims"] = extended_dims

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            if verbose:
                print_rank_0(f"  ✓ Saved extended_linear_dims to {metadata_path.name}")
                print_rank_0(f"    {len(extended_dims)} layers with extended dimensions")

        # For vLLM target: remove _salient_indices from checkpoint
        # vLLM's llama.py load_weights doesn't know how to handle these keys
        # The indices are already in residue_kernel_metadata.json
        if export_target == "vllm":
            removed_count = self._remove_salient_indices_from_checkpoint(
                export_path, verbose
            )

        return export_path

    def _remove_salient_indices_from_checkpoint(
        self, export_path: Path, verbose: bool = False
    ) -> int:
        """Remove input_quantizer._salient_indices from checkpoint safetensors.

        vLLM's llama.py load_weights doesn't know how to handle these keys,
        causing KeyError during model loading. The indices are already saved
        in residue_kernel_metadata.json by save_kernel_metadata.

        This is a post-processing step for vLLM target exports.

        Args:
            export_path: Path to the exported model directory
            verbose: Enable debug logging

        Returns:
            Number of keys removed
        """
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError:
            self.logger.warning("  ⚠ safetensors not installed, skipping filter")
            return 0

        import json

        safetensor_files = list(export_path.glob("*.safetensors"))
        if not safetensor_files:
            return 0

        total_removed = 0

        for sf_path in safetensor_files:
            # Load all tensors from safetensors file
            tensors = {}
            keys_to_remove = []

            with safe_open(sf_path, framework="pt") as f:
                for key in f.keys():
                    if "input_quantizer._salient_indices" in key:
                        keys_to_remove.append(key)
                    else:
                        tensors[key] = f.get_tensor(key)

            if not keys_to_remove:
                continue

            # Save filtered tensors back
            save_file(tensors, sf_path)

            total_removed += len(keys_to_remove)
            if verbose:
                print_rank_0(
                    f"  ✓ Removed {len(keys_to_remove)} _salient_indices keys from {sf_path.name}"
                )

        # Update model.safetensors.index.json if it exists
        index_path = export_path / "model.safetensors.index.json"
        if index_path.exists() and total_removed > 0:
            with open(index_path) as f:
                index = json.load(f)

            # Filter weight_map
            if "weight_map" in index:
                original_count = len(index["weight_map"])
                index["weight_map"] = {
                    k: v for k, v in index["weight_map"].items()
                    if "input_quantizer._salient_indices" not in k
                }
                filtered_count = original_count - len(index["weight_map"])

                if filtered_count > 0:
                    with open(index_path, "w") as f:
                        json.dump(index, f, indent=2)
                    if verbose:
                        print_rank_0(
                            f"  ✓ Removed {filtered_count} entries from {index_path.name}"
                        )

        if total_removed > 0:
            print_rank_0(
                f"  ✓ Removed {total_removed} _salient_indices keys for vLLM compatibility"
            )

        return total_removed
