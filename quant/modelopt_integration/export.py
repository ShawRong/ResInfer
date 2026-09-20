"""Export wrapper for residue-quantized checkpoints.

This module provides a wrapper around ModelOpt's export_hf_checkpoint
that ensures config.json accurately reflects the extended weight dimensions.

This is a UNIVERSAL correctness fix - not specific to any inference framework.
The problem: ModelOpt exports weights with extended dimensions, but config.json
still has the original model dimensions. This causes loaders that read config.json
(like vLLM) to pre-allocate tensors with wrong shapes.

Usage:
    from quant.modelopt_integration.export import export_residue_checkpoint

    export_residue_checkpoint(
        model,
        export_dir="./exported",
        dtype=torch.float16,
    )
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch


def export_residue_checkpoint(
    model: torch.nn.Module,
    export_dir: str,
    dtype: Optional[torch.dtype] = None,
    residue_ratio: float = 0.125,
    **export_kwargs: Any,
) -> Path:
    """Export residue-quantized model checkpoint.

    This wrapper calls ModelOpt's export_hf_checkpoint() for proper FP4 packing.

    IMPORTANT: config.json is NOT modified. The original dimensions (hidden_size,
    intermediate_size) are preserved because:
    - Embedding layer uses hidden_size and is NOT extended
    - Linear layers have extended dimensions but get them from metadata

    vLLM's residue NVFP4 linear method reads extended_linear_dims from
    residue_kernel_metadata.json to override input_size_per_partition
    in create_weights().

    Args:
        model: Quantized model to export
        export_dir: Directory to save checkpoint
        dtype: Export dtype (default: model dtype)
        residue_ratio: Ratio used for residue extension (for metadata)
        **export_kwargs: Additional args for export_hf_checkpoint

    Returns:
        Path to exported checkpoint directory
    """
    from .mxfp4_export import has_mxfp4_weights, export_mxfp4_checkpoint
    if has_mxfp4_weights(model):
        export_kwargs.pop("save_modelopt_state", None)
        return export_mxfp4_checkpoint(model, export_dir, dtype=dtype, **export_kwargs)

    from modelopt.torch.export import export_hf_checkpoint

    export_path = Path(export_dir)

    # Standard ModelOpt export (handles FP4 packing)
    # DO NOT update config.json - dimensions must stay original for embedding
    # Extended dimensions are handled per-layer via metadata
    with torch.inference_mode():
        export_hf_checkpoint(
            model,
            export_dir=str(export_path),
            dtype=dtype,
            **export_kwargs,
        )

    # vLLM's ModelOpt quantization loader currently only whitelists "NVFP4"
    # (see QUANT_ALGOS in vllm/model_executor/layers/quantization/modelopt.py).
    # ModelOpt may export AWQ-lite checkpoints as "NVFP4_AWQ", while the actual
    # packed weights are still NVFP4. Runtime pre_quant_scale is not supported
    # by the native vLLM real-quant contract; formal AWQ exports must fold it
    # before vLLM loading.
    _normalize_quant_algo_for_vllm(export_path)

    return export_path


def _normalize_quant_algo_for_vllm(export_path: Path) -> None:
    """Rewrite exported quant_algo to "NVFP4" for vLLM compatibility.

    This is a lightweight metadata patch (does NOT touch any weights).
    It preserves the original quant_algo under an auxiliary key for traceability.
    """

    def _patch_json(path: Path, *, key_path: list[str], aux_key: str) -> None:
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"[residue] Warning: Failed to read {path}: {e}")
            return

        cur = data
        for k in key_path[:-1]:
            if not isinstance(cur, dict) or k not in cur:
                return
            cur = cur[k]
        leaf = key_path[-1]
        if not isinstance(cur, dict) or leaf not in cur:
            return

        quant_algo = cur.get(leaf)
        if not isinstance(quant_algo, str):
            return

        if quant_algo.upper() != "NVFP4_AWQ":
            return

        cur[aux_key] = quant_algo
        cur[leaf] = "NVFP4"
        try:
            path.write_text(json.dumps(data, indent=4) + "\n")
        except Exception as e:
            print(f"[residue] Warning: Failed to write {path}: {e}")

    # config.json: quantization_config.quant_algo
    _patch_json(
        export_path / "config.json",
        key_path=["quantization_config", "quant_algo"],
        aux_key="_original_quant_algo",
    )
    # hf_quant_config.json: quantization.quant_algo
    _patch_json(
        export_path / "hf_quant_config.json",
        key_path=["quantization", "quant_algo"],
        aux_key="_original_quant_algo",
    )


def _update_config_with_extended_dims(export_path: Path, residue_ratio: float) -> None:
    """Update config.json to match actual weight dimensions.

    Reads actual weight shapes from safetensors and updates
    hidden_size/intermediate_size to reflect the true dimensions.

    Args:
        export_path: Path to exported checkpoint directory
        residue_ratio: Fallback ratio if can't read from weights
    """
    config_path = export_path / "config.json"
    if not config_path.exists():
        print(f"[residue] Warning: config.json not found at {config_path}")
        return

    # Read current config
    with open(config_path) as f:
        config = json.load(f)

    original_hidden = config.get("hidden_size", 4096)
    original_intermediate = config.get("intermediate_size", 14336)

    # Read actual dimensions from exported weights (authoritative source)
    actual_dims = _read_weight_dims_from_safetensors(export_path)

    if actual_dims:
        extended_hidden = actual_dims.get("hidden_size", original_hidden)
        extended_intermediate = actual_dims.get("intermediate_size", original_intermediate)
    else:
        # Fallback: compute from ratio
        print("[residue] Warning: Could not read weight dims from safetensors, using ratio fallback")
        extended_hidden = original_hidden + int(original_hidden * residue_ratio)
        extended_intermediate = original_intermediate + int(original_intermediate * residue_ratio)

    # Only update if dimensions actually changed
    if extended_hidden == original_hidden and extended_intermediate == original_intermediate:
        print("[residue] Dimensions unchanged, skipping config.json update")
        return

    # Update config with actual dimensions
    config["hidden_size"] = extended_hidden
    config["intermediate_size"] = extended_intermediate

    # Preserve original dimensions for reference
    config["_original_hidden_size"] = original_hidden
    config["_original_intermediate_size"] = original_intermediate
    config["_residue_extended"] = True

    # Write updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[residue] Updated config.json with actual dimensions:")
    print(f"  hidden_size: {original_hidden} → {extended_hidden}")
    print(f"  intermediate_size: {original_intermediate} → {extended_intermediate}")


def _read_weight_dims_from_safetensors(export_path: Path) -> Dict[str, int]:
    """Read actual weight dimensions from exported safetensors.

    Returns dict with extended hidden_size and intermediate_size
    based on actual weight shapes.

    The logic:
    - gate_proj.weight shape: [intermediate_size, hidden_size_packed]
      → hidden_size = packed_dim * 2 (FP4 nibble packing)
    - down_proj.weight shape: [hidden_size, intermediate_size_packed]
      → intermediate_size = packed_dim * 2

    Args:
        export_path: Path to exported checkpoint directory

    Returns:
        Dict with "hidden_size" and/or "intermediate_size" keys
    """
    try:
        from safetensors import safe_open
    except ImportError:
        print("[residue] Warning: safetensors not installed, cannot read weight dims")
        return {}

    safetensor_files = list(export_path.glob("*.safetensors"))
    if not safetensor_files:
        return {}

    result: Dict[str, int] = {}

    for sf_path in safetensor_files:
        try:
            with safe_open(sf_path, framework="pt") as f:
                for key in f.keys():
                    # Look for gate_proj to determine extended hidden_size
                    # gate_proj shape: [intermediate_size, hidden_size_packed]
                    if "gate_proj.weight" in key and "hidden_size" not in result:
                        tensor = f.get_tensor(key)
                        # Unpack: hidden_size = packed_dim * 2 (FP4 nibble packing)
                        result["hidden_size"] = tensor.shape[1] * 2

                    # Look for down_proj to determine extended intermediate_size
                    # down_proj shape: [hidden_size, intermediate_size_packed]
                    if "down_proj.weight" in key and "intermediate_size" not in result:
                        tensor = f.get_tensor(key)
                        # Unpack: intermediate_size = packed_dim * 2
                        result["intermediate_size"] = tensor.shape[1] * 2

                    if len(result) == 2:
                        break
        except Exception as e:
            print(f"[residue] Warning: Error reading {sf_path}: {e}")
            continue

        if len(result) == 2:
            break

    return result


def is_residue_quantized(model: torch.nn.Module) -> bool:
    """Check if model has residue quantization applied.

    Detects residue quantization by looking for the _original_in_features
    attribute that is set during residue weight extension.

    Args:
        model: Model to check

    Returns:
        True if model has residue quantization applied
    """
    for module in model.modules():
        if hasattr(module, "_original_in_features"):
            return True
    return False
