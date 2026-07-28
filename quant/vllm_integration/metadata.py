"""Metadata loading utilities for residue quantization.

INACTIVE: All functions in this file are unused by the main vLLM path.
The main path uses patch.py + linear_utils.py for metadata loading. Kept as
reference for potential future use.

Handles loading salient_indices and other residue-specific metadata
from exported model directories.
"""

import json
import os
from typing import Optional

import torch


def load_residue_metadata(model_dir: str) -> dict:
    """Load residue_kernel_metadata.json from model directory.

    Args:
        model_dir: Path to exported model directory

    Returns:
        Metadata dict with layer-specific salient indices, or empty dict if not found
    """
    path = os.path.join(model_dir, "residue_kernel_metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def get_salient_indices(
    metadata: dict,
    layer_name: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Extract salient indices for a specific layer.

    Args:
        metadata: Loaded metadata dict from load_residue_metadata()
        layer_name: Full layer name (e.g., "model.layers.0.self_attn.q_proj")
        device: Target device for the tensor

    Returns:
        1D int64 tensor of salient channel indices
    """
    layers = metadata.get("layers", {})
    layer_meta = layers.get(layer_name, {})
    indices = layer_meta.get("salient_indices", [])
    return torch.tensor(indices, dtype=torch.int64, device=device)


def compute_channel_mask(
    in_features: int,
    salient_indices: torch.Tensor,
) -> torch.Tensor:
    """Pre-compute bitmask for fast kernel path.

    The mask indicates which channels are salient (1) vs non-salient (0).
    Used by nvfp4.scaled_fp4_quant_with_mask for efficient input extension.

    Args:
        in_features: Original input dimension K
        salient_indices: 1D tensor of salient channel indices

    Returns:
        uint8 tensor of shape [in_features] with 1s at salient positions
    """
    mask = torch.zeros(in_features, dtype=torch.uint8, device=salient_indices.device)
    mask[salient_indices] = 1
    return mask


def infer_num_salient(
    packed_weight_shape: tuple,
    original_in_features: int,
) -> int:
    """Infer number of salient channels from packed weight shape.

    Args:
        packed_weight_shape: Weight tensor shape [out, (K+S)/2]
        original_in_features: Original input dimension K

    Returns:
        Number of salient channels S
    """
    # Weight is FP4 nibble-packed: 2 values per byte
    extended_in_features = packed_weight_shape[1] * 2
    num_salient = extended_in_features - original_in_features
    return num_salient
