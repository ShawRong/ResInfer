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
