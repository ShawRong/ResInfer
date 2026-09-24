"""Shared sidecar reader and compatibility helpers for residue metadata.

Readers keep format-specific validation and caching in their callers. Importing
this module does not initialize PyTorch or a quantization backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


def read_residue_metadata_json(metadata_path: str | Path) -> dict:
    """Read one sidecar; propagate file and JSON errors without caching."""
    with open(metadata_path) as f:
        return json.load(f)


def load_residue_metadata(model_dir: str | Path) -> dict:
    """Load a model's optional sidecar; plain checkpoints may omit it."""
    path = Path(model_dir) / "residue_kernel_metadata.json"
    return read_residue_metadata_json(path) if path.exists() else {}


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
    import torch

    layers = metadata.get("layers", {})
    layer_meta = layers.get(layer_name, {})
    indices = layer_meta.get("salient_indices", [])
    return torch.tensor(indices, dtype=torch.int64, device=device)
