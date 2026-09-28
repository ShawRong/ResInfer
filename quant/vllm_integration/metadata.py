"""Shared sidecar reader and compatibility helpers for residue metadata.

Readers keep format-specific validation and caching in their callers. Importing
this module does not initialize PyTorch or a quantization backend.
"""

from __future__ import annotations

import json
from pathlib import Path
def read_residue_metadata_json(metadata_path: str | Path) -> dict:
    """Read one sidecar; propagate file and JSON errors without caching."""
    with open(metadata_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Residue metadata must be an object: {metadata_path}")
    return data


def load_residue_metadata(model_dir: str | Path) -> dict:
    """Load a model's optional sidecar; plain checkpoints may omit it."""
    path = Path(model_dir) / "residue_kernel_metadata.json"
    return read_residue_metadata_json(path) if path.exists() else {}
