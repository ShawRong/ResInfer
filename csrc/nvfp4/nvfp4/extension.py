"""C++ extension loader for the local NVFP4 kernels."""

from __future__ import annotations

import importlib.util

import torch

_LOADED = False


def ensure_loaded() -> None:
    """Load nvfp4._C once so torch.ops.nvfp4.* is registered."""
    global _LOADED
    if _LOADED:
        return

    spec = importlib.util.find_spec("nvfp4._C")
    if spec is None or spec.origin is None:
        raise ImportError(
            "nvfp4 extension not found. Please build and install the extension first:\n"
            "  ./build.sh"
        )
    torch.ops.load_library(spec.origin)
    _LOADED = True


ensure_loaded()


__all__ = ["ensure_loaded"]
