# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Load the NVFP4 CUDA extension."""

from __future__ import annotations

import importlib.util
import warnings

import torch

_LOADED = False
_WARNED = False


def _warn_if_stale(so_path: str) -> None:
    """Warn if the compiled extension predates its sources."""
    import os
    from pathlib import Path

    try:
        src = Path(so_path).resolve().parent.parent
        if not (src / "bindings" / "pybind.cpp").is_file():
            return
        so_mtime = os.path.getmtime(so_path)
        newest, newest_f = 0.0, None
        native_dirs = ("bindings", "common/cuda", "common/fp4", "nvfp4/cuda",
                       "mxfp4/cuda")
        for f in (p for directory in native_dirs
                  for p in (src / directory).rglob("*")):
            if f.suffix in (".cu", ".cuh", ".cpp", ".h") :
                m = os.path.getmtime(f)
                if m > newest:
                    newest, newest_f = m, f
        if newest > so_mtime:
            warnings.warn(
                f"nvfp4._C at {so_path} is OLDER than its C++ sources "
                f"(newest: {newest_f}). The stale binary still exports its "
                f"old ops, so hasattr fallbacks will NOT catch this -- "
                f"results can be silently wrong. Rebuild with "
                f"`python -m pip install --no-build-isolation --no-deps ./csrc` "
                f"in the target Torch environment.",
                RuntimeWarning,
                stacklevel=3,
            )
    except OSError:
        pass

_MISSING_MSG = (
    "nvfp4 extension (nvfp4._C) not found or unloadable. Build it with "
    "`python -m pip install --no-build-isolation --no-deps ./csrc`. "
    "ResAttn residue-Q decode requires this extension."
)


def ensure_loaded(required: bool = True) -> bool:
    """Load the extension and report whether its operators are available."""
    global _LOADED, _WARNED
    if _LOADED:
        return True

    spec = importlib.util.find_spec("nvfp4._C")
    err: Exception | None = None
    if spec is None or spec.origin is None:
        err = ImportError(_MISSING_MSG)
    else:
        _warn_if_stale(spec.origin)
        try:
            torch.ops.load_library(spec.origin)
            _LOADED = True
            return True
        except OSError as exc:
            err = exc

    if required:
        raise ImportError(_MISSING_MSG) from err
    if not _WARNED:
        _WARNED = True
        warnings.warn(f"{_MISSING_MSG} ({err})", RuntimeWarning, stacklevel=2)
    return False


ensure_loaded(required=False)

__all__ = ["ensure_loaded"]
