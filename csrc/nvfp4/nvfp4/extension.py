"""C++ extension loader for the local NVFP4 kernels.

The compiled extension backs the fold / quant / GEMM ops. It is NOT needed by
``nvfp4.cute_attn`` (paged NVFP4 attention), which is pure CuTeDSL + torch, so
importing this package must not hard-fail when the extension is absent: the
module-level load is soft, and callers that actually need an op get a clear
error from ``ensure_loaded()``.
"""

from __future__ import annotations

import importlib.util
import warnings

import torch

_LOADED = False
_WARNED = False


def _warn_if_stale(so_path: str) -> None:
    'Warn when the built extension is OLDER than the C++ it was built from.'
    import os
    from pathlib import Path

    try:
        src = Path(so_path).resolve().parent.parent / "src"
        if not src.is_dir():
            return
        so_mtime = os.path.getmtime(so_path)
        newest, newest_f = 0.0, None
        for f in src.rglob("*"):
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
                f"csrc/nvfp4/build.sh (a bare `pip install -e csrc/nvfp4` "
                f"fails under default PEP-517 isolation: setup.py needs "
                f"torch at build time).",
                RuntimeWarning,
                stacklevel=3,
            )
    except OSError:
        pass

_MISSING_MSG = (
    "nvfp4 extension (nvfp4._C) not found or unloadable. Build it with "
    "`csrc/nvfp4/build.sh` if you need the fold / quant / GEMM ops. "
    "nvfp4.cute_attn (paged NVFP4 attention) does not require it."
)


def ensure_loaded(required: bool = True) -> bool:
    """Load ``nvfp4._C`` once so ``torch.ops.nvfp4.*`` is registered.

    Returns True iff the extension is loaded. ``required=False`` downgrades a
    missing or ABI-incompatible extension to a one-time warning, which is what
    the module-level import does.
    """
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
        except OSError as exc:  # built against a different libstdc++/torch ABI
            err = exc

    if required:
        raise ImportError(_MISSING_MSG) from err
    if not _WARNED:
        _WARNED = True
        warnings.warn(f"{_MISSING_MSG} ({err})", RuntimeWarning, stacklevel=2)
    return False


ensure_loaded(required=False)

__all__ = ["ensure_loaded"]
