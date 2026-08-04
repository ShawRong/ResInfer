"""C++ extension loader for the local NVFP4 kernels.

The compiled extension backs the fold / quant / GEMM ops. Importing this
package must not hard-fail when the extension is absent: the module-level
load is soft, and callers that actually need an op get a clear error from
``ensure_loaded()``.
"""

from __future__ import annotations

import importlib.util
import warnings

import torch

_LOADED = False
_WARNED = False

_MISSING_MSG = (
    "nvfp4 extension (nvfp4._C) not found or unloadable. Build it with "
    "`csrc/nvfp4/build.sh` if you need the fold / quant / GEMM ops."
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
