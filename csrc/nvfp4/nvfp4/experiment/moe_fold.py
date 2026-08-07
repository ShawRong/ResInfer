# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""JIT-built row-pair fold kernel. EXPERIMENT -- see this folder's README.

`fold_pairs(t)` replaces `t[0::2] + t[1::2]` and falls back to exactly that if
the kernel will not build, so it cannot break a caller.

Built with torch.utils.cpp_extension.load into its OWN module, not into
nvfp4._C. An experiment must not be able to change which ops the production
extension exports -- a stale _C.so has already produced both a
cudaErrorNoKernelImageForDevice and an AttributeError for an op that was in
fact registered, and both pointed at the wrong cause.
"""

from __future__ import annotations

from pathlib import Path

import torch

_SOURCE = Path(__file__).resolve().parent / "moe_fold_reduce.cu"

_module = None
_tried = False
_error: str | None = None


def load(verbose: bool = False):
    """Build once; return the module, or None with `load_error()` set."""
    global _module, _tried, _error
    if _tried:
        return _module
    _tried = True
    try:
        from torch.utils.cpp_extension import load as _load

        flags = ["-O3", "--use_fast_math"]
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            # Current device only. An experiment does not need a fat binary,
            # and one arch keeps the first-call compile short.
            flags.append(
                f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"
            )
        _module = _load(
            name="nvfp4_experiment_moe_fold",
            sources=[str(_SOURCE)],
            extra_cuda_cflags=flags,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        _error = f"{type(exc).__name__}: {exc}"
        _module = None
    return _module


def load_error() -> str | None:
    return _error


def available() -> bool:
    return load() is not None


def fold_pairs(t: torch.Tensor, *, impl: str = "auto") -> torch.Tensor:
    """Sum adjacent rows along dim 0: (2M, ...) -> (M, ...).

    impl='cuda' pins the kernel and raises if it is unavailable; 'torch' pins
    the slice-add reference. Benchmarks must pin, or a silent fallback makes
    the kernel look exactly as fast as the thing it replaces.
    """
    if t.shape[0] % 2 != 0:
        raise ValueError(
            f"expected an even leading dim to fold row pairs, got {tuple(t.shape)}"
        )
    if impl == "torch":
        return t[0::2] + t[1::2]
    if impl == "cuda":
        mod = load()
        if mod is None:
            raise RuntimeError(f"moe_fold kernel unavailable: {load_error()}")
        return mod.moe_fold_pairs(t)
    if impl != "auto":
        raise ValueError(f"impl must be auto|cuda|torch, got {impl!r}")

    if t.is_cuda:
        mod = load()
        if mod is not None:
            return mod.moe_fold_pairs(t)
    return t[0::2] + t[1::2]