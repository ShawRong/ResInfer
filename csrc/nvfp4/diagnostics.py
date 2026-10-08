"""Fold logging and passive compile-cache inspection; no kernel imports."""
from __future__ import annotations

import contextlib
import os
import sys
import warnings


def silence_default_notice() -> bool:
    """Suppress the static-tactic notice without changing the selected tactic."""
    return os.environ.get("RESIDUE_FOLD_SILENCE_DEFAULT") == "1"


def log_compilation() -> bool:
    """Report SM120 Fold compilations without changing cache behavior."""
    return os.environ.get("NVFP4_FOLD_COMPILE_LOG") == "1"


_DEFAULT_PATH_WARNED = False
_SM10X_HOST = "nvfp4.cute_fold.host"
_SM120_HOST = "nvfp4.cute_fold_sm120.host"


def compile_stats() -> dict[str, dict | None]:
    """Fold kernels JIT-compiled by this process so far, per arch.

    Side-effect free: never imports a module that is not already loaded.
    `None` for an arch means its host module has not been imported at all,
    which is the normal state for the arch this GPU is not.
    """
    out: dict[str, dict | None] = {"sm10x": None, "sm120": None}

    for arch, module_name in (("sm10x", _SM10X_HOST), ("sm120", _SM120_HOST)):
        mod = sys.modules.get(module_name)
        cache = getattr(mod, "_KERNEL_CACHE", None)
        if cache is not None:
            out[arch] = {"compiled": len(cache), "keys": list(cache)}

    return out


@contextlib.contextmanager
def observe_compiles(where: str, *, warn: bool = True):
    """Report any fold kernel JIT-compiled inside this block.

    Wrap the regions that must already be warm -- cudagraph capture above all,
    where a JIT is both slow and, for the sm10x path, a sync inside capture.
    A clean block prints nothing; the whole point is that the silent case
    stays silent and the broken case does not.
    """
    before = compile_stats()
    try:
        yield
    finally:
        after = compile_stats()
        for arch in ("sm10x", "sm120"):
            b, a = before.get(arch), after.get(arch)
            if not a:
                continue
            n = a["compiled"] - (b["compiled"] if b else 0)
            if n <= 0:
                continue
            detail = ""
            if a["keys"] is not None:
                fresh = a["keys"][len(b["keys"]) if b else 0:]
                detail = "".join(f"\n[residue]     {k}" for k in fresh[:8])
                if len(fresh) > 8:
                    detail += f"\n[residue]     ... and {len(fresh) - 8} more"
            if warn:
                print(
                    f"[residue] WARMUP UNDER-COVERAGE: {n} {arch} fold "
                    f"kernel(s) JIT-compiled during {where}. Each costs ~1.3 s "
                    f"and blocks the caller. The pre-compile missed:{detail}",
                    flush=True,
                )


def warn_default_path(arch: str, m_tok: int, detail: str = "") -> None:
    """Report the SM100 static fallback once; tuning belongs to FlashInfer."""
    global _DEFAULT_PATH_WARNED
    if _DEFAULT_PATH_WARNED:
        return
    _DEFAULT_PATH_WARNED = True
    warnings.warn(
        f"[residue] Fold uses the static fallback "
        f"(arch={arch}, m_tok={m_tok}{', ' + detail if detail else ''}). "
        "Use the FlashInfer integration autotune path to measure candidates, "
        "or set RESIDUE_FOLD_SILENCE_DEFAULT=1 to silence this notice.",
        RuntimeWarning,
        stacklevel=3,
    )
