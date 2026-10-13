"""Compatibility fixes loaded only by the isolated lighteval scorer."""

from __future__ import annotations

try:
    import xxhash
except ImportError:  # The scorer preflight reports the missing dependency.
    xxhash = None


if xxhash is not None and not getattr(xxhash.xxh64, "_resinfer_text_compat", False):
    _original_xxh64 = xxhash.xxh64

    def _xxh64(data=b"", *args, **kwargs):
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _original_xxh64(data, *args, **kwargs)

    _xxh64._resinfer_text_compat = True
    xxhash.xxh64 = _xxh64


# Opt-in only for our pinned scorer process; failures must stop evaluation.
import os
if os.environ.get("RESINFER_LIGHTEVAL_CUSTOM_ONLY") == "1":
    try:
        from importlib.metadata import version
        if version("lighteval") != "0.13.0":
            raise RuntimeError("custom-only discovery requires reviewed lighteval 0.13.0")
        from lighteval.tasks.registry import Registry
        from benchmarks.quality.lighteval.compat.custom_registry import install
        install(Registry)
    except Exception as exc:
        raise SystemExit(f"Cannot install custom-only task discovery: {exc}") from exc

if os.environ.get("RESINFER_LIGHTEVAL_SAMPLING_COMPAT") == "1":
    try:
        from importlib.metadata import version
        if version("lighteval") != "0.13.0":
            raise RuntimeError("sampling compatibility requires reviewed lighteval 0.13.0")
        from lighteval.models.model_input import GenerationParameters
        from benchmarks.quality.lighteval.compat.sampling import install
        install(GenerationParameters)
    except Exception as exc:
        raise SystemExit(f"Cannot install explicit evaluation sampling protocol: {exc}") from exc
