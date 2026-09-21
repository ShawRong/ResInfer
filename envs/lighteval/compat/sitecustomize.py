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
