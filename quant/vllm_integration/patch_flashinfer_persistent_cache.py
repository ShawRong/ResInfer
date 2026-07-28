# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"Standalone patch: re-enable vLLM's FlashInfer PERSISTENT autotune cache."

_INSTALLED_ATTR = "_residue_persistent_cache_enabled"


def _flashinfer_file_key_has_extras() -> bool:
    """True iff FlashInfer carries the #3367 fix: the persistent file-cache key
    is the 4-tuple that includes per-runner ``get_cache_key_extras`` (so
    ``use_8x4_sf_layout`` True/False no longer collide).

    Detected structurally rather than by version string: the fix makes
    ``load_from_file`` build its lookup from ``key[3]`` (the extras slot). We
    require BOTH that marker and that the autotuner threads
    ``get_cache_key_extras`` into its cache key.
    """
    try:
        import inspect

        import flashinfer.autotuner as fa

        src = inspect.getsource(fa)
    except Exception as e:  # pragma: no cover - flashinfer layout drift
        print(f"[residue] persistent-cache patch: cannot introspect flashinfer "
              f"autotuner ({e}); refusing to enable persistence.", flush=True)
        return False
    return ("key[3]" in src) and ("get_cache_key_extras" in src)


def install() -> bool:
    """Enable vLLM's FlashInfer persistent autotune cache. Idempotent,
    best-effort: returns True iff persistence is (now) enabled. Any failure or a
    missing #3367 fix leaves vLLM's default behavior untouched."""
    try:
        import vllm.model_executor.warmup.kernel_warmup as kw
    except Exception as e:  # pragma: no cover - vLLM layout drift / not installed
        print(f"[residue] persistent-cache patch: vLLM kernel_warmup "
              f"unavailable ({e}); skipping.", flush=True)
        return False

    if getattr(kw, _INSTALLED_ATTR, False):
        return True

    if getattr(kw, "_FLASHINFER_USE_PERSISTENT_CACHE", None) is None:
        # The flag was renamed/removed -> vLLM may already persist, or the
        # surface changed. Do not guess; leave it alone.
        print("[residue] persistent-cache patch: vLLM has no "
              "_FLASHINFER_USE_PERSISTENT_CACHE flag (version drift); skipping.",
              flush=True)
        return False

    if getattr(kw, "_FLASHINFER_USE_PERSISTENT_CACHE") is True:
        # Upstream already re-enabled it -> nothing to do.
        setattr(kw, _INSTALLED_ATTR, True)
        print("[residue] FlashInfer persistent autotune cache already enabled "
              "by vLLM; nothing to patch.", flush=True)
        return True

    if not _flashinfer_file_key_has_extras():
        print("[residue] persistent-cache patch: installed FlashInfer lacks the "
              "#3367 file-key fix (use_8x4_sf_layout would collide) -- NOT "
              "enabling persistence; falling back to re-tune-every-boot.",
              flush=True)
        return False

    kw._FLASHINFER_USE_PERSISTENT_CACHE = True
    setattr(kw, _INSTALLED_ATTR, True)
    print("[residue] FlashInfer persistent autotune cache ENABLED "
          "(_FLASHINFER_USE_PERSISTENT_CACHE=True; safe: flashinfer #3367 "
          "4-tuple key present). First boot tunes+writes; later boots load.",
          flush=True)
    return True


# Auto-install on import -- mirrors how the residue patch is delivered into the
# worker (import side effect). Explicit install() stays available for callers
# that prefer to gate it themselves.
install()
