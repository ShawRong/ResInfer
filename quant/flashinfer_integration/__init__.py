"""ResInfer operators adapted to FlashInfer's AutoTuner, independent of vLLM.

Use FlashInfer's ``autotune`` context to profile these operators. Framework
startup, distributed cache broadcast, and graph capture belong to callers.
Imports stay lazy so importing this package does not load GPU dependencies.
"""

from importlib import import_module

_ENTRIES = {
    "tuned_resattn_decode": "resattn_tuner",
    "tuned_linear": "linear_tuner",
    "tuned_nvfp4_linear": "nvfp4_linear_tuner",
    "tuned_mxfp4_fold": "mxfp4_fold_tuner",
}
__all__ = list(_ENTRIES)


def __getattr__(name):
    if name not in _ENTRIES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_ENTRIES[name]}", __name__), name)
    globals()[name] = value
    return value
