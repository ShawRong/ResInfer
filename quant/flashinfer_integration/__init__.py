# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResInfer operators for FlashInfer autotuning."""

from importlib import import_module

_ENTRIES = {
    "tuned_linear": "linear_tuner",
    "tuned_nvfp4_linear": "nvfp4_linear_adapter",
    "tuned_mxfp4_fold": "mxfp4_linear_adapter",
}
__all__ = list(_ENTRIES)


def __getattr__(name):
    if name not in _ENTRIES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_ENTRIES[name]}", __name__), name)
    globals()[name] = value
    return value
