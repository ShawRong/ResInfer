# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResInfer linear-method integration for vLLM."""

from importlib import import_module


def __getattr__(name):
    if name == "ResidueNvFp4LinearMethodV2":
        module = import_module(".linear_method_v2", __name__)
    elif name in ("ModelOptMxfp4Config", "ResidueMxfp4LinearMethod"):
        module = import_module(".linear_method_mxfp4", __name__)
    elif name in __all__:
        module = import_module(".patch", __name__)
    else:
        raise AttributeError(name)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    # Linear methods
    "ModelOptMxfp4Config",
    "ResidueMxfp4LinearMethod",
    "ResidueNvFp4LinearMethodV2",
    # Patch functions (recommended way to use)
    "patch_vllm_modelopt",
    "patch_vllm_modelopt_mext_r1",
    "patch_vllm_modelopt_v2",
    "is_patched",
    "is_patched_mext_r1",
]
