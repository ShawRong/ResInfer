"""vLLM integration for Residue NVFP4 quantization.

This module provides a custom LinearMethod for residue-quantized models,
allowing vLLM to load and serve models exported with residue quantization.

Usage (with patch - recommended):
    # Patch vLLM's ModelOpt to use our linear method
    from quant.vllm_integration import patch_vllm_modelopt
    patch_vllm_modelopt()  # Call BEFORE creating LLM

    # Then use vLLM's native ModelOpt quantization entry.
    llm = LLM(model="./exported", quantization="modelopt_fp4")
"""

# Keep independent quantization integrations importable without initializing
# the NVFP4/ModelOpt patch stack. Existing public names stay lazy.
from importlib import import_module


def __getattr__(name):
    if name == "ResidueNvFp4LinearMethodV2":
        module = import_module(".linear_method_v2", __name__)
    elif name in ("ResidueMxfp4Config", "ResidueMxfp4LinearMethod"):
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
    "ResidueMxfp4Config",
    "ResidueMxfp4LinearMethod",
    "ResidueNvFp4LinearMethodV2",
    # Patch functions (recommended way to use)
    "patch_vllm_modelopt",
    "patch_vllm_modelopt_mext_r1",
    "patch_vllm_modelopt_v2",
    "unpatch_vllm_modelopt",
    "is_patched",
    "is_patched_mext_r1",
]
