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

from .linear_method_v2 import ResidueNvFp4LinearMethodV2
from .patch import (
    is_patched,
    is_patched_mext_r1,
    patch_vllm_modelopt,
    patch_vllm_modelopt_mext_r1,
    patch_vllm_modelopt_v2,
    unpatch_vllm_modelopt,
)

__all__ = [
    # Linear methods
    "ResidueNvFp4LinearMethodV2",
    # Patch functions (recommended way to use)
    "patch_vllm_modelopt",
    "patch_vllm_modelopt_mext_r1",
    "patch_vllm_modelopt_v2",
    "unpatch_vllm_modelopt",
    "is_patched",
    "is_patched_mext_r1",
]
