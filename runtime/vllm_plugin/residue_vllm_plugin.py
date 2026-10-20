# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Initialize ResInfer integrations through vLLM's plugin hook."""
from __future__ import annotations

import os


def initialize() -> None:
    """Install enabled ResInfer integrations in the current vLLM process."""
    nvfp4 = os.environ.get("RESIDUE_NVFP4") == "1"
    resattn = os.environ.get("RESIDUE_ATTN") == "1"
    mxfp4 = os.environ.get("RESIDUE_MXFP4") == "1"
    if not (nvfp4 or resattn or mxfp4):
        return
    pid = os.getpid()
    try:
        if mxfp4:
            from quant.vllm_integration import linear_method_mxfp4  # noqa: F401
        if nvfp4:
            from quant.vllm_integration.patch import patch_vllm_modelopt

            model_dir = os.environ.get("RESIDUE_VLLM_MODEL_DIR")
            # patch mode (mext_r1 / v2) is read from RESIDUE_PATCH_MODE inside
            # patch_vllm_modelopt via version="auto".
            patch_vllm_modelopt(model_dir=model_dir)
        if resattn:
            from quant.vllm_integration.attn.register import install

            if not install():
                raise RuntimeError("ResAttn backend registration was refused")
        print(f"[residue:plugin] applied in pid={pid}", flush=True)
    except Exception as exc:
        print(
            f"[residue:plugin] FAILED in pid={pid}: {type(exc).__name__}: {exc}\n"
            f"[residue:plugin] refusing to serve without the residue kernels "
            f"(a residue/ResAttn gate was set). Unset the gate to run plain "
            f"vLLM deliberately.",
            flush=True,
        )
        raise


def apply() -> None:
    """Stable ``vllm.general_plugins`` entry point, called in every worker."""
    initialize()
