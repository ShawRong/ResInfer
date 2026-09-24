# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Initialize enabled ResInfer integrations through vLLM's general plugin hook.

vLLM calls this entry point before model construction in its serving processes,
including spawned workers. The worker must be able to import ``quant``.

Each gate is independent and enabled only by the literal string "1":
    RESIDUE_NVFP4=1    install the NVFP4 LinearMethod patch
    RESIDUE_MXFP4=1    register the ModelOpt MXFP4 LinearMethod adapter
    RESIDUE_ATTN=1     register the ResAttn attention backend

RESIDUE_PATCH_MODE selects the NVFP4 patch mode (default v2).
RESIDUE_VLLM_MODEL_DIR supplies the checkpoint directory for residue metadata.
Importing this module does not install integrations; vLLM calls ``apply()``.
"""
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
            from quant.vllm_integration.patch import (is_patched,
                                                      patch_vllm_modelopt)

            model_dir = os.environ.get("RESIDUE_VLLM_MODEL_DIR")
            # patch mode (mext_r1 / v2) is read from RESIDUE_PATCH_MODE inside
            # patch_vllm_modelopt via version="auto".
            patch_vllm_modelopt(model_dir=model_dir)
            # patch_vllm_modelopt returns False BOTH on "already patched" and
            # on a swallowed ImportError -- so the return value alone cannot
            # distinguish success from silent failure. Verify the class
            # attribute instead: if the patch is not actually installed,
            # vLLM would serve stock NVFP4 with numbers that look normal.
            if not is_patched():
                raise RuntimeError(
                    "patch_vllm_modelopt did not install (see the warning "
                    "it printed above)")
        if resattn:
            from quant.vllm_integration.attn.register import install

            if not install():
                raise RuntimeError("ResAttn backend registration was refused")
        print(f"[residue:plugin] applied in pid={pid}", flush=True)
    except Exception as exc:
        # RAISE, do not just report. Printing is not loud enough: vLLM would
        # carry on and serve the model as plain NVFP4, producing numbers that
        # look completely normal. A residue arm silently becomes its own
        # control, and the results table shows "residue == plain" -- which
        # reads as a kernel bug and sends the next person to debug the fold.
        # That has already cost someone an evening.
        #
        # Failing is safe to do here because the caller explicitly enabled at
        # least one residue/ResAttn component. Refusing to start beats a fake
        # success with a partially installed pipeline.
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
