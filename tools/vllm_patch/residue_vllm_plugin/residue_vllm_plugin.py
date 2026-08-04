# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""vLLM general-plugin shim that installs the Residue NVFP4 patch in EVERY
vLLM process, including the V1 EngineCore worker subprocess.

Why this exists
---------------
vLLM V1 (>= the dev260 wheel we pin) runs EngineCore in a *spawned subprocess*
by default (``VLLM_ENABLE_V1_MULTIPROCESSING=1``). That subprocess — not the
main process — is what builds the model executor and loads weights
(``vllm/v1/engine/core.py``: ``load_general_plugins()`` at ctor line 110, then
``executor_class(vllm_config)`` at line 123). Patching ``ModelOptNvFp4Config``
only in the main process therefore never reaches the worker, and weight loading
falls back to stock ModelOpt → the qkv_proj shape assert.

Older scripts (``vllm==0.15.0``) "worked" only because that version's offline
``LLM()`` ran EngineCore in-process, so there was no subprocess to miss. It was
never a real fix for worker propagation.

The canonical, version- and container-layout-independent mechanism is the
``vllm.general_plugins`` entry-point group: vLLM loads it in *all* processes,
before the model is built. That is strictly more robust than sitecustomize/.pth
(which depend on the container's python site layout and did not run in Modal's
spawned worker).

Enable (opt-in, same env knobs the old scripts used):
    RESIDUE_PATCH_VLLM=1          turn the plugin on
    RESIDUE_PATCH_MODE=mext_r1    patch mode (patch.py reads this; default v2)
    RESIDUE_VLLM_MODEL_DIR=...     model dir for residue metadata lookup

``quant`` must be importable in the worker. Put the repo root on ``PYTHONPATH``
(env is inherited by spawned subprocesses) so ``import quant`` works there.

The function is idempotent — ``patch_vllm_modelopt`` guards re-entry, and vLLM
may call plugins more than once per process.
"""
from __future__ import annotations

import os


def apply() -> None:
    """Entry point for the ``vllm.general_plugins`` group."""
    if not os.environ.get("RESIDUE_PATCH_VLLM"):
        return
    pid = os.getpid()
    try:
        from quant.vllm_integration.patch import patch_vllm_modelopt

        model_dir = os.environ.get("RESIDUE_VLLM_MODEL_DIR")
        # patch mode (mext_r1 / v2) is read from RESIDUE_PATCH_MODE inside
        # patch_vllm_modelopt via version="auto".
        patch_vllm_modelopt(model_dir=model_dir)
        print(f"[residue:plugin] applied in pid={pid}", flush=True)
    except Exception as exc:
        # RAISE, do not just report. Printing is not loud enough: vLLM would
        # carry on and serve the model as plain NVFP4, producing numbers that
        # look completely normal. A residue arm silently becomes its own
        # control, and the results table shows "residue == plain" -- which
        # reads as a kernel bug and sends the next person to debug the fold.
        # That has already cost someone an evening.
        #
        # Failing is safe to do here because reaching this point means
        # RESIDUE_PATCH_VLLM=1, i.e. the caller explicitly asked for residue.
        # Refusing to start beats a fake success.
        print(
            f"[residue:plugin] FAILED in pid={pid}: {type(exc).__name__}: {exc}\n"
            f"[residue:plugin] refusing to serve without the residue kernels "
            f"(RESIDUE_PATCH_VLLM=1 was set). Unset it to run plain NVFP4 "
            f"deliberately.",
            flush=True,
        )
        raise
