"""Site customization for Residue vLLM patch (repo-local).

This module is automatically imported by Python at startup if it is importable
on sys.path. We keep this in-repo (instead of writing into `.venv/`) so the
patch behavior is:

- Version-controlled
- Reproducible across machines / fresh environments
- Applied in vLLM spawn worker processes (when PYTHONPATH includes repo root)

Enable by setting:
  RESIDUE_PATCH_VLLM=1

and (recommended) also setting:
  RESIDUE_VLLM_MODEL_DIR=/path/to/exported_model
"""

from __future__ import annotations

import os


def _maybe_patch_vllm() -> None:
    if not os.environ.get("RESIDUE_PATCH_VLLM"):
        return

    # Pass model_dir if available; worker processes inherit this env var.
    model_dir = os.environ.get("RESIDUE_VLLM_MODEL_DIR")

    try:
        from quant.vllm_integration.patch import patch_vllm_modelopt

        patch_vllm_modelopt(model_dir=model_dir)
    except Exception:
        # Best-effort only: vLLM may not be installed for non-vLLM workflows.
        return


_maybe_patch_vllm()

