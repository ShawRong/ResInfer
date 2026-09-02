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
    except Exception as exc:
        # Best-effort only: vLLM may not be installed for non-vLLM workflows.
        # But when RESIDUE_PATCH_VLLM is explicitly on, a silent skip means the
        # residue kernels never activate — surface the reason instead of hiding
        # it (this was the e2e "patch didn't reach the worker" root cause).
        # ALWAYS loud (audit F10): the comment above promises "surface the
        # reason instead of hiding it", but the print was gated behind a
        # debug flag -- so a real patch failure silently served plain NVFP4.
        # The plugin path re-raises; sitecustomize at least must print.
        if True:
            import traceback

            print(
                f"[residue:sitecustomize] patch FAILED in pid={os.getpid()}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
        return


_maybe_patch_vllm()

