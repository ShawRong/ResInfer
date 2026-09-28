"""Site customization for Residue vLLM patch (repo-local).

This module is automatically imported by Python at startup if it is importable
on sys.path. We keep this in-repo (instead of writing into `.venv/`) so the
patch behavior is:

- Version-controlled
- Reproducible across machines / fresh environments
- Applied in vLLM spawn worker processes (when PYTHONPATH includes repo root)

Enable by setting:
  RESIDUE_NVFP4=1

and (recommended) also setting:
  RESIDUE_VLLM_MODEL_DIR=/path/to/exported_model
"""

from __future__ import annotations

import os


def _maybe_patch_vllm() -> None:
    import sys
    from pathlib import Path

    if os.environ.get("RESIDUE_NVFP4") != "1":
        return

    # Pass model_dir if available; worker processes inherit this env var.
    model_dir = os.environ.get("RESIDUE_VLLM_MODEL_DIR")

    try:
        from quant.vllm_integration.patch import patch_vllm_modelopt

        patch_vllm_modelopt(model_dir=model_dir)
    except Exception as exc:
        import traceback

        print(
            f"[residue:sitecustomize] patch FAILED in pid={os.getpid()}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        # Exceptions raised by sitecustomize are otherwise reported and then
        # ignored by Python. SystemExit makes an explicit patch request fail
        # closed before the serving process can fall back to stock vLLM.
        raise SystemExit(
            "RESIDUE_NVFP4=1 but the patch was not installed") from exc


_maybe_patch_vllm()
