# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Register ResAttn as vLLM's CUSTOM attention backend."""

import inspect
import os

from .. import diagnostics

_INSTALLED_ATTR = "_residue_res_attn_registered"
_BACKEND_CLASS_PATH = (
    "quant.vllm_integration.attn.backend.ResAttnFlashInferBackend"
)


def _registry_shape_ok(reg) -> bool:
    """Check vLLM's custom-backend registration API."""
    if not hasattr(reg, "register_backend") or not hasattr(
        reg, "AttentionBackendEnum"
    ):
        diagnostics.log_registration_failure("registry lacks register_backend/AttentionBackendEnum")
        return False
    if "CUSTOM" not in reg.AttentionBackendEnum.__members__:
        diagnostics.log_registration_failure("registry has no CUSTOM member")
        return False
    try:
        params = inspect.signature(reg.register_backend).parameters
    except (TypeError, ValueError) as e:
        diagnostics.log_registration_failure("cannot introspect register_backend", e)
        return False
    if "backend" not in params or "class_path" not in params:
        diagnostics.log_registration_failure("register_backend signature changed", list(params))
        return False
    return True


def install(force: bool = False) -> bool:
    """Register ResAttn when enabled, or when force=True; return registration status."""
    if not force and os.environ.get("RESIDUE_ATTN") != "1":
        return False

    try:
        import vllm.v1.attention.backends.registry as reg
    except Exception as e:  # pragma: no cover - vLLM missing/drifted
        diagnostics.log_registration_failure("vLLM attention registry unavailable", e)
        return False

    if getattr(reg, _INSTALLED_ATTR, False):
        return True

    if not _registry_shape_ok(reg):
        return False

    if reg.AttentionBackendEnum.CUSTOM.is_overridden():
        try:
            other = reg.AttentionBackendEnum.CUSTOM.get_path()
        except Exception:
            other = "<unknown>"
        diagnostics.log_registration_failure("CUSTOM already registered; refusing to overwrite", other)
        return False

    try:
        from quant.vllm_integration.attn.backend import (
            ResAttnFlashInferBackend,
        )
        from vllm.v1.attention.backend import AttentionBackend

        assert issubclass(ResAttnFlashInferBackend, AttentionBackend)
    except Exception as e:
        diagnostics.log_registration_failure("backend class failed to import/validate", e)
        return False

    reg.register_backend(
        reg.AttentionBackendEnum.CUSTOM, class_path=_BACKEND_CLASS_PATH
    )
    setattr(reg, _INSTALLED_ATTR, True)
    diagnostics.log_registration_success(_BACKEND_CLASS_PATH)
    return True
