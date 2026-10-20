# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"Register the ResAttn backend as vLLM's CUSTOM attention backend."

import inspect
import os

from .. import diagnostics

_INSTALLED_ATTR = "_residue_res_attn_registered"
_BACKEND_CLASS_PATH = (
    "quant.vllm_integration.attn.backend.ResAttnFlashInferBackend"
)


def _registry_shape_ok(reg) -> bool:
    """Structural check: the vLLM registry still has the surface we target.

    Verified structurally rather than by version string (survives the
    0.23 -> 0.26 move unchanged): a ``register_backend(backend,
    class_path=...)`` callable,
    an ``AttentionBackendEnum`` with a ``CUSTOM`` member, and CUSTOM's
    registered-before-use contract (``get_path`` raising on unregistered
    CUSTOM is vLLM's own guard; we only need member + setter).
    """
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
    """Register the ResAttn backend on CUSTOM.

    Idempotent; returns True iff the backend is (now) registered. Gated on
    ``RESIDUE_ATTN=1`` unless ``force=True``. Any structural mismatch refuses
    loudly and leaves vLLM untouched.
    """
    if not force and os.environ.get("RESIDUE_ATTN") != "1":
        return False  # normal off state -- quiet by design

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
        # Someone else claimed the single CUSTOM slot -- do not clobber it.
        try:
            other = reg.AttentionBackendEnum.CUSTOM.get_path()
        except Exception:
            other = "<unknown>"
        diagnostics.log_registration_failure("CUSTOM already registered; refusing to overwrite", other)
        return False

    # Import our backend BEFORE registering: if the vLLM interfaces our class
    # subclasses have drifted, fail now with a real traceback message instead
    # of at first attention call inside the engine.
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


# NO import-time install. The plugin (residue_vllm_plugin.apply) is the one
# sanctioned installer; in-process launchers call install() themselves. An
# unrelated `import quant.vllm_integration.attn.register` in a test mutating
# vLLM's process-global _ATTN_OVERRIDES was exactly the kind of spooky action
# the plugin exists to prevent.
