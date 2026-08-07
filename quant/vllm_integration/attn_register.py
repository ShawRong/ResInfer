# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"Register the ResAttn backend as vLLM's CUSTOM attention backend."

import inspect
import os

_INSTALLED_ATTR = "_residue_res_attn_registered"
_BACKEND_CLASS_PATH = "quant.vllm_integration.attn_backend.ResAttnBackend"


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
        print(
            "[residue] ResAttn register: vLLM registry lacks "
            "register_backend/AttentionBackendEnum (layout drift); refusing.",
            flush=True,
        )
        return False
    if "CUSTOM" not in reg.AttentionBackendEnum.__members__:
        print(
            "[residue] ResAttn register: AttentionBackendEnum has no CUSTOM "
            "member (vLLM drift); refusing.",
            flush=True,
        )
        return False
    try:
        params = inspect.signature(reg.register_backend).parameters
    except (TypeError, ValueError) as e:
        print(
            f"[residue] ResAttn register: cannot introspect register_backend "
            f"({e}); refusing.",
            flush=True,
        )
        return False
    if "backend" not in params or "class_path" not in params:
        print(
            "[residue] ResAttn register: register_backend signature changed "
            f"(params={list(params)}); refusing.",
            flush=True,
        )
        return False
    return True


def install(force: bool = False) -> bool:
    """Register ResAttnBackend on AttentionBackendEnum.CUSTOM.

    Idempotent; returns True iff the backend is (now) registered. Gated on
    ``RESIDUE_ATTN=1`` unless ``force=True``. Any structural mismatch refuses
    loudly and leaves vLLM untouched.
    """
    if not force and os.environ.get("RESIDUE_ATTN") != "1":
        return False  # normal off state -- quiet by design

    try:
        import vllm.v1.attention.backends.registry as reg
    except Exception as e:  # pragma: no cover - vLLM missing/drifted
        print(
            f"[residue] ResAttn register: vLLM v1 attention registry "
            f"unavailable ({e}); refusing -- vLLM runs with its default "
            "backend.",
            flush=True,
        )
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
        print(
            f"[residue] ResAttn register: CUSTOM already registered to "
            f"{other!r}; refusing to overwrite. Unset the other registration "
            "or clear_override() first.",
            flush=True,
        )
        return False

    # Import our backend BEFORE registering: if the vLLM interfaces our class
    # subclasses have drifted, fail now with a real traceback message instead
    # of at first attention call inside the engine.
    try:
        from quant.vllm_integration.attn_backend import ResAttnBackend
        from vllm.v1.attention.backend import AttentionBackend

        assert issubclass(ResAttnBackend, AttentionBackend)
    except Exception as e:
        print(
            f"[residue] ResAttn register: backend class failed to import/"
            f"validate ({e!r}); refusing -- vLLM runs with its default "
            "backend.",
            flush=True,
        )
        return False

    reg.register_backend(
        reg.AttentionBackendEnum.CUSTOM, class_path=_BACKEND_CLASS_PATH
    )
    setattr(reg, _INSTALLED_ATTR, True)
    print(
        "[residue] ResAttn backend REGISTERED on AttentionBackendEnum.CUSTOM "
        f"({_BACKEND_CLASS_PATH}). Select it with --attention-backend CUSTOM "
        "--kv-cache-dtype nvfp4 --block-size 128. Decode runs the fp4 kernel; "
        "prefill goes through the v0 bridge (correct, not fast).",
        flush=True,
    )
    return True


# NO import-time install. The plugin (residue_vllm_plugin.apply) is the one
# sanctioned installer; in-process launchers call install() themselves. An
# unrelated `import quant.vllm_integration.attn_register` in a test mutating
# vLLM's process-global _ATTN_OVERRIDES was exactly the kind of spooky action
# the plugin exists to prevent.
