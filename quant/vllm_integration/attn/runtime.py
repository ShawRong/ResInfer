# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Lazy kernel loading and the decode residue feature gate."""
import os

from .. import diagnostics

_DECODE_KERNEL_HOST = None
_DECODE_KERNEL_CHECKED = False
_QK_RESIDUE = os.environ.get("RESIDUE_ATTN_DISABLE_QK_RESIDUE", "0") != "1"


def _qk_residue_for_geometry(num_qo_heads: int, num_kv_heads: int) -> bool:
    """Use the default residue path where its Q8/Q16 mapping exists."""
    if num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        return False
    return _QK_RESIDUE and num_qo_heads // num_kv_heads in (4, 6, 8)


def _load_decode_kernel():
    """Return the promoted `nvfp4.cute_attn.host` module, or None (guarded)."""
    global _DECODE_KERNEL_HOST, _DECODE_KERNEL_CHECKED
    if _DECODE_KERNEL_CHECKED:
        return _DECODE_KERNEL_HOST
    _DECODE_KERNEL_CHECKED = True
    try:
        from nvfp4.cute_attn import host  # type: ignore[attr-defined]

        _DECODE_KERNEL_HOST = host
    except Exception as exc:
        _DECODE_KERNEL_HOST = None
        diagnostics.log_decode_import_failure(exc)
    return _DECODE_KERNEL_HOST
