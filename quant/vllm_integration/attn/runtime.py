# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Lazy kernel loading and the decode residue feature gate."""
import os

_QK_RESIDUE = os.environ.get("RESIDUE_ATTN_DISABLE_QK_RESIDUE", "0") != "1"


def _validate_residue_geometry(num_qo_heads: int, num_kv_heads: int) -> None:
    """Validate the serving Q8/Q16 contract; ordinary QK is a dev experiment."""
    if not _QK_RESIDUE:
        raise NotImplementedError("ordinary QK moved to .dev/tools/resattn/experimental")
    if (num_kv_heads <= 0 or num_qo_heads % num_kv_heads
            or num_qo_heads // num_kv_heads not in (4, 6, 8)):
        raise NotImplementedError("production ResAttn requires GQA4/6/8")


def _load_decode_kernel():
    """Import on first use; Python caches successful imports, failures propagate."""
    try:
        from nvfp4.cute_attn import host
    except Exception as exc:
        raise RuntimeError("Failed to load ResAttn kernel host nvfp4.cute_attn.host") from exc
    return host
