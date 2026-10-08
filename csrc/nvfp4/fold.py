# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 Fold families exposed to the FlashInfer tuner.

Each candidate includes its activation layout; FlashInfer times quantization
and GEMM together. Kernel tables and static fallback live in cute_fold.tactics.
"""
import os

PRODUCTION_FAMILY = ("row_pair", "std")
ENABLE_MEXT_FAMILY = os.environ.get("RESIDUE_FOLD_MEXT", "0") == "1"


def candidate_families() -> list[tuple[str, str]]:
    """Keep A4/A6 tile8 candidates behind the existing opt-in gate."""
    families = [PRODUCTION_FAMILY]
    if ENABLE_MEXT_FAMILY:
        families.extend((("tile8", "std"), ("tile8", "pi")))
    return families
