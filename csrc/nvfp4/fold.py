# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 Fold candidates for FlashInfer autotuning."""
import os

PRODUCTION_FAMILY = ("row_pair", "std")
ENABLE_MEXT_FAMILY = os.environ.get("RESIDUE_FOLD_MEXT", "0") == "1"


def candidate_families() -> list[tuple[str, str]]:
    """Return enabled Fold candidate families."""
    families = [PRODUCTION_FAMILY]
    if ENABLE_MEXT_FAMILY:
        families.extend((("tile8", "std"), ("tile8", "pi")))
    return families
