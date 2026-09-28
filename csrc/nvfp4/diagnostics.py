"""Output-only switches shared by the SM10x and SM120 Fold implementations."""
from __future__ import annotations

import os


def silence_default_notice() -> bool:
    """Suppress the static-tactic notice without changing the selected tactic."""
    return os.environ.get("RESIDUE_FOLD_SILENCE_DEFAULT") == "1"


def log_compilation() -> bool:
    """Report SM120 Fold compilations without changing cache behavior."""
    return os.environ.get("NVFP4_FOLD_COMPILE_LOG") == "1"
