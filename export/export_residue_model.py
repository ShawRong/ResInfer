#!/usr/bin/env python3
"""Export a residue NVFP4 checkpoint.

This is the public CLI shim for the current AWQ-clean -> residue auto-search ->
metadata injection -> HF export flow.  The implementation lives in
export.residue_export_flow so tests, suite wrappers, and users exercise the
same export path.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export.residue_export_flow import main


if __name__ == "__main__":
    raise SystemExit(main())
