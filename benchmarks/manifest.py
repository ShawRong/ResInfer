# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Load benchmark workloads and resolve architecture grids and checkpoint paths."""
from __future__ import annotations

import pathlib
from typing import Any

DEFAULT_SPEC = (pathlib.Path(__file__).with_name("suites")
                / "speed_dense.yaml")


def load(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Parse a spec. `None` loads the one shipped beside this module."""
    import yaml                       # a vLLM dependency; nothing new is added

    p = pathlib.Path(path) if path else DEFAULT_SPEC
    if not p.exists():
        raise SystemExit(f"spec not found: {p}")
    with open(p) as fh:
        spec = yaml.safe_load(fh)
    if "name" not in spec:
        raise SystemExit(f"{p}: manifest has no `name`")
    spec["_path"] = str(p.resolve())
    return spec


def stamp(spec: dict) -> dict:
    """The fields every result row carries so its spec is checkable later.

    Recorded rather than assumed: two files can then be compared without
    anyone having to remember which command produced them.
    """
    return {
        "manifest": spec.get("name"),
        "manifest_path": spec.get("_path"),
    }


def m_grid_for(spec: dict, family: str, sm: str) -> list[int]:
    """Filter each family's grid by its architecture-specific dispatch range."""
    fam = (spec.get("kernel", {}).get("families", {}) or {}).get(family)
    if not isinstance(fam, dict):
        raise SystemExit(
            f"{spec.get('_path')}: kernel.families.{family} must be a mapping "
            f"with its own m_grid; a shared grid cannot express that the two "
            f"families run over different M.")
    grid = fam.get("m_grid") or []
    lo = (fam.get("m_min_by_arch") or {}).get(sm)
    hi = (fam.get("m_max_by_arch") or {}).get(sm)
    if lo is None and hi is None:
        raise SystemExit(
            f"{spec.get('_path')}: family {family!r} sets neither "
            f"m_min_by_arch nor m_max_by_arch for {sm}. Add it -- an unsized "
            f"architecture must not silently inherit another one's range.")
    return [m for m in grid
            if (lo is None or m >= lo) and (hi is None or m <= hi)]


def ratios_for(spec: dict, family: str) -> list[float]:
    """Ratios a family sweeps. `kext` includes ratio-1 prefill; `mext` covers the small-M grid."""
    fam = (spec.get("kernel", {}).get("families", {}) or {}).get(family) or {}
    return [float(x) for x in (fam.get("ratios") or [])]


def resolve_arms(spec: dict, models_root: str | pathlib.Path) -> list[dict]:
    """Resolve every declared arm before starting GPU work."""
    root = pathlib.Path(models_root).expanduser().resolve()
    arms = spec.get("arms") or []
    if not arms:
        raise SystemExit(f"{spec.get('_path')}: manifest defines no arms")

    resolved, missing = [], []
    for arm in arms:
        if not {"id", "path", "kind"} <= arm.keys():
            raise SystemExit(f"{spec.get('_path')}: arm needs `id`, `path` and `kind`: {arm}")
        full = root / arm["path"]
        if not full.exists():
            missing.append(f"    {arm['id']:<8} -> {full}")
            continue
        resolved.append(dict(arm, abs_path=str(full)))

    if missing:
        raise SystemExit(
            f"manifest {spec['name']}: {len(missing)} arm path(s) not found "
            f"under --models-root {root}:\n" + "\n".join(missing) +
            "\n  Every arm is checked before any GPU time is spent, because a "
            "table\n  missing a column still looks like a complete table.")
    return resolved
