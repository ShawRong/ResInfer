# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Load a benchmark spec and reconcile it with what the call site supplied.

A spec makes two runs comparable, but only if the run actually obeyed it. So
loading is not just parsing: `resolve_arms` refuses a run whose arms disagree
with the spec in EITHER direction. A missing arm silently shrinks the table and
an extra one silently widens it, and both produce something that looks like a
complete result.

Paths deliberately live outside the spec (see the header of
`suites/speed_dense.yaml`): they are site-specific, and putting them in the
file would destroy the portability the file exists to provide.
"""
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
    """The M values this manifest says are meaningful for one family on one arch.

    The two families do NOT share a range, and giving them one produced rows
    describing a path no deployment takes. A K-extended layer is fold-eligible
    in production, so below the arch's fold band it FOLDS -- it reads the
    k_base prefix and never touches its salient columns, which is why decode is
    ratio-blind. The K-extended GEMM only runs above the band. So `kext`
    declares a floor and `mext` declares a ceiling, and each is read here.

    An architecture the manifest has never sized is refused by the caller
    rather than handed another arch's numbers.
    """
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
    """Ratios a family sweeps. Only `kext` has any; mext IS ratio 1.0."""
    fam = (spec.get("kernel", {}).get("families", {}) or {}).get(family) or {}
    return [float(x) for x in (fam.get("ratios") or [])]


def resolve_arms(spec: dict, models_root: str | pathlib.Path) -> list[dict]:
    """Turn the manifest's relative arm paths into absolute ones, or refuse.

    Same convention as the quality suites in this directory: each arm's `path`
    is relative to `--models-root`, so the manifest itself holds no
    machine-specific path and is the spec you commit.

    Every arm must resolve to something that exists. A manifest whose arms are
    partly missing would otherwise produce a table that is short a column and
    looks complete -- so the whole set is checked up front, before any GPU time
    is spent, and every missing arm is reported at once rather than one per
    failed run.
    """
    root = pathlib.Path(models_root).expanduser().resolve()
    arms = spec.get("arms") or []
    if not arms:
        raise SystemExit(f"{spec.get('_path')}: manifest defines no arms")

    resolved, missing = [], []
    for arm in arms:
        if "id" not in arm or "path" not in arm:
            raise SystemExit(f"{spec.get('_path')}: arm needs `id` and `path`: {arm}")
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
