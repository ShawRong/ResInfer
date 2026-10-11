# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Implementation details.'
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Summary field, ratio prefix, numerator, comparison arms (None means all).
_COMPARISONS = (
    ("by_m", "mext", "mext_r1", None),
    ("mxfp4_by_m", "mxfp4_mext", "resinfer_mxfp4_mext_r1",
     ("fi_mxfp4_mxfp4_stock", "fi_mxfp4_mxfp4_cutedsl_quant", "fi_mxfp4_mxfp8_w4a8")),
    ("mxfp4_tuned_by_m", "mxfp4_tuned", "resinfer_mxfp4_mext_r1_tuned",
     ("resinfer_mxfp4_mext_r1", "fi_mxfp4_mxfp4_stock", "fi_mxfp4_mxfp4_cutedsl_quant")),
)


def _rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        if row.get("schema") != 1 or row.get("kind") != "measure":
            raise ValueError(f"{path}:{number}: unexpected row schema/kind")
        rows.append(row)
    return rows


def aggregate(files: list[Path], shapes: list[dict], m_values: list[int],
              arms: list[str], required_arms: list[str] | None = None) -> dict:
    """Check sample completeness and summarize; the sweep validates run identity.

    All requested arms are required unless required_arms is supplied. An optional
    arm can be skipped in every repetition; partial samples cannot form a median.
    """
    if not files or not shapes or not m_values or not arms:
        raise ValueError("empty measurement matrix")
    required = set(arms if required_arms is None else required_arms)
    if required - set(arms):
        raise ValueError("required arms missing from suite")
    expected = {(rep, shape["shape_id"], m, arm)
                for rep in range(len(files)) for shape in shapes
                for m in m_values for arm in arms}
    observed = {}
    for rep, path in enumerate(files):
        for row in _rows(path):
            key = (row.get("rep"), row.get("shape_id"), row.get("m"), row.get("arm"))
            if key[0] != rep or key not in expected:
                raise ValueError(f"unexpected cell in {path}: {key}")
            if key in observed:
                raise ValueError(f"duplicate cell: {key}")
            status = row.get("status")
            if key[3] in required and status != "passed":
                raise ValueError(f"required arm did not pass: {key}")
            if status == "passed":
                us = row.get("us")
                if type(us) not in (int, float) or not math.isfinite(us) or us <= 0:
                    raise ValueError(f"invalid latency: {key}")
            elif status != "skipped" or not row.get("reason"):
                raise ValueError(f"failed or invalid measurement: {key}")
            observed[key] = row
    if expected - observed.keys():
        raise ValueError(f"missing {len(expected - observed.keys())} cells")

    points = []
    macro = {field: defaultdict(list) for field, *_ in _COMPARISONS}
    for shape in shapes:
        for m in m_values:
            latency, process_range, ratios = {}, {}, {}
            for arm in arms:
                rows = [observed[(rep, shape["shape_id"], m, arm)] for rep in range(len(files))]
                samples = [row["us"] for row in rows if row["status"] == "passed"]
                if not samples:
                    continue
                if len(samples) != len(files):
                    raise ValueError(f"incomplete repetitions: {shape['shape_id']} M={m} {arm}")
                latency[arm] = statistics.median(samples)
                process_range[arm] = [min(samples), max(samples)]
            for field, prefix, numerator, comparisons in _COMPARISONS:
                for arm in (arms if comparisons is None else comparisons):
                    if numerator in latency and arm in latency and arm != numerator:
                        ratio = latency[numerator] / latency[arm]
                        ratios[f"{prefix}_over_{arm}"] = ratio
                        macro[field][(arm, m)].append(ratio)
            points.append({**shape, "m": m, "latency_us": latency,
                           "process_range_us": process_range, "ratios": ratios})
    summary = {"schema": 1, "status": "complete",
               "run_id": next(iter(observed.values())).get("run_id"),
               "process_reps": len(files), "unique_shapes": len(shapes), "points": points}
    for field, groups in macro.items():
        summary[field] = [
            {"arm": arm, "m": m, "n_unique_shapes": len(values),
             "macro_median_ratio": statistics.median(values)}
            for (arm, m), values in sorted(groups.items())]
    return summary



def write_outputs(summary: dict, directory: Path) -> None:
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ratio_columns = [f"{prefix}_over_arm" for _, prefix, *_ in _COMPARISONS]
    with (directory / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "shape_id", "n", "k", "m", "models_projections_tp", "arm",
            "latency_us", "process_min_us", "process_max_us", *ratio_columns])
        writer.writeheader()
        for point in summary["points"]:
            base = {key: point[key] for key in ("shape_id", "n", "k", "m")}
            base["models_projections_tp"] = ";".join(
                f"{a['model']}/{a['projection']}/tp{a['tp']}" for a in point["aliases"])
            for arm, us in point["latency_us"].items():
                low, high = point["process_range_us"][arm]
                writer.writerow({**base, "arm": arm, "latency_us": us,
                                 "process_min_us": low, "process_max_us": high,
                                 **{f"{prefix}_over_arm": point["ratios"].get(f"{prefix}_over_{arm}")
                                    for _, prefix, *_ in _COMPARISONS}})
