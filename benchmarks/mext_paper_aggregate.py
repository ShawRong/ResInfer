# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Strict, CPU-only aggregation of per-process MExt paper JSONL files."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

REQUIRED_ARMS = (
    "mext_r1", "fi_nvfp4_cutlass", "fi_nvfp4_cutedsl",
    "fi_mxfp4_mxfp8_w4a8",
)
OPTIONAL_ARMS = ("trtllm_nvfp4_fp8_w4a8",)
SCHEMA = 1


def _rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        if row.get("schema") != SCHEMA or row.get("kind") != "measure":
            raise ValueError(f"{path}:{number}: unexpected row schema/kind")
        rows.append(row)
    return rows


def aggregate(files: list[Path], shapes: list[dict], m_values: list[int],
              arms: list[str], required_arms: list[str] | None = None) -> dict:
    required = set(REQUIRED_ARMS if required_arms is None else required_arms)
    if required - set(arms):
        raise ValueError(f"required arms missing from suite: {sorted(required - set(arms))}")
    expected_shape = {shape["shape_id"]: shape for shape in shapes}
    if len(expected_shape) != len(shapes):
        raise ValueError("duplicate physical shapes in frozen manifest")
    expected = {(rep, sid, m, arm)
                for rep in range(len(files)) for sid in expected_shape
                for m in m_values for arm in arms}
    observed: dict[tuple, dict] = {}
    run_ids = set()
    for rep, path in enumerate(files):
        if not path.exists():
            raise ValueError(f"missing rep file: {path}")
        for row in _rows(path):
            key = (row.get("rep"), row.get("shape_id"), row.get("m"), row.get("arm"))
            if row.get("rep") != rep or key not in expected:
                raise ValueError(f"unexpected cell in {path}: {key}")
            if key in observed:
                raise ValueError(f"duplicate cell: {key}")
            if row.get("n") != expected_shape[key[1]]["n"] or row.get("k") != expected_shape[key[1]]["k"]:
                raise ValueError(f"physical shape mismatch: {key}")
            if row.get("aliases") != expected_shape[key[1]]["aliases"]:
                raise ValueError(f"alias mismatch: {key}")
            if row.get("status") not in {"passed", "skipped", "failed"}:
                raise ValueError(f"missing status: {key}")
            if row["status"] == "passed" and not (isinstance(row.get("us"), (float, int)) and row["us"] > 0):
                raise ValueError(f"invalid latency: {key}")
            if row["status"] == "skipped" and not row.get("reason"):
                raise ValueError(f"missing skip reason: {key}")
            run_ids.add(row.get("run_id"))
            observed[key] = row
    if len(run_ids) != 1 or None in run_ids:
        raise ValueError(f"mixed run ids: {run_ids}")
    missing = expected - set(observed)
    if missing:
        raise ValueError(f"missing {len(missing)} cells; first: {sorted(missing)[:3]}")
    for key, row in observed.items():
        if key[3] in required and row["status"] != "passed":
            raise ValueError(f"required arm did not pass: {key}: {row.get('reason')}")
        if key[3] not in required and row["status"] == "failed":
            raise ValueError(f"optional arm failed after engagement: {key}: {row.get('reason')}")
    for sid in expected_shape:
        for m in m_values:
            for arm in arms:
                if arm not in required:
                    states = {observed[(rep, sid, m, arm)]["status"]
                              for rep in range(len(files))}
                    if len(states) != 1:
                        raise ValueError(f"optional arm changed availability: {sid} M={m} {arm}: {states}")

    points = []
    macro: dict[tuple[str, int], list[float]] = defaultdict(list)
    mx_macro: dict[tuple[str, int], list[float]] = defaultdict(list)
    for sid, shape in expected_shape.items():
        for m in m_values:
            latency = {arm: statistics.median(
                observed[(rep, sid, m, arm)]["us"]
                for rep in range(len(files)))
                for arm in arms if all(
                    observed[(rep, sid, m, arm)]["status"] == "passed"
                    for rep in range(len(files)))}
            process_range = {arm: [min(observed[(rep, sid, m, arm)]["us"]
                                       for rep in range(len(files))),
                                   max(observed[(rep, sid, m, arm)]["us"]
                                       for rep in range(len(files)))]
                             for arm in latency}
            ratios = {}
            for arm in arms:
                if arm == "mext_r1" or arm not in latency or "mext_r1" not in latency:
                    continue
                ratio = latency["mext_r1"] / latency[arm]
                ratios[f"mext_over_{arm}"] = ratio
                macro[(arm, m)].append(ratio)
            mx_arm = "resinfer_mxfp4_mext_r1"
            for arm in ("fi_mxfp4_mxfp4_stock",
                        "fi_mxfp4_mxfp4_cutedsl_quant",
                        "fi_mxfp4_mxfp8_w4a8"):
                if mx_arm in latency and arm in latency:
                    ratio = latency[mx_arm] / latency[arm]
                    ratios[f"mxfp4_mext_over_{arm}"] = ratio
                    mx_macro[(arm, m)].append(ratio)
            points.append({"shape_id": sid, "n": shape["n"], "k": shape["k"],
                           "m": m, "aliases": shape["aliases"],
                           "latency_us": latency, "process_range_us": process_range,
                           "ratios": ratios})
    by_m = [{"arm": arm, "m": m, "n_unique_shapes": len(values),
             "macro_median_ratio": statistics.median(values)}
            for (arm, m), values in sorted(macro.items())]
    mx_by_m = [{"arm": arm, "m": m, "n_unique_shapes": len(values),
                "macro_median_ratio": statistics.median(values)}
               for (arm, m), values in sorted(mx_macro.items())]
    return {"schema": SCHEMA, "status": "complete", "run_id": next(iter(run_ids)),
            "process_reps": len(files), "unique_shapes": len(shapes),
            "points": points, "by_m": by_m, "mxfp4_by_m": mx_by_m}


def write_outputs(summary: dict, directory: Path) -> None:
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (directory / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "shape_id", "n", "k", "m", "models_projections_tp", "arm",
            "latency_us", "process_min_us", "process_max_us", "mext_over_arm",
            "mxfp4_mext_over_arm"))
        writer.writeheader()
        for point in summary["points"]:
            aliases = ";".join(f"{a['model']}/{a['projection']}/tp{a['tp']}"
                               for a in point["aliases"])
            for arm, us in point["latency_us"].items():
                writer.writerow({"shape_id": point["shape_id"], "n": point["n"],
                                 "k": point["k"], "m": point["m"],
                                 "models_projections_tp": aliases, "arm": arm,
                                 "latency_us": us,
                                 "process_min_us": point["process_range_us"][arm][0],
                                 "process_max_us": point["process_range_us"][arm][1],
                                 "mext_over_arm": point["ratios"].get(f"mext_over_{arm}"),
                                 "mxfp4_mext_over_arm": point["ratios"].get(
                                     f"mxfp4_mext_over_{arm}")})
