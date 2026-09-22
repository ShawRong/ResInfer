"""Summarize immutable ResAttn kernel-grid cell JSON files as CSV and Markdown."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from resattn_grid import (DEFAULT_BATCHES, DEFAULT_CONTRACTS,
                          DEFAULT_GQAS, DEFAULT_SEQS)


def row_for(path: Path) -> dict:
    record = json.loads(path.read_text())
    result = record.get("result") or {}
    reference = (result.get("reference_diagnostic") or {}).get("all_heads") or {}
    medians = result.get("median_us") or {}
    aa = result.get("stock_aa_ratio")
    status = record["status"]
    if status == "ok" and aa is not None and abs(aa - 1) > .02:
        status = "timing_drift"
    return {"gqa": record["gqa"], "batch": record["batch"],
            "seq": record["seq"], "output_contract": record["output_contract"],
            "status": status, "plan": "/".join(map(str, result.get("plan", []))),
            "correctness_gate_kind": (result.get("correctness_gate") or {}).get(
                "kind"),
            "stock_us": medians.get("stock0"),
            "resattn_us": medians.get("resattn"),
            "stock_over_resattn": result.get("stock_over_resattn"),
            "stock_aa_ratio": aa,
            "stock_ref_min_cosine": reference.get("stock_min_cosine"),
            "resattn_ref_min_cosine": reference.get("resattn_min_cosine"),
            "cross_min_cosine": result.get("min_cross_cosine"),
            "peak_allocated_bytes": result.get("peak_allocated_bytes"),
            "elapsed_seconds": record.get("elapsed_seconds"),
            "probe_sha256": record.get("probe_sha256"),
            "file": str(path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--output-prefix", type=Path, required=True)
    ap.add_argument("--supplement-dir", type=Path,
                    action="append", help="replace invalid cells with valid diagnostics; repeat in order")
    args = ap.parse_args()
    primary = [row_for(path) for path in sorted(args.result_dir.glob("*.json"))]
    rows_by_key = {(row["gqa"], row["batch"], row["seq"],
                    row["output_contract"]): row for row in primary}
    if args.supplement_dir:
        for directory in args.supplement_dir:
            for path in sorted(directory.glob("*.json")):
                row = row_for(path)
                key = (row["gqa"], row["batch"], row["seq"],
                       row["output_contract"])
                if (key in rows_by_key and rows_by_key[key]["status"] != "ok"
                        and row["status"] in ("ok", "timing_drift")):
                    rows_by_key[key] = row
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_file = args.output_prefix.with_suffix(".csv")
    with csv_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [
            "gqa", "batch", "seq", "output_contract", "status"],
            lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    by_key = {(row["gqa"], row["batch"], row["seq"],
               row["output_contract"]): row for row in rows}
    lines = ["# ResAttn decode-only attention-chain grid", "",
             "Synthetic NVFP4 KV context is built before timing. CUDA Graph "
             "replay times only BF16 Q to the selected attention output; "
             "prefill, KV construction, projection/FFN layers and serving "
             "are excluded.", "",
             "Cell values are stock / ResAttn median latency. `D` marks stock "
             "A/A drift above 2%; `F` marks a failed or invalid cell; `·` "
             "marks an unrun cell. `†` marks a supplemental ResAttn-only "
             "development gate chosen after the initial scan. Ratio >1 "
             "favors ResAttn only in valid cells.",
             "", f"Completed cell files: {len(rows)}/120.", ""]
    for contract in DEFAULT_CONTRACTS:
        for gqa in DEFAULT_GQAS:
            lines += [f"## {contract}, GQA{gqa}", "",
                      "| Batch \\ KV | " + " | ".join(
                          f"{seq // 1024}K" for seq in DEFAULT_SEQS) + " |",
                      "| ---: | " + " | ".join("---:" for _ in DEFAULT_SEQS) + " |"]
            for batch in DEFAULT_BATCHES:
                cells = []
                for seq in DEFAULT_SEQS:
                    row = by_key.get((gqa, batch, seq, contract))
                    if row is None:
                        cells.append("·")
                    elif row["status"] == "timing_drift":
                        cells.append("D")
                    elif row["status"] != "ok":
                        cells.append("F")
                    else:
                        marker = ("†" if row["correctness_gate_kind"] ==
                                  "resattn_vs_dequantized_kv_reference" else "")
                        cells.append(f"{row['stock_over_resattn']:.3f}{marker}")
                lines.append(f"| {batch} | " + " | ".join(cells) + " |")
            lines.append("")
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
