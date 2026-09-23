"""Summarize ABBA real-model ResAttn E2E grid results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def texts(summary: dict) -> list[str]:
    result = json.loads(Path(summary["result_file"]).read_text())
    return result["generated_texts"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    groups = defaultdict(dict)
    for path in args.result_dir.glob("round*-*/*/summary.json"):
        summary = json.loads(path.read_text())
        evidence_file = path.parents[1] / "server-evidence.json"
        summary["server_evidence"] = (
            json.loads(evidence_file.read_text()) if evidence_file.exists() else {})
        groups[(summary["input_len"], summary["concurrency"])][
            summary["round"]] = summary
    lines = ["# ResAttn real-model BF16-output E2E grid", "",
             "Qwen3-8B on NVFP4 KV, CUDA Graph; each cell uses four ABBA "
             "rounds. Input length and request concurrency are workload "
             "settings, not measured decode batch. Ratios >1 favor ResAttn.",
             "", "| Input | Concurrency | Rounds | Stock TPOT ms | ResAttn TPOT ms | "
             "Stock / ResAttn | Stock A/A | ResAttn A/A | Timing | Exact text agreement |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |"]
    for (length, concurrency), rounds in sorted(groups.items()):
        if sorted(rounds) != [1, 2, 3, 4] or any(
                item["status"] != "ok" for item in rounds.values()):
            lines.append(f"| {length} | {concurrency} | {len(rounds)}/4 | · | · | · | · | · | · | · |")
            continue
        if (not all(rounds[n]["server_evidence"].get(
                "stock_trtllm_gen_selected") and rounds[n]["server_evidence"].get(
                "cuda_graph_mentioned") for n in (1, 4)) or
                not all(rounds[n]["server_evidence"].get(
                    "custom_backend_registered") and rounds[n]["server_evidence"].get(
                    "cuda_graph_mentioned") for n in (2, 3))):
            lines.append(f"| {length} | {concurrency} | 4/4 | E | E | E | E | E | E | E |")
            continue
        stock = [rounds[1]["mean_tpot_ms"], rounds[4]["mean_tpot_ms"]]
        res = [rounds[2]["mean_tpot_ms"], rounds[3]["mean_tpot_ms"]]
        stock_aa = stock[0] / stock[1]
        res_aa = res[0] / res[1]
        timing = ("drift" if max(abs(stock_aa - 1), abs(res_aa - 1)) > .05
                  else "paired")
        agreement = []
        for stock_round, res_round in ((1, 2), (4, 3)):
            a, b = texts(rounds[stock_round]), texts(rounds[res_round])
            if len(a) != len(b):
                agreement.append("length mismatch")
            else:
                agreement.append(f"{sum(x == y for x, y in zip(a, b))}/{len(a)}")
        lines.append(
            f"| {length} | {concurrency} | 4/4 | "
            f"{statistics.mean(stock):.3f} | {statistics.mean(res):.3f} | "
            f"{statistics.mean(stock)/statistics.mean(res):.3f} | "
            f"{stock_aa:.3f} | {res_aa:.3f} | {timing} | "
            f"{', '.join(agreement)} |")
    lines += ["", "Exact text agreement compares seeded greedy requests "
              "between arms; it is not a task accuracy score. Engine decode "
              "batch and per-step KV lengths require a separate scheduler "
              "trace before mapping these E2E cells to kernel-grid cells. `E` "
              "marks missing backend or CUDA Graph evidence; that row has no "
              "performance conclusion. `drift` means at least one same-arm "
              "TPOT pair differs by more than 5%; its cross-arm ratio is "
              "diagnostic only.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
