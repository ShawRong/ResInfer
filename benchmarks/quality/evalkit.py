"""CLI for upstream lm-eval tasks and saved-result analysis.

See benchmarks/quality/offline/README.md for task protocols and inference settings.
"""
from __future__ import annotations

import argparse

from benchmarks.quality.common.results import cmd_analyze, cmd_analyze_ppl
from benchmarks.quality.common.runtime import add_runtime_arguments
from benchmarks.quality.offline.lm_eval_tasks import summarize


def cmd_task(a) -> int:
    from benchmarks.quality.offline.lm_eval_tasks import run
    return run(a)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    z = sub.add_parser("analyze", help="table + conditioning gate + paired tests")
    z.add_argument("--out", required=True)
    z.add_argument("--reference", default="bf16")
    z.add_argument("--baseline", default="plain")
    z.add_argument("--order", default="bf16,plain,prefill_off,uniform,b45,b80")
    z.add_argument("--compare", default="b45 vs prefill_off; b45 vs uniform")
    z.set_defaults(fn=cmd_analyze)

    k = sub.add_parser("task", help="official tasks: downstream, longbench_* and wikitext")
    k.add_argument("--arm", required=True)
    k.add_argument("--tasks", default="gsm8k,arc_challenge,hellaswag,piqa,winogrande")
    k.add_argument("--max-model-len", type=int, default=4096)
    k.add_argument("--max-gen", type=int, default=None)
    k.add_argument("--limit", type=int, default=None)
    k.add_argument("--enforce-eager", action="store_true")
    k.add_argument("--num-fewshot", type=int, default=None,
                   help="Unset uses each official task default; changing this creates a new protocol")
    k.add_argument("--task-path", help="Official lm-eval TaskManager include_path, e.g. staged offline task YAML")
    k.add_argument("--apply-chat-template", action="store_true",
                   help="Use lm-eval chat formatting; not valid for wikitext")
    k.add_argument("--truncation-side", choices=("left", "right", "middle"), default="left")
    k.add_argument("--prefill-only", type=int, metavar="FOLD_BAND")
    k.set_defaults(fn=cmd_task)

    summary = sub.add_parser("summarize", help="display native lm-eval aggregate metrics")
    summary.add_argument("--out", required=True)
    summary.set_defaults(fn=summarize)

    y = sub.add_parser("analyze-ppl", help="historical fixed-window token PPL only")
    y.add_argument("--out", required=True)
    y.add_argument("--corpus", default="wikitext2")
    y.add_argument("--reference", default="bf16")
    y.add_argument("--order", default="bf16,plain,prefill_off,uniform,b45,b50,b60,b80")
    y.add_argument("--compare", default="b45 vs plain; b45 vs uniform")
    y.set_defaults(fn=cmd_analyze_ppl)

    add_runtime_arguments(k)
    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
