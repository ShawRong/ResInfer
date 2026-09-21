"""Lighteval tasks backed by explicitly staged local datasets."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.aime import aime24_avg, aime25_avg
from lighteval.tasks.tasks.gpqa import gpqa_diamond_instruct

from benchmarks.gpqa_deterministic import ordered_answers


def _data_root() -> Path:
    value = os.environ.get("RESINFER_EVAL_DATA_ROOT")
    if not value:
        raise RuntimeError(
            "RESINFER_EVAL_DATA_ROOT must point to the staged AIME/GPQA bundle")
    root = Path(value).expanduser().resolve()
    missing = [name for name in ("aime24", "aime25", "gpqa")
               if not (root / name).is_dir()]
    if missing:
        raise RuntimeError(
            f"offline dataset root {root} is missing: {', '.join(missing)}")
    return root


def gpqa_instruct_prompt_deterministic(line, task_name=None):
    question = line["Question"].strip()
    answers, correct = ordered_answers(line)
    instruction = (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'Answer: $LETTER' "
        "(without quotes) where LETTER is one of ABCD. Think step by step "
        "before answering."
    )
    query = (
        "{instruction}\n\n{question}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
    ).format(
        instruction=instruction,
        question=question,
        a=answers[0].strip(),
        b=answers[1].strip(),
        c=answers[2].strip(),
        d=answers[3].strip(),
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase[:4]),
        gold_index=correct,
        instruction=instruction,
    )


DATA_ROOT = _data_root()

aime24_avg_local = replace(
    aime24_avg, name="aime24_avg_local", hf_repo=str(DATA_ROOT / "aime24"))
aime25_avg_local = replace(
    aime25_avg, name="aime25_avg_local", hf_repo=str(DATA_ROOT / "aime25"))
aime24_avg8_local = replace(
    aime24_avg_local,
    name="aime24_avg8_local",
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 8})],
)
aime25_avg8_local = replace(
    aime25_avg_local,
    name="aime25_avg8_local",
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 8})],
)
gpqa_diamond_local = replace(
    gpqa_diamond_instruct,
    name="gpqa_diamond_local",
    hf_repo=str(DATA_ROOT / "gpqa"),
    prompt_function=gpqa_instruct_prompt_deterministic,
)

TASKS_TABLE = [
    aime24_avg_local,
    aime25_avg_local,
    aime24_avg8_local,
    aime25_avg8_local,
    gpqa_diamond_local,
]
