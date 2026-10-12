"""Deterministic answer placement for paired GPQA comparisons."""

from __future__ import annotations

import hashlib


def gold_index(question: str) -> int:
    """Return a stable answer slot derived only from the question text."""
    normalized = question.strip().encode("utf-8")
    return hashlib.sha256(normalized).digest()[0] % 4


def ordered_answers(row: dict[str, str]) -> tuple[list[str], int]:
    """Place the correct answer in a deterministic slot among four answers."""
    index = gold_index(row["Question"])
    answers = [
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    answers.insert(index, row["Correct Answer"])
    return answers, index
