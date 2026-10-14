"""Shared result identities, completeness checks and cohort pairing."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys


def input_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def claim_arm_output(a):
    """Each arm is a new experiment; never resume or overwrite its files."""
    directory = pathlib.Path(a.out)
    directory.mkdir(parents=True, exist_ok=True)
    if not a.arm or pathlib.Path(a.arm).name != a.arm or "__" in a.arm:
        raise ValueError("arm must be a filename component without '__'")
    if any(p.name.startswith(a.arm + "__") for p in directory.iterdir()):
        raise ValueError(f"Results already exist for arm {a.arm!r}; use a new output directory/arm. "
                         "--force does not overwrite historical results.")
    # Exclusive creation also prevents two processes from claiming the same arm.
    with (directory / f"{a.arm}__run.json").open("x") as stream:
        json.dump({"model": a.model, "argv": sys.argv[1:]}, stream)
    return directory


def target_logprob_sum(ids, output, start=1):
    """Require every target token in the requested span; no missing-as-zero."""
    if list(output.prompt_token_ids) != list(ids):
        raise ValueError("Returned prompt tokens do not match the scoring input")
    lps = output.prompt_logprobs
    if lps is None or len(lps) != len(ids) or not 1 <= start < len(ids):
        raise ValueError("Incomplete logprob coverage or zero scored tokens")
    values = []
    for i in range(start, len(ids)):
        entry = lps[i]
        value = entry.get(ids[i]) if entry else None
        if value is None:
            raise ValueError(f"Missing target logprob at token {i}")
        value = float(value.logprob if hasattr(value, "logprob") else value)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite target logprob at token {i}")
        values.append(value)
    total = math.fsum(values)
    if not math.isfinite(total):
        raise ValueError("Nonfinite summed logprob")
    return total, len(values)


def require_output_count(inputs, outputs):
    if len(inputs) != len(outputs):
        raise ValueError(f"Incomplete generation: {len(outputs)} outputs for {len(inputs)} inputs")


def pair_quality_rows(left, right):
    """Align complete cohorts by identity; never infer pairs from row order."""
    def index(rows):
        result = {}
        for row in rows:
            key = row.get("item_id")
            if not isinstance(key, str) or not key or key in result:
                raise ValueError("Quality pairing requires unique, nonempty item_id values; "
                                 "legacy results without identities cannot be paired")
            score = row.get("score")
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("Quality pairing requires finite scores")
            result[key] = row
        return result

    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError("Quality pairing requires identical nonempty item cohorts")
    aligned = [b[key] for key in a]
    for x, y in zip(a.values(), aligned):
        if "prompt_token_ids" in x or "prompt_token_ids" in y:
            if (not x.get("prompt_token_ids") or
                    x["prompt_token_ids"] != y.get("prompt_token_ids")):
                raise ValueError("Quality pairing requires identical rendered prompt tokens")
    return list(a.values()), aligned
