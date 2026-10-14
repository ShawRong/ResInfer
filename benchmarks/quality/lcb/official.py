"""Pinned LiveCodeBench scoring with audited IO and task-specific repairs."""

from __future__ import annotations

from functools import partial
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
import signal
import sys

from benchmarks.quality.common.results import input_hash

COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
PROTOCOL = "lcb-release_v6-audited-20261013"
CHECKS = json.loads(Path(__file__).with_name("checks.json").read_text())


class _CheckerFailure(BaseException):
    """Bypass upstream's candidate-error catch for a failing judge."""


def load_official(source):
    """Import a verified external checkout; never vendor or patch it at runtime."""
    source = Path(source).resolve()
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "environments/livecodebench/source.json"
        ).read_text()
    )
    for name, expected in manifest["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Unreviewed LCB source: {name}; see environments/livecodebench"
            )
    sys.path.insert(0, str(source))
    module = importlib.import_module("lcb_runner.evaluation.testing_util")
    if (
        Path(module.__file__).resolve()
        != source / "lcb_runner/evaluation/testing_util.py"
    ):
        raise ValueError("Another LCB checkout is already imported")
    return module


def question_key(row):
    return f"{row['platform']}:{row['question_id']}"


def checked_key(row, release, sample=None):
    key = question_key(row)
    if key in CHECKS["questions"]:
        identity = {
            k: row[k]
            for k in (
                "question_title",
                "question_content",
                "platform",
                "question_id",
                "public_test_cases",
            )
        }
        if (
            release != CHECKS["release"]
            or input_hash(identity) != CHECKS["questions"][key]["problem_sha256"]
            or (
                sample is not None
                and input_hash(sample) != CHECKS["questions"][key]["sample_sha256"]
            )
        ):
            raise ValueError(f"Audited checker identity changed: {key}")
        return key
    return None


def _compare(key, inputs, prediction, reference):
    from benchmarks.quality.lcb.checks import check

    # A slow/broken checker must not be counted as a candidate's time limit.
    signal.alarm(60)
    try:
        return check(key, inputs, prediction, reference)
    except BaseException as exc:
        raise _CheckerFailure(f"{type(exc).__name__}: {exc}") from exc
    finally:
        signal.alarm(0)


def _execute(source, sample, code, key, timeout, connection):
    try:
        executor = load_official(source)
        from benchmarks.quality.lcb.io import install

        install(executor)
        executor.OUTPUT_COMPARATOR = partial(_compare, key) if key else None
        tests, metadata = executor.run_test(sample, test=code, timeout=timeout)
        connection.send({"tests": [int(x) for x in tests], "metadata": metadata})
    except BaseException as exc:
        connection.send({"infrastructure_error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def score_code(source, sample, code, key=None, timeout=6):
    """Official candidate execution in a bounded child; parent faults stay errors."""
    data = json.loads(sample["input_output"])
    count = len(data["inputs"])
    if count == 0 or count != len(data["outputs"]):
        raise ValueError("Invalid or empty LCB test set")
    if not code:
        return {
            "score": 0.0,
            "tests": [],
            "expected_tests": count,
            "metadata": {"error_message": "empty final code"},
        }
    ctx = multiprocessing.get_context("spawn")
    parent, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_execute, args=(str(source), sample, code, key, timeout, sender)
    )
    process.start()
    sender.close()
    try:
        if not parent.poll(min(600, (timeout + 1) * count + 65)):
            raise RuntimeError("LCB scorer exceeded parent deadline")
        try:
            result = parent.recv()
        except EOFError as exc:
            raise RuntimeError("LCB scorer exited without results") from exc
    finally:
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
    if "infrastructure_error" in result:
        raise RuntimeError(result["infrastructure_error"])
    tests = result["tests"]
    if (
        not tests
        or len(tests) > count
        or (all(x > 0 for x in tests) and len(tests) != count)
    ):
        raise RuntimeError("LCB scorer returned an incomplete passing test set")
    result.update(
        score=float(len(tests) == count and all(x > 0 for x in tests)),
        expected_tests=count,
    )
    return result
