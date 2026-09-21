#!/usr/bin/env python3
"""Install the pinned offline registry assets into a lighteval environment."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import subprocess
import tarfile
import tempfile


TINYBENCHMARKS_SHA256 = (
    "c3b6e426dfe7b100fe6d0ee960398e10a8763254bcead3be80cc6bc15abca284")
NLTK_ARCHIVE_SHA256 = (
    "cf0a576b771afbe73bdd89eda94ea44aeeac474fe0e8d42945895cc449db4dc8")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: pathlib.Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def _lighteval_task_dir(python: pathlib.Path) -> pathlib.Path:
    code = (
        "import importlib.util, pathlib; "
        "s=importlib.util.find_spec('lighteval'); "
        "assert s and s.origin; "
        "print(pathlib.Path(s.origin).resolve().parent/'tasks'/'tasks')"
    )
    result = subprocess.run(
        [str(python), "-c", code], check=True, capture_output=True, text=True)
    return pathlib.Path(result.stdout.strip())


def _safe_extract(archive: pathlib.Path, destination: pathlib.Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    f"unsupported archive member type: {member.name}")
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        bundle.extractall(root)


def _install_nltk(archive: pathlib.Path, destination: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="resinfer-nltk-") as tmp:
        extracted = pathlib.Path(tmp)
        _safe_extract(archive, extracted)
        source = extracted / "nltk_data"
        if not source.is_dir():
            source = extracted
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            target = destination / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=pathlib.Path, required=True)
    parser.add_argument("--tinybenchmarks", type=pathlib.Path, required=True)
    parser.add_argument("--nltk-archive", type=pathlib.Path, required=True)
    parser.add_argument("--nltk-data", type=pathlib.Path, required=True)
    args = parser.parse_args()

    _require_hash(args.tinybenchmarks, TINYBENCHMARKS_SHA256)
    _require_hash(args.nltk_archive, NLTK_ARCHIVE_SHA256)
    task_dir = _lighteval_task_dir(args.python)
    task_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.tinybenchmarks, task_dir / "tinyBenchmarks.pkl")
    _install_nltk(args.nltk_archive, args.nltk_data)
    print(f"installed tinyBenchmarks.pkl in {task_dir}")
    print(f"installed NLTK data in {args.nltk_data.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
