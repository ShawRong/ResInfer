#!/usr/bin/env python3
'Implementation details.'
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PUBLIC_RELEASE_EXCLUDES.json"


def tracked_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    )
    return output.splitlines()


def findings():
    manifest = json.loads(MANIFEST.read_text())
    tracked = tracked_files()
    result = []
    for pattern in manifest["remove_globs"]:
        matches = sorted(path for path in tracked if fnmatch.fnmatch(path, pattern))
        if matches:
            result.append({"kind": "remove", "pattern": pattern, "matches": matches})
    for entry in manifest["clean_references"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        matches = [token for token in entry["tokens"] if token in text]
        if matches:
            result.append({
                "kind": "clean_reference",
                "path": entry["path"],
                "tokens": matches,
            })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inventory", "check"))
    args = parser.parse_args()
    result = findings()
    print(json.dumps(result, indent=2))
    if args.mode == "check" and result:
        raise SystemExit(
            f"public release blocked by {len(result)} internal MoE findings"
        )


if __name__ == "__main__":
    main()
