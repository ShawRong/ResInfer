"""CPU-only gates for the paper MExt sweep contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.mext_paper_aggregate import aggregate
from benchmarks.mext_paper_shapes import load_suite, projection_shapes, resolve
from benchmarks.mext_paper_sweep import SUITE


def test_qwen3_and_qwen35_real_projection_shapes():
    suite = load_suite(SUITE)
    four, _ = resolve(suite, {}, ["qwen3_4b"])
    physical = {(shape["n"], shape["k"]) for shape in four}
    assert (6144, 2560) in physical  # fused qkv
    assert (2560, 1024) in physical  # TP4 o_proj
    assert (4864, 2560) in physical  # TP4 gate_up
    assert (2560, 1216) in physical  # TP8 down
    text = suite["models"]["qwen35_27b"]["config"]
    assert projection_shapes(text, ["qkv", "o", "gate_up", "down"]) == {
        "qkv": (8192, 5120), "o": (5120, 6144),
        "gate_up": (34816, 5120), "down": (5120, 17408),
    }


def test_shared_physical_shape_is_timed_once():
    suite = load_suite(SUITE)
    suite["models"]["qwen3_4b_clone"] = suite["models"]["qwen3_4b"]
    shapes, _ = resolve(suite, {}, ["qwen3_4b", "qwen3_4b_clone"])
    shared = next(s for s in shapes if (s["n"], s["k"]) == (2560, 1024))
    assert len(shared["aliases"]) > 1
    assert len({(s["n"], s["k"]) for s in shapes}) == len(shapes)


def _sample_files(tmp_path: Path, shapes: list[dict], duplicate=False) -> list[Path]:
    files = []
    for rep in range(2):
        path = tmp_path / f"rep{rep}.jsonl"
        rows = []
        for shape in shapes:
            for arm, us in (("mext_r1", 8.0), ("fi_nvfp4_cutlass", 10.0),
                            ("fi_nvfp4_cutedsl", 7.0),
                            ("fi_mxfp4_mxfp8_w4a8", 20.0),
                            ("trtllm_nvfp4_fp8_w4a8", None)):
                rows.append({"schema": 1, "kind": "measure", "run_id": "run",
                             "rep": rep, "shape_id": shape["shape_id"],
                             "n": shape["n"], "k": shape["k"], "m": 8,
                             "aliases": shape["aliases"], "arm": arm,
                             "status": "passed" if us is not None else "skipped",
                             "us": us, "reason": "unavailable" if us is None else None})
        if duplicate and rep == 0:
            rows.append(rows[0])
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files.append(path)
    return files


def test_aggregate_macro_uses_unique_physical_shapes(tmp_path: Path):
    suite = load_suite(SUITE)
    suite["models"]["qwen3_4b_clone"] = suite["models"]["qwen3_4b"]
    shapes, _ = resolve(suite, {}, ["qwen3_4b", "qwen3_4b_clone"])
    # Keep two shapes, one of which has more than one logical alias.
    shapes = shapes[:2]
    shapes[0]["aliases"].append(dict(shapes[0]["aliases"][0], model="third_alias"))
    files = _sample_files(tmp_path, shapes)
    got = aggregate(files, shapes, [8], suite["arms"])
    assert got["unique_shapes"] == 2
    assert next(r for r in got["by_m"] if r["arm"] == "fi_nvfp4_cutlass")["n_unique_shapes"] == 2
    assert next(r for r in got["by_m"] if r["arm"] == "fi_nvfp4_cutlass")["macro_median_ratio"] == 0.8


def test_aggregate_rejects_duplicate_missing_and_required_failure(tmp_path: Path):
    suite = load_suite(SUITE)
    shapes, _ = resolve(suite, {}, ["qwen3_4b"])
    shapes = shapes[:1]
    files = _sample_files(tmp_path, shapes, duplicate=True)
    with pytest.raises(ValueError, match="duplicate cell"):
        aggregate(files, shapes, [8], suite["arms"])
    files = _sample_files(tmp_path, shapes)
    rows = files[1].read_text().splitlines()
    files[1].write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="missing"):
        aggregate(files, shapes, [8], suite["arms"])
    files = _sample_files(tmp_path, shapes)
    rows = [json.loads(r) for r in files[0].read_text().splitlines()]
    rows[0].update(status="failed", reason="broken", us=None)
    files[0].write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="required arm did not pass"):
        aggregate(files, shapes, [8], suite["arms"])
