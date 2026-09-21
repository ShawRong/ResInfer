"""CPU-only gates for the paper MExt sweep contract."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import types

import pytest

from benchmarks.mext_paper_aggregate import aggregate
from benchmarks.mext_paper_shapes import (load_suite, parse_tp_overrides,
                                          projection_shapes, resolve)
from benchmarks.mext_paper_sweep import SUITE


def test_qwen3_and_qwen35_real_projection_shapes():
    suite = load_suite(SUITE)
    four, _ = resolve(suite, {}, ["qwen3_4b"])
    physical = {(shape["n"], shape["k"]) for shape in four}
    assert (6144, 2560) in physical  # fused qkv
    assert (2560, 4096) in physical  # TP1 o_proj
    assert (19456, 2560) in physical  # TP1 gate_up
    assert (2560, 9728) in physical  # TP1 down
    assert all(alias["tp"] == 1 for shape in four for alias in shape["aliases"])
    text = suite["models"]["qwen35_27b"]["config"]
    assert projection_shapes(text, ["qkv", "o", "gate_up", "down"]) == {
        "qkv": (8192, 5120), "o": (5120, 6144),
        "gate_up": (34816, 5120), "down": (5120, 17408),
    }


def test_profiled_static_fallback_is_distinct_from_missing_tune(monkeypatch):
    """The profiled fallback runner is a real winner, not a cache miss."""
    path = Path(__file__).resolve().parents[1] / "quant/vllm_integration/fi_tuner_base.py"
    spec = importlib.util.spec_from_file_location("fi_tuner_base_contract", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__path__ = []
        autotuner = types.ModuleType("flashinfer.autotuner")
        patch.setitem(sys.modules, "flashinfer", flashinfer)
        patch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
        spec.loader.exec_module(module)

        fallback = types.SimpleNamespace(forward=lambda *a, **kw: "fallback")
        candidate = types.SimpleNamespace(forward=lambda *a, **kw: "candidate")
        class FakeTuner:
            is_tuning_mode = False
            selected = fallback
            tactic = 0

            @classmethod
            def get(cls):
                return cls

            @classmethod
            def choose_one(cls, *args):
                return cls.selected, cls.tactic

        autotuner.AutoTuner = FakeTuner
        sources = []
        def call():
            return module.tuned_call(
                "test", runners_getter=lambda: [fallback, candidate],
                config_getter=lambda: object(), inputs=[], fallback_forward=lambda x: None,
                kill_env_value="1", announce_flag=[True], record=sources.append)

        assert call() == "fallback"
        assert sources.pop() == "autotuned_fallback"
        FakeTuner.selected = candidate
        assert call() == "candidate"
        assert sources.pop() == "autotuned"
        FakeTuner.tactic = -1
        assert call() == "candidate"
        assert sources.pop() == "offline_table"


def test_shared_physical_shape_is_timed_once():
    suite = load_suite(SUITE)
    suite["models"]["qwen3_4b_clone"] = suite["models"]["qwen3_4b"]
    shapes, _ = resolve(suite, {}, ["qwen3_4b", "qwen3_4b_clone"])
    shared = next(s for s in shapes if (s["n"], s["k"]) == (2560, 4096))
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
                            ("resinfer_mxfp4_mext_r1", 12.0),
                            ("fi_mxfp4_mxfp4_stock", 11.0),
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
    assert next(r for r in got["mxfp4_by_m"]
                if r["arm"] == "fi_mxfp4_mxfp4_stock")["macro_median_ratio"] == 12 / 11


def test_mxfp4_only_diagnostic_has_own_ratios(tmp_path: Path):
    suite = load_suite(SUITE)
    shapes, _ = resolve(suite, {}, ["qwen3_4b"])
    arms = ["resinfer_mxfp4_mext_r1", "fi_mxfp4_mxfp4_stock"]
    files = _sample_files(tmp_path, shapes[:1])
    for path in files:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows
                                if row["arm"] in arms))
    got = aggregate(files, shapes[:1], [8], arms, arms)
    assert got["by_m"] == []
    assert got["mxfp4_by_m"][0]["macro_median_ratio"] == 12 / 11


def test_cutedsl_quantizer_companion_has_separate_mxfp4_ratio(tmp_path: Path):
    path = SUITE.with_name("mext_paper_sm100_v4_mxfp4_producer.json")
    suite = load_suite(path)
    alternate = "fi_mxfp4_mxfp4_cutedsl_quant"
    assert alternate in suite["arms"] and alternate in suite["required_arms"]
    shapes, _ = resolve(suite, {}, ["qwen3_4b"])
    shape = shapes[0]
    arms = ["resinfer_mxfp4_mext_r1", "fi_mxfp4_mxfp4_stock", alternate]
    files = _sample_files(tmp_path, [shape])
    for file in files:
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        rows = [row for row in rows if row["arm"] in arms]
        rows.append(dict(rows[0], arm=alternate, us=6.0))
        file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    got = aggregate(files, [shape], [8], arms, arms)
    ratios = {row["arm"]: row["macro_median_ratio"]
              for row in got["mxfp4_by_m"]}
    assert ratios["fi_mxfp4_mxfp4_stock"] == 12 / 11
    assert ratios[alternate] == 2.0


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


def test_formal_suite_requires_real_trtllm_measurement(tmp_path: Path):
    suite = load_suite(SUITE)
    shapes, _ = resolve(suite, {}, ["qwen3_4b"])
    files = _sample_files(tmp_path, shapes[:1])
    with pytest.raises(ValueError, match="required arm did not pass"):
        aggregate(files, shapes[:1], [8], suite["arms"], suite["required_arms"])
