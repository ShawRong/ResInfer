"""CPU-only contract tests for the MXFP4 FlashInfer Fold adapter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


MODULE = (Path(__file__).resolve().parents[1] /
          "quant/vllm_integration/mxfp4_fold_tuner.py")


class TensorStub:
    def __init__(self, shape, strides, dtype="bf16", device="cuda:0"):
        self.shape = shape
        self._strides = strides
        self.dtype = dtype
        self.device = device

    def stride(self):
        return self._strides

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@pytest.fixture
def tuner(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.cuda = types.SimpleNamespace(synchronize=lambda device: None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    base = types.ModuleType("quant.vllm_integration.fi_tuner_base")
    base.bucket_of = lambda buckets, m: next((b for b in buckets if m <= b), buckets[-1])
    base.make_tuning_config = lambda buckets, mapper: (buckets, mapper)
    base.make_runner_pair = lambda op, **kw: kw
    base.tuned_call = lambda *args, **kw: kw
    monkeypatch.setitem(sys.modules, base.__name__, base)

    spec = importlib.util.spec_from_file_location(
        "quant.vllm_integration.mxfp4_fold_tuner", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._CANDIDATES = [
        (("row_pair", "std"), "tma", (128, 32), (1, 1), None),
        (("row_pair", "std"), "tma", (128, 64), (1, 1), None),
        (("tile8", "std"), "tma_mext", (128, 32), (1, 1), None),
        (("tile8", "pi"), "direct_mext", (128, 64), (1, 1), None),
    ]
    return module


def _inputs(m=1, stride=2048):
    return (TensorStub((m, 4096), (4096, 1)),
            TensorStub((2560, 2048), (stride, 1), "uint8"),
            TensorStub((163840,), (1,), "uint8"))


def test_preflight_filters_wrong_output_and_failed_launch_without_mutating_inputs(tuner,
                                                                                     monkeypatch):
    inputs = _inputs()
    snapshots = [(t.shape, t.stride(), t.dtype) for t in inputs]
    calls = []

    def run(received, candidate, pdl, dtype):
        calls.append((candidate, pdl, dtype))
        assert received == inputs
        if candidate == tuner._CANDIDATES[1]:
            raise RuntimeError("cudaErrorInvalidValue")
        return "wrong" if candidate == tuner._CANDIDATES[2] else "good"

    monkeypatch.setattr(tuner, "_run_tactic", run)
    monkeypatch.setattr(tuner, "_check_output", lambda actual, reference:
                        0.01 if actual == "good" else (_ for _ in ()).throw(
                            ValueError("BF16 relative max 1.0 > 0.25")))
    report = tuner.preflight_tactics(inputs, "reference")
    assert report["valid_indices"] == [0, 3]
    assert set(report["failed"]) == {1, 2}
    assert "cudaErrorInvalidValue" in report["failed"][1]
    assert report["relative_max"] == {0: 0.01, 3: 0.01}
    assert all(pdl == "early" and dtype == "bf16" for _, pdl, dtype in calls)
    assert [(t.shape, t.stride(), t.dtype) for t in inputs] == snapshots
    json.dumps(report)


def test_preflight_is_scoped_to_logical_m_pdl_and_weight_stride(tuner, monkeypatch):
    monkeypatch.setattr(tuner, "_run_tactic", lambda *args: "good")
    monkeypatch.setattr(tuner, "_check_output", lambda *args: 0.0)
    inputs = _inputs()
    runners = tuner._make_runners("early", "bf16")
    assert runners["valid_tactics"](inputs, True) == []
    tuner.preflight_tactics(inputs, "ref")
    assert runners["valid_tactics"](inputs, True) == [0, 1, 2, 3]
    assert runners["valid_tactics"](inputs, False) == [0, 1, 2, 3]
    assert runners["valid_tactics"](_inputs(m=2), True) == []
    assert runners["valid_tactics"](_inputs(stride=4096), True) == []
    assert tuner._make_runners("off", "bf16")["valid_tactics"](inputs, True) == []
    assert tuner._map_to_tuning_buckets(1) == 1
    assert tuner._map_to_tuning_buckets(3) == 4
    assert runners["candidate_extras"](inputs)[0:2] == ("mxfp4", 32)


def test_kernel_receives_group32_format_and_family_specific_quant(tuner, monkeypatch):
    quant_calls = []
    gemm_calls = []
    quant = types.ModuleType("nvfp4.mxfp4")
    quant.quantize_mext = lambda x, **kw: (quant_calls.append(kw) or ("packed", "scales"))
    host = types.ModuleType("nvfp4.cute_fold.host")
    host.mext_fold_gemm_sm103 = lambda *args, **kw: (
        gemm_calls.append((args, kw)) or ["out"])
    monkeypatch.setitem(sys.modules, "nvfp4", types.ModuleType("nvfp4"))
    monkeypatch.setitem(sys.modules, "nvfp4.mxfp4", quant)
    monkeypatch.setitem(sys.modules, "nvfp4.cute_fold", types.ModuleType("nvfp4.cute_fold"))
    monkeypatch.setitem(sys.modules, "nvfp4.cute_fold.host", host)

    tuner._run_tactic(_inputs(), tuner._CANDIDATES[3], "early", "bf16")
    assert quant_calls == [{"mext_layout": "tile8", "m_pad_multiple": 64,
                            "pdl_mode": "early"}]
    assert gemm_calls[0][1]["fp4_format"] == "mxfp4"
    assert gemm_calls[0][1]["pi_via_layout"] is True
    assert gemm_calls[0][0][4] is None  # MXFP4 has no global alpha.
    assert tuner.last_tactic_tile() == tuner._CANDIDATES[3]

    tuner._run_tactic(_inputs(), tuner._CANDIDATES[2], "off", "bf16")
    assert quant_calls[1] == {"mext_layout": "tile8", "m_pad_multiple": 8,
                              "pdl_mode": "off"}
    assert gemm_calls[1][1]["pi_via_layout"] is False


def test_bad_fallback_aborts_preflight_and_does_not_publish_candidates(tuner,
                                                                         monkeypatch):
    inputs = _inputs()
    monkeypatch.setattr(tuner, "_run_tactic", lambda *args: "wrong")
    monkeypatch.setattr(tuner, "_check_output", lambda *args: (_ for _ in ()).throw(
        ValueError("wrong output")))
    with pytest.raises(RuntimeError, match="fallback failed preflight"):
        tuner.preflight_tactics(inputs, "ref")
    assert tuner._make_runners("early", "bf16")["valid_tactics"](inputs, True) == []
