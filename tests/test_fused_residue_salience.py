from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="requires Torch")
pytest.importorskip("modelopt", reason="requires ModelOpt")
from torch import nn

from quant.modelopt_integration.fused_salience import align_runtime_fused_salience
from quant.vllm_integration.linear_utils import get_residue_layer_candidates


class _Quantizer:
    pass


class _Linear(nn.Module):
    def __init__(self, score: list[float]):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, len(score)))
        self.input_quantizer = _Quantizer()
        self._residue_salience_score = torch.tensor(score)
        self._residue_salient_indices_by_ratio = {"0.5": torch.tensor([0, 1])}


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.mlp = nn.Module()
        self.block.mlp.gate_proj = _Linear([4, 3, 2, 1])
        self.block.mlp.up_proj = _Linear([1, 2, 3, 4])


def test_align_runtime_fused_salience_uses_joint_energy() -> None:
    model = _Model()
    summary = align_runtime_fused_salience(
        model,
        ratio=0.5,
        block_size=4,
        salience_metric="nvfp4_error_weighted",
    )
    gate = model.block.mlp.gate_proj.input_quantizer._salient_indices
    up = model.block.mlp.up_proj.input_quantizer._salient_indices
    assert torch.equal(gate, up)
    assert gate.numel() == 2
    assert summary["aligned_groups"] == 1
    assert summary["aligned_layers"] == 2
    assert not hasattr(model.block.mlp.gate_proj, "_residue_salience_score")


def test_qwen35_runtime_prefix_expands_to_hf_fused_members() -> None:
    candidates = get_residue_layer_candidates(
        "language_model.model.layers.0.linear_attn.in_proj_qkvz"
    )
    assert "model.language_model.layers.0.linear_attn.in_proj_qkv" in candidates
    assert "model.language_model.layers.0.linear_attn.in_proj_z" in candidates

    candidates = get_residue_layer_candidates(
        "language_model.model.layers.3.mlp.gate_up_proj"
    )
    assert "model.language_model.layers.3.mlp.gate_proj" in candidates
    assert "model.language_model.layers.3.mlp.up_proj" in candidates
