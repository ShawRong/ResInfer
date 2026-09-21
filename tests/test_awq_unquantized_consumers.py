import unittest
from types import SimpleNamespace
from pathlib import Path
import importlib.util
from unittest.mock import patch
import pytest

torch = pytest.importorskip("torch", reason="requires Torch")
pytest.importorskip("modelopt", reason="requires ModelOpt")
from export.awq_unquantized_consumers import capture_consumers, compensate_consumers


class Qwen3_5RMSNorm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(width) * .1)
    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return (y * (1 + self.weight.float())).to(x.dtype)


def toy():
    torch.manual_seed(42)
    block = torch.nn.Module()
    block.input_layernorm = Qwen3_5RMSNorm(32)
    block.linear_attn = torch.nn.Module()
    block.linear_attn.in_proj_a = torch.nn.Linear(32, 4)
    block.linear_attn.in_proj_b = torch.nn.Linear(32, 4)
    model = torch.nn.Module()
    model.block = block
    return model, block


def test_shared_norm_function_preservation():
    model, block = toy()
    x = torch.randn(8, 32)
    ref = block.linear_attn.in_proj_a(block.input_layernorm(x)).detach()
    snapshot = capture_consumers(model)
    with torch.no_grad():
        block.input_layernorm.weight.copy_((block.input_layernorm.weight + 1) * torch.linspace(.2, 5, 32) - 1)
    broken = block.linear_attn.in_proj_a(block.input_layernorm(x))
    assert (broken-ref).norm()/ref.norm() > .1
    changed = compensate_consumers(snapshot)
    assert len(changed) == 2
    torch.testing.assert_close(block.linear_attn.in_proj_a(block.input_layernorm(x)), ref, atol=1e-6, rtol=1e-5)
    assert compensate_consumers(capture_consumers(model)) == []


def test_skips_quantized_consumer_and_rejects_erased_channel():
    model, block = toy()
    block.linear_attn.in_proj_a.weight_quantizer = SimpleNamespace(is_enabled=True)
    snapshot = capture_consumers(model)
    assert len(snapshot[0]["consumers"]) == 1
    with torch.no_grad():
        block.input_layernorm.weight[0] = -1
    with unittest.TestCase().assertRaisesRegex(ValueError, "erased"):
        compensate_consumers(snapshot)


def test_export_fusion_calls_compensation():
    import modelopt.torch.export.quant_utils as qu
    import modelopt.torch.export.unified_export_hf as hf
    path = Path(__file__).parents[1] / "export/stages/awq_scale_fusion.py"
    spec = importlib.util.spec_from_file_location("export.stages.awq_scale_fusion_unit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, block = toy()
    x = torch.randn(8, 32)
    ref = block.linear_attn.in_proj_b(block.input_layernorm(x)).detach()
    mappings = list(qu.PQS_FUSE_MODULE_MAPPING)
    original_detector = qu._layernorm_uses_weight_plus_one
    def fake_native_fusion(model):
        norm = model.block.input_layernorm
        norm.weight.copy_((norm.weight + 1) * 2.5 - 1)
    with patch.object(hf, "requantize_resmooth_fused_llm_layers", fake_native_fusion):
        assert module._apply_awq_scale_fusion(model) == 2
    torch.testing.assert_close(block.linear_attn.in_proj_b(block.input_layernorm(x)), ref, atol=1e-6, rtol=1e-5)
    assert qu.PQS_FUSE_MODULE_MAPPING == mappings
    assert qu._layernorm_uses_weight_plus_one is original_detector


if __name__ == "__main__":
    test_shared_norm_function_preservation()
    test_skips_quantized_consumer_and_rejects_erased_channel()
    test_export_fusion_calls_compensation()
    print("PASS: shared-norm excluded-consumer algebra and safety")
