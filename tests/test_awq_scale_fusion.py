import pytest

pytest.importorskip("modelopt", reason="requires the ModelOpt export environment")

import torch
import torch.nn as nn

from export.stages.awq_scale_fusion import _apply_awq_scale_fusion


def test_qwen35_zero_centered_rmsnorm_fusion_is_function_preserving(monkeypatch):
    import modelopt.torch.export.quant_utils as quant_utils
    import modelopt.torch.export.unified_export_hf as export_hf

    class Qwen3_5RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.25, -0.125]))

    class InputQuantizer:
        def __init__(self, scale):
            self._pre_quant_scale = scale

    class Consumer(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.input_quantizer = InputQuantizer(scale)

    norm = Qwen3_5RMSNorm()
    scale = torch.tensor([2.0, 0.5])
    consumer = Consumer(scale)
    expected_effective_weight = (1.0 + norm.weight.detach()) * scale
    stock_detector = quant_utils._layernorm_uses_weight_plus_one

    monkeypatch.setattr(quant_utils, "fuse_prequant_to_linear", lambda *_a, **_k: None)

    def exercise_layernorm_fusion(_model):
        quant_utils.fuse_prequant_layernorm(norm, [consumer])

    monkeypatch.setattr(
        export_hf,
        "requantize_resmooth_fused_llm_layers",
        exercise_layernorm_fusion,
    )

    _apply_awq_scale_fusion(nn.Module())

    torch.testing.assert_close(1.0 + norm.weight.detach(), expected_effective_weight)
    assert not hasattr(consumer.input_quantizer, "_pre_quant_scale")
    assert consumer.fused_with_prequant is True
    assert quant_utils._layernorm_uses_weight_plus_one is stock_detector
