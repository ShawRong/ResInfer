import sys
from types import ModuleType


_FAKE_TORCH = False
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    _FAKE_TORCH = True
    torch_module = ModuleType("torch")
    nn_module = ModuleType("torch.nn")

    class _Module:
        pass

    nn_module.Module = _Module
    torch_module.nn = nn_module
    sys.modules["torch"] = torch_module
    sys.modules["torch.nn"] = nn_module


from quant.vllm_integration.linear_utils import (
    get_extended_dim,
    get_layer_metadata,
    get_residue_layer_candidates,
)

if _FAKE_TORCH:
    sys.modules.pop("torch.nn", None)
    sys.modules.pop("torch", None)


FUSED = "model.language_model.layers.0.linear_attn.in_proj_qkvz"
QKV = "model.language_model.layers.0.linear_attn.in_proj_qkv"
Z = "model.language_model.layers.0.linear_attn.in_proj_z"


class _Config:
    _extended_linear_dims = {QKV: 6400, Z: 6400}

    def __init__(self):
        self.layers = {
            QKV: {
                "runtime_mode": "extended_k",
                "num_salient": 1280,
                "salient_indices": [0, 4, 9],
            },
            Z: {
                "runtime_mode": "extended_k",
                "num_salient": 1280,
                "salient_indices": [0, 4, 9],
            },
        }

    def get_layer_metadata(self, name):
        return self.layers.get(name)


def test_qwen35_fused_projection_uses_checkpoint_names():
    assert get_residue_layer_candidates(FUSED) == [FUSED, QKV, Z]


def test_qwen35_fused_projection_recovers_extended_input_contract():
    config = _Config()
    assert get_extended_dim(config, FUSED) == 6400
    assert get_layer_metadata(config, FUSED) == config.layers[QKV]


def test_qwen35_ignored_ba_projection_has_checkpoint_candidates():
    fused = "model.language_model.layers.0.linear_attn.in_proj_ba"
    assert get_residue_layer_candidates(fused) == [
        fused,
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.in_proj_a",
    ]
