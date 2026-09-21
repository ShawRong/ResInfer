from types import SimpleNamespace

import pytest

from export.stages.hf_model_class import resolve_hf_auto_model_class


class CausalAuto:
    pass


class MultimodalAuto:
    pass


def fake_transformers(*, multimodal=True):
    module = SimpleNamespace(AutoModelForCausalLM=CausalAuto)
    if multimodal:
        module.AutoModelForMultimodalLM = MultimodalAuto
    return module


def test_auto_keeps_text_checkpoint_on_causal_loader():
    config = SimpleNamespace(vision_config=None, audio_config=None)
    assert (
        resolve_hf_auto_model_class(
            config, transformers_module=fake_transformers()
        )
        is CausalAuto
    )


def test_auto_preserves_outer_multimodal_checkpoint_structure():
    config = SimpleNamespace(vision_config=object(), audio_config=None)
    assert (
        resolve_hf_auto_model_class(
            config, transformers_module=fake_transformers()
        )
        is MultimodalAuto
    )


def test_multimodal_loader_requires_new_enough_transformers():
    config = SimpleNamespace(vision_config=object(), audio_config=None)
    with pytest.raises(RuntimeError, match="AutoModelForMultimodalLM"):
        resolve_hf_auto_model_class(
            config, transformers_module=fake_transformers(multimodal=False)
        )
