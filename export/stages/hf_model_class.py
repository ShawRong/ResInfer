"""Select the Hugging Face auto-model class for an export checkpoint."""

from __future__ import annotations

from typing import Any


MODEL_LOADER_CHOICES = ("auto", "causal_lm", "multimodal_lm")


def resolve_hf_auto_model_class(
    config: Any,
    model_loader: str = "auto",
    *,
    transformers_module: Any | None = None,
):
    """Return the HF auto-model class that preserves the checkpoint structure.

    Qwen3.5-27B has a dense language trunk, but its outer checkpoint class also
    carries non-language modules.  Loading it through ``AutoModelForCausalLM``
    drops that outer structure.  Auto mode therefore selects the multimodal
    loader only when the config actually advertises a vision/audio component;
    ordinary dense text checkpoints keep the existing causal-LM path.
    """
    if model_loader not in MODEL_LOADER_CHOICES:
        raise ValueError(
            f"Unsupported model_loader={model_loader!r}; "
            f"expected one of {MODEL_LOADER_CHOICES}"
        )

    if transformers_module is None:
        import transformers as transformers_module

    if model_loader == "causal_lm":
        return transformers_module.AutoModelForCausalLM

    has_multimodal_config = any(
        getattr(config, name, None) is not None
        for name in ("vision_config", "audio_config")
    )
    if model_loader == "multimodal_lm" or has_multimodal_config:
        auto_cls = getattr(transformers_module, "AutoModelForMultimodalLM", None)
        if auto_cls is None:
            raise RuntimeError(
                "This checkpoint requires AutoModelForMultimodalLM, but the "
                "installed Transformers version does not provide it."
            )
        return auto_cls

    return transformers_module.AutoModelForCausalLM
