# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
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
    """Select a loader that preserves the outer multimodal checkpoint structure."""
    if model_loader not in MODEL_LOADER_CHOICES:
        raise ValueError(
            f"Unsupported model_loader={model_loader!r}; "
            f"expected one of {MODEL_LOADER_CHOICES}"
        )

    if transformers_module is None:
        import transformers as transformers_module

    has_multimodal_config = any(
        getattr(config, name, None) is not None
        for name in ("vision_config", "audio_config")
    )
    if model_loader == "causal_lm":
        if has_multimodal_config:
            raise ValueError(
                "causal_lm would discard the outer multimodal checkpoint "
                "structure; use model_loader='auto' or 'multimodal_lm'"
            )
        return transformers_module.AutoModelForCausalLM

    if model_loader == "multimodal_lm" or has_multimodal_config:
        auto_cls = getattr(transformers_module, "AutoModelForMultimodalLM", None)
        if auto_cls is None and getattr(config, "vision_config", None) is not None:
            auto_cls = getattr(transformers_module, "AutoModelForImageTextToText", None)
        if auto_cls is None:
            raise RuntimeError(
                "This checkpoint requires AutoModelForMultimodalLM, but the "
                "installed Transformers version does not provide it."
            )
        return auto_cls

    return transformers_module.AutoModelForCausalLM
