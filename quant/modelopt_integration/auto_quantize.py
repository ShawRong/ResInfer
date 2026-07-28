# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Runtime compatibility hooks for using residue candidates in ModelOpt auto_quantize."""

from __future__ import annotations

from functools import wraps

from .tensor_quantizer import ResidueTensorQuantizer

_PATCHED = False
_ORIGINAL_QUANT_RECIPE_HPARAM_INIT = None


def _modelopt_version() -> str:
    try:
        import modelopt

        return getattr(modelopt, "__version__", "unknown")
    except Exception:
        return "unknown"


def _compat_error(message: str) -> RuntimeError:
    return RuntimeError(
        f"{message} This residue auto_quantize compatibility hook relies on "
        "ModelOpt auto_quantize internals; please update the hook for "
        f"ModelOpt {_modelopt_version()}."
    )


def _is_residue_recipe(recipe) -> bool:
    algorithm = getattr(recipe.config, "algorithm", None)
    if not isinstance(algorithm, dict):
        return False
    return algorithm.get("method") == "residue_max"


def _repair_residue_quantizer_choices(hparam) -> None:
    all_choices = getattr(hparam, "_all_quantizer_choices", None)
    if all_choices is None:
        raise _compat_error(
            "ModelOpt QuantRecipeHparam no longer exposes _all_quantizer_choices."
        )

    for recipe, module_to_quantizers in all_choices.items():
        if not _is_residue_recipe(recipe):
            continue

        for quantizers in module_to_quantizers.values():
            input_quantizer = quantizers.get("input_quantizer")
            weight_quantizer = quantizers.get("weight_quantizer")

            if input_quantizer is not None:
                input_quantizer.__class__ = ResidueTensorQuantizer
                input_quantizer._enable_salient_indices = True

            if weight_quantizer is not None:
                weight_quantizer.__class__ = ResidueTensorQuantizer
                weight_quantizer._enable_salient_indices = True


def enable_residue_auto_quantize_support() -> None:
    """Enable residue_max candidates in ModelOpt auto_quantize.

    This must be called before ``mtq.auto_quantize`` constructs
    ``QuantRecipeHparam`` objects. The patch is intentionally explicit so
    normal quantize/export paths are unaffected by importing this package.
    """
    global _PATCHED, _ORIGINAL_QUANT_RECIPE_HPARAM_INIT
    if _PATCHED:
        return

    from modelopt.torch.quantization import algorithms, model_quant

    supported_algorithms = getattr(model_quant, "_AUTO_QUANTIZE_SUPPORTED_ALGORITHMS", None)
    if supported_algorithms is None:
        # ModelOpt <=0.42 did not expose this allowlist. Custom algorithm
        # configs are accepted there as long as the calibration mode exists.
        pass
    elif hasattr(supported_algorithms, "add"):
        supported_algorithms.add("residue_max")
    elif hasattr(supported_algorithms, "append"):
        if "residue_max" not in supported_algorithms:
            supported_algorithms.append("residue_max")
    else:
        raise _compat_error(
            "ModelOpt auto_quantize no longer exposes a patchable algorithm allowlist."
        )

    if not hasattr(algorithms, "QuantRecipeHparam"):
        raise _compat_error("ModelOpt no longer exposes QuantRecipeHparam.")

    _ORIGINAL_QUANT_RECIPE_HPARAM_INIT = algorithms.QuantRecipeHparam.__init__

    @wraps(_ORIGINAL_QUANT_RECIPE_HPARAM_INIT)
    def _patched_quant_recipe_hparam_init(self, *args, **kwargs):
        _ORIGINAL_QUANT_RECIPE_HPARAM_INIT(self, *args, **kwargs)
        _repair_residue_quantizer_choices(self)

    algorithms.QuantRecipeHparam.__init__ = _patched_quant_recipe_hparam_init
    algorithms.QuantRecipeHparam._residue_auto_quantize_patched = True
    _PATCHED = True


__all__ = ["enable_residue_auto_quantize_support"]
