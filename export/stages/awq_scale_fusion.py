# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Apply ModelOpt AWQ pre-quant-scale fusion before residue injection."""
from __future__ import annotations

from typing import Any, Dict

import torch
from modelopt.torch.utils.logging import print_rank_0

from .base import Stage, StageContext


class ApplyAwqScaleFusionStage(Stage):
    """Fuse AWQ pre-quant scales into the in-memory model.

    External residue injection extends linear weights from K to K+S.  If AWQ
    pre-quant scales are still attached when export runs, ModelOpt may try to
    apply original-K scales to extended-K weights.  This stage mirrors the
    export-time AWQ fusion step before weight extension, while preserving the
    live quantizer state for the later snapshot/injection checks.
    """

    def __init__(self, name: str = "awq_scale_fusion"):
        super().__init__(name)

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        if ctx.get_ref("model") is None:
            raise ValueError("ApplyAwqScaleFusionStage requires ctx.refs['model']")

    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        model = ctx.get_ref("model")
        verbose = params.get("verbose", False)
        allow_unfused_pre_quant_scale = params.get("allow_unfused_pre_quant_scale", False)
        run_when_empty = params.get("run_awq_scale_fusion_when_empty", False)

        summary = fold_awq_pre_quant_scale_for_external_injection(
            model,
            run_when_empty=run_when_empty,
            verbose=verbose,
        )
        ctx.metrics["awq_pre_quant_scale_before_fusion"] = summary[
            "awq_pre_quant_scale_before_fusion"
        ]
        ctx.metrics["awq_pre_quant_scale_after_fusion"] = summary[
            "awq_pre_quant_scale_after_fusion"
        ]
        ctx.metrics["awq_fused_with_prequant_markers_cleared"] = summary[
            "awq_fused_with_prequant_markers_cleared"
        ]
        ctx.metrics["awq_pre_quant_scale_resmooth_fallback"] = summary[
            "awq_pre_quant_scale_resmooth_fallback"
        ]
        ctx.metrics["awq_pre_quant_scale_resmooth_fallback_modules"] = summary[
            "awq_pre_quant_scale_resmooth_fallback_modules"
        ]
        ctx.metrics["awq_unquantized_consumers_compensated"] = summary[
            "awq_unquantized_consumers_compensated"
        ]

        if summary["awq_pre_quant_scale_after_fusion"] and not allow_unfused_pre_quant_scale:
            preview = ", ".join(summary["awq_pre_quant_scale_after_fusion_modules"][:10])
            raise RuntimeError(
                "AWQ scale fusion left unfused pre_quant_scale tensors before "
                f"residue injection: {preview}"
            )

        print_rank_0(
            "AWQ pre-quant-scale fusion complete: "
            f"{summary['awq_pre_quant_scale_before_fusion']} before, "
            f"{summary['awq_pre_quant_scale_after_fusion']} after, "
            f"resmoothed {summary['awq_pre_quant_scale_resmooth_fallback']} "
            "non-fusible modules, "
            f"cleared {summary['awq_fused_with_prequant_markers_cleared']} "
            "fused_with_prequant markers"
        )
        return ctx


def fold_awq_pre_quant_scale_for_external_injection(
    model,
    *,
    run_when_empty: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fold AWQ pre-quant-scale state and remove AWQ-only export side effects.

    External residue injection materializes K+S weights after a base calibration
    pass.  For AWQ variants, any pre_quant_scale must be absorbed before that
    extension, otherwise ModelOpt export may try to apply original-K scales to
    extended-K weights.  After the fold, ModelOpt's ``fused_with_prequant``
    marker must also be removed; that marker makes get_quantization_format()
    report ``nvfp4_awq`` and re-enter AWQ-specific export code even though the
    scale tensors have already been consumed.
    """
    before = _find_pre_quant_scale_modules(model)
    fallback_modules = _resmooth_nonfusible_pre_quant_scale_modules(model)
    compensated_consumers = 0
    if before or run_when_empty:
        if verbose:
            print_rank_0(
                "Applying AWQ pre-quant-scale fusion before residue injection "
                f"({len(before)} modules with pre_quant_scale)"
            )
        compensated_consumers = _apply_awq_scale_fusion(model) or 0
    elif verbose:
        print_rank_0("AWQ scale fusion no-op: no pre_quant_scale modules found")

    after = _find_pre_quant_scale_modules(model)
    fused_marker_count = _clear_fused_with_prequant_markers(model)
    return {
        "awq_pre_quant_scale_before_fusion": len(before),
        "awq_pre_quant_scale_after_fusion": len(after),
        "awq_pre_quant_scale_after_fusion_modules": after,
        "awq_fused_with_prequant_markers_cleared": fused_marker_count,
        "awq_pre_quant_scale_resmooth_fallback": len(fallback_modules),
        "awq_pre_quant_scale_resmooth_fallback_modules": fallback_modules,
        "awq_unquantized_consumers_compensated": compensated_consumers,
    }


def _apply_awq_scale_fusion(model) -> int:
    import modelopt.torch.export.unified_export_hf as export_hf
    import modelopt.torch.export.quant_utils as quant_utils
    from modelopt.torch.export.quant_utils import (
        PQS_FUSE_MODULE_MAPPING,
        fuse_prequant_to_linear,
    )
    from export.awq_unquantized_consumers import capture_consumers, compensate_consumers

    excluded_consumers = capture_consumers(model)

    # ModelOpt 0.45 predates the Qwen3.5 class names.  These two modules have
    # the same algebraic producer/consumer pairs as their Qwen3 counterparts,
    # so reusing the existing mappings is exact.  Keep the extension local to
    # this call instead of mutating ModelOpt process-global policy forever.
    qwen35_mappings = [
        (["Qwen3_5Attention"], ("v_proj", "o_proj")),
        (["Qwen3_5MLP"], ("up_proj", "down_proj")),
    ]
    added_mappings = []
    for mapping in qwen35_mappings:
        if mapping not in PQS_FUSE_MODULE_MAPPING:
            PQS_FUSE_MODULE_MAPPING.append(mapping)
            added_mappings.append(mapping)

    try:
        with torch.inference_mode():
            # The grouped-head path is required for Llama/Qwen GQA/MQA o_proj.
            fuse_prequant_to_linear(model, fuse_grouped_heads=True)

            orig_fuse_prequant_to_linear = export_hf.fuse_prequant_to_linear
            orig_layernorm_uses_weight_plus_one = (
                quant_utils._layernorm_uses_weight_plus_one
            )

            def layernorm_uses_weight_plus_one(module) -> bool:
                # transformers' Qwen3.5 RMSNorm stores a zero-centred gamma and
                # computes norm(x) * (1 + weight).  ModelOpt 0.45 predates this
                # class, so its stock detector treats the parameter as an
                # ordinary multiplicative gamma.  That makes AWQ fusion use
                # ``weight * scale`` instead of the function-preserving
                # ``(1 + weight) * scale - 1`` and corrupts the clean model.
                if type(module).__name__ == "Qwen3_5RMSNorm":
                    return True
                return orig_layernorm_uses_weight_plus_one(module)

            export_hf.fuse_prequant_to_linear = lambda *_args, **_kwargs: None
            quant_utils._layernorm_uses_weight_plus_one = (
                layernorm_uses_weight_plus_one
            )
            try:
                export_hf.requantize_resmooth_fused_llm_layers(model)
            finally:
                export_hf.fuse_prequant_to_linear = orig_fuse_prequant_to_linear
                quant_utils._layernorm_uses_weight_plus_one = (
                    orig_layernorm_uses_weight_plus_one
                )
    finally:
        for mapping in added_mappings:
            PQS_FUSE_MODULE_MAPPING.remove(mapping)
    compensated = compensate_consumers(excluded_consumers)
    if compensated:
        print_rank_0(f"AWQ shared-norm fusion: compensated {len(compensated)} excluded GDN consumers")
    return len(compensated)


def _resmooth_nonfusible_pre_quant_scale_modules(model) -> list[str]:
    """Undo AWQ smoothing where Qwen3.5 has no exact upstream fusion target.

    ``Qwen3_5GatedDeltaNet.out_proj`` consumes a recurrent/gated result whose
    final RMSNorm shares 128 parameters across 48 heads.  AWQ learns 6,144
    independent input scales, so those scales cannot be folded into that norm
    unless they repeat across heads (real calibration shows they do not).
    ModelOpt's official resmooth helper restores the original floating weight,
    recalibrates it with max calibration, and lets clean-HF export remain exact.
    """
    from modelopt.torch.quantization.model_calib import (
        disable_pre_quant_scale_and_resmooth,
    )

    fallback_modules = []
    for name, module in model.named_modules():
        if not name.endswith(".linear_attn.out_proj"):
            continue
        input_quantizer = getattr(module, "input_quantizer", None)
        if input_quantizer is None or not hasattr(input_quantizer, "_pre_quant_scale"):
            continue
        disable_pre_quant_scale_and_resmooth(
            module,
            delete_pre_quant_scale=True,
        )
        fallback_modules.append(name)
    return fallback_modules


def _find_pre_quant_scale_modules(model) -> list[str]:
    modules = []
    for name, module in model.named_modules():
        input_quantizer = getattr(module, "input_quantizer", None)
        if input_quantizer is not None and hasattr(input_quantizer, "_pre_quant_scale"):
            modules.append(name)
    return modules


def _clear_fused_with_prequant_markers(model) -> int:
    """Clear ModelOpt AWQ export markers after pre-quant scales are folded.

    ModelOpt's get_quantization_format() treats fused_with_prequant=True as
    nvfp4_awq even when _pre_quant_scale has already been deleted.  For the
    external-injection path, scales are already folded before residue weight
    extension, so the final checkpoint should export as normal NVFP4.
    """
    count = 0
    for module in model.modules():
        if getattr(module, "fused_with_prequant", False):
            delattr(module, "fused_with_prequant")
            count += 1
    return count
