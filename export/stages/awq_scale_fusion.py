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
    if before or run_when_empty:
        if verbose:
            print_rank_0(
                "Applying AWQ pre-quant-scale fusion before residue injection "
                f"({len(before)} modules with pre_quant_scale)"
            )
        _apply_awq_scale_fusion(model)
    elif verbose:
        print_rank_0("AWQ scale fusion no-op: no pre_quant_scale modules found")

    after = _find_pre_quant_scale_modules(model)
    fused_marker_count = _clear_fused_with_prequant_markers(model)
    return {
        "awq_pre_quant_scale_before_fusion": len(before),
        "awq_pre_quant_scale_after_fusion": len(after),
        "awq_pre_quant_scale_after_fusion_modules": after,
        "awq_fused_with_prequant_markers_cleared": fused_marker_count,
    }


def _apply_awq_scale_fusion(model) -> None:
    import modelopt.torch.export.unified_export_hf as export_hf
    from modelopt.torch.export.quant_utils import fuse_prequant_to_linear

    with torch.inference_mode():
        # The grouped-head path is required for Llama GQA/MQA o_proj.
        fuse_prequant_to_linear(model, fuse_grouped_heads=True)

        orig_fuse_prequant_to_linear = export_hf.fuse_prequant_to_linear
        export_hf.fuse_prequant_to_linear = lambda *_args, **_kwargs: None
        try:
            export_hf.requantize_resmooth_fused_llm_layers(model)
        finally:
            export_hf.fuse_prequant_to_linear = orig_fuse_prequant_to_linear


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
