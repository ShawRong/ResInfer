# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ModelOpt calibration and export integration for residue quantization."""


from . import algo  # noqa: F401

# ==============================================================================
# Core Components
# ==============================================================================

# Calibrators/helpers for residue_max algorithm
from .algo_helper import ResidueHelper

# Tensor Quantizers
from .tensor_quantizer import ResidueTensorQuantizer

# QTensor extensions live in ModelOpt; residue uses their APIs in util.py

# Residue-aware linear modules
from .residue_linear import (
    ResidueQuantLinear,
    ResidueLinear,
)

# Registry and configuration utilities
from .registry import (
    apply_quant_scope,
    build_residue_cfg,
    build_awq_lite_cfg,
    build_awq_full_cfg,
    build_awq_clip_cfg,
    build_gptq_cfg,
)

from .auto_quantize import enable_residue_auto_quantize_support

# Algorithm functions (for direct use)
from .algo import (
    residue_max_calibrate,
    inject_residue_salient_map,
    create_and_replace_residue_linear_on_the_fly,
    trigger_weight_extension,
    finalize_restore_residue_model,
    residue_restore,
    ResidueMaxCalibConfig,
    ResidueMaxCalibrateModeDescriptor,
    ResidueQuantModuleRegistry,
)

# ModelOpt helpers (imported from ModelOpt, not algo.py)
from modelopt.torch.quantization.model_calib import (
    enable_stats_collection,
    finish_stats_collection,
)

# Utility functions
from .util import (
    weight_extension,
    precompute_channel_masks,
)

# Export functions
from .export import (
    export_residue_checkpoint,
    is_residue_quantized,
)

from .kernel_metadata import (
    collect_extended_linear_dims,
    collect_kernel_metadata,
    collect_salient_indices,
    save_kernel_metadata,
)

# ==============================================================================
# Optional Components
# ==============================================================================

# ==============================================================================
# Public API
# ==============================================================================

__all__ = [
    # Calibrators (helpers)
    "ResidueHelper",

    # Tensor Quantizers
    "ResidueTensorQuantizer",

    # QTensor extensions are provided by ModelOpt directly.

    # Quantized Modules
    "ResidueQuantLinear",
    "ResidueLinear",

    # Registry & Configuration
    "apply_quant_scope",
    "build_residue_cfg",
    "build_awq_lite_cfg",
    "build_awq_full_cfg",
    "build_awq_clip_cfg",
    "build_gptq_cfg",
    "enable_residue_auto_quantize_support",

    # Algorithm Components
    "residue_max_calibrate",
    "inject_residue_salient_map",
    "create_and_replace_residue_linear_on_the_fly",
    "trigger_weight_extension",
    "finalize_restore_residue_model",
    "residue_restore",
    "enable_stats_collection",
    "finish_stats_collection",
    "ResidueMaxCalibConfig",
    "ResidueMaxCalibrateModeDescriptor",
    "ResidueQuantModuleRegistry",

    # Utilities
    "weight_extension",
    "precompute_channel_masks",

    # Export functions
    "export_residue_checkpoint",
    "is_residue_quantized",
    "collect_extended_linear_dims",
    "collect_kernel_metadata",
    "collect_salient_indices",
    "save_kernel_metadata",
]

# ==============================================================================
# Linear Implementations Registration (for runner system)
# ==============================================================================

# Register concrete Linear implementations to runner's LinearRegistry
# This is optional - only needed when using the runner system
try:
    import os
    from . import linear_impls  # noqa: F401
    if os.getenv("RESIDUE_DEBUG", "0") == "1":
        print("[modelopt_integration] Linear implementations registered to runner")
except ImportError as e:
    # Runner system not available (tests.runner.model_io)
    # This is fine - linear_impls only needed for bundle export/import
    pass
