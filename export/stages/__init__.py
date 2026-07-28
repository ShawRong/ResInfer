# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Reusable export/calibration pipeline stages."""

from .awq_scale_fusion import ApplyAwqScaleFusionStage, fold_awq_pre_quant_scale_for_external_injection
from .base import ConditionalStage, Stage, StageContext
from .calibrate_modelopt import ModelOptCalibrateStage
from .dump_clean_hf import DumpCleanHFStage
from .export_hf import ExportHFStage
from .inject_residue_metadata import InjectResidueMetadataStage
from .inspect_quant_export import (
    CheckQuantizerStateStage,
    InspectQuantExportStage,
    SnapshotQuantizerStateStage,
)
from .load_hf import LoadHFStage

__all__ = [
    "ApplyAwqScaleFusionStage",
    "CheckQuantizerStateStage",
    "ConditionalStage",
    "DumpCleanHFStage",
    "ExportHFStage",
    "InjectResidueMetadataStage",
    "InspectQuantExportStage",
    "LoadHFStage",
    "ModelOptCalibrateStage",
    "SnapshotQuantizerStateStage",
    "Stage",
    "StageContext",
    "fold_awq_pre_quant_scale_for_external_injection",
]
