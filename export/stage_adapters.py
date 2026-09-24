#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Export-stage adapter boundary.

Legacy import path forwarding to export.stages on demand. The symbol list
is retained for compatibility; stage resolution belongs to export.stages.
"""

from __future__ import annotations

from export import stages

__all__ = [
    "CheckQuantizerStateStage",
    "DumpCleanHFStage",
    "ExportHFStage",
    "InjectResidueMetadataStage",
    "InspectQuantExportStage",
    "LoadHFStage",
    "ModelOptCalibrateStage",
    "SnapshotQuantizerStateStage",
    "StageContext",
    "fold_awq_pre_quant_scale_for_external_injection",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(stages, name)


def __dir__():
    return sorted(__all__)
