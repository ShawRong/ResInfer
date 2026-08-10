# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Reusable export/calibration pipeline stages.

Submodules are resolved on FIRST ATTRIBUTE ACCESS (PEP 562), not at package
import. Six of the seven stages import ModelOpt at module level, and ModelOpt
lives only in the export venv -- so an eager `__init__` made
`import export.stages.inspect_quant_export` fail from the runtime venv even
though that module needs nothing but json, torch and safetensors.

That mattered off this machine. A pod-side task probes this package to answer
"does this checkout carry the tensor-opening export guard?", and it probes with
the runtime interpreter. The eager import turned a missing optional dependency
into `FATAL: stale export guard ... sync the pod`, so a dependency problem
presented as a version problem and no amount of syncing could clear it.

`from export.stages import ExportHFStage` still works and still needs ModelOpt
-- it just fails when you ask for a stage that needs it, instead of when you
import its neighbour.
"""

from __future__ import annotations

import importlib
from typing import Any

_SOURCE: dict[str, str] = {
    "ApplyAwqScaleFusionStage": "awq_scale_fusion",
    "fold_awq_pre_quant_scale_for_external_injection": "awq_scale_fusion",
    "ConditionalStage": "base",
    "Stage": "base",
    "StageContext": "base",
    "ModelOptCalibrateStage": "calibrate_modelopt",
    "DumpCleanHFStage": "dump_clean_hf",
    "ExportHFStage": "export_hf",
    "InjectResidueMetadataStage": "inject_residue_metadata",
    "CheckQuantizerStateStage": "inspect_quant_export",
    "InspectQuantExportStage": "inspect_quant_export",
    "SnapshotQuantizerStateStage": "inspect_quant_export",
    "LoadHFStage": "load_hf",
}


def __getattr__(name: str) -> Any:
    module = _SOURCE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(_SOURCE)


__all__ = sorted(_SOURCE)
