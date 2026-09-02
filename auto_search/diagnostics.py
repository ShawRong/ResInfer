from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChoiceDebug:
    index: int
    format: str
    modelopt_cost: float
    storage_bits: float
    group_cost: dict[str, float]
    valid_cost: bool
    cost_error: str | None = None
    extra_cost_vs_nvfp4: float | None = None
    score_gain_vs_nvfp4: float | None = None
    payback_vs_nvfp4: float | None = None

    def selected_json(self) -> dict[str, Any]:
        return {
            "storage_bits": self.storage_bits,
            "extra_cost_vs_nvfp4": self.extra_cost_vs_nvfp4,
            "score_gain_vs_nvfp4": self.score_gain_vs_nvfp4,
            "payback_vs_nvfp4": self.payback_vs_nvfp4,
            "valid_cost": self.valid_cost,
        }


@dataclass
class ChoiceGroupDiagnostics:
    skipped_hparams: list[dict[str, Any]] = field(default_factory=list)
    cost_warnings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FrontierDiagnostics:
    skipped_hparams: list[dict[str, Any]] = field(default_factory=list)
    cost_warnings: list[dict[str, Any]] = field(default_factory=list)


def missing_cost_warning(
    *,
    hparam_name: str,
    choice_format: str,
    method: str | None,
    ratio: float | None,
    module_names: list[str],
    error: str,
) -> dict[str, Any]:
    return {
        "hparam": hparam_name,
        "format": choice_format,
        "algorithm_method": method,
        "residue_ratio": ratio,
        "module_names": module_names,
        "reason": "missing_module_shape",
        "error": error,
    }


def skip_hparam(
    *,
    hparam_name: str,
    stats: dict[str, Any],
    reason: str,
    num_valid_choices: int,
) -> dict[str, Any]:
    return {
        "hparam": hparam_name,
        "reason": reason,
        "module_names": stats["module_names"],
        "formats": stats["formats"],
        "num_valid_choices": num_valid_choices,
    }
