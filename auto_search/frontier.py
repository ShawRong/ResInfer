from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from auto_search.common import fmt_float, is_none_format
from auto_search.cost import (
    INVALID_COST,
    budget_cost,
    candidate_cost,
    cost_to_effective_bits,
    normalize_cost_model,
)
from auto_search.diagnostics import (
    ChoiceDebug,
    FrontierDiagnostics,
    ChoiceGroupDiagnostics,
    missing_cost_warning,
    skip_hparam,
)


@dataclass
class Choice:
    # Core fields used by solver and policy metadata.
    format: str
    is_none: bool
    algorithm_method: str | None
    residue_ratio: float | None
    score: float
    storage_cost: float
    module_names: list[str]
    debug: ChoiceDebug

    @property
    def valid_cost(self) -> bool:
        return self.debug.valid_cost

    @property
    def frontier_cost(self) -> float:
        # Backward-compatible name for older summary consumers.
        return self.storage_cost

    def selected_json(self) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "is_none": self.is_none,
            "algorithm_method": self.algorithm_method,
            "residue_ratio": self.residue_ratio,
            "score": self.score,
            "storage_cost": self.storage_cost,
            "frontier_cost": self.storage_cost,
            "module_names": self.module_names,
        }
        payload.update(self.debug.selected_json())
        return payload


@dataclass
class ChoiceGroup:
    hparam_name: str
    choices: list[Choice]
    baseline_score: float
    baseline_cost: float


@dataclass
class FrontierInputs:
    groups: list[ChoiceGroup]
    diagnostics: FrontierDiagnostics


def _candidate_method_for_cost(is_none: bool, method: str | None) -> str | None:
    return None if is_none else method


def _invalid_storage_bits(is_none: bool) -> float:
    return 16.0 if is_none else 4.0


def _make_choice(
    *,
    hparam_name: str,
    stats: dict[str, Any],
    index: int,
    module_shapes: dict[str, list[int] | tuple[int, int]],
    cost_model: str,
) -> tuple[Choice, dict[str, Any] | None]:
    fmt = str(stats["formats"][index])
    is_none = is_none_format(fmt)
    method = stats["algorithm_methods"][index]
    ratio = stats["residue_ratios"][index]
    warning = None

    try:
        cost, bits, group_cost = candidate_cost(
            stats["module_names"],
            module_shapes,
            _candidate_method_for_cost(is_none, method),
            ratio,
            cost_model,
        )
        cost_error = None
    except KeyError as exc:
        cost = INVALID_COST
        bits = _invalid_storage_bits(is_none)
        group_cost = {}
        cost_error = str(exc)
        warning = missing_cost_warning(
            hparam_name=hparam_name,
            choice_format=fmt,
            method=method,
            ratio=ratio,
            module_names=stats["module_names"],
            error=cost_error,
        )

    return (
        Choice(
            format=fmt,
            is_none=is_none,
            algorithm_method=method,
            residue_ratio=ratio,
            score=float(stats["scores"][index]),
            storage_cost=cost,
            module_names=list(stats["module_names"]),
            debug=ChoiceDebug(
                index=index,
                format=fmt,
                modelopt_cost=float(stats["modelopt_costs"][index]),
                storage_bits=bits,
                group_cost=group_cost,
                valid_cost=math.isfinite(cost),
                cost_error=cost_error,
            ),
        ),
        warning,
    )


def _find_baseline(choices: list[Choice]) -> Choice | None:
    for choice in choices:
        if (
            choice.algorithm_method == "max"
            and not choice.is_none
            and choice.valid_cost
        ):
            return choice
    return None


def _annotate_vs_baseline(choices: list[Choice], baseline: Choice) -> list[Choice]:
    valid_choices = [choice for choice in choices if choice.valid_cost]
    for choice in valid_choices:
        choice.debug.extra_cost_vs_nvfp4 = choice.storage_cost - baseline.storage_cost
        choice.debug.score_gain_vs_nvfp4 = baseline.score - choice.score
        extra = choice.debug.extra_cost_vs_nvfp4
        choice.debug.payback_vs_nvfp4 = (
            choice.debug.score_gain_vs_nvfp4 / extra if extra > 0 else None
        )
    return valid_choices


def _build_choice_group(
    hparam_name: str,
    stats: dict[str, Any],
    *,
    module_shapes: dict[str, list[int] | tuple[int, int]],
    cost_model: str,
    include_none_candidates: bool,
) -> tuple[ChoiceGroup | None, ChoiceGroupDiagnostics]:
    choices: list[Choice] = []
    diagnostics = ChoiceGroupDiagnostics()

    for idx, fmt in enumerate(stats["formats"]):
        if is_none_format(str(fmt)) and not include_none_candidates:
            continue
        choice, warning = _make_choice(
            hparam_name=hparam_name,
            stats=stats,
            index=idx,
            module_shapes=module_shapes,
            cost_model=cost_model,
        )
        choices.append(choice)
        if warning is not None:
            diagnostics.cost_warnings.append(warning)

    baseline = _find_baseline(choices)
    valid_choices = [choice for choice in choices if choice.valid_cost]
    if baseline is None:
        diagnostics.skipped_hparams.append(
            skip_hparam(
                hparam_name=hparam_name,
                stats=stats,
                reason="missing_valid_standard_nvfp4_candidate",
                num_valid_choices=len(valid_choices),
            )
        )
        return None, diagnostics

    return (
        ChoiceGroup(
            hparam_name=hparam_name,
            choices=_annotate_vs_baseline(choices, baseline),
            baseline_score=baseline.score,
            baseline_cost=baseline.frontier_cost,
        ),
        diagnostics,
    )


def _build_frontier_inputs(
    candidate_summary: dict[str, Any],
    module_shapes: dict[str, list[int] | tuple[int, int]],
    *,
    cost_model: str,
    include_none_candidates: bool,
) -> FrontierInputs:
    result = FrontierInputs(groups=[], diagnostics=FrontierDiagnostics())
    for hparam_name, stats in candidate_summary.items():
        group, diagnostics = _build_choice_group(
            hparam_name,
            stats,
            module_shapes=module_shapes,
            cost_model=cost_model,
            include_none_candidates=include_none_candidates,
        )
        result.diagnostics.skipped_hparams.extend(diagnostics.skipped_hparams)
        result.diagnostics.cost_warnings.extend(diagnostics.cost_warnings)
        if group is not None:
            result.groups.append(group)
    return result


def _selected_module_names(groups: list[ChoiceGroup]) -> set[str]:
    return {
        module_name
        for group in groups
        for choice in group.choices
        for module_name in choice.module_names
    }


def _total_weight_numel(
    module_names: set[str],
    module_shapes: dict[str, list[int] | tuple[int, int]],
) -> int:
    return sum(
        int(module_shapes[name][0]) * int(module_shapes[name][1])
        for name in module_names
    )


def _lps_inputs(groups: list[ChoiceGroup]) -> tuple[list[list[float]], list[list[float]]]:
    return (
        [[choice.score for choice in group.choices] for group in groups],
        [[choice.storage_cost for choice in group.choices] for group in groups],
    )


def _solve_budget(
    groups: list[ChoiceGroup],
    *,
    budget_bits: float,
    baseline_cost: float,
    cost_model: str,
) -> dict[str, Any]:
    from modelopt.torch.opt.searcher import LPS

    cost_model = normalize_cost_model(cost_model)
    candidate_scores, candidate_costs = _lps_inputs(groups)
    max_cost = budget_cost(baseline_cost, budget_bits, cost_model)
    lps = LPS(
        name=f"ResidueEffectiveBitBudget_{cost_model}_{budget_bits:g}",
        constraints={"storage_cost": max_cost},
        constraints_to_candidate_costs={"storage_cost": candidate_costs},
        candidate_scores=candidate_scores,
        objective_type="minimize",
        verbose=False,
    )
    selected_indices, status = lps()
    if status != "Optimal":
        return {
            "budget_effective_bits": budget_bits,
            "budget_storage_cost": max_cost,
            "budget_cost": max_cost,
            "feasible": False,
            "solver": "modelopt_lps",
            "solver_status": status,
        }

    selected_choices = [
        group.choices[choice_idx]
        for group, choice_idx in zip(groups, selected_indices)
    ]
    total_cost = sum(choice.storage_cost for choice in selected_choices)
    total_score = sum(choice.score for choice in selected_choices)
    return {
        "budget_effective_bits": budget_bits,
        "budget_storage_cost": max_cost,
        "budget_cost": max_cost,
        "feasible": True,
        "solver": "modelopt_lps",
        "solver_status": status,
        "objective": "minimize_raw_sensitivity_score",
        "total_storage_cost": total_cost,
        "total_cost": total_cost,
        "total_effective_bits": cost_to_effective_bits(
            total_cost, baseline_cost, cost_model
        ),
        "total_score": total_score,
        "selected": {
            group.hparam_name: choice.selected_json()
            for group, choice in zip(groups, selected_choices)
        },
    }


def _top_payback(groups: list[ChoiceGroup]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        for choice in group.choices:
            if (
                choice.debug.payback_vs_nvfp4 is None
                or choice.debug.score_gain_vs_nvfp4 <= 0
            ):
                continue
            rows.append(
                {
                    "hparam": group.hparam_name,
                    "format": choice.format,
                    "algorithm_method": choice.algorithm_method,
                    "residue_ratio": choice.residue_ratio,
                    "extra_cost_vs_nvfp4": choice.debug.extra_cost_vs_nvfp4,
                    "score_gain_vs_nvfp4": choice.debug.score_gain_vs_nvfp4,
                    "payback_vs_nvfp4": choice.debug.payback_vs_nvfp4,
                    "module_names": choice.module_names,
                }
            )
    rows.sort(key=lambda row: row["payback_vs_nvfp4"], reverse=True)
    return rows[:50]


def _choices_json(groups: list[ChoiceGroup]) -> dict[str, list[dict[str, Any]]]:
    return {
        group.hparam_name: [choice.selected_json() for choice in group.choices]
        for group in groups
    }


def build_frontier(
    candidate_summary: dict[str, Any],
    module_shapes: dict[str, list[int] | tuple[int, int]],
    budgets: list[float],
    cost_model: str,
    include_none_candidates: bool = False,
) -> dict[str, Any]:
    cost_model = normalize_cost_model(cost_model)
    inputs = _build_frontier_inputs(
        candidate_summary,
        module_shapes,
        cost_model=cost_model,
        include_none_candidates=include_none_candidates,
    )
    module_names = _selected_module_names(inputs.groups)
    total_weight_numel = _total_weight_numel(module_names, module_shapes)
    baseline_cost = sum(group.baseline_cost for group in inputs.groups)

    frontier = {
        fmt_float(budget_bits): _solve_budget(
            inputs.groups,
            budget_bits=budget_bits,
            baseline_cost=baseline_cost,
            cost_model=cost_model,
        )
        for budget_bits in budgets
    }

    return {
        "cost_model": cost_model,
        "budget_model": cost_model,
        "baseline_storage_cost": baseline_cost,
        "baseline_cost": baseline_cost,
        "total_weight_numel": total_weight_numel,
        "num_hparams": len(inputs.groups),
        "num_combinations": math.prod(len(group.choices) for group in inputs.groups),
        "solver": "modelopt_lps",
        "objective": "minimize_raw_sensitivity_score",
        "constraint_name": "storage_cost",
        "skipped_hparams": inputs.diagnostics.skipped_hparams,
        "cost_warnings": inputs.diagnostics.cost_warnings,
        "choices": _choices_json(inputs.groups),
        "budgets": frontier,
        "top_payback": _top_payback(inputs.groups),
    }
