"""Residue auto-search helpers.

This package keeps the reusable search/frontier/policy logic out of one-off
runner scripts.  CLI scripts should only load models, run ModelOpt, and call
these helpers to summarize scores and emit policy metadata.
"""

from auto_search.common import fmt_float, parse_csv, parse_float_list, safe_token
from auto_search.cost import (
    ShapeCostModel,
    budget_cost,
    candidate_cost,
    cost_to_effective_bits,
    normalize_cost_model,
)
from auto_search.frontier import build_frontier
from auto_search.modelopt_summary import (
    collect_linear_module_shapes,
    collect_residue_salient_choices,
    collect_residue_salient_indices_by_hparam,
    compare_salient_choices_to_references,
    parse_reference_metadata_paths,
    summarize_candidate_stats,
    summarize_raw_hparam_candidate_stats,
)
from auto_search.policy_metadata import (
    build_policy_metadata_for_budget,
    choice_to_policy_ratio,
    emit_policy_metadata_outputs,
)

__all__ = [
    "budget_cost",
    "build_frontier",
    "build_policy_metadata_for_budget",
    "candidate_cost",
    "choice_to_policy_ratio",
    "collect_linear_module_shapes",
    "collect_residue_salient_choices",
    "collect_residue_salient_indices_by_hparam",
    "compare_salient_choices_to_references",
    "cost_to_effective_bits",
    "emit_policy_metadata_outputs",
    "fmt_float",
    "normalize_cost_model",
    "parse_csv",
    "parse_float_list",
    "parse_reference_metadata_paths",
    "safe_token",
    "ShapeCostModel",
    "summarize_candidate_stats",
    "summarize_raw_hparam_candidate_stats",
]
