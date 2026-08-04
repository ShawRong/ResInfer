from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from auto_search.common import fmt_float, safe_token
from auto_search.cost import normalize_cost_model


def choice_to_policy_ratio(
    choice: dict[str, Any],
) -> tuple[float | None, str]:
    if choice.get("algorithm_method") == "residue_max":
        ratio = choice.get("residue_ratio")
        if ratio is None:
            raise ValueError(f"Residue choice is missing residue_ratio: {choice}")
        return float(ratio), f"r{fmt_float(float(ratio))}"

    if choice.get("is_none"):
        return None, "bf16_none_omitted"

    return None, "nvfp4_omitted"


def build_policy_metadata_for_budget(
    frontier_budget: dict[str, Any],
    salient_indices_by_hparam: dict[str, Any],
    module_shapes: dict[str, list[int]],
    *,
    cost_model: str,
    budget_key: str,
    block_size: int,
    source_summary: str | None = None,
) -> dict[str, Any]:
    if not frontier_budget.get("feasible", False):
        raise ValueError(f"Cannot build policy metadata for infeasible budget: {budget_key}")

    layers: dict[str, Any] = {}
    extended_linear_dims: dict[str, int] = {}
    omitted_layers: dict[str, Any] = {}
    layer_policies: dict[str, Any] = {}

    for hparam_name, choice in frontier_budget.get("selected", {}).items():
        ratio, source_policy_choice = choice_to_policy_ratio(
            choice,
        )
        module_names = list(choice.get("module_names", []))
        if ratio is None:
            mode = "none" if choice.get("is_none") else "nvfp4"
            for module_name in module_names:
                shape = module_shapes.get(module_name)
                out_features = int(shape[0]) if shape is not None else None
                in_features = int(shape[1]) if shape is not None else None
                policy_meta = {
                    "mode": mode,
                    "source_hparam": hparam_name,
                    "source_policy_choice": source_policy_choice,
                    "source_format": choice.get("format"),
                    "source_algorithm_method": choice.get("algorithm_method"),
                    "source_budget": budget_key,
                    "source_budget_model": cost_model,
                    "source_cost_model": cost_model,
                    "score": choice.get("score"),
                    "storage_cost": choice.get("storage_cost", choice.get("frontier_cost")),
                    "frontier_cost": choice.get("frontier_cost"),
                    "storage_bits": choice.get("storage_bits"),
                }
                if in_features is not None:
                    policy_meta["original_in_features"] = in_features
                if out_features is not None:
                    policy_meta["out_features"] = out_features
                omitted_layers[module_name] = dict(policy_meta)
                layer_policies[module_name] = policy_meta
            continue

        ratio_key = fmt_float(ratio)
        hparam_salient = salient_indices_by_hparam.get(hparam_name, {})
        ratio_salient = hparam_salient.get(ratio_key)
        if not ratio_salient:
            available = sorted(hparam_salient)
            raise KeyError(
                f"Missing full salient indices for hparam={hparam_name}, ratio={ratio_key}. "
                f"Available ratios: {available}. Include this ratio in --ratios."
            )

        for module_name in module_names:
            salient_indices = ratio_salient.get(module_name)
            if not salient_indices:
                raise KeyError(
                    f"Missing full salient indices for module={module_name}, "
                    f"hparam={hparam_name}, ratio={ratio_key}"
                )
            if module_name in layers:
                raise ValueError(f"Duplicate module in policy metadata: {module_name}")
            if module_name in layer_policies:
                raise ValueError(f"Duplicate module policy metadata: {module_name}")

            shape = module_shapes.get(module_name)
            out_features = int(shape[0]) if shape is not None else None
            in_features = int(shape[1]) if shape is not None else None
            num_salient = int(len(salient_indices))
            # ratio 1.0 layers are served by the M-extension path (mext_r1):
            # the weight is NOT K-extended, so they must not contribute an
            # extended_in_features / extended_linear_dims entry.
            #
            # This restates the invariant "mext_r1 IS ratio 1.0", whose single
            # definition is quant.modelopt_integration.kernel_metadata.
            # is_mext_r1_layer -- read that docstring before changing this
            # line. It is duplicated rather than imported on purpose:
            # auto_search does not depend on quant/ and one predicate is not
            # worth the edge. If they ever disagree, that one is right.
            is_mext_r1 = in_features is not None and num_salient == in_features
            extended_in_features = (
                int(in_features + num_salient)
                if in_features is not None and not is_mext_r1
                else None
            )

            layer_meta = {
                "salient_indices": salient_indices,
                "num_salient": num_salient,
                "source_hparam": hparam_name,
                "source_format": choice.get("format"),
                "source_policy_choice": source_policy_choice,
                "source_algorithm_method": choice.get("algorithm_method"),
                "source_ratio": ratio,
                "source_budget": budget_key,
                "source_budget_model": cost_model,
                "source_cost_model": cost_model,
                "score": choice.get("score"),
                "storage_cost": choice.get("storage_cost", choice.get("frontier_cost")),
                "frontier_cost": choice.get("frontier_cost"),
                "storage_bits": choice.get("storage_bits"),
            }
            if is_mext_r1:
                layer_meta["runtime_mode"] = "mext_r1"
            if in_features is not None:
                layer_meta["original_in_features"] = in_features
            if extended_in_features is not None:
                layer_meta["extended_in_features"] = extended_in_features
                extended_linear_dims[module_name] = extended_in_features
            if out_features is not None:
                layer_meta["out_features"] = out_features

            layers[module_name] = layer_meta
            policy_meta = dict(layer_meta)
            policy_meta.update({"mode": "residue", "source_mode": "residue"})
            layer_policies[module_name] = policy_meta

    # vLLM merges q/k/v and gate/up into single fused layers and resolves ONE
    # metadata entry for the merged module, so every member of a merged group
    # must share the same runtime mode (mext_r1 vs extended_k). A mix would
    # load one weight layout while serving assumes another.
    _MERGED_GROUPS = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"))
    grouped: dict[tuple[str, int], dict[str, str]] = {}
    for module_name, layer_meta in layers.items():
        parent, _, leaf = module_name.rpartition(".")
        for gi, group in enumerate(_MERGED_GROUPS):
            if leaf in group:
                mode = layer_meta.get("runtime_mode", "extended_k")
                grouped.setdefault((parent, gi), {})[leaf] = mode
    for (parent, gi), members in grouped.items():
        if len(set(members.values())) > 1:
            raise ValueError(
                f"Merged-layer runtime-mode mismatch under '{parent}': {members}. "
                "All members of a vLLM-merged group must share the same mode "
                "(ratio 1.0 -> mext_r1 must apply to the whole group or none)."
            )

    metadata = {
        "layers": layers,
        "global": {
            "policy": "auto_quantize_effective_bit_budget",
            "mixed_residue_ratio": True,
            "block_size": int(block_size),
            "source_budget_model": cost_model,
            "source_cost_model": cost_model,
            "source_budget": budget_key,
            "source_budget_effective_bits": frontier_budget.get("budget_effective_bits"),
            "source_total_effective_bits": frontier_budget.get("total_effective_bits"),
            "source_total_score": frontier_budget.get("total_score"),
            "score_source": "raw_hparam_scores",
            "solver": frontier_budget.get("solver"),
            "solver_status": frontier_budget.get("solver_status"),
        },
        "extended_linear_dims": extended_linear_dims,
        "layer_policies": layer_policies,
        "omitted_layers": omitted_layers,
    }
    if source_summary is not None:
        metadata["global"]["source_summary"] = source_summary
    return metadata


def emit_policy_metadata_outputs(
    output_dir: Path,
    frontiers: dict[str, Any],
    salient_indices_by_hparam: dict[str, Any],
    module_shapes: dict[str, list[int]],
    *,
    method: str,
    source_summary: Path,
    block_size: int,
    cost_model_filter: list[str],
    budget_filter: list[str],
) -> dict[str, Any]:
    requested_cost_models = {
        normalize_cost_model(item) for item in cost_model_filter if item
    }
    requested_budget_keys = {fmt_float(float(item)) for item in budget_filter}
    policy_dir = output_dir / "policy_metadata"
    policy_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {}
    for cost_model, frontier in frontiers.items():
        if requested_cost_models and cost_model not in requested_cost_models:
            continue
        cost_model_outputs: dict[str, Any] = {}
        for budget_key, budget_data in frontier.get("budgets", {}).items():
            if requested_budget_keys and budget_key not in requested_budget_keys:
                continue
            if not budget_data.get("feasible", False):
                continue

            metadata = build_policy_metadata_for_budget(
                budget_data,
                salient_indices_by_hparam,
                module_shapes,
                cost_model=cost_model,
                budget_key=budget_key,
                block_size=block_size,
                source_summary=str(source_summary),
            )
            filename = (
                f"{method}_{safe_token(cost_model)}_budget{safe_token(budget_key)}_"
                "residue_metadata.json"
            )
            out_path = policy_dir / filename
            out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
            cost_model_outputs[budget_key] = {
                "path": str(out_path),
                "layers": len(metadata["layers"]),
                "layer_policies": len(metadata.get("layer_policies", {})),
                "omitted_layers": len(metadata.get("omitted_layers", {})),
            }
        if cost_model_outputs:
            outputs[cost_model] = cost_model_outputs
    return outputs
