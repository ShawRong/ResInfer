from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from auto_search.common import fmt_float


def algorithm_method(recipe: Any) -> str | None:
    algorithm = getattr(recipe.config, "algorithm", None)
    if isinstance(algorithm, dict):
        return algorithm.get("method")
    return algorithm


def residue_ratio(recipe: Any) -> float | None:
    algorithm = getattr(recipe.config, "algorithm", None)
    if isinstance(algorithm, dict) and algorithm.get("method") == "residue_max":
        value = algorithm.get("residue_ratio")
        return None if value is None else float(value)
    return None


def summarize_candidate_stats(candidate_stats: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, stats in candidate_stats.items():
        formats = list(stats.get("formats", []))
        summary[name] = {
            "formats": [str(fmt) for fmt in formats],
            "algorithm_methods": [algorithm_method(fmt) for fmt in formats],
            "residue_ratios": [residue_ratio(fmt) for fmt in formats],
            "scores": [float(score) for score in stats.get("scores", [])],
            "modelopt_costs": [float(cost) for cost in stats.get("costs", [])],
            "module_names": list(stats.get("module_names", [])),
        }
    return summary


def summarize_raw_hparam_candidate_stats(model: torch.nn.Module) -> dict[str, Any]:
    """Collect unclamped auto_quantize scores from QuantRecipeHparam objects."""
    from modelopt.torch.opt.utils import named_hparams
    from modelopt.torch.quantization.algorithms import QuantRecipeHparam

    module_names_by_id = {id(module): name for name, module in model.named_modules()}
    summary: dict[str, Any] = {}
    for name, hparam in named_hparams(model, unique=True):
        if not isinstance(hparam, QuantRecipeHparam):
            continue
        formats = list(hparam.choices)
        summary[name] = {
            "formats": [str(fmt) for fmt in formats],
            "algorithm_methods": [algorithm_method(fmt) for fmt in formats],
            "residue_ratios": [residue_ratio(fmt) for fmt in formats],
            "scores": [float(hparam.get_score(fmt)) for fmt in formats],
            "modelopt_costs": [float(hparam.get_cost(fmt)) for fmt in formats],
            "module_names": _quant_module_names(hparam, module_names_by_id),
        }
    return summary


def _quant_module_names(hparam: Any, module_names_by_id: dict[int, str]) -> list[str]:
    names = getattr(hparam, "quant_module_names", None)
    if names is not None:
        return list(names)

    modules = getattr(hparam, "quant_modules", None)
    if modules is not None:
        return [
            module_names_by_id.get(id(module), type(module).__name__)
            for module in modules
        ]

    # Last-resort fallback for ModelOpt versions that only expose the internal
    # recipe -> module map.
    result: list[str] = []
    for module_to_quantizers in getattr(hparam, "_all_quantizer_choices", {}).values():
        for module in module_to_quantizers:
            name = module_names_by_id.get(id(module), type(module).__name__)
            if name not in result:
                result.append(name)
    return result


def salient_digest(indices: torch.Tensor) -> dict[str, Any]:
    values = indices.detach().cpu().to(torch.int64).contiguous()
    digest = hashlib.sha256(values.numpy().tobytes()).hexdigest()
    flat = values.view(-1)
    return {
        "num_salient": int(flat.numel()),
        "min": int(flat.min().item()) if flat.numel() else None,
        "max": int(flat.max().item()) if flat.numel() else None,
        "sha256": digest,
        "first16": [int(x) for x in flat[:16].tolist()],
    }


def collect_residue_salient_choices(model: torch.nn.Module) -> dict[str, Any]:
    from modelopt.torch.opt.utils import named_hparams
    from modelopt.torch.quantization.algorithms import QuantRecipeHparam

    module_names = {id(module): name for name, module in model.named_modules()}
    choices: dict[str, Any] = {}
    for hparam_name, hparam in named_hparams(model, unique=True):
        if not isinstance(hparam, QuantRecipeHparam):
            continue

        hparam_choices = []
        for recipe, module_to_quantizers in hparam._all_quantizer_choices.items():
            method = algorithm_method(recipe)
            ratio = residue_ratio(recipe)
            if method != "residue_max":
                continue

            modules = []
            for module, quantizers in module_to_quantizers.items():
                input_quantizer = quantizers.get("input_quantizer")
                salient = (
                    getattr(input_quantizer, "salient_indices", None)
                    if input_quantizer is not None
                    else None
                )
                if salient is None or salient.numel() == 0:
                    salient = getattr(
                        module, "_residue_salient_indices_by_ratio", {}
                    ).get(fmt_float(ratio))
                modules.append(
                    {
                        "module_name": module_names.get(id(module), "<unknown>"),
                        "input_quantizer": type(input_quantizer).__name__
                        if input_quantizer is not None
                        else None,
                        "residue_fake_forward_counter": int(
                            getattr(input_quantizer, "_residue_fake_forward_counter", 0)
                        )
                        if input_quantizer is not None
                        else 0,
                        "salient": salient_digest(salient)
                        if salient is not None and salient.numel() > 0
                        else None,
                    }
                )

            hparam_choices.append(
                {
                    "recipe": str(recipe),
                    "residue_ratio": ratio,
                    "modules": modules,
                }
            )

        if hparam_choices:
            choices[hparam_name] = hparam_choices
    return choices


def collect_residue_salient_indices_by_hparam(model: torch.nn.Module) -> dict[str, Any]:
    """Collect full salient indices for policy metadata emission."""
    from modelopt.torch.opt.utils import named_hparams
    from modelopt.torch.quantization.algorithms import QuantRecipeHparam

    module_names = {id(module): name for name, module in model.named_modules()}
    result: dict[str, Any] = {}
    for hparam_name, hparam in named_hparams(model, unique=True):
        if not isinstance(hparam, QuantRecipeHparam):
            continue

        hparam_result: dict[str, Any] = {}
        for recipe, module_to_quantizers in hparam._all_quantizer_choices.items():
            method = algorithm_method(recipe)
            ratio = residue_ratio(recipe)
            if method != "residue_max":
                continue

            ratio_key = fmt_float(ratio)
            modules: dict[str, list[int]] = {}
            for module, quantizers in module_to_quantizers.items():
                input_quantizer = quantizers.get("input_quantizer")
                salient = (
                    getattr(input_quantizer, "salient_indices", None)
                    if input_quantizer is not None
                    else None
                )
                if salient is None or salient.numel() == 0:
                    salient = getattr(
                        module, "_residue_salient_indices_by_ratio", {}
                    ).get(ratio_key)
                if salient is None or salient.numel() == 0:
                    continue
                values = salient.detach().cpu().to(torch.int64).contiguous().view(-1)
                modules[module_names.get(id(module), "<unknown>")] = [
                    int(value) for value in values.tolist()
                ]

            if modules:
                hparam_result[ratio_key] = modules

        if hparam_result:
            result[hparam_name] = hparam_result
    return result


def collect_linear_module_shapes(model: torch.nn.Module) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            shapes[name] = [int(weight.shape[0]), int(weight.shape[1])]
    return shapes


def parse_reference_metadata_paths(value: str) -> dict[str, Path]:
    refs: dict[str, Path] = {}
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        if "=" not in item:
            raise ValueError(
                "reference metadata entries must use ratio=/path syntax, "
                f"got: {item}"
            )
        ratio_text, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if path.is_dir():
            path = path / "residue_kernel_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"reference residue metadata does not exist: {path}")
        refs[fmt_float(float(ratio_text))] = path
    return refs


def load_reference_salient_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    layers = data.get("layers", {})
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"reference metadata has no non-empty layers dict: {path}")

    result: dict[str, Any] = {}
    for layer_name, layer_meta in layers.items():
        raw = layer_meta.get("salient_indices")
        if not raw:
            continue
        tensor = torch.tensor(raw, dtype=torch.int64)
        digest = salient_digest(tensor)
        digest["declared_num_salient"] = layer_meta.get("num_salient")
        result[layer_name] = digest
    return result


def compare_salient_choices_to_references(
    residue_salient_choices: dict[str, Any],
    reference_paths: dict[str, Path],
) -> dict[str, Any]:
    references = {
        ratio: load_reference_salient_metadata(path)
        for ratio, path in reference_paths.items()
    }
    report: dict[str, Any] = {
        ratio: {
            "path": str(reference_paths[ratio]),
            "reference_layers": len(reference),
            "matched_modules": 0,
            "missing_modules": 0,
            "same_count": 0,
            "same_hash": 0,
            "mismatches": [],
        }
        for ratio, reference in references.items()
    }

    for hparam_name, choices in residue_salient_choices.items():
        for choice in choices:
            ratio = fmt_float(choice.get("residue_ratio"))
            if ratio not in references:
                continue
            ref = references[ratio]
            dst = report[ratio]
            for module in choice["modules"]:
                module_name = module["module_name"]
                salient = module.get("salient")
                ref_salient = ref.get(module_name)
                if ref_salient is None:
                    dst["missing_modules"] += 1
                    if len(dst["mismatches"]) < 20:
                        dst["mismatches"].append(
                            {
                                "hparam": hparam_name,
                                "module_name": module_name,
                                "reason": "missing_in_reference",
                            }
                        )
                    continue
                dst["matched_modules"] += 1
                same_count = (
                    salient is not None
                    and salient["num_salient"] == ref_salient["num_salient"]
                )
                same_hash = salient is not None and salient["sha256"] == ref_salient["sha256"]
                dst["same_count"] += int(same_count)
                dst["same_hash"] += int(same_hash)
                if (not same_count or not same_hash) and len(dst["mismatches"]) < 20:
                    dst["mismatches"].append(
                        {
                            "hparam": hparam_name,
                            "module_name": module_name,
                            "reason": "count_or_hash_mismatch",
                            "auto": salient,
                            "reference": ref_salient,
                        }
                    )
    return report
