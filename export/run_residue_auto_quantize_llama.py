#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Run a Llama/Qwen auto_quantize search with residue candidates.

The reusable solver, score summarization, and policy-metadata logic lives in
``auto_search``.  This file is intentionally just the CLI/model-runner glue.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auto_search import (
    build_frontier as _build_frontier,
    build_policy_metadata_for_budget as _build_policy_metadata_for_budget,
    collect_linear_module_shapes as _collect_linear_module_shapes,
    collect_residue_salient_choices as _collect_residue_salient_choices,
    collect_residue_salient_indices_by_hparam as _collect_residue_salient_indices_by_hparam,
    compare_salient_choices_to_references as _compare_salient_choices_to_references,
    emit_policy_metadata_outputs as _emit_policy_metadata_outputs,
    fmt_float as _fmt_float,
    parse_csv as _parse_csv,
    parse_float_list as _parse_float_list,
    parse_reference_metadata_paths as _parse_reference_metadata_paths,
    normalize_cost_model as _normalize_cost_model,
    safe_token as _safe_token,
    summarize_candidate_stats as _summarize_candidate_stats,
    summarize_raw_hparam_candidate_stats as _summarize_raw_hparam_candidate_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--dataset_name", default="c4")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_sample_length", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_calib_steps", type=int, default=1)
    parser.add_argument("--num_score_steps", type=int, default=1)
    parser.add_argument(
        "--auto_quantize_checkpoint",
        default="",
        help=(
            "Optional ModelOpt auto_quantize checkpoint path. If the file exists, "
            "ModelOpt restores the search state and skips expensive score estimation."
        ),
    )
    parser.add_argument("--effective_bits", type=float, default=8.0)
    parser.add_argument("--frontier_effective_bits", default="4.0,4.5,5.0,6.0,8.0,16.0")
    parser.add_argument(
        "--frontier_cost_models",
        default="effective_bits",
        help=(
            "Deprecated compatibility option. Auto-search now always uses the "
            "effective_bits/storage budget model; legacy latency-table names are "
            "accepted and normalized to effective_bits."
        ),
    )
    parser.add_argument("--ratios", default="0.125,0.25,0.5,1.0")
    parser.add_argument("--calib_block_size", type=int, default=8)
    parser.add_argument("--method", choices=["gradient", "kl_div"], default="gradient")
    parser.add_argument("--quant_scope", choices=["all", "ffn_only", "qkv_only"], default="all")
    parser.add_argument("--disabled_layers", default="*lm_head*")
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trust remote model/tokenizer code when loading the search model.",
    )
    parser.add_argument(
        "--reference_residue_metadata_paths",
        default="",
        help=(
            "Optional comma-separated references for salient sanity checks. "
            "Each item is ratio=/path/to/residue_kernel_metadata.json or ratio=/export_dir."
        ),
    )
    parser.add_argument(
        "--external_residue_metadata_paths",
        default="",
        help=(
            "Optional comma-separated external salient metadata used by residue "
            "candidates during scoring. Each item is ratio=/path/to/"
            "residue_kernel_metadata.json. Ratios without a path fall back to "
            "residue_max activation top-k."
        ),
    )
    parser.add_argument(
        "--recompute_from_summary",
        default="",
        help="Existing auto_quantize_llama_summary.json. Recompute frontier only, no GPU/model load.",
    )
    parser.add_argument(
        "--emit_policy_metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Emit mixed-ratio residue metadata for feasible frontier budgets. "
            "This requires a live model run because full salient_indices are not stored in summaries."
        ),
    )
    parser.add_argument(
        "--include_none_candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include ModelOpt NONE/BF16 candidates in the effective-bit frontier. Disabled by "
            "default because residue injection metadata cannot faithfully materialize per-layer "
            "BF16 choices yet."
        ),
    )
    parser.add_argument(
        "--policy_metadata_cost_models",
        default="",
        help=(
            "Deprecated compatibility option. Optional comma-separated budget models "
            "to emit metadata for; all entries normalize to effective_bits."
        ),
    )
    parser.add_argument(
        "--policy_metadata_budgets",
        default="",
        help="Optional comma-separated budget keys/effective bits to emit metadata for. Empty means all feasible budgets.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype


class _SyntheticTextDataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer,
        num_samples: int,
        max_sample_length: int,
        device: torch.device,
        include_labels: bool,
    ) -> None:
        texts = [
            (
                "Synthetic auto-search calibration sample "
                f"{idx}. The quick brown fox jumps over the lazy dog. "
                "FP4 residue quantization smoke text."
            )
            for idx in range(int(num_samples))
        ]
        encoded = tokenizer(
            texts,
            max_length=int(max_sample_length),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self._batches = {
            key: value.to(device)
            for key, value in encoded.items()
        }
        self._include_labels = include_labels
        self._pad_token_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return int(self._batches["input_ids"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = {
            key: value[index]
            for key, value in self._batches.items()
        }
        if self._include_labels:
            labels = batch["input_ids"].clone()
            if self._pad_token_id is not None:
                labels = labels.masked_fill(labels == self._pad_token_id, -100)
            batch["labels"] = labels
        return batch


def _build_data_loader(
    *,
    args: argparse.Namespace,
    tokenizer,
    device: torch.device,
    include_labels: bool,
):
    if args.dataset_name != "synthetic":
        from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

        return get_dataset_dataloader(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_sample_length=args.max_sample_length,
            device=device,
            include_labels=include_labels,
        )

    dataset = _SyntheticTextDataset(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        max_sample_length=args.max_sample_length,
        device=device,
        include_labels=include_labels,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def _frontier_budget_models(value: str) -> list[str]:
    models = []
    for item in _parse_csv(value) or ["effective_bits"]:
        model = _normalize_cost_model(item)
        if model not in models:
            models.append(model)
    return models


def _recompute_from_summary(args: argparse.Namespace, output_dir: Path) -> int:
    source_path = Path(args.recompute_from_summary)
    data = json.loads(source_path.read_text())
    module_shapes = data.get("module_shapes")
    if not module_shapes:
        raise ValueError(
            f"{source_path} does not contain module_shapes. Re-run search with the updated script."
        )

    frontier_budget_models = _frontier_budget_models(args.frontier_cost_models)
    frontiers = {
        budget_model: _build_frontier(
            data["candidate_stats"],
            {name: [int(shape[0]), int(shape[1])] for name, shape in module_shapes.items()},
            budgets=_parse_float_list(args.frontier_effective_bits),
            cost_model=budget_model,
            include_none_candidates=args.include_none_candidates,
        )
        for budget_model in frontier_budget_models
    }
    data["frontiers"] = frontiers
    data["frontier"] = frontiers[frontier_budget_models[-1]]
    data["frontier_effective_bits"] = _parse_float_list(args.frontier_effective_bits)
    data["frontier_budget_models"] = frontier_budget_models
    data["policy_metadata_outputs"] = {}
    data["policy_metadata_note"] = (
        "Skipped in --recompute_from_summary mode because summaries store only "
        "salient digests, not full salient_indices."
    )
    out_path = output_dir / "auto_quantize_llama_summary.json"
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def _build_quantization_formats(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import modelopt.torch.quantization as mtq

    from quant.modelopt_integration import build_residue_cfg
    from quant.modelopt_integration.registry import _disable_lm_head, apply_quant_scope

    external_paths = _parse_reference_metadata_paths(args.external_residue_metadata_paths)
    residue_cfgs = [
        build_residue_cfg(
            "nvfp4",
            ratio=ratio,
            calib_block_size=args.calib_block_size,
            quant_scope=args.quant_scope,
            external_salient_metadata_path=str(external_paths[_fmt_float(ratio)])
            if _fmt_float(ratio) in external_paths
            else None,
        )
        for ratio in _parse_float_list(args.ratios)
    ]
    nvfp4_cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    _disable_lm_head(nvfp4_cfg)
    apply_quant_scope(nvfp4_cfg, args.quant_scope)
    return [*residue_cfgs, nvfp4_cfg], external_paths


def _run_modelopt_search(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    import modelopt
    import modelopt.torch.quantization as mtq
    import modelopt.torch.quantization.plugins.huggingface  # noqa: F401
    from modelopt.torch.quantization import model_quant
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quant.modelopt_integration import enable_residue_auto_quantize_support

    enable_residue_auto_quantize_support()
    if "residue_max" not in model_quant._AUTO_QUANTIZE_SUPPORTED_ALGORITHMS:
        raise AssertionError("residue_max was not added to auto_quantize allowlist")

    dtype = _torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = str(next(model.parameters()).device)
    include_labels = args.method == "gradient"
    data_loader = _build_data_loader(
        args=args,
        tokenizer=tokenizer,
        device=torch.device(device),
        include_labels=include_labels,
    )

    def forward_step(active_model, batch):
        outputs = active_model(**batch)
        return outputs if args.method == "gradient" else outputs.logits

    def loss_func(outputs, _batch):
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError("Model output did not include loss; use --method kl_div or include labels.")
        return loss

    quantization_formats, external_paths = _build_quantization_formats(args)
    module_numels = {
        name: int(module.weight.numel())
        for name, module in model.named_modules()
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor)
    }
    module_shapes = _collect_linear_module_shapes(model)

    t0 = time.perf_counter()
    model, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": args.effective_bits},
        quantization_formats=quantization_formats,
        data_loader=data_loader,
        forward_step=forward_step,
        loss_func=loss_func if args.method == "gradient" else None,
        num_calib_steps=args.num_calib_steps,
        num_score_steps=args.num_score_steps,
        disabled_layers=_parse_csv(args.disabled_layers),
        verbose=args.verbose,
        method=args.method,
        checkpoint=args.auto_quantize_checkpoint or None,
    )
    elapsed_s = time.perf_counter() - t0

    candidate_summary = _summarize_raw_hparam_candidate_stats(model)
    if not candidate_summary:
        raise AssertionError("auto_quantize returned no QuantRecipeHparam candidate stats")

    frontiers = {
        budget_model: _build_frontier(
            candidate_summary,
            module_shapes,
            budgets=_parse_float_list(args.frontier_effective_bits),
            cost_model=budget_model,
            include_none_candidates=args.include_none_candidates,
        )
        for budget_model in _frontier_budget_models(args.frontier_cost_models)
    }

    residue_salient_choices = _collect_residue_salient_choices(model)
    reference_paths = _parse_reference_metadata_paths(args.reference_residue_metadata_paths)
    if not reference_paths and external_paths:
        reference_paths = external_paths
    salient_reference_compare = (
        _compare_salient_choices_to_references(residue_salient_choices, reference_paths)
        if reference_paths
        else {}
    )

    return {
        "elapsed_s": elapsed_s,
        "modelopt_version": getattr(modelopt, "__version__", None),
        "model": model,
        "search_state": search_state,
        "module_numels": module_numels,
        "module_shapes": module_shapes,
        "candidate_summary": candidate_summary,
        "modelopt_candidate_summary": _summarize_candidate_stats(
            search_state.get("candidate_stats", {})
        ),
        "residue_salient_choices": residue_salient_choices,
        "residue_salient_indices_by_hparam": _collect_residue_salient_indices_by_hparam(model)
        if args.emit_policy_metadata
        else {},
        "salient_reference_compare": salient_reference_compare,
        "frontiers": frontiers,
        "external_paths": external_paths,
    }


def _build_summary(
    args: argparse.Namespace,
    run: dict[str, Any],
    policy_metadata_outputs: dict[str, Any],
) -> dict[str, Any]:
    search_state = run["search_state"]
    frontier_budget_models = _frontier_budget_models(args.frontier_cost_models)
    return {
        "status": "PASS",
        "elapsed_s": run["elapsed_s"],
        "modelopt_version": run["modelopt_version"],
        "model_path": args.model_path,
        "dtype": args.dtype,
        "dataset_name": args.dataset_name,
        "batch_size": args.batch_size,
        "max_sample_length": args.max_sample_length,
        "num_samples": args.num_samples,
        "num_calib_steps": args.num_calib_steps,
        "num_score_steps": args.num_score_steps,
        "method": args.method,
        "quant_scope": args.quant_scope,
        "disabled_layers": args.disabled_layers,
        "trust_remote_code": args.trust_remote_code,
        "ratios": _parse_float_list(args.ratios),
        "external_residue_metadata_paths": {
            ratio: str(path) for ratio, path in sorted(run["external_paths"].items())
        },
        "calib_block_size": args.calib_block_size,
        "frontier_budget_models": frontier_budget_models,
        "frontier_cost_models": frontier_budget_models,
        "frontier_effective_bits": _parse_float_list(args.frontier_effective_bits),
        "frontier_include_none_candidates": args.include_none_candidates,
        "module_numels": run["module_numels"],
        "module_shapes": run["module_shapes"],
        "candidate_stats": run["candidate_summary"],
        "modelopt_clamped_candidate_stats": run["modelopt_candidate_summary"],
        "candidate_stats_source": "raw_hparam_scores_external_salient_metadata"
        if run["external_paths"]
        else "raw_hparam_scores",
        "residue_salient_choices": run["residue_salient_choices"],
        "salient_reference_compare": run["salient_reference_compare"],
        "frontiers": run["frontiers"],
        "frontier": run["frontiers"][frontier_budget_models[-1]],
        "policy_metadata_outputs": policy_metadata_outputs,
        "best": {
            "recipe": {name: str(recipe) for name, recipe in search_state["best"]["recipe"].items()},
            "constraints": search_state["best"].get("constraints", {}),
            "score": float(search_state["best"].get("score", 0.0)),
            "is_satisfied": bool(search_state["best"].get("is_satisfied", False)),
        },
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.recompute_from_summary:
        return _recompute_from_summary(args, output_dir)

    run = _run_modelopt_search(args)
    summary_path = output_dir / "auto_quantize_llama_summary.json"
    policy_metadata_outputs = (
        _emit_policy_metadata_outputs(
            output_dir,
            run["frontiers"],
            run["residue_salient_indices_by_hparam"],
            run["module_shapes"],
            method=args.method,
            source_summary=summary_path,
            block_size=args.calib_block_size,
            cost_model_filter=_parse_csv(args.policy_metadata_cost_models),
            budget_filter=_parse_csv(args.policy_metadata_budgets),
        )
        if args.emit_policy_metadata
        else {}
    )

    summary = _build_summary(args, run, policy_metadata_outputs)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
