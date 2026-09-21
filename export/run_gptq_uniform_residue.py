#!/usr/bin/env python3
'Build a fixed-ratio GPTQ + residue checkpoint in two resumable stages.'

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--model-loader",
        choices=["auto", "causal_lm", "multimodal_lm"],
        default="auto",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-gptq")
    common_parser(prepare)
    prepare.add_argument("--percdamp", type=float, default=0.01)
    prepare.add_argument("--gptq-block-size", type=int, default=128)
    prepare.add_argument(
        "--gptq-layerwise",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    residue = subparsers.add_parser("residue-arm")
    common_parser(residue)
    residue.add_argument("--ratio", type=float, default=0.125)
    residue.add_argument("--calib-block-size", type=int, default=8)
    residue.add_argument(
        "--expected-extended-linear-dims-min",
        type=int,
        default=1,
        help="Set to 0 for smoke models whose linear dimensions need no padding.",
    )
    residue.add_argument(
        "--salience-metric",
        choices=["activation_max", "nvfp4_error_weighted"],
        required=True,
    )
    return parser.parse_args()


def validate_common(args: argparse.Namespace) -> None:
    args.model_path = args.model_path.resolve()
    args.dataset_jsonl = args.dataset_jsonl.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not args.dataset_jsonl.is_file():
        raise FileNotFoundError(f"Dataset JSONL not found: {args.dataset_jsonl}")
    if args.num_samples <= 0 or args.seq_len <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples, seq-len, and batch-size must be positive")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stage_api():
    from export.stage_adapters import (
        DumpCleanHFStage,
        ExportHFStage,
        InspectQuantExportStage,
        LoadHFStage,
        ModelOptCalibrateStage,
        StageContext,
    )

    return {
        "DumpCleanHFStage": DumpCleanHFStage,
        "ExportHFStage": ExportHFStage,
        "InspectQuantExportStage": InspectQuantExportStage,
        "LoadHFStage": LoadHFStage,
        "ModelOptCalibrateStage": ModelOptCalibrateStage,
        "StageContext": StageContext,
    }


def load_params(args: argparse.Namespace, model_path: Path) -> dict:
    return {
        "model_path": str(model_path),
        "device": "cuda",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "model_loader": args.model_loader,
        "verbose": args.verbose,
    }


def prepare_gptq(args: argparse.Namespace) -> dict:
    api = load_stage_api()
    clean_dir = args.output_dir / "gptq_clean_hf"
    if (clean_dir / "config.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing GPTQ checkpoint: {clean_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = api["StageContext"](case_dir=args.output_dir)
    common = load_params(args, args.model_path)
    started = time.time()
    context = api["LoadHFStage"]("load_base")(context, common)
    context = api["ModelOptCalibrateStage"]("gptq_calibration")(
        context,
        {
            **common,
            "dataset_ref": str(args.dataset_jsonl),
            "calib_samples": args.num_samples,
            "calib_seq_len": args.seq_len,
            "calib_batch_size": args.batch_size,
            "quant_mode": "gptq",
            "quant_fmt": "nvfp4",
            "quant_scope": "all",
            "percdamp": args.percdamp,
            "gptq_block_size": args.gptq_block_size,
            "gptq_layerwise": args.gptq_layerwise,
            "artifact_root": str(args.output_dir / "artifacts_gptq"),
            "seed": args.seed,
        },
    )
    context = api["DumpCleanHFStage"]("dump_clean_gptq_hf")(
        context,
        {
            "model_path": str(args.model_path),
            "model_loader": args.model_loader,
            "clean_hf_export_dir": str(clean_dir),
            "dtype": "bfloat16",
            "apply_awq_scale_fusion": False,
            "safe_serialization": True,
            "validate_load": False,
            "free_source_model": False,
            "verbose": args.verbose,
        },
    )
    summary = {
        "status": "PASS",
        "stage": "prepare-gptq",
        "model_path": str(args.model_path),
        "model_loader": args.model_loader,
        "dataset_jsonl": str(args.dataset_jsonl),
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "percdamp": args.percdamp,
        "gptq_block_size": args.gptq_block_size,
        "gptq_layerwise": args.gptq_layerwise,
        "gptq_clean_hf": str(clean_dir),
        "elapsed_seconds": time.time() - started,
        "metrics": context.metrics,
    }
    return summary


def residue_arm(args: argparse.Namespace) -> dict:
    api = load_stage_api()
    export_dir = args.output_dir / "exported_residue"
    if (export_dir / "config.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing residue checkpoint: {export_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = api["StageContext"](case_dir=args.output_dir)
    common = load_params(args, args.model_path)
    started = time.time()
    context = api["LoadHFStage"]("load_clean_gptq")(context, common)
    model = context.get_ref("model")
    model.config.use_cache = False

    import modelopt.torch.quantization as mtq
    from modelopt.torch.utils import create_forward_loop
    from quant.modelopt_integration.registry import build_residue_cfg

    quant_cfg = build_residue_cfg(
        fmt="nvfp4",
        ratio=args.ratio,
        calib_block_size=args.calib_block_size,
        quant_scope="all",
        salience_metric=args.salience_metric,
    )
    forward_loop = create_forward_loop(
        model=model,
        dataset_name=str(args.dataset_jsonl),
        tokenizer=context.get_ref("tokenizer"),
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        max_sample_length=args.seq_len,
        device=str(next(model.parameters()).device),
    )
    print(
        f"[residue-arm] metric={args.salience_metric} ratio={args.ratio} "
        f"samples={args.num_samples} seq_len={args.seq_len}",
        flush=True,
    )
    model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    context.set_ref("model", model)
    context = api["ExportHFStage"]("export_residue")(
        context,
        {
            "export_dir": str(export_dir),
            "export_target": "vllm",
            "save_kernel_metadata": True,
            "safe_serialization": True,
            "calib_block_size": args.calib_block_size,
            "residue_ratio": args.ratio,
            "verbose": args.verbose,
        },
    )
    context = api["InspectQuantExportStage"]("inspect_export")(
        context,
        {
            "forbid_salient_indices_in_safetensors": True,
            "require_finite_export_tensors": True,
            "expected_residue_metadata_min_layers": 1,
            "expected_extended_linear_dims_min": args.expected_extended_linear_dims_min,
        },
    )
    metadata = json.loads((export_dir / "residue_kernel_metadata.json").read_text())
    summary = {
        "status": "PASS",
        "stage": "residue-arm",
        "model_path": str(args.model_path),
        "model_loader": args.model_loader,
        "dataset_jsonl": str(args.dataset_jsonl),
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "ratio": args.ratio,
        "calib_block_size": args.calib_block_size,
        "expected_extended_linear_dims_min": args.expected_extended_linear_dims_min,
        "salience_metric": args.salience_metric,
        "export_dir": str(export_dir),
        "metadata_layers": len(metadata.get("layers", {})),
        "extended_linear_dims": len(metadata.get("extended_linear_dims", {})),
        "elapsed_seconds": time.time() - started,
        "metrics": context.metrics,
    }
    return summary


def main() -> int:
    args = parse_args()
    validate_common(args)
    seed_everything(args.seed)
    summary = prepare_gptq(args) if args.command == "prepare-gptq" else residue_arm(args)
    summary_path = args.output_dir / f"{args.command}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
