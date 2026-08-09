#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Export a residue NVFP4 checkpoint through the AWQ-clean pipeline.

This script validates the export-time control flow used by storage-budget
checkpoints:

1. AWQ-lite calibrate a base HF model.
2. Fold AWQ pre-quant scales and dump a clean HF checkpoint.
3. Run residue auto-search on the clean AWQ checkpoint and emit policy metadata.
4. Calibrate plain NVFP4 scales on the clean checkpoint.
5. Inject selected residue metadata, preserving calibrated scales.
6. Export packed FP4 weights and inspect metadata/safetensors.

The default sizes are intentionally tiny so the same entrypoint can be used as
a wiring/correctness smoke. For quality runs, pass a real calibration dataset
and increase the sample and step counts.

THIS FILE DOES NOT RUN GPTQ. The weight optimiser here is AWQ-lite, and there
is no `--weight_opt` to change that. For a GPTQ + residue checkpoint use
`run_gptq_uniform_residue.py` in this directory. Asking for GPTQ here fails as
an unrecognised argument rather than as a missing file, which is a slow thing
to diagnose -- hence this paragraph.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("/tmp") / f"resinfer_awq_clean_residue_export_smoke_{stamp}"


def _default_model_path() -> str:
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        candidate = Path(env_path)
        if not (candidate.is_absolute() and not candidate.exists()):
            return env_path
    return "Qwen/Qwen3-0.6B"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=_default_model_path())
    parser.add_argument("--output_dir", default=str(_default_output_dir()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--dataset_name",
        default="synthetic",
        help=(
            "Calibration/search dataset. The default synthetic path is local "
            "and deterministic so the smoke does not depend on HF downloads."
        ),
    )
    parser.add_argument("--quant_scope", default="all", choices=["all", "ffn_only", "qkv_only"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--awq_samples", type=int, default=2)
    parser.add_argument("--awq_seq_len", type=int, default=64)
    parser.add_argument("--awq_batch_size", type=int, default=1)
    parser.add_argument("--awq_alpha_step", type=float, default=0.5)

    parser.add_argument("--search_samples", type=int, default=2)
    parser.add_argument("--search_seq_len", type=int, default=64)
    parser.add_argument("--search_batch_size", type=int, default=1)
    parser.add_argument("--search_calib_steps", type=int, default=1)
    parser.add_argument("--search_score_steps", type=int, default=1)
    parser.add_argument("--search_method", default="gradient", choices=["gradient", "kl_div"])
    parser.add_argument("--ratios", default="0.25")
    parser.add_argument(
        "--budget",
        default="5.0",
        help=(
            "Effective-bit budget used for the synthetic smoke. With only "
            "ratio=0.25 residue candidates, 5.0 is the stable feasible point."
        ),
    )
    parser.add_argument("--calib_block_size", type=int, default=8)

    parser.add_argument("--nvfp4_samples", type=int, default=2)
    parser.add_argument("--nvfp4_seq_len", type=int, default=64)
    parser.add_argument("--nvfp4_batch_size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _run_stage(stage, ctx, params: dict[str, Any]):
    print(f"[smoke] stage={stage.name}", flush=True)
    return stage(ctx, params)


def _run_subprocess(cmd: list[str], *, cwd: Path, log_path: Path | None = None) -> None:
    print("[smoke] + " + " ".join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        )
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=f"See {log_path}\n{tail}",
        )
    print(f"[smoke] wrote subprocess log: {log_path}", flush=True)


def _normalize_budget_key(budget: str) -> str:
    """Match the emitter's key format (`auto_search.common.fmt_float`)."""
    try:
        return f"{float(budget):g}"
    except (TypeError, ValueError):
        return str(budget)


def _load_policy_path(auto_summary_path: Path, budget: str) -> Path:
    """Resolve `--budget` to the policy file the search emitted for it.

    Two faults used to compound here into "you asked for one arm and got a
    different one, with one line of print as the only sign".

    1. KEY MISMATCH. The emitter keys by `fmt_float(value)` = `f"{v:g}"`,
       which drops a trailing `.0`; this side looked up the raw command-line
       string. So `--budget 5.0` was emitted under `'5'`, looked up as
       `'5.0'`, and always missed. `--budget 5` matched, and `4.5` matched
       only because `f"{4.5:g}"` is already `'4.5'`.
    2. SILENT FALLBACK to `sorted(effective.items())[0]`. A single-budget run
       lands back on the intended plan by luck. Emit several -- which is what
       comparing budgets requires -- and asking for `5.0` hands back `4.5`,
       because the sort is LEXICOGRAPHIC: `sorted(['4.5','5','8'])[0]` is
       `'4.5'`. It is not even nearest-budget: `sorted(['4.5','10'])[0]` is
       `'10'`.

    Now: normalise the key the way the emitter writes it, and if it is still
    absent, fail with the list of what was emitted. Exporting an arm the
    caller did not ask for is not a degraded answer, it is a wrong one.
    """
    data = json.loads(auto_summary_path.read_text())
    outputs = data.get("policy_metadata_outputs", {})
    effective = outputs.get("effective_bits", {})
    if not effective:
        raise RuntimeError(
            f"No policy metadata was emitted. Check frontier feasibility in "
            f"{auto_summary_path}"
        )

    for key in (budget, _normalize_budget_key(budget)):
        if key in effective:
            return Path(effective[key]["path"])

    raise RuntimeError(
        f"budget={budget!r} was not emitted by the search. Emitted: "
        f"{sorted(effective)}. Refusing to substitute another budget -- that "
        f"would export a different arm under the requested name. "
        f"(Note {budget!r} normalises to {_normalize_budget_key(budget)!r}; "
        f"the emitter keys with '%g'.)"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO_ROOT))

    import torch

    from export.stage_adapters import (
        CheckQuantizerStateStage,
        DumpCleanHFStage,
        ExportHFStage,
        InjectResidueMetadataStage,
        InspectQuantExportStage,
        LoadHFStage,
        ModelOptCalibrateStage,
        SnapshotQuantizerStateStage,
        StageContext,
        fold_awq_pre_quant_scale_for_external_injection,
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    awq_clean_dir = output_dir / "awq_clean_hf"
    auto_dir = output_dir / "residue_auto_search"
    export_dir = output_dir / "exported_residue"

    # ------------------------------------------------------------------
    # 1. AWQ-lite calibration and clean HF dump.
    # ------------------------------------------------------------------
    awq_ctx = StageContext(case_dir=output_dir)
    common_load = {
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "verbose": args.verbose,
    }
    awq_ctx = _run_stage(LoadHFStage("load_base"), awq_ctx, common_load)
    awq_ctx = _run_stage(
        ModelOptCalibrateStage("awq_lite_calib"),
        awq_ctx,
        {
            **common_load,
            "dataset_ref": args.dataset_name,
            "calib_samples": args.awq_samples,
            "calib_seq_len": args.awq_seq_len,
            "calib_batch_size": args.awq_batch_size,
            "quant_mode": "awq_lite",
            "quant_fmt": "nvfp4",
            "quant_scope": args.quant_scope,
            "alpha_step": args.awq_alpha_step,
            "artifact_root": str(output_dir / "artifacts_awq"),
            "seed": args.seed,
        },
    )

    # Explicitly call the shared helper once before clean dump. DumpCleanHFStage
    # then sees no remaining pre_quant_scale and should be a no-op if the helper
    # remains compatible with the current ModelOpt version.
    awq_model = awq_ctx.get_ref("model")
    fusion_summary = fold_awq_pre_quant_scale_for_external_injection(
        awq_model,
        verbose=args.verbose,
    )
    awq_ctx.metrics.update(fusion_summary)

    awq_ctx = _run_stage(
        DumpCleanHFStage("dump_clean_awq_hf"),
        awq_ctx,
        {
            "model_path": args.model_path,
            "clean_hf_export_dir": str(awq_clean_dir),
            "dtype": args.dtype,
            "apply_awq_scale_fusion": True,
            "allow_unfused_pre_quant_scale": False,
            "safe_serialization": True,
            "validate_load": False,
            "free_source_model": False,
            "verbose": args.verbose,
        },
    )
    awq_ctx.set_ref("model", None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. Residue auto-search on the clean AWQ checkpoint.
    # ------------------------------------------------------------------
    auto_summary = auto_dir / "auto_quantize_llama_summary.json"
    _run_subprocess(
        [
            sys.executable,
            "export/run_residue_auto_quantize_llama.py",
            "--output_dir",
            str(auto_dir),
            "--model_path",
            str(awq_clean_dir),
            "--dtype",
            args.dtype,
            "--dataset_name",
            args.dataset_name,
            "--batch_size",
            str(args.search_batch_size),
            "--max_sample_length",
            str(args.search_seq_len),
            "--num_samples",
            str(args.search_samples),
            "--num_calib_steps",
            str(args.search_calib_steps),
            "--num_score_steps",
            str(args.search_score_steps),
            "--method",
            args.search_method,
            "--effective_bits",
            "8.0",
            "--frontier_effective_bits",
            args.budget,
            "--frontier_cost_models",
            "effective_bits",
            "--ratios",
            args.ratios,
            "--calib_block_size",
            str(args.calib_block_size),
            "--quant_scope",
            args.quant_scope,
            "--disabled_layers",
            "*lm_head*",
            "--trust_remote_code" if args.trust_remote_code else "--no-trust_remote_code",
            "--policy_metadata_cost_models",
            "effective_bits",
            "--policy_metadata_budgets",
            args.budget,
            "--seed",
            str(args.seed),
        ],
        cwd=REPO_ROOT,
        log_path=auto_dir / "run.log",
    )
    policy_path = _load_policy_path(auto_summary, args.budget)

    # ------------------------------------------------------------------
    # 3. Plain NVFP4 calibration, external residue injection, FP4 export.
    # ------------------------------------------------------------------
    export_ctx = StageContext(case_dir=output_dir)
    export_ctx = _run_stage(
        LoadHFStage("load_clean_awq"),
        export_ctx,
        {**common_load, "model_path": str(awq_clean_dir)},
    )
    export_ctx = _run_stage(
        ModelOptCalibrateStage("nvfp4_calib_on_clean_awq"),
        export_ctx,
        {
            **common_load,
            "model_path": str(awq_clean_dir),
            "dataset_ref": args.dataset_name,
            "calib_samples": args.nvfp4_samples,
            "calib_seq_len": args.nvfp4_seq_len,
            "calib_batch_size": args.nvfp4_batch_size,
            "quant_mode": "w4a4",
            "quant_fmt": "nvfp4",
            "quant_scope": args.quant_scope,
            "artifact_root": str(output_dir / "artifacts_nvfp4"),
            "seed": args.seed,
        },
    )
    export_ctx = _run_stage(SnapshotQuantizerStateStage("snapshot_scales"), export_ctx, {})
    export_ctx = _run_stage(
        InjectResidueMetadataStage("inject_policy_metadata"),
        export_ctx,
        {
            "residue_metadata_path": str(policy_path),
            "verbose": args.verbose,
        },
    )
    export_ctx = _run_stage(CheckQuantizerStateStage("check_scales_after_injection"), export_ctx, {})
    export_ctx = _run_stage(
        ExportHFStage("export_residue"),
        export_ctx,
        {
            "export_dir": str(export_dir),
            "export_target": "vllm",
            "save_kernel_metadata": True,
            "safe_serialization": True,
            "calib_block_size": args.calib_block_size,
            "residue_ratio": float(args.ratios.split(",")[0]),
            "verbose": args.verbose,
        },
    )
    export_ctx = _run_stage(
        InspectQuantExportStage("inspect_export"),
        export_ctx,
        {
            "forbid_salient_indices_in_safetensors": True,
            "require_finite_export_tensors": True,
            "expected_residue_metadata_min_layers": 1,
            "expected_extended_linear_dims_min": 1,
        },
    )

    metadata = _read_json(export_dir / "residue_kernel_metadata.json")
    hf_quant_config = _read_json(export_dir / "hf_quant_config.json")
    quant_algo = hf_quant_config.get("quantization", {}).get("quant_algo")
    if quant_algo != "NVFP4":
        raise AssertionError(f"Expected exported quant_algo NVFP4, got {quant_algo!r}")

    summary = {
        "status": "PASS",
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "awq_clean_hf": str(awq_clean_dir),
        "auto_search_summary": str(auto_summary),
        "policy_metadata": str(policy_path),
        "export_dir": str(export_dir),
        "export_quant_algo": quant_algo,
        "export_metadata_layers": len(metadata.get("layers", {})),
        "export_extended_linear_dims": len(metadata.get("extended_linear_dims", {})),
        "awq_metrics": awq_ctx.metrics,
        "export_metrics": export_ctx.metrics,
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(summary_text, encoding="utf-8")
    # Backward-compatible alias for older smoke wrappers.
    (output_dir / "smoke_summary.json").write_text(summary_text, encoding="utf-8")
    print(summary_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
