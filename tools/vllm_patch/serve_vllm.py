#!/usr/bin/env python3
"""
serve_vllm.py - Launch a vLLM OpenAI-compatible server for local testing.

Usage:
    source env.sh

    uv run python tools/vllm_patch/serve_vllm.py \
        --model_id llama3_8b_instruct \
        --port 8000

    uv run python tools/vllm_patch/serve_vllm.py \
        --model_path /path/to/exported_residue \
        --quantization residue_nvfp4 \
        --max_model_len 32768 \
        --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_current_pythonpath = os.environ.get("PYTHONPATH", "")
if _project_root not in _current_pythonpath:
    os.environ["PYTHONPATH"] = (
        f"{_project_root}:{_current_pythonpath}"
        if _current_pythonpath
        else _project_root
    )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _merge_model_registry(base: dict, override: dict) -> dict:
    merged = {
        "models": dict(base.get("models") or {}),
        "default_model": base.get("default_model"),
    }
    merged["models"].update(override.get("models") or {})
    if override.get("default_model") is not None:
        merged["default_model"] = override["default_model"]
    return merged


def resolve_model_id(model_id: str) -> tuple[str, dict]:
    config_dir = Path(_project_root) / "configs"
    registry = _merge_model_registry(
        _load_yaml(config_dir / "models.yaml"),
        _load_yaml(config_dir / "models.local.yaml"),
    )
    models = registry.get("models") or {}
    if model_id not in models:
        known = ", ".join(sorted(models))
        raise KeyError(f"Unknown model_id={model_id!r}. Known model IDs: {known}")

    entry = models[model_id]
    if isinstance(entry, str):
        path = entry
        metadata = {"path": path}
    elif isinstance(entry, dict) and "path" in entry:
        path = entry["path"]
        metadata = dict(entry)
    else:
        raise ValueError(
            f"Model registry entry for {model_id!r} must be a string path or "
            f"a mapping with a 'path' field, got {entry!r}"
        )

    resolved = os.path.expanduser(os.path.expandvars(str(path)))
    metadata["path"] = resolved
    return resolved, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a vLLM OpenAI-compatible server for a local/exported model."
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_id",
        type=str,
        help="Model ID from configs/models.yaml",
    )
    model_group.add_argument(
        "--model_path",
        type=str,
        help="Path to a local/exported model directory",
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="vLLM quantization mode (e.g. modelopt_fp4 or residue_nvfp4)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="vLLM dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction (default: 0.9)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Max context length to pass to vLLM",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help="Optional vLLM max_num_batched_tokens override",
    )
    parser.add_argument(
        "--linear_backend",
        type=str,
        default=None,
        help=(
            "Optional vLLM linear backend override passed as --linear-backend "
            "(for residue NVFP4, use 'cutlass' for the production path)."
        ),
    )
    parser.add_argument(
        "--enable_flashinfer_autotune",
        action="store_true",
        help=(
            "Pass --enable-flashinfer-autotune to vLLM's native "
            "FlashInfer kernels. MExt fold GEMM autotune is enabled by "
            "default separately."
        ),
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable torch.compile / cudagraph in vLLM",
    )
    parser.add_argument(
        "--allow_deprecated_quantization",
        action="store_true",
        help="Allow deprecated quantization methods in vLLM",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trust remote model code (default: False)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--served_model_name",
        type=str,
        default=None,
        help="OpenAI API served model name (default: model_id or model dir name)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Optional API key required by the server",
    )
    parser.add_argument(
        "--residue_swap",
        action="store_true",
        help="Set RESIDUE_SWAP=1 before launching the server",
    )
    parser.add_argument(
        "--residue_patch_mode",
        type=str,
        choices=("legacy", "mext_r1"),
        default=None,
        help="Optional RESIDUE_PATCH_MODE override for patched vLLM workers.",
    )
    parser.add_argument(
        "--enable_decode_mext",
        action="store_true",
        help=(
            "Set RESIDUE_ENABLE_DECODE_MEXT=1 so legacy residue prefill keeps "
            "the ratio path and decode-only forwards use MExt."
        ),
    )
    parser.add_argument(
        "--share_autotune_cache",
        action="store_true",
        help=(
            'Share the FlashInfer autotune cache across models instead of keying the directory by model-config hash. Do not use with concurrently started servers: concurrent cache writes are best-effort.'
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the final server command and exit",
    )
    return parser.parse_args()


def resolve_model(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.model_id:
        model_path, _metadata = resolve_model_id(args.model_id)
        return model_path, args.model_id
    assert args.model_path is not None
    return args.model_path, None


def detect_patch_mode(model_path: str, requested_quantization: str | None) -> tuple[bool, str | None]:
    quantization = requested_quantization
    export_dir = Path(model_path)
    needs_patch = requested_quantization == "residue_nvfp4"

    if export_dir.is_dir():
        has_residue_files = (
            (export_dir / "custom_linear_spec.json").is_file()
            and (export_dir / "channel_masks.pt").is_file()
        )
        if has_residue_files:
            needs_patch = True
            if quantization is None:
                quantization = "residue_nvfp4"

        residue_metadata = export_dir / "residue_kernel_metadata.json"
        if residue_metadata.exists():
            metadata = json.loads(residue_metadata.read_text())
            layers = metadata.get("layers", {})
            has_mext_r1 = any(
                layer_meta.get("runtime_mode") == "mext_r1"
                for layer_meta in layers.values()
                if isinstance(layer_meta, dict)
            )
            if has_mext_r1:
                needs_patch = True
                if quantization is None:
                    quantization = "residue_nvfp4"

    return needs_patch, quantization


def build_server_command(args: argparse.Namespace, model_path: str, served_model_name: str, vllm_quantization: str | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--served-model-name",
        served_model_name,
    ]

    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.allow_deprecated_quantization:
        command.append("--allow-deprecated-quantization")
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.linear_backend is not None:
        command.extend(["--linear-backend", args.linear_backend])
    if args.enable_flashinfer_autotune:
        command.append("--enable-flashinfer-autotune")
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if vllm_quantization:
        if vllm_quantization == "residue_nvfp4":
            vllm_quantization = "modelopt_fp4"
        command.extend(["--quantization", vllm_quantization])

    return command


def main() -> None:
    args = parse_args()
    model_path, resolved_model_id = resolve_model(args)
    needs_patch, detected_quantization = detect_patch_mode(model_path, args.quantization)
    if args.residue_patch_mode is not None:
        needs_patch = True
        if detected_quantization is None:
            detected_quantization = "residue_nvfp4"

    served_model_name = (
        args.served_model_name
        or resolved_model_id
        or Path(model_path).name
    )

    if args.residue_swap:
        os.environ["RESIDUE_SWAP"] = "1"

    if needs_patch:
        os.environ["RESIDUE_PATCH_VLLM"] = "1"
        os.environ["RESIDUE_VLLM_MODEL_DIR"] = model_path
        if args.residue_patch_mode is not None:
            os.environ["RESIDUE_PATCH_MODE"] = args.residue_patch_mode
        else:
            os.environ.pop("RESIDUE_PATCH_MODE", None)
        if args.enable_decode_mext:
            os.environ["RESIDUE_ENABLE_DECODE_MEXT"] = "1"
        if args.share_autotune_cache:
            os.environ["RESIDUE_SHARE_AUTOTUNE_CACHE"] = "1"
    else:
        os.environ.pop("RESIDUE_PATCH_VLLM", None)
        os.environ.pop("RESIDUE_VLLM_MODEL_DIR", None)
        os.environ.pop("RESIDUE_PATCH_MODE", None)
        os.environ.pop("RESIDUE_ENABLE_DECODE_MEXT", None)
        os.environ.pop("RESIDUE_SHARE_AUTOTUNE_CACHE", None)

    command = build_server_command(
        args=args,
        model_path=model_path,
        served_model_name=served_model_name,
        vllm_quantization=detected_quantization,
    )

    print("=" * 80)
    print("serve_vllm.py")
    print("=" * 80)
    print(f"Model:            {model_path}")
    print(f"Served name:      {served_model_name}")
    print(f"Quantization:     {detected_quantization or 'auto'}")
    print(f"Residue patch:    {'enabled' if needs_patch else 'disabled'}")
    print(f"Patch mode:       {os.environ.get('RESIDUE_PATCH_MODE', '(default)')}")
    print(
        "MExt clean store: "
        f"{os.environ.get('RESIDUE_ENABLE_MEXT_CLEAN_VEC_STORE', '(default)')}"
    )
    print(
        "Decode MExt:      "
        f"{os.environ.get('RESIDUE_ENABLE_DECODE_MEXT', '(default)')}"
    )
    print(f"Linear backend:   {args.linear_backend or 'auto'}")
    print(
        "vLLM FI autotune: "
        f"{'enabled' if args.enable_flashinfer_autotune else 'disabled'}"
    )
    print(f"RESIDUE_SWAP:     {os.environ.get('RESIDUE_SWAP', '(not set)')}")
    print(f"Endpoint:         http://{args.host}:{args.port}/v1")
    print("Command:")
    print("  " + " ".join(command))

    if args.dry_run:
        return

    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
