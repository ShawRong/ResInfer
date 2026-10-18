"""Validate local checkpoint identity before starting expensive benchmark runs."""
from __future__ import annotations

import json
from pathlib import Path


def detect_quantization(path: str | Path) -> str | None:
    directory = Path(path)
    config = json.loads((directory / "config.json").read_text())
    quant = config.get("quantization_config")
    sidecar = directory / "hf_quant_config.json"
    if quant is None and sidecar.exists():
        quant = json.loads(sidecar.read_text())
        quant.setdefault("quant_method", "modelopt")
    residue = (directory / "residue_kernel_metadata.json").exists()
    if not quant and not residue:
        return None
    if not isinstance(quant, dict):
        raise ValueError(f"Missing quantization config: {directory}")
    method = str(quant.get("quant_method", "")).lower()
    algo = str(quant.get("quantization", quant).get("quant_algo", "")).upper()
    if method in ("modelopt", "modelopt_mxfp4") and algo == "MXFP4":
        return "modelopt_mxfp4"
    if method not in ("modelopt", "modelopt_fp4") or algo != "NVFP4":
        raise ValueError(f"Unsupported quantization in {directory}: {method}/{algo}")
    return "residue_nvfp4" if residue else "modelopt_fp4"


def validate_context(path: str | Path, length: int) -> None:
    config = json.loads((Path(path) / "config.json").read_text())
    config = config.get("text_config", config)
    limit = config.get("max_position_embeddings")
    if not isinstance(limit, int) or length > limit:
        raise ValueError(f"Requested context {length} exceeds declared model limit {limit}")


def validate_kind(path: str | Path, kind: str) -> str | None:
    actual = detect_quantization(path)
    expected = {"bf16": None, "fp16": None, "modelopt_fp4": "modelopt_fp4",
                "residue": "residue_nvfp4", "modelopt_mxfp4": "modelopt_mxfp4"}[kind]
    if actual != expected:
        raise ValueError(f"Checkpoint {path}: kind={kind} expects {expected}, found {actual}")
    return actual
