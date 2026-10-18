# SPDX-License-Identifier: Apache-2.0
"""Read kernel sweep JSON and command-line configuration overrides."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


MODEL_CATALOG = Path(__file__).resolve().parents[1] / "suites" / "linear_models.json"


def load_models(names: list[str]) -> dict:
    """Select model definitions from the shared Linear benchmark catalog."""
    catalog = json.loads(MODEL_CATALOG.read_text())
    unknown = set(names) - catalog.keys()
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"unknown or duplicate models: {names}")
    return {name: dict(catalog[name], config_source=str(MODEL_CATALOG)) for name in names}


def load_suite(path: str | Path) -> dict:
    suite = json.loads(Path(path).read_text())
    if not suite.get("name") or not suite.get("models"):
        raise ValueError("suite needs name and models")
    if isinstance(suite["models"], list):
        suite["models"] = load_models(suite["models"])
    for model, spec in suite["models"].items():
        tp_values = spec.get("tp_values")
        if tp_values is None:
            if not spec.get("tp_required"):
                raise ValueError(f"{model} needs tp_values or tp_required")
        elif (not tp_values or any(type(tp) is not int or tp < 1 for tp in tp_values)
              or len(set(tp_values)) != len(tp_values)):
            raise ValueError(f"{model} tp_values must be positive and unique")
    if sorted(set(suite["m_values"])) != sorted(suite["m_values"]):
        raise ValueError("m_values must be unique")
    return suite


def parse_config_overrides(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model-config expects MODEL=PATH: {value}")
        model, raw_path = value.split("=", 1)
        if not model or not raw_path or model in out:
            raise ValueError(f"bad or duplicate --model-config: {value}")
        out[model] = Path(raw_path).expanduser().resolve()
    return out


def parse_tp_overrides(values: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model-tp expects MODEL=TP[,TP]: {value}")
        model, raw = value.split("=", 1)
        try:
            tps = [int(tp) for tp in raw.split(",")]
        except ValueError as exc:
            raise ValueError(f"bad --model-tp: {value}") from exc
        if (not model or model in out or not tps or
            any(tp < 1 for tp in tps) or len(set(tps)) != len(tps)):
            raise ValueError(f"bad or duplicate --model-tp: {value}")
        out[model] = tps
    return out


def load_model_config(model: str, spec: dict, path: Path | None) -> tuple[dict, dict]:
    """Load a model configuration and retain its origin and content hash."""
    if path is not None:
        raw = path.read_bytes()
        config = json.loads(raw)
        origin = str(path)
    elif "config" in spec:
        config = spec["config"]
        raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        origin = spec.get("config_source", "suite:inline")
    else:
        raise ValueError(
            f"{model} needs --model-config {model}=<checkpoint/config.json>; "
            "formal results must not guess dimensions")
    return config, {"source": origin, "sha256": hashlib.sha256(raw).hexdigest()}
