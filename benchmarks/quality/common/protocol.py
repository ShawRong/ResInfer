"""Shared manifest parsing for the maintained reasoning evaluations."""

from __future__ import annotations
import json
import pathlib

SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "max_new_tokens",
    "seed",
)


def load_manifest(path: str) -> dict:
    import yaml  # a vLLM dependency of the main venv; nothing new is added

    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(f"manifest not found: {p}")
    man = yaml.safe_load(p.read_text())
    man["_path"] = str(p.resolve())
    for key in ("name", "arms"):
        if key not in man:
            raise SystemExit(f"{p}: manifest has no `{key}`")
    # Accept framework task specs or the explicit official benchmark name.
    man["_tasks"] = man.get("task") or man.get("tasks") or man.get("benchmark")
    if not man["_tasks"]:
        raise SystemExit(f"{p}: manifest names no task (`task:`, `tasks:` or `benchmark:`)")
    # Decoding params are the manifest's to set, not this file's to default:
    # a silent default here would let two runs of "the same manifest" sample
    # differently and still look comparable.
    missing = [
        k
        for k in (
            *SAMPLING_FIELDS,
            "max_model_len",
            "enable_thinking",
            "model_type",
            "protocol_source",
        )
        if k not in man
    ]
    if missing:
        raise SystemExit(
            f"{p}: manifest is missing decoding params: {', '.join(missing)}"
        )
    if type(man["enable_thinking"]) is not bool:
        raise SystemExit("enable_thinking must be an explicit YAML boolean")
    if not man["model_type"] or not man["protocol_source"]:
        raise SystemExit("model_type and protocol_source must be nonempty")
    if "reasoning_effort" in man and (
        not isinstance(man["reasoning_effort"], str)
        or not man["reasoning_effort"].strip()
    ):
        raise SystemExit("reasoning_effort must be a nonempty model-supported string")
    return man


def resolve_arms(man: dict, models_root: str, only: str | None) -> list[dict]:
    """Absolute-ize arm paths, refusing up front on ANY missing arm -- a table
    short one column still looks like a complete table."""
    root = pathlib.Path(models_root).expanduser().resolve()
    arms = man["arms"]
    if only:
        wanted = [s.strip() for s in only.split(",") if s.strip()]
        by_id = {a["id"]: a for a in arms}
        unknown = [w for w in wanted if w not in by_id]
        if unknown:
            raise SystemExit(
                f"--arms names arm(s) not in the manifest: "
                f"{', '.join(unknown)} (have: {', '.join(by_id)})"
            )
        arms = [by_id[w] for w in wanted]
    resolved, missing = [], []
    for arm in arms:
        if "id" not in arm or "path" not in arm or "kind" not in arm:
            raise SystemExit(f"{man['_path']}: arm needs id/path/kind: {arm}")
        if arm["kind"] not in ("fp16", "residue", "modelopt_fp4"):
            raise SystemExit(
                f"arm {arm['id']}: unknown kind {arm['kind']!r} "
                f"(expected fp16 | residue | modelopt_fp4)"
            )
        full = root / arm["path"]
        if not full.exists():
            missing.append(f"    {arm['id']:<16} -> {full}")
            continue
        config = json.loads((full / "config.json").read_text())
        quant = config.get("quantization_config") or {}
        quant_details = quant.get("quantization", quant)
        if (
            quant.get("quant_method") == "residue_mxfp4"
            or str(quant_details.get("quant_algo", "")).upper() == "MXFP4"
        ):
            raise SystemExit(
                f"arm {arm['id']}: this manifest runner has no MXFP4 "
                "serving path; do not route MXFP4 through modelopt_fp4. "
                "Native ModelOpt MXFP4 uses the offline evalkit; legacy "
                "residue_mxfp4 requires its isolated historical adapter."
            )
        actual_type = config.get("text_config", config).get("model_type")
        if actual_type != man["model_type"]:
            raise SystemExit(
                f"arm {arm['id']}: model_type {actual_type!r} does not match "
                f"protocol {man['model_type']!r}; choose a model-specific manifest"
            )
        resolved.append(dict(arm, abs_path=str(full)))
    if missing:
        raise SystemExit(
            f"manifest {man['name']}: {len(missing)} arm path(s) not found "
            f"under --models-root {root}:\n" + "\n".join(missing)
        )
    return resolved
