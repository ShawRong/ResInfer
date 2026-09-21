"""Preserve non-tokenizer Hugging Face runtime assets across exports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


_RUNTIME_ASSET_PATTERNS = (
    "*processor_config.json",
    "chat_template*.jinja",
    "chat_template*.json",
)


def _asset_sources(model_path: Path) -> list[Path]:
    """Return the checkpoint followed by its recorded clean-dump source."""
    sources = [model_path]
    summary_path = model_path / "clean_hf_dump_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            base = summary.get("base_model_path")
            if base:
                base_path = Path(base)
                if base_path.is_dir() and base_path not in sources:
                    sources.append(base_path)
        except (OSError, ValueError, TypeError):
            # The checkpoint can still be text-only. Export validation will
            # surface a genuinely required missing processor in the runtime.
            pass
    return sources


def copy_hf_runtime_assets(
    model_path: str | Path,
    export_path: str | Path,
) -> list[str]:
    """Copy processor/chat assets without overwriting exported model config.

    ``save_pretrained`` on an ``AutoTokenizer`` does not preserve multimodal
    processor files.  A clean checkpoint records the original model path, so
    a later quantized export can recover those files even when the clean dump
    predates this helper.
    """
    source = Path(model_path)
    destination = Path(export_path)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for candidate_root in _asset_sources(source):
        for pattern in _RUNTIME_ASSET_PATTERNS:
            for candidate in sorted(candidate_root.glob(pattern)):
                if not candidate.is_file():
                    continue
                target = destination / candidate.name
                if target.exists():
                    continue
                shutil.copy2(candidate, target)
                copied.append(candidate.name)
    return sorted(copied)
