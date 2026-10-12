"""Small vLLM configuration helpers; no runtime imports or process management."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


def merge_engine_config(base: dict, overrides: dict) -> dict:
    """Merge nested vLLM mappings; lists and scalar values replace, never append."""
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_engine_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_engine_config(path: str | None) -> dict:
    """Read raw vLLM engine kwargs, shared by benchmark entrypoints."""
    if not path:
        return {}
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("engine config must be a JSON object of vLLM arguments")
    # These describe the experiment arm, not its execution policy.
    reserved = {"model", "tokenizer", "quantization", "dtype", "seed"} & value.keys()
    if reserved:
        raise ValueError(f"Set arm/task fields outside engine config: {sorted(reserved)}")
    return value


def accuracy_engine_config(base: dict, overrides: dict) -> dict:
    """Apply accuracy defaults, then explicit experiment overrides.

    These Inductor controls address observed compiler variation on the tested
    runtime; they do not establish whole-inference determinism.
    """
    defaults = {"compilation_config": {"inductor_compile_config": {
        "deterministic": True, "benchmark_combo_kernel": False,
    }}}
    return merge_engine_config(merge_engine_config(defaults, base), overrides)
