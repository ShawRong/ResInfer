from __future__ import annotations

from dataclasses import dataclass

from auto_search.common import fmt_float

INVALID_COST = float("inf")
EFFECTIVE_BITS_COST_MODEL = "effective_bits"

_LEGACY_COST_MODEL_ALIASES = {
    "storage",
    "storage_bits",
}


def normalize_cost_model(value: str | None) -> str:
    """Normalize supported aliases to the storage-bit budget model."""

    if value is None or not str(value).strip():
        return EFFECTIVE_BITS_COST_MODEL
    text = str(value).strip()
    if text == EFFECTIVE_BITS_COST_MODEL or text in _LEGACY_COST_MODEL_ALIASES:
        return EFFECTIVE_BITS_COST_MODEL
    raise ValueError(
        f"Unsupported auto-search cost model {text!r}. "
        f"Only {EFFECTIVE_BITS_COST_MODEL!r} is supported."
    )


def storage_bits(method: str | None, ratio: float | None) -> float:
    if method == "residue_max":
        if ratio is None:
            raise ValueError("residue candidate is missing residue_ratio")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        if ratio >= 1.0:
            return 8.0
        return 4.0 * (1.0 + ratio)
    if method is None:
        return 16.0
    return 4.0


@dataclass(frozen=True)
class ShapeCostModel:
    """Compatibility shim for older callers.

    Auto-search now uses only storage/effective-bit cost.  The class is kept so
    older scripts importing ``ShapeCostModel`` fail less abruptly, but it no
    longer reads hardware latency tables.
    """

    name: str = EFFECTIVE_BITS_COST_MODEL

    @classmethod
    def from_name_or_path(cls, value: str) -> "ShapeCostModel":
        return cls(normalize_cost_model(value))


def _candidate_mode(method: str | None) -> str:
    if method == "residue_max":
        return "residue_max"
    if method is None:
        return "none"
    return "nvfp4"


def candidate_cost(
    module_names: list[str],
    module_shapes: dict[str, list[int] | tuple[int, int]],
    method: str | None,
    ratio: float | None,
    cost_model: str = EFFECTIVE_BITS_COST_MODEL,
) -> tuple[float, float, dict[str, float]]:
    """Return storage-bit cost for a candidate group.

    Cost unit is total stored activation-weight bits for the selected modules:
    ``sum(out_features * in_features * bits_per_weight)``.  Therefore the
    effective-bit budget is directly the average bits per original weight.
    """

    normalize_cost_model(cost_model)
    bits = storage_bits(method, ratio)
    mode = _candidate_mode(method)
    shape_cost: dict[str, float] = {}
    total = 0.0
    for name in module_names:
        if name not in module_shapes:
            raise KeyError(f"Missing module shape for {name}")
        n, k = (int(value) for value in module_shapes[name])
        cost = float(n * k) * bits
        shape_key = (
            f"n{n}_k{k}_{mode}_"
            f"{fmt_float(ratio) if mode == 'residue_max' else 'none'}"
        )
        shape_cost[shape_key] = shape_cost.get(shape_key, 0.0) + cost
        total += cost
    return total, bits, shape_cost


def budget_cost(
    baseline_cost: float,
    budget_bits: float,
    cost_model: str = EFFECTIVE_BITS_COST_MODEL,
) -> float:
    normalize_cost_model(cost_model)
    return float(baseline_cost) * (float(budget_bits) / 4.0)


def cost_to_effective_bits(
    cost: float,
    baseline_cost: float,
    cost_model: str = EFFECTIVE_BITS_COST_MODEL,
) -> float:
    normalize_cost_model(cost_model)
    return (float(cost) / float(baseline_cost)) * 4.0
