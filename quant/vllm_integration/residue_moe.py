# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Residue coverage for MoE expert layers.'
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field


@dataclass
class MoECoverageReport:
    """What the residue patch would have to cover on this model."""

    layers: dict[str, dict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, name: str, **facts) -> None:
        with self._lock:
            self.layers.setdefault(name, {}).update(facts)

    def summary(self) -> dict:
        n_experts = sum(v.get("num_experts", 0) for v in self.layers.values())
        params = sum(v.get("expert_params", 0) for v in self.layers.values())
        applies = sum(v.get("applies", 0) for v in self.layers.values())
        return {
            "moe_layers": len(self.layers),
            "total_experts": n_experts,
            "expert_params": params,
            "apply_calls": applies,
        }

    def render(self) -> str:
        s = self.summary()
        head = (
            f"[residue:moe] {s['moe_layers']} MoE layers, "
            f"{s['total_experts']} experts total, "
            f"{s['expert_params'] / 1e9:.2f} B expert params, "
            f"{s['apply_calls']} apply() calls -- "
            "NONE of this is under residue today"
        )
        rows = []
        for name, v in sorted(self.layers.items())[:3]:
            rows.append(
                f"    {name}: E={v.get('num_experts')} "
                f"w13={v.get('w13_shape')} w2={v.get('w2_shape')}"
            )
        return "\n".join([head, *rows])


REPORT = MoECoverageReport()


def observer_enabled() -> bool:
    return os.environ.get("RESIDUE_MOE_OBSERVE", "0") == "1"


def make_observing_moe_cls(base_cls: type) -> type:
    """Subclass `base_cls`, adding only observation.

    Built at patch time from whatever vLLM's current NVFP4 MoE method is, so
    this does not pin a vLLM version or duplicate any of its logic.
    """

    class ResidueObservingNvFp4FusedMoE(base_cls):  # type: ignore[misc,valid-type]
        __doc__ = (
            "vLLM's NVFP4 FusedMoE method, instrumented. Identical numerics; "
            "records what residue is not covering."
        )

        def process_weights_after_loading(self, layer) -> None:
            super().process_weights_after_loading(layer)
            try:
                w13 = getattr(layer, "w13_weight", None)
                w2 = getattr(layer, "w2_weight", None)
                name = getattr(layer, "prefix", None) or f"moe@{id(layer):x}"
                # w13/w2 are FP4-packed: the last dim is half the logical K.
                params = 0
                for w in (w13, w2):
                    if w is not None:
                        params += w.numel() * 2
                REPORT.record(
                    name,
                    num_experts=int(getattr(layer, "local_num_experts", 0)
                                    or getattr(layer, "global_num_experts", 0)),
                    w13_shape=tuple(w13.shape) if w13 is not None else None,
                    w2_shape=tuple(w2.shape) if w2 is not None else None,
                    expert_params=params,
                )
            except Exception as e:  # observation must never break a model load
                REPORT.record(f"error@{id(layer):x}", error=repr(e)[:200])

        def apply(self, layer, *args, **kwargs):
            name = getattr(layer, "prefix", None) or f"moe@{id(layer):x}"
            REPORT.record(name, applies=REPORT.layers.get(name, {})
                          .get("applies", 0) + 1)
            return super().apply(layer, *args, **kwargs)

    return ResidueObservingNvFp4FusedMoE


def patch_moe_observer() -> bool:
    """Install the observer on `ModelOptNvFp4Config.FusedMoEMethodCls`.

    Mirrors patch.py's handling of LinearMethodCls, including keeping the
    original around so unpatching is possible.
    """
    if not observer_enabled():
        return False
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )
    except ImportError:
        return False

    current = getattr(ModelOptNvFp4Config, "FusedMoEMethodCls", None)
    if current is None:
        print("[residue:moe] vLLM has no FusedMoEMethodCls hook; skipping",
              flush=True)
        return False
    if getattr(current, "__name__", "") == "ResidueObservingNvFp4FusedMoE":
        return True

    if not hasattr(ModelOptNvFp4Config, "_original_FusedMoEMethodCls"):
        ModelOptNvFp4Config._original_FusedMoEMethodCls = current
    ModelOptNvFp4Config.FusedMoEMethodCls = make_observing_moe_cls(current)
    print(f"[residue:moe] observing FusedMoEMethodCls (was {current.__name__})",
          flush=True)
    return True


def unpatch_moe_observer() -> bool:
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )
    except ImportError:
        return False
    orig = getattr(ModelOptNvFp4Config, "_original_FusedMoEMethodCls", None)
    if orig is not None:
        ModelOptNvFp4Config.FusedMoEMethodCls = orig
        return True
    return False
