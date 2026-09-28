# SPDX-License-Identifier: Apache-2.0
"""One FlashInfer selection lifecycle for NVFP4 and MXFP4 Linear operators.

Format adapters own numerics and legality.  Only the activation's M dimension
is synthesized by FlashInfer; packed weights and scales remain caller-owned.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch

from .fi_tuner_base import make_runner_pair, make_tuning_config, tuned_call


@dataclass(frozen=True)
class LinearSpec:
    format: str
    policy: tuple
    buckets: tuple[int, ...]
    enabled: bool
    fallback: Callable
    valid: Callable
    run: Callable
    route: Callable
    fallback_key: str
    record: Callable | None = None

    def __post_init__(self):
        if self.format not in ("nvfp4", "mxfp4") or not self.buckets:
            raise ValueError("Linear tuning requires a supported format and buckets")
        if self.buckets != tuple(sorted(set(self.buckets))) or self.buckets[0] < 1:
            raise ValueError("Linear tuning buckets must be sorted positive values")
        if not self.fallback_key:
            raise ValueError("Linear fallback needs a stable backend key")


def _map_m(spec: LinearSpec, m: int) -> int:
    route = spec.route(int(m), spec.policy)
    for bucket in spec.buckets:
        if bucket >= m and spec.route(bucket, spec.policy) == route:
            return bucket
    
    
    return int(m)


@lru_cache(maxsize=None)
def _config(spec: LinearSpec):
    def map_bucket(m: int) -> int:
        return _map_m(spec, m)

    def profile_buckets(m: int):
        # A single vLLM max-token forward profiles the configured small
        # buckets and its actual large M, even when M exceeds 2048.
        return tuple(sorted({*spec.buckets, int(m)}))

    generated = profile_buckets if spec.format == "nvfp4" else spec.buckets
    return make_tuning_config(generated, map_bucket,
                              use_cold_l2_cache=True)


@lru_cache(maxsize=None)
def _runners(spec: LinearSpec):
    def fallback(inputs):
        return spec.fallback(inputs, spec.policy)

    def valid(inputs, tuning):
        return spec.valid(inputs, spec.policy, tuning)

    def run(inputs, tactic):
        return spec.run(inputs, spec.policy, tactic)

    def extras(inputs):
        # M belongs to the optimization profile.  Static operand geometry,
        # format and policy distinguish otherwise identical profile shapes.
        return ("linear_v1", spec.format, spec.policy, spec.fallback_key,
                tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype))
                      if isinstance(t, torch.Tensor) else None
                      for t in inputs[1:]))

    return make_runner_pair(
        f"residue_linear_{spec.format}_v1", fallback,
        ("linear_v1", spec.format, spec.policy, spec.fallback_key),
        valid, run, extras, record=spec.record,
    )


def tuned_linear(inputs: list[torch.Tensor], spec: LinearSpec):
    """Run one format's legal candidates through vLLM's FlashInfer context."""
    with torch.cuda.device(inputs[0].device):
        return tuned_call(
            f"residue_linear_{spec.format}_v1",
            lambda: _runners(spec), lambda: _config(spec), inputs,
            lambda args: spec.fallback(args, spec.policy),
            kill_env_value="1" if spec.enabled else "0", record=spec.record,
        )
