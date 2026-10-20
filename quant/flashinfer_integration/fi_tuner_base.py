# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Shared FlashInfer runners and tuning configuration."""
from __future__ import annotations

from typing import Callable

import torch

_ANNOUNCED: set[str] = set()
_DEGRADED: set[str] = set()


def tile8_pad_multiple(mode: str, tiler: tuple) -> int:
    """A6 direct stores need a full MMA tile; A4 TMA stores clamp M tails."""
    return max(8, tiler[0] // 2) if mode == "direct_mext" else 8


def weight_geometry(weight: torch.Tensor) -> tuple[int, int, int | None]:
    """Packed weight's output width, logical K, and optional stored row pitch."""
    n_w, packed_k = map(int, weight.shape)
    stored_k = int(weight.stride(0)) * 2
    return n_w, packed_k * 2, None if stored_k == packed_k * 2 else stored_k


def make_tuning_config(buckets, map_fn, *, use_cold_l2_cache=True):
    """Profile whole runners as graphs; FlashInfer 0.7 owns per-sample L2 eviction.

    Keep map_fn stable: FlashInfer hashes it into the configuration identity.
    """
    from flashinfer.autotuner import DynamicTensorSpec, TuningConfig

    return TuningConfig(
        use_cuda_graph=True,
        use_cold_l2_cache=use_cold_l2_cache,
        dynamic_tensor_specs=(DynamicTensorSpec(
            input_idx=(0,), dim_idx=(0,),
            gen_tuning_buckets=buckets if callable(buckets) else tuple(buckets),
            map_to_tuning_buckets=map_fn),),
    )


def make_runner_pair(
    op_name: str,
    fallback_forward: Callable,
    fallback_extras: tuple,
    valid_tactics: Callable,
    run_tactic: Callable,
    candidate_extras: Callable,
    record: Callable[[str], None] | None = None,
):
    """Runner zero is the fallback; stale cached candidates also use it."""
    from flashinfer.autotuner import AutoTuner, TunableRunner

    class FallbackRunner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [0]

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            # FlashInfer calls preparation eagerly: tune nested GEMMs before capture.
            return fallback_forward(inputs)

        def get_cache_key_extras(self, inputs):
            return fallback_extras

    class CandidateRunner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return valid_tactics(inputs, AutoTuner.get().is_tuning_mode)

        def forward(self, inputs, tactic=-1, **kwargs):
            if tactic < 0:
                return fallback_forward(inputs)
            if (not AutoTuner.get().is_tuning_mode
                    and tactic not in valid_tactics(inputs, False)):
                if op_name not in _DEGRADED:
                    _DEGRADED.add(op_name)
                    print(f"[residue] {op_name}: cached tactic {tactic} is not "
                          "ready; using fallback", flush=True)
                if record is not None:
                    record("fallback")
                return fallback_forward(inputs)
            return run_tactic(inputs, tactic)

        def get_cache_key_extras(self, inputs):
            return candidate_extras(inputs)

    return [FallbackRunner(), CandidateRunner()]


def tuned_call(op_name, runners_getter, config_getter, inputs,
               fallback_forward, enabled=True, record=None):
    """Disabled/unavailable tuner uses fallback; otherwise delegate to FlashInfer."""
    if not enabled:
        if record is not None:
            record("offline_table")
        return fallback_forward(inputs)
    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError:
        if op_name not in _ANNOUNCED:
            _ANNOUNCED.add(op_name)
            print(f"[residue] {op_name}: FlashInfer unavailable; using fallback", flush=True)
        if record is not None:
            record("offline_table")
        return fallback_forward(inputs)

    runners, config = runners_getter(), config_getter()
    tuner = AutoTuner.get()
    runner, tactic = tuner.choose_one(op_name, runners, config, inputs)
    source = ("offline_table" if tactic < 0 else
              "autotuned_fallback" if runner is runners[0] else "autotuned")
    if record is not None:
        record(source)
    if op_name not in _ANNOUNCED:
        _ANNOUNCED.add(op_name)
        print(f"[residue] {op_name}: {source}, tactic={tactic}", flush=True)
    return runner.forward(inputs, tactic=tactic)
