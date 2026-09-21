# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared scaffold for residue ops tuned through FlashInfer's AutoTuner.

Three tuners share one shape: a FALLBACK runner at index 0 (FlashInfer's
designated cache-miss target -- serving it must never compile) and a
CANDIDATE runner whose tactics are indices into a stable list, offered in
full inside the tuning window and filtered to precompiled kernels outside
it. This module holds that shape once; the per-op files keep only what is
genuinely theirs: the candidate list, the kernel invocation, the precompile
call, and the cache-key extras.

Two contract points deliberately stay OUT of this module:

* the bucket-mapping function must be a module-level named function in the
  OP'S OWN module -- TuningConfig is hashed into the cache key, so the
  mapper must be one stable object per op, not a factory product;
* get_cache_key_extras stays per-op: what invalidates a persisted entry
  (a tile-list hash, an a_ld, a dtype) is exactly the per-op knowledge.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch


def bucket_of(buckets: Sequence[int], m: int) -> int:
    for b in buckets:
        if m <= b:
            return b
    return buckets[-1]


def weight_geometry(weight: torch.Tensor) -> tuple[int, int, int | None]:
    """(n_w, k_logical, a_ld) off a packed-FP4 weight, view-aware.

    a_ld is the stored row pitch in FP4 elements, None when contiguous --
    the convention both fold hosts take. It must be part of any whitelist
    key and cache-key extras: a plain weight and an ext-K prefix view have
    the SAME shape but different read cost.
    """
    n_w = int(weight.shape[0])
    k_logical = int(weight.shape[1]) * 2
    k_stored = int(weight.stride(0)) * 2
    return n_w, k_logical, (None if k_stored == k_logical else k_stored)


def make_tuning_config(buckets: Sequence[int], map_fn: Callable[[int], int]):
    'Build the one-dynamic-dim (input 0, dim 0) TuningConfig every residue'
    from flashinfer.autotuner import DynamicTensorSpec, TuningConfig

    return TuningConfig(
        use_cold_l2_cache=True,
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=tuple(buckets),
                map_to_tuning_buckets=map_fn,
            ),
        ),
    )


# What the last tuning window actually achieved, per op. Reported, never
# assumed -- see cold_tuning_state.
_COLD_STATE: dict[str, dict] = {}


def cold_tuning_state(op_name: str | None = None):
    'Did the tuning window actually run cold? Ask; do not infer.'
    if op_name is None:
        return dict(_COLD_STATE)
    return _COLD_STATE.get(op_name)


def prime_cold_tuning(op_name: str, inputs: list) -> dict:
    """Raise the AutoTuner's `repeat` until the clone set clears L2.

    `repeat` lives on the AutoTuner singleton, not on a TuningConfig, so
    this is process-wide and therefore SYMMETRIC: the baseline tunes with
    the same value. That is the property to preserve -- a per-arm repeat
    would hand the two arms different profiling conditions, which is the
    fairness rule arriving through a parameter nobody thinks of as an arm
    property.

    Returns the state, and records it for `cold_tuning_state`.
    """
    import torch
    from flashinfer.autotuner import AutoTuner

    tuner = AutoTuner.get()
    one_buffer = sum(
        t.numel() * t.element_size() for t in inputs if isinstance(t, torch.Tensor)
    )
    try:
        l2 = tuner._get_l2_cache_size_in_bytes()
    except Exception:
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
    state = {"one_buffer_bytes": one_buffer, "l2_bytes": l2,
             "repeat_before": tuner.repeat}
    if one_buffer <= 0:
        state.update(cold=False, why="no tensor inputs measured")
        _COLD_STATE[op_name] = state
        return state
    # Enough clones that the working set exceeds L2 outright; the +1 is the
    # original, and one extra is margin against a partial last touch.
    need_buffers = int(l2 // one_buffer) + 2
    uncapped = l2 * 3 // one_buffer + 1
    need_repeat = min(uncapped, need_buffers) - 1
    if need_repeat > tuner.repeat:
        tuner.repeat = need_repeat
    buffers = min(uncapped, tuner.repeat + 1)
    workset = buffers * one_buffer
    state.update(repeat=tuner.repeat, num_buffers=buffers,
                 workset_bytes=workset, workset_over_l2=workset / l2,
                 cold=workset > l2)
    if not state["cold"]:
        state["why"] = ("clone count capped below what L2 needs -- "
                        f"{buffers} x {one_buffer / 1e6:.1f}MB does not clear "
                        f"{l2 / 1e6:.1f}MB")
    _COLD_STATE[op_name] = state
    return state


def make_runner_pair(
    op_name: str,
    fallback_forward: Callable,          # (inputs) -> Tensor
    fallback_extras: tuple,
    valid_tactics: Callable,             # (inputs, tuning: bool) -> list[int]
    run_tactic: Callable,                # (inputs, idx) -> Tensor
    candidate_extras: Callable,          # (inputs) -> tuple
    degraded_flag: list,                 # [bool], one per op module
    on_degrade: Callable[[], None] | None = None,
):
    """[FallbackRunner(), CandidateRunner()] with the shared guarantees.

    The candidate runner re-checks validity at serving time: a persisted
    tactic whose kernel is absent (whitelist drift, partial warmup) degrades
    to the fallback and SAYS SO once -- a silent degrade lets the caller's
    announce report "autotuned" while the fallback serves, which is the
    stale-copy failure shape.
    """
    from flashinfer.autotuner import AutoTuner, TunableRunner

    class FallbackRunner(TunableRunner):
        """runners[0]: any cache miss or skipped op serves this -- the
        never-JIT guarantee."""

        def get_valid_tactics(self, inputs, profile):
            return [0]

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            return fallback_forward(inputs)

        def get_cache_key_extras(self, inputs):
            return fallback_extras

    class CandidateRunner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return valid_tactics(inputs, AutoTuner.get().is_tuning_mode)

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            if tactic < 0:
                return fallback_forward(inputs)
            if (not AutoTuner.get().is_tuning_mode
                    and tactic not in valid_tactics(inputs, False)):
                if not degraded_flag[0]:
                    degraded_flag[0] = True
                    print(f"[residue] {op_name}: cached tactic {tactic} has "
                          f"no precompiled kernel -- DEGRADING to the "
                          f"fallback (once per process; later shapes may do "
                          f"the same silently)", flush=True)
                if on_degrade is not None:
                    on_degrade()
                return fallback_forward(inputs)
            return run_tactic(inputs, tactic)

        def get_cache_key_extras(self, inputs):
            return candidate_extras(inputs)

    return [FallbackRunner(), CandidateRunner()]


def tuned_call(
    op_name: str,
    runners_getter: Callable[[], list],
    config_getter: Callable[[], object],
    inputs: list,
    fallback_forward: Callable,
    kill_env_value: str | None,
    announce_flag: list,
    announce: Callable[[object, int, list], str] | None = None,
    record: Callable[[str], None] | None = None,
):
    """The shared serving entry: kill switch -> import guard -> choose_one.

    `kill_env_value` is the already-read env value ("0" disables); reading
    the env stays in the op module so the contract table has one owner.

    `record`, when given, receives the resolved tactic source on EVERY call
    ("autotuned" | "autotuned_fallback" | "offline_table"); a degrade inside the candidate runner
    overrides it afterwards via make_runner_pair's on_degrade. Exists so a
    bench can report the ACTUAL source per row instead of an architecture
    constant -- a field that looks like a measurement but is a constant is
    this week's recurring bug shape.
    """
    if kill_env_value == "0":
        if record is not None:
            record("offline_table")
        return fallback_forward(inputs)
    try:
        from flashinfer.autotuner import AutoTuner
    except Exception:                                      # noqa: BLE001
        if not announce_flag[0]:
            announce_flag[0] = True
            print(f"[residue] {op_name}: flashinfer.autotuner unavailable "
                  f"-- serving the fallback", flush=True)
        if record is not None:
            record("offline_table")
        return fallback_forward(inputs)

    runners = runners_getter()
    # Only inside a tuning window does choose_one actually profile, so that
    # is the only place raising `repeat` costs anything or buys anything.
    # Outside it this is a cache lookup and the state stays as the window
    # left it -- which is what the bench should report.
    if AutoTuner.get().is_tuning_mode:
        prime_cold_tuning(op_name, inputs)
    runner, tactic = AutoTuner.get().choose_one(
        op_name, runners, config_getter(), inputs)
    if announce is not None and not announce_flag[0]:
        announce_flag[0] = True
        print(f"[residue] {op_name}: {announce(runner, tactic, runners)}",
              flush=True)
    if record is not None:
        if tactic < 0:
            record("offline_table")
        elif runner is runners[0]:
            record("autotuned_fallback")
        else:
            record("autotuned")
    return runner.forward(inputs, tactic=tactic)
