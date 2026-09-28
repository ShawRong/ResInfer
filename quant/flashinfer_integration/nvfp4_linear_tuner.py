# SPDX-License-Identifier: Apache-2.0
"""NVFP4 candidate routing inside the shared Linear AutoTuner call.

The policy is made from per-layer constants by the caller.  Route selection
uses the activation M synthesized by FlashInfer, rather than the M of vLLM's
outer dummy forward.
"""
from __future__ import annotations

from typing import NamedTuple

from .linear_tuner import LinearSpec, tuned_linear

_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512,
            1024, 2048)


class NVFP4Policy(NamedTuple):
    k_base: int
    num_salient: int
    padding: int
    output_size: int
    fold_eligible: bool
    use_swap: bool
    ratio1: bool
    arch: str | None
    fold_max_m: int
    ratio1_fold_max_m: int
    force_fold: bool
    residue_off: bool
    off_min_m: int | None
    off_max_m: int | None
    prefill: str
    sm10x_tuner: bool
    sm120_tuner: bool
    candidates: tuple = ()


def _residue_off(m, policy):
    if not policy.residue_off:
        return False
    if policy.off_max_m is not None and m <= policy.off_max_m:
        return True
    if policy.off_min_m is not None:
        return m > policy.off_min_m
    return policy.off_max_m is None and m > policy.fold_max_m


def _route(m, policy):
    """Choose the numerical path for both synthesized and serving M."""
    off = _residue_off(int(m), policy)
    fold = not off and policy.fold_eligible and policy.arch is not None and (
        policy.ratio1 or policy.force_fold or m <= policy.fold_max_m)
    if fold:
        if policy.ratio1 and m > policy.ratio1_fold_max_m:
            return "ratio1_prefill"
        return "sm10x_fold" if policy.arch in ("sm100", "sm103") else "sm120_fold"
    if policy.ratio1 and not off:
        return "invalid_mext"
    return "plain" if policy.num_salient == 0 or (policy.ratio1 and off) else "k_ext"


def _fold_inputs(inputs, policy):
    x, weight, scale_inv, _full_sf, base_sf, _mask, alpha = inputs
    k_base = policy.k_base
    packed_k = k_base // 2
    if packed_k > weight.shape[1]:
        raise ValueError("base K exceeds packed weight width")
    view = weight if packed_k == weight.shape[1] else weight[:, :packed_k]
    return [x, view, scale_inv, base_sf, alpha]


def _valid(inputs, policy, tuning):
    route = _route(int(inputs[0].shape[0]), policy)
    if route == "sm10x_fold":
        if not policy.sm10x_tuner:
            return []
        from .sm10x_fold_tuner import valid_tactics
        return valid_tactics(_fold_inputs(inputs, policy), inputs[0].dtype, tuning)
    if route == "sm120_fold":
        if not policy.sm120_tuner:
            return []
        from .sm120_fold_tuner import valid_tactics
        return valid_tactics(_fold_inputs(inputs, policy), inputs[0].dtype, tuning)
    if route == "ratio1_prefill" and policy.prefill == "auto":
        from .ratio1_prefill_tuner import _kloop_valid_tactics
        return _kloop_valid_tactics(_fold_inputs(inputs, policy), tuning)
    return []


def _run(inputs, policy, tactic):
    route = _route(int(inputs[0].shape[0]), policy)
    args = _fold_inputs(inputs, policy)
    if route == "sm10x_fold":
        from .sm10x_fold_tuner import _run_tactic
        return _run_tactic(args, tactic, inputs[0].dtype)
    if route == "sm120_fold":
        from .sm120_fold_tuner import _run_fold_tile, FOLD_TILES
        return _run_fold_tile(args, FOLD_TILES[tactic], inputs[0].dtype)
    if route == "ratio1_prefill":
        from .ratio1_prefill_tuner import _run_kloop
        return _run_kloop(args, tactic)
    raise ValueError(f"tactic {tactic} is not valid for {route}")


def _fallback(inputs, policy):
    # The vLLM caller owns the existing CUTLASS and static Fold paths.  Import
    # only when that callback is needed; this module remains standalone.
    from quant.vllm_integration.residue_linear_op import _nvfp4_linear_impl

    return _nvfp4_linear_impl(inputs, policy)


def _record(source):
    # Keep the source available to existing benchmark instrumentation.
    from .sm10x_fold_tuner import _record_source as record_sm10x
    from .sm120_fold_tuner import _record_source as record_sm120
    record_sm10x(source)
    record_sm120(source)


def tuned_nvfp4_linear(inputs, policy):
    if _route(int(inputs[0].shape[0]), policy) == "invalid_mext":
        raise RuntimeError("M-extension requires a prepared Fold path")
    boundaries = {policy.fold_max_m, policy.fold_max_m + 1,
                  policy.ratio1_fold_max_m, policy.ratio1_fold_max_m + 1}
    for switch in (policy.off_min_m, policy.off_max_m):
        if switch is not None:
            boundaries.update((switch, switch + 1))
    buckets = tuple(sorted({b for b in (*_BUCKETS, *boundaries)
                            if 0 < b <= max(_BUCKETS)}))
    # All candidate families share this call and its cold-L2 measurement.
    fold_tuner = policy.sm10x_tuner if policy.arch in ("sm100", "sm103") else policy.sm120_tuner
    enabled = policy.fold_eligible and (fold_tuner or (policy.ratio1 and policy.prefill == "auto"))
    if not enabled:
        return _fallback(inputs, policy)
    from .sm10x_fold_tuner import candidates_hash
    from .sm120_fold_tuner import fold_tiles_hash
    from .ratio1_prefill_tuner import KLOOP_SM100_CFGS, KLOOP_TILES

    cache_policy = policy._replace(candidates=(
        candidates_hash(), fold_tiles_hash(), KLOOP_SM100_CFGS, KLOOP_TILES))
    spec = LinearSpec(
        "nvfp4", cache_policy, buckets, enabled, _fallback, _valid, _run,
        _route, "vllm_cutlass_and_static_fold_v2", _record,
    )
    return tuned_linear(inputs, spec)
