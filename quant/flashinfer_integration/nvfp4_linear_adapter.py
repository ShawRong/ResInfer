# SPDX-License-Identifier: Apache-2.0
"""NVFP4 candidate routing inside the shared Linear AutoTuner call.

The policy is made from per-layer constants by the caller.  Route selection
uses the activation M synthesized by FlashInfer, rather than the M of vLLM's
outer dummy forward.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from .linear_tuner import LINEAR_BUCKETS, LinearSpec, tuned_linear


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
    prefill: str
    sm10x_tuner: bool
    sm120_tuner: bool
    candidates: tuple = ()


def _route(m, policy):
    """Choose the numerical path for both synthesized and serving M."""
    fold = policy.fold_eligible and policy.arch in ("sm100", "sm120") and (
        policy.ratio1 or policy.force_fold or m <= policy.fold_max_m)
    if fold:
        if policy.ratio1 and m > policy.ratio1_fold_max_m:
            return "ratio1_prefill"
        return "sm10x_fold" if policy.arch == "sm100" else "sm120_fold"
    if policy.ratio1:
        return "invalid_mext"
    return "plain" if policy.num_salient == 0 else "k_ext"


def _fold_inputs(inputs, policy):
    x, weight, scale_inv, _full_sf, base_sf, _mask, alpha = inputs
    k_base = policy.k_base
    packed_k = k_base // 2
    if packed_k > weight.shape[1]:
        raise ValueError("base K exceeds packed weight width")
    view = weight if packed_k == weight.shape[1] else weight[:, :packed_k]
    return (x, view, scale_inv, base_sf, alpha)


def _valid(inputs, policy, tuning):
    # FlashInfer profiles candidates after eager warmup calls. vLLM also runs
    # each graph size eagerly before capture, including when the FI cache hits.
    # Let those calls load the chosen kernel; capture only uses resident ones.
    allow_uncompiled = not torch.cuda.is_current_stream_capturing()
    route = _route(int(inputs[0].shape[0]), policy)
    if route == "sm10x_fold":
        if not policy.sm10x_tuner:
            return []
        from .sm100_fold_candidates import valid_tactics
        return valid_tactics(_fold_inputs(inputs, policy), inputs[0].dtype, allow_uncompiled)
    if route == "sm120_fold":
        if not policy.sm120_tuner:
            return []
        from .sm120_fold_candidates import valid_tactics
        return valid_tactics(_fold_inputs(inputs, policy), inputs[0].dtype, allow_uncompiled)
    if route == "ratio1_prefill" and policy.prefill == "auto":
        from .ratio1_prefill_candidates import _kloop_valid_tactics
        return _kloop_valid_tactics(_fold_inputs(inputs, policy), allow_uncompiled)
    return []


def _run(inputs, policy, tactic):
    route = _route(int(inputs[0].shape[0]), policy)
    args = _fold_inputs(inputs, policy)
    if route == "sm10x_fold":
        from .sm100_fold_candidates import _run_tactic
        return _run_tactic(args, tactic, inputs[0].dtype)
    if route == "sm120_fold":
        from .sm120_fold_candidates import _run_fold_tile, FOLD_TILES
        return _run_fold_tile(args, FOLD_TILES[tactic], inputs[0].dtype)
    if route == "ratio1_prefill":
        from .ratio1_prefill_candidates import _run_kloop
        return _run_kloop(args, tactic)
    raise ValueError(f"tactic {tactic} is not valid for {route}")


def _fallback(inputs, policy):
    # The vLLM caller owns the existing CUTLASS and static Fold paths.  Import
    # only when that callback is needed; this module remains standalone.
    from quant.vllm_integration.residue_linear_op import _nvfp4_linear_impl

    return _nvfp4_linear_impl(inputs, policy)


def _record(source):
    # Keep the source available to existing benchmark instrumentation.
    from .sm100_fold_candidates import _record_source as record_sm10x
    from .sm120_fold_candidates import _record_source as record_sm120
    record_sm10x(source)
    record_sm120(source)


def tuned_nvfp4_linear(inputs, policy):
    if _route(int(inputs[0].shape[0]), policy) == "invalid_mext":
        raise RuntimeError("M-extension requires a prepared Fold path")
    boundaries = {policy.fold_max_m, policy.fold_max_m + 1,
                  policy.ratio1_fold_max_m, policy.ratio1_fold_max_m + 1}
    buckets = tuple(sorted({b for b in (*LINEAR_BUCKETS, *boundaries)
                            if 0 < b <= max(LINEAR_BUCKETS)}))
    # All candidate families share this call and its cold-L2 measurement.
    fold_tuner = policy.sm10x_tuner if policy.arch == "sm100" else policy.sm120_tuner
    enabled = (policy.fold_eligible and policy.arch in ("sm100", "sm120")
               and (fold_tuner or (policy.ratio1 and policy.prefill == "auto")))
    if not enabled:
        return _fallback(inputs, policy)
    from .sm100_fold_candidates import candidates_hash
    from .sm120_fold_candidates import fold_tiles_hash
    from .ratio1_prefill_candidates import KLOOP_SM100_CFGS, KLOOP_TILES

    cache_policy = policy._replace(candidates=(
        candidates_hash(), fold_tiles_hash(), KLOOP_SM100_CFGS, KLOOP_TILES))
    spec = LinearSpec(
        "nvfp4", cache_policy, buckets, enabled, _fallback, _valid, _run,
        _route, "vllm_cutlass_and_static_fold_v2", _record,
    )
    return tuned_linear(inputs, spec)
