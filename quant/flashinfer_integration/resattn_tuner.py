"""ResAttn candidates and host calls for the shared FlashInfer tuner scaffold.

FlashInfer owns timing and its winner cache. Compile readiness is recorded
per specialization and device. Persistent scratch
belongs to the caller; profiling keeps only one temporary candidate workspace.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import importlib.metadata
import os

from .fi_tuner_base import make_runner_pair, tuned_call

_OP = "resattn_decode_v2"
_PRECOMPILED = set()
_LAST_SOURCE = [None]
_LAST_PLAN = [None]


def tuner_enabled():
    value = os.environ.get("RESIDUE_ATTN_TUNER", "0")
    if value not in ("0", "1"):
        raise ValueError("RESIDUE_ATTN_TUNER must be 0 or 1")
    return value == "1"


def last_tactic_source():
    return _LAST_SOURCE[0]


def last_decode_plan():
    return _LAST_PLAN[0]


def _record_source(source):
    _LAST_SOURCE[0] = source


@lru_cache(maxsize=1)
def _versions():
    return tuple((name, importlib.metadata.version(name)) for name in
                 ("torch", "flashinfer-python", "nvidia-cutlass-dsl", "nvfp4"))


@lru_cache(maxsize=1)
def _config():
    from flashinfer.autotuner import TuningConfig
    # Static inputs preserve valid page IDs, lengths and borrowed view strides.
    return TuningConfig(use_cuda_graph=True, use_cold_l2_cache=False)


def _compile_key(spec, device):
    return (str(device), spec["args"], tuple(sorted(spec["kwargs"].items())))


def precompile_resattn_variant(spec, device):
    """Callable from the existing vLLM CuTe warmup provider on every worker."""
    import torch
    from nvfp4.cute_attn import host

    key = _compile_key(spec, device)
    if key not in _PRECOMPILED:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("precompile ResAttn before CUDA graph capture")
        with torch.cuda.device(device):
            host.compile_decode_variant(spec)
        _PRECOMPILED.add(key)


def tuned_resattn_decode(q, pool, block_tables, seq_lens, sm_scale, *, out,
                         max_seq_len, residue_fold=False, k_sf_mode="linear",
                         value_scale=1.0, output_scale=None, output_block_scale=None,
                         workspace_getter=None):
    """Tune within FlashInfer's context, otherwise use its cache/default.

    ``workspace_getter(plan)`` supplies persistent scratch for the chosen plan.
    It is not called while profiling; tuning scratch dies with this invocation.
    Callers without a getter use the host's ordinary per-call allocation.
    Metadata must describe initialized KV. This operator never invents pages
    or copies sequence lengths to the CPU to decide a plan.
    """
    import torch
    from flashinfer.autotuner import AutoTuner
    from nvfp4.cute_attn import host
    from nvfp4.cute_attn.plan import decode_plan_candidates

    if max_seq_len <= 0:
        raise ValueError("ResAttn tuning needs a host-side positive max_seq_len")
    packed = output_scale is not None
    if packed != (output_block_scale is not None):
        raise ValueError("packed output requires both output scale buffers")
    batch, heads, dim = q.shape
    hkv = pool.num_kv_heads
    tile = (host._residue_padded_group_size(heads, hkv) if residue_fold
            else host._padded_group_size(heads, hkv))
    policy = os.environ.get("RESIDUE_ATTN_REDUCTION", "legacy")
    plans = decode_plan_candidates(max_seq_len, output_nvfp4=packed,
                                  grouped_head_tile=tile, residue_fold=residue_fold,
                                  reduction_policy=policy)
    fallback = (host.plan_qmul4_decode_nvfp4(
        batch, heads, hkv, max_seq_len, q.device, residue_fold=residue_fold)
        if packed else host.plan_qmul4_decode(batch, hkv, max_seq_len, q.device))
    if policy == "deterministic" and fallback[1] not in ("none", "kernel"):
        fallback = (1, "none")
    
    
    if fallback not in plans:
        plans = (*plans, fallback)
    specs = {plan: host.decode_compile_spec(
        heads, hkv, dim, plan, output_nvfp4=packed, residue_fold=residue_fold,
        k_sf_mode=k_sf_mode) for plan in plans}
    inputs = [q, pool.k_data, pool.v_data, pool.k_sf, pool.v_sf,
              block_tables, seq_lens, out]
    if packed:
        inputs += [output_scale, output_block_scale]
    device = torch.cuda.get_device_properties(q.device)
    extras = (2, _versions(), (device.major, device.minor, device.multi_processor_count),
              int(max_seq_len), hkv, dim, pool.page_size, tile, packed,
              residue_fold, k_sf_mode, policy, plans, fallback,
              tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in inputs))
    tuning = AutoTuner.get().is_tuning_mode
    if tuning and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("autotune ResAttn before CUDA graph capture")
    scratch_plan, scratch = None, None

    def execute(tensors, plan):
        nonlocal scratch_plan, scratch
        qi, kd, vd, ks, vs, table, lengths, output = tensors[:8]
        view = replace(pool, k_data=kd, v_data=vd, k_sf=ks, v_sf=vs)
        splits, reduction = plan
        workspace = None
        if reduction in ("kernel", "gmem_fused"):
            if tuning:
                if scratch_plan != plan:
                    # Warmup visits a tactic before graph profiling. Do not
                    # retain a separate scratch allocation for every tactic.
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("warm up ResAttn tactic before profiling capture")
                    scratch = None
                    allocator = (host.allocate_qmul4_gmem_reduction_workspace
                                 if reduction == "gmem_fused"
                                 else host.allocate_qmul4_reduction_workspace)
                    scratch = allocator(splits, batch, hkv * tile, dim, qi.device)
                    scratch_plan = plan
                workspace = scratch
            elif workspace_getter is not None:
                workspace = workspace_getter(plan)
        kwargs = dict(out=output, max_seq_len=max_seq_len, k_sf_mode=k_sf_mode,
                      value_scale=value_scale, reduction_workspace=workspace,
                      plan=plan, residue_fold=residue_fold)
        if packed:
            result = host.run_paged_fp4_decode_qmul4_tmem_nvfp4(
                qi, view, vd, vs, table, lengths, sm_scale, tensors[8],
                output_block_scale=tensors[9], **kwargs)
        else:
            result = host.run_paged_fp4_decode_qmul4_tmem(
                qi, view, vd, vs, table, lengths, sm_scale,
                direct_output=residue_fold, **kwargs)
        _LAST_PLAN[0] = plan
        return result

    def valid_tactics(tensors, is_tuning):
        if is_tuning:
            for spec in specs.values():
                precompile_resattn_variant(spec, q.device)
        return [i for i, plan in enumerate(plans)
                if _compile_key(specs[plan], q.device) in _PRECOMPILED]

    fallback_forward = lambda tensors: execute(tensors, fallback)
    runners = lambda: make_runner_pair(
        _OP, fallback_forward=fallback_forward, fallback_extras=extras,
        valid_tactics=valid_tactics,
        run_tactic=lambda tensors, i: execute(tensors, plans[i]),
        candidate_extras=lambda tensors: extras, record=_record_source)
    try:
        return tuned_call(_OP, runners, _config, inputs, fallback_forward,
                          record=_record_source)
    finally:
        scratch = None
