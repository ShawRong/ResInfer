"""Run ResAttn samples inside vLLM's FlashInfer tuning lifecycle."""
from __future__ import annotations

from contextlib import contextmanager
import functools
import os

import torch

from .. import diagnostics


def _sample_shapes(runner):
    cfg = runner.vllm_config
    limit = min(cfg.scheduler_config.max_num_seqs,
                cfg.scheduler_config.max_num_batched_tokens)
    batches = {1, limit}
    batches.update(b for b in cfg.compilation_config.cudagraph_capture_sizes or []
                   if 0 < b <= limit)
    length = int(os.environ.get("RESIDUE_ATTN_CG_CAPTURE_LEN", "0")) or cfg.model_config.max_model_len
    if not 0 < length <= cfg.model_config.max_model_len:
        raise ValueError("ResAttn tuning length must fit max_model_len")
    return [(b, length) for b in sorted(batches)]


def _layers(runner):
    from .backend import ResAttnFlashInferImpl

    return [layer for layer in runner.compilation_config.static_forward_context.values()
            if isinstance(getattr(layer, "impl", None), ResAttnFlashInferImpl)]


@contextmanager
def _sample_pages(runner, layers, batch, length):
    """Initialize distinct pages in the unused startup cache and restore tables.

    vLLM's dummy forward supplies lengths/query offsets and suppresses KV
    writes with slot -1. It does not initialize page IDs or cached values.
    This helper is startup-only: no live requests or hybrid cache groups.
    """
    if runner.input_batch.num_reqs:
        raise RuntimeError("ResAttn tuning must run before admitting requests")
    tables = runner.input_batch.block_table.block_tables
    if len(tables) != 1:
        raise NotImplementedError("ResAttn startup tuning requires one KV cache group")
    table = tables[0]
    if table.block_size != 128 or table.use_hybrid_blocks:
        raise ValueError("ResAttn startup tuning requires page128 without block splitting")
    width = (length + 127) // 128
    count = batch * width
    if width > table.max_num_blocks_per_req:
        raise ValueError("ResAttn tuning length exceeds the allocated page table")
    caches = {layer.kv_cache.data_ptr(): layer.kv_cache for layer in layers}
    if any(cache.shape[0] < count for cache in caches.values()):
        raise ValueError("KV capacity cannot hold the ResAttn startup sample")
    saved = table.block_table.cpu[:batch].clone()
    saved_counts = table.num_blocks_per_row[:batch].copy()
    try:
        # Zero packed FP4 values and scale bytes produce finite zero KV. Pages
        # remain distinct so timing sees the sample's actual memory footprint.
        for cache in caches.values():
            cache[:count].zero_()
        for row in range(batch):
            table.add_row(list(range(row * width, (row + 1) * width)), row)
        yield
    finally:
        # CPU buffers participate in async H2D; wait before restoring them.
        torch.cuda.synchronize(runner.device)
        table.block_table.cpu[:batch].copy_(saved)
        table.num_blocks_per_row[:batch] = saved_counts
        table.commit_block_table(batch)
        torch.cuda.synchronize(runner.device)


def _run_samples(runner, dummy_run, layers, shapes, phase):
    from vllm.config import CUDAGraphMode

    records = []
    for batch, length in shapes:
        with _sample_pages(runner, layers, batch, length):
            dummy_run(num_tokens=batch, uniform_decode=True,
                      force_attention=True, profile_seq_lens=length,
                      cudagraph_runtime_mode=CUDAGraphMode.NONE,
                      allow_microbatching=False, skip_eplb=True, is_profile=True)
            statuses = [getattr(layer.impl, "_decode_tuning_status", None)
                        for layer in layers]
            if any(status is None or status[2:] != (batch, length)
                   for status in statuses):
                raise RuntimeError("ResAttn startup sample did not execute every decode layer")
            if phase == "prepare" and any(status[1] not in
                    ("autotuned", "autotuned_fallback") for status in statuses):
                raise RuntimeError("ResAttn startup sample missed its tuned cache entry")
            records.append(dict(batch=batch, length=length, layers=len(layers),
                                selections=statuses))
    diagnostics.log_tuning_report(phase, records)
    return records


def install_resattn_autotune_warmup():
    """Extend the existing cache context; never start tuning during capture."""
    import vllm.model_executor.warmup.kernel_warmup as kw
    from .warmup import _DECODE_WARMUP_PROVIDER

    if getattr(kw, "_resattn_tuning_installed", False):
        return
    original = kw.flashinfer_autotune

    @functools.wraps(original)
    def with_resattn(runner):
        from quant.flashinfer_integration.resattn_tuner import tuner_enabled

        layers = _layers(runner)
        if not tuner_enabled() or not layers:
            return original(runner)
        if runner.uniform_decode_query_len != 1:
            raise NotImplementedError("ResAttn tuning requires single-token decode")
        shapes = _sample_shapes(runner)
        # Disk-cache hits can skip candidate compilation. Every worker needs
        # local compiled variants before consuming the shared winner cache.
        for unit in _DECODE_WARMUP_PROVIDER.get_cutedsl_warmup_compile_units():
            unit.compile()
        dummy = runner._dummy_run
        done = False

        def with_samples(*args, **kwargs):
            nonlocal done
            result = dummy(*args, **kwargs)
            if not done and kwargs.get("is_profile"):
                done = True
                # All workers execute matching model forwards, including
                # non-tuning ranks, so model collectives stay synchronized.
                # AutoTuner's enclosing context owns whether profiling occurs.
                with torch.inference_mode():
                    _run_samples(runner, dummy, layers, shapes, "tune")
            return result

        runner._dummy_run = with_samples
        try:
            original(runner)
        finally:
            runner._dummy_run = dummy
        if not done:
            raise RuntimeError("vLLM FlashInfer warmup did not invoke its profiling forward")
        # This is outside the tuning context and after cache loading/broadcast.
        # The ordinary decode path now allocates only the selected persistent
        # workspaces, before vLLM begins capturing model graphs.
        with torch.inference_mode():
            runner._resattn_tuning_report = _run_samples(
                runner, dummy, layers, shapes, "prepare")

    kw.flashinfer_autotune = with_resattn
    kw._resattn_tuning_installed = True
