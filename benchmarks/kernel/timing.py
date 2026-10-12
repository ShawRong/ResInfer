# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Measurement primitives shared by the benchmarks in this directory.

Official FlashInfer timing, warmup and basic output checks.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from typing import Any

# Match FlashInfer's default minimum; cold inputs may require a longer graph.
ITERS_WITHIN_GRAPH = 10
TIMING_METHOD = "flashinfer_cudagraph_rotation_v2"


def iters_for_cold_l2(args: Sequence[Any], device: Any, *,
                      floor: int = ITERS_WITHIN_GRAPH) -> int:
    """Visit all copies chosen by FlashInfer's cold-L2 rotation policy."""
    import torch
    from flashinfer.testing.utils import calculate_rotation_count

    tensors = [a for a in args if torch.is_tensor(a) and a.is_cuda]
    if not tensors:
        raise ValueError("cold-L2 timing requires explicit CUDA tensor inputs")
    return max(floor, int(calculate_rotation_count(tensors, device)))


def l2_accounting(args: Sequence[Any], device: Any,
                  iters_within_graph: int) -> dict:
    """Record buffer coverage, not a claim about hardware cache residency."""
    import torch
    from flashinfer.testing.utils import calculate_rotation_count

    tensors = [a for a in args if torch.is_tensor(a) and a.is_cuda]
    if not tensors:
        raise ValueError("cold-L2 timing requires explicit CUDA tensor inputs")
    rotations = int(calculate_rotation_count(tensors, device))
    return {
        "arg_mb": sum(t.numel() * t.element_size() for t in tensors) / 1e6,
        "rotations": rotations,
        "visited_rotations": min(rotations, iters_within_graph),
        "rotation_covered": iters_within_graph >= rotations,
        "iters_within_graph": iters_within_graph,
    }


def timed_detail(fn: Callable[..., Any], args: Sequence[Any] = (),
                 *, repeat_iters: int = 25,
                 iters_within_graph: int | None = None,
                 repeats: int = 5) -> dict:
    """Use official Graph timing; retain samples and repeat medians in us.

    Callers comparing arms pass one graph length sufficient for every arm.
    FlashInfer owns buffer cloning, warmup, capture, replay and CUDA events.
    """
    from flashinfer.testing.utils import bench_gpu_time_with_cudagraph
    import torch

    if repeats < 1 or repeat_iters < 1:
        raise ValueError("timing repeat counts must be positive")
    device = next((a.device for a in args if torch.is_tensor(a) and a.is_cuda), None)
    required = iters_for_cold_l2(args, device)
    if iters_within_graph is None:
        iters_within_graph = required
    if iters_within_graph < required:
        raise ValueError(f"graph needs at least {required} calls for buffer rotation")
    samples_us = [
        [float(ms) * 1000.0 for ms in bench_gpu_time_with_cudagraph(
            fn=fn, repeat_iters=repeat_iters,
            num_iters_within_graph=iters_within_graph,
            input_args=list(args), cold_l2_cache=True)]
        for _ in range(repeats)
    ]
    if any(not batch or any(not math.isfinite(t) or t <= 0 for t in batch)
           for batch in samples_us):
        raise ValueError("FlashInfer returned empty or invalid timing samples")
    ordered = [statistics.median(batch) for batch in samples_us]
    medians = sorted(ordered)
    p50 = statistics.median(medians)
    return {
        "timing_method": TIMING_METHOD,
        "us": p50, "us_stat": "p50", "p50_us": p50,
        "us_mink": medians[0], "max_us": medians[-1],
        "samples_us": samples_us,
        "medians_us": medians, "medians_order": ordered,
        "spread_pct": 100.0 * (medians[-1] - medians[0]) / medians[0],
        "repeats": repeats, "repeat_iters": repeat_iters,
        "iters_within_graph": iters_within_graph,
    }


def fi_autotuned(fn: Callable[..., Any], args: Sequence[Any],
                 m_bucket: int, warmup: int = 3) -> None:
    'Implementation details.'
    import torch
    from flashinfer.autotuner import autotune

    with autotune(True, tuning_buckets=(m_bucket,)):
        for _ in range(warmup):
            fn(*args)
    torch.cuda.synchronize()


def warm_equally(fn: Callable[..., Any], args: Sequence[Any] = (),
                 *, calls: int = 400) -> None:
    'Run the same warmup count for each arm, outside the timing window.'
    import torch

    for _ in range(int(calls)):
        fn(*args)
    torch.cuda.synchronize()


def check_output(output, reference, *, max_rel_error=0.25) -> float:
    """Basic shape/finiteness/BF16 sanity gate, outside the timing window."""
    import torch
    if output.shape != reference.shape or not bool(torch.isfinite(output).all()):
        raise ValueError("wrong output shape or non-finite values")
    error = float((output.float() - reference.float()).abs().max()) / (
        float(reference.float().abs().max()) + 1e-9)
    if error > max_rel_error:
        raise ValueError(f"BF16 sanity gate failed: relative max={error:.4f}")
    return error
