# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Measurement primitives shared by the benchmarks in this directory.

These three functions are the reason a number in this repository can be
believed, so they ship with the numbers rather than living in a private tree.
Each encodes a finding that cost a measurement campaign to learn, and each is
easy to get wrong in a way that produces a plausible answer.

Distilled from the project's internal `_bench_common.py`; only the parts a
published benchmark needs are here.
"""
from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Correctness gates.
#
# A reproducer needs to know what counts as a successful reproduction, so the
# thresholds are constants rather than judgement calls. `reldiff` is a MAXIMUM
# relative error, not a mean -- a single wrong element is a wrong kernel.
#
#   BITWISE       Two spellings of the same arithmetic must agree exactly.
#                 NVFP4 quantisation is invariant to row permutation, so a
#                 fold that reorders rows and a reference that does not are
#                 still required to produce identical bits. Anything above 0.0
#                 here means the two are not, in fact, the same math.
#
#   SAME_FAMILY   Different kernels for the same operation inside this repo
#                 (fold vs two-GEMM vs wrapped K-loop). They differ only in
#                 accumulation order, so the gap is rounding.
#
#   CROSS_IMPL    Our kernel vs an external one (FlashInfer, CUTLASS, cuDNN).
#                 Different tile shapes and split-K choices accumulate in
#                 different orders; agreement to 1e-2 is what "computes the
#                 same function" means across implementations, and a tighter
#                 gate would fail on correct kernels.
# ---------------------------------------------------------------------------
RELDIFF_BITWISE = 0.0
RELDIFF_SAME_FAMILY = 1e-3
RELDIFF_CROSS_IMPL = 1e-2


def timed(fn: Callable[..., Any], args: Sequence[Any] = (),
          *, repeat_iters: int = 25, iters_within_graph: int = 8,
          repeats: int = 5) -> float:
    'Microseconds per call: MIN over `repeats` cudagraph medians, cold L2.'
    from flashinfer.testing.utils import bench_gpu_time_with_cudagraph

    medians = []
    for _ in range(repeats):
        samples = bench_gpu_time_with_cudagraph(
            fn=fn, repeat_iters=repeat_iters,
            num_iters_within_graph=iters_within_graph,
            input_args=list(args), cold_l2_cache=True)
        medians.append(statistics.median(samples))
    return min(medians) * 1000.0            # ms -> us


def timed_detail(fn: Callable[..., Any], args: Sequence[Any] = (),
                 *, repeat_iters: int = 25, iters_within_graph: int = 8,
                 repeats: int = 5) -> dict:
    """`timed`, keeping the individual medians instead of collapsing them.

    The spread between them is the within-process noise floor, and a table
    produced by this module is expected to report it: an effect smaller than
    its own floor is not a result. Returning only the minimum would make that
    impossible to compute after the fact.
    """
    from flashinfer.testing.utils import bench_gpu_time_with_cudagraph

    medians = []
    for _ in range(repeats):
        samples = bench_gpu_time_with_cudagraph(
            fn=fn, repeat_iters=repeat_iters,
            num_iters_within_graph=iters_within_graph,
            input_args=list(args), cold_l2_cache=True)
        medians.append(statistics.median(samples) * 1000.0)
    medians.sort()
    return {
        "us": medians[0],                   # the headline number: min-of-k
        "p50_us": medians[len(medians) // 2],
        "max_us": medians[-1],
        "medians_us": medians,
        "spread_pct": (100.0 * (medians[-1] - medians[0]) / medians[0]
                       if medians[0] else float("nan")),
        "repeats": repeats,
        "repeat_iters": repeat_iters,
        "iters_within_graph": iters_within_graph,
    }


def fi_autotuned(fn: Callable[..., Any], args: Sequence[Any],
                 m_bucket: int, warmup: int = 3) -> None:
    """Let a FlashInfer-backed callable tune for the exact point it is timed at.

    The fairness rule this benchmark holds itself to: **every contestant tunes
    for the exact point it is timed at.** FlashInfer's advantage over a fixed
    kernel is precisely its per-shape search, so timing it with a cold or
    mismatched cache does not measure FlashInfer -- it measures FlashInfer
    handicapped, and every number compared against it inherits the flattery.

    `tuning_buckets=(m_bucket,)` matters: tuning at one M and timing at another
    gives the tactic chosen for a different problem. Tune per M.

    This is the mirror image of not letting our own fold kernel run on its
    static fallback tactic table. Both sides get warmed; both facts get
    recorded.
    """
    import torch
    from flashinfer.autotuner import autotune

    with autotune(True, tuning_buckets=(m_bucket,)):
        for _ in range(warmup):
            fn(*args)
    torch.cuda.synchronize()


def reldiff(out, ref) -> float:
    """Maximum relative error against a reference, the gate metric.

    Compare with `RELDIFF_BITWISE` / `RELDIFF_SAME_FAMILY` /
    `RELDIFF_CROSS_IMPL` above depending on what the two sides are.

    Maximum, not mean: a layout mismatch in a fold kernel can leave most
    elements correct while a minority are catastrophically wrong, and a mean
    absorbs exactly that. A timing-only harness reports such a kernel as a
    win, which is the most expensive failure available to a benchmark.
    """
    import torch

    o = out.float()
    r = ref.float()
    denom = float(r.abs().max())
    if denom == 0.0:
        return float(torch.abs(o).max())
    return float((o - r).abs().max()) / denom
