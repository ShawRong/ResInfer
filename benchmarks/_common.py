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


def assert_args_consumed(fn: Callable[..., Any], args: Sequence[Any],
                         *, label: str = "") -> None:
    'Fail loudly when a timed callable ignores the buffers handed to it.'
    import torch

    idxs = [i for i, a in enumerate(args)
            if torch.is_tensor(a) and a.is_cuda]
    if not idxs:
        return
    ref = fn(*args)
    if not torch.is_tensor(ref):
        return                          # cannot decide; do not pretend to
    ref = ref.clone()

    ignored = []
    for i in idxs:
        probe = list(args)
        # Zero, not negate: the arguments span bf16 activations, uint8-packed
        # FP4 weights and FP8 scale factors, and `neg` is not defined on the
        # last of those. Zero is a legal encoding in all three and drives the
        # output to zero (or to NaN, if a scale is divided by) -- either way it
        # moves, which is all this needs.
        probe[i] = torch.zeros_like(args[i])
        try:
            got = fn(*probe)
        except Exception:
            continue                    # it looked at the tensor; that is the
            #                             question being asked here
        if torch.is_tensor(got) and torch.equal(ref, got):
            ignored.append(i)
    fn(*args)                           # restore state the callable mutated

    if ignored:
        raise RuntimeError(
            f"timed callable {label or fn!r} ignores input_args at index "
            f"{ignored}: perturbing them did not change the output. "
            f"cold_l2_cache rotates exactly these buffers, so they would be "
            f"cloned and never read, and every measurement here would be a "
            f"warm-L2 timing wearing a cold-L2 label. Take the tensors as "
            f"parameters instead of reading them from the closure.")


def kernels_per_call(fn: Callable[..., Any], args: Sequence[Any],
                     *, reps: int = 5) -> float | None:
    'CUDA kernel launches inside ONE invocation of a timed callable.'
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    try:
        fn(*args)                       # warm: never count a JIT
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn(*args)
            torch.cuda.synchronize()
        n = sum(e.count for e in prof.key_averages()
                if e.device_type == DeviceType.CUDA
                and e.self_device_time_total > 0)
        return n / reps
    except Exception:
        return None                     # an instrument, never a blocker


def l2_accounting(args: Sequence[Any], device: Any) -> dict:
    """How the rotating buffers landed, recorded per row rather than recalled.

    `calculate_rotation_count` returns 1 when the arguments already exceed L2,
    and the helper then disables rotation because the cache is cold anyway. So
    `rotations == 1` and `rotations > 1` are both correct and mean opposite
    things; a row cannot be read without knowing which it was.
    """
    import torch
    from flashinfer.testing.utils import calculate_rotation_count

    tensors = [a for a in args if torch.is_tensor(a) and a.is_cuda]
    if not tensors:
        return {"arg_mb": 0.0, "rotations": 0, "l2_resident": None}
    nbytes = sum(t.numel() * t.element_size() for t in tensors)
    try:
        rot = int(calculate_rotation_count(tensors, device))
    except Exception:                   # helper is test-only; never fatal
        rot = -1
    return {
        "arg_mb": nbytes / 1e6,
        "rotations": rot,
        # True  -> one copy fits, rotation is what keeps it cold
        # False -> the working set alone exceeds L2; cold either way
        "l2_resident": (rot > 1) if rot > 0 else None,
    }


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


def pid_identity(pid: int) -> dict:
    'Who is that neighbour? Read /proc AT DETECTION TIME, not later.'
    import os
    import pathlib

    out: dict = {"pid": int(pid)}
    base = pathlib.Path(f"/proc/{int(pid)}")
    try:
        raw = (base / "cmdline").read_bytes()
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        if cmd:
            out["cmdline"] = cmd[:300]
    except OSError:
        pass
    try:
        # os.readlink, not Path.resolve(): resolve() is non-strict by default
        # and happily returns "/proc/<dead pid>/cwd" for a process that no
        # longer exists -- a plausible-looking string standing in for a
        # missing fact, which is the one thing this helper must not produce.
        out["cwd"] = os.readlink(str(base / "cwd"))
    except OSError:
        pass
    try:
        # Field 22 of /proc/pid/stat is starttime in clock ticks since boot.
        # comm may contain spaces/parens, so split after the last ')'.
        stat = (base / "stat").read_text()
        fields = stat[stat.rfind(")") + 1:].split()
        out["start_ticks"] = int(fields[19])
    except (OSError, IndexError, ValueError):
        pass
    if len(out) == 1:
        out["note"] = "process already exited -- identity unrecoverable"
    return out
