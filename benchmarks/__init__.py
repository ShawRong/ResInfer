"""Public reproduction methods for ResInfer's published numbers.

Two independent halves, no cross-imports:

- kernel speed: ``bench_residue_nvfp4.py`` (timing primitives in
  ``_common.py`` -- cudagraph min-of-k, per-M autotune, reldiff gates).
- model quality: ``evalkit.py`` (LongBench / teacher-forced PPL /
  downstream; run ``python -m benchmarks.evalkit selftest`` first).

Package marker so ``python -m benchmarks.<half>`` works from the repo root.
"""
