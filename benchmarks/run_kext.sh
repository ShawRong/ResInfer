#!/usr/bin/env bash
# Selective KEXT + ratio1 auto + stock/W4A8 latency; synthetic per-8 masks.
# Usage: bash benchmarks/run_kext.sh --gpu 0 --rep 0 --out /local/results/kext-rep0.jsonl
# One process per invocation; use distinct --rep/--out for independent repetitions.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${PYTHON:-python3}" -m benchmarks.kernel.kext_paper_sweep --family kext "$@"
