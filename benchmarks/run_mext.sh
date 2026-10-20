#!/usr/bin/env bash
# Full MEXT latency matrix: 9 arms, 5 processes.
# Usage: bash benchmarks/run_mext.sh --gpu 0 --outdir /local/results/mext-run
# Add --dry-run to inspect; all remaining options go to the existing sweep.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${PYTHON:-python3}" -m benchmarks.kernel.mext_paper_sweep \
  "$@"
