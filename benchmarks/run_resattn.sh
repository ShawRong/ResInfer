#!/usr/bin/env bash
# ResAttn vs stock TRT-LLM decode chain.
# Usage: CUDA_VISIBLE_DEVICES=0 bash benchmarks/run_resattn.sh \
#   --batch 8 --seq 32768 --gqa 4 --kv-heads 8 > /local/results/resattn-run.jsonl
# BF16 output by default; override with --output-contract nvfp4 if needed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${PYTHON:-python3}" -m benchmarks.resattn.stock_ab \
  --output-contract bf16 --vllm-kv-layout \
  --value-scale 256 --output-scale-inv 256 "$@"
