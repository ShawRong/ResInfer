#!/usr/bin/env bash
# setup-export-venv.sh — Provision the FP4 export venv at env-export/.venv.
#
# Usage:
#   source ./env.sh
#   ./setup-export-venv.sh
#
# Most of the venv is built declaratively by `uv sync --project env-export`
# from env-export/pyproject.toml — including nvidia-modelopt==0.45.0rc1 and
# tensorrt-llm 1.1.0, which uses a [[tool.uv.dependency-metadata]] override to
# strip its modelopt~=0.37 pin.
#
# This script then layers in nvfp4 (csrc), which needs install-time flags
# uv cannot express in pyproject.toml:
#   - --no-build-isolation (so the build sees torch)
#   - non-editable (avoids sharing the built .so with the main .venv, which
#     uses a different torch ABI)
#   - quant/export-only by default (no local CUTLASS GEMM), because this venv
#     is for ModelOpt calibration/export and TRT-LLM pins old FlashInfer headers.
#
# Idempotent: re-running upgrades/repairs the venv in place.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$REPO_ROOT/env-export"
VENV="$PROJECT_DIR/.venv"
PY="$VENV/bin/python"

echo "[setup-export-venv] Project: $PROJECT_DIR"
echo "[setup-export-venv] Target:  $VENV"

# ── 1. Bulk install via uv sync (torch + modelopt + trtllm + HF stack) ──
echo "[setup-export-venv] Step 1/2: uv sync --project env-export"
uv sync --project "$PROJECT_DIR"

# ── 2. nvfp4 local CUDA kernel (non-editable, isolated from main .venv) ──
echo "[setup-export-venv] Step 2/2: nvfp4 (non-editable, --no-build-isolation)"
#
# Keep the extension build independent from the caller's interactive shell.
# In particular, env.sh may export a broad TORCH_CUDA_ARCH_LIST that includes
# sm_100; nvfp4 kernels use FP4 PTX forms that are only valid for the Blackwell
# arch-specific targets we compile through setup.py. The 5090 export env only
# needs sm_120 here.
NVFP4_CUDA_HOME="${NVFP4_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}"
NVFP4_TOOLCHAIN_PREFIX="${NVFP4_TOOLCHAIN_PREFIX:-${CONDA_PREFIX:-}}"
NVFP4_CC="${NVFP4_CC:-${NVFP4_TOOLCHAIN_PREFIX:+$NVFP4_TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-gcc}}"
if [[ -z "$NVFP4_CC" || ! -x "$NVFP4_CC" ]]; then
    NVFP4_CC="$(command -v x86_64-conda-linux-gnu-gcc 2>/dev/null || command -v gcc 2>/dev/null || true)"
fi
NVFP4_CXX="${NVFP4_CXX:-${NVFP4_TOOLCHAIN_PREFIX:+$NVFP4_TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-g++}}"
if [[ -z "$NVFP4_CXX" || ! -x "$NVFP4_CXX" ]]; then
    NVFP4_CXX="$(command -v x86_64-conda-linux-gnu-g++ 2>/dev/null || command -v g++ 2>/dev/null || true)"
fi
NVFP4_TORCH_CUDA_ARCH_LIST="${NVFP4_TORCH_CUDA_ARCH_LIST:-12.0}"
NVFP4_CUDAARCHS="${NVFP4_CUDAARCHS:-120-real}"
NVFP4_BUILD_ARCHS="${NVFP4_BUILD_ARCHS:-sm120}"
NVFP4_BUILD_GEMM="${NVFP4_BUILD_GEMM:-0}"

for tool in "$NVFP4_CUDA_HOME/bin/nvcc" "$NVFP4_CC" "$NVFP4_CXX"; do
    if [[ ! -x "$tool" ]]; then
        echo "[setup-export-venv] Missing required build tool: $tool" >&2
        exit 1
    fi
done

NVFP4_BUILD_PATH="$NVFP4_CUDA_HOME/bin:$PATH"
if [[ -n "$NVFP4_TOOLCHAIN_PREFIX" ]]; then
    NVFP4_BUILD_PATH="$NVFP4_TOOLCHAIN_PREFIX/bin:$NVFP4_BUILD_PATH"
fi

PATH="$NVFP4_BUILD_PATH" \
CC="$NVFP4_CC" \
CXX="$NVFP4_CXX" \
TORCH_CUDA_ARCH_LIST="$NVFP4_TORCH_CUDA_ARCH_LIST" \
CUDAARCHS="$NVFP4_CUDAARCHS" \
NVFP4_BUILD_ARCHS="$NVFP4_BUILD_ARCHS" \
NVFP4_BUILD_GEMM="$NVFP4_BUILD_GEMM" \
uv pip install --python "$PY" \
    "$REPO_ROOT/csrc" \
    --no-build-isolation \
    --no-deps

# ── Verify ────────────────────────────────────────────────────────────────
echo ""
echo "[setup-export-venv] Verifying imports..."
"$PY" - <<'PY'
import torch
print(f"  torch        {torch.__version__}")

import tensorrt_llm
print(f"  trtllm       {tensorrt_llm.__version__}")

import tensorrt_llm._torch  # registers torch.ops.trtllm.*
assert hasattr(torch.ops.trtllm, "fp4_quantize"), "fp4_quantize op missing"
assert hasattr(torch.ops.trtllm, "nvfp4_gemm"),   "nvfp4_gemm op missing"
print("  trtllm ops   fp4_quantize, nvfp4_gemm OK")

import modelopt
print(f"  modelopt     {modelopt.__version__}")

import transformers
print(f"  transformers {transformers.__version__}")

import nvfp4
print("  nvfp4        OK")
assert hasattr(torch.ops.nvfp4, "scaled_fp4_quant"), "scaled_fp4_quant op missing"
assert hasattr(torch.ops.nvfp4, "scaled_fp4_quant_mext_r1"), "scaled_fp4_quant_mext_r1 op missing"
print("  nvfp4 ops    quant OK")

from modelopt.torch.quantization.backends.nvfp4_gemm import Nvfp4Linear
print("  Nvfp4Linear  OK")

# Sanity-check gptq() availability in the export ModelOpt build.
try:
    from modelopt.torch.quantization.model_calib import gptq
    print("  modelopt.gptq()  OK")
except ImportError:
    print("  modelopt.gptq()  NOT FOUND (commit may be too old)")
PY

echo ""
echo "[setup-export-venv] Done. Reference this venv in suite YAML:"
echo "    execution:"
echo "      launcher: direct"
echo "      python_executable: $VENV/bin/python"
