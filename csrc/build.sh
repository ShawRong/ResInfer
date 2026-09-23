#!/usr/bin/env bash
# build.sh — Build nvfp4 _C.*.so against a target venv's torch.
#
# Direct setuptools build_ext --inplace; does NOT call uv (avoids implicit
# `uv sync` that would re-evaluate the venv before we even start the build).
# Idempotent: removes prior build artifacts then rebuilds the .so in place.
#
# Default target is the repository's main .venv when present, otherwise the
# active virtualenv or python3. Override via env:
#
#   # build into env-export/.venv (different torch ABI)
#   PY=$(realpath ../env-export/.venv/bin/python) ./build.sh
#
#   # multi-arch build (sm_100 + sm_120 + sm_90)
#   NVFP4_TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0" \
#   NVFP4_CUDAARCHS="90-real;100-real;120-real" \
#     ./build.sh
#
# Inputs (all optional):
#   PY                          target venv python
#   NVFP4_CUDA_HOME             CUDA toolkit root; otherwise infer from CUDA_HOME/nvcc
#   NVFP4_TOOLCHAIN_PREFIX      conda/micromamba toolchain prefix
#   NVFP4_CC                    C compiler; otherwise conda gcc or system gcc
#   NVFP4_CXX                   C++ compiler; otherwise conda g++ or system g++
#   NVFP4_TORCH_CUDA_ARCH_LIST  12.0
#   NVFP4_CUDAARCHS             120-real

set -euo pipefail

KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$KERNEL_DIR/.." && pwd)"

find_executable() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 0
}

find_in_path() {
    local name="$1"
    command -v "$name" 2>/dev/null || true
}

if [[ -z "${PY:-}" ]]; then
    PY="$(find_executable \
        "$REPO_ROOT/.venv/bin/python" \
        "${VIRTUAL_ENV:-}/bin/python" \
        "$(find_in_path python3)" \
        "$(find_in_path python)")"
fi

if [[ -z "${NVFP4_CUDA_HOME:-}" ]]; then
    if [[ -n "${CUDA_HOME:-}" ]]; then
        NVFP4_CUDA_HOME="$CUDA_HOME"
    elif [[ -n "$(find_in_path nvcc)" ]]; then
        NVFP4_CUDA_HOME="$(cd "$(dirname "$(dirname "$(find_in_path nvcc)")")" && pwd)"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        NVFP4_CUDA_HOME="/usr/local/cuda"
    elif [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
        NVFP4_CUDA_HOME="/usr/local/cuda-13.0"
    else
        NVFP4_CUDA_HOME=""
    fi
fi

NVFP4_TOOLCHAIN_PREFIX="${NVFP4_TOOLCHAIN_PREFIX:-${CONDA_PREFIX:-}}"
NVFP4_CC="${NVFP4_CC:-$(find_executable \
    "$NVFP4_TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
    "$(find_in_path x86_64-conda-linux-gnu-gcc)" \
    "$(find_in_path gcc)")}"
NVFP4_CXX="${NVFP4_CXX:-$(find_executable \
    "$NVFP4_TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
    "$(find_in_path x86_64-conda-linux-gnu-g++)" \
    "$(find_in_path g++)")}"
NVFP4_TORCH_CUDA_ARCH_LIST="${NVFP4_TORCH_CUDA_ARCH_LIST:-12.0}"
NVFP4_CUDAARCHS="${NVFP4_CUDAARCHS:-120-real}"

if [[ -z "$PY" ]]; then
    echo "[build] Could not find a Python interpreter. Set PY=/path/to/python." >&2
    exit 1
fi
if [[ -z "$NVFP4_CUDA_HOME" ]]; then
    echo "[build] Could not find nvcc. Set NVFP4_CUDA_HOME or CUDA_HOME." >&2
    exit 1
fi
if [[ -z "$NVFP4_CC" || -z "$NVFP4_CXX" ]]; then
    echo "[build] Could not find gcc/g++. Set NVFP4_CC and NVFP4_CXX, or activate a toolchain env." >&2
    exit 1
fi

echo "[build] kernel dir: $KERNEL_DIR"
echo "[build] python:     $PY"
echo "[build] cuda:       $NVFP4_CUDA_HOME"
echo "[build] toolchain:  ${NVFP4_TOOLCHAIN_PREFIX:-system PATH}"
echo "[build] cc:         $NVFP4_CC"
echo "[build] cxx:        $NVFP4_CXX"
echo "[build] arch:       sm_$NVFP4_TORCH_CUDA_ARCH_LIST  ($NVFP4_CUDAARCHS)"

for t in "$NVFP4_CUDA_HOME/bin/nvcc" "$NVFP4_CC" "$NVFP4_CXX" "$PY"; do
    if [[ ! -x "$t" ]]; then
        echo "[build] missing required tool: $t" >&2
        exit 1
    fi
done

"$PY" -c "import torch; print(f'[build] torch:      {torch.__version__} (cuda {torch.version.cuda})')"

cd "$KERNEL_DIR"
echo "[build] cleaning previous artifacts..."
rm -rf build/ nvfp4/_C*.so nvfp4.egg-info/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "[build] running setup.py build_ext --inplace ..."
BUILD_PATH="$NVFP4_CUDA_HOME/bin:$PATH"
if [[ -n "$NVFP4_TOOLCHAIN_PREFIX" ]]; then
    BUILD_PATH="$NVFP4_TOOLCHAIN_PREFIX/bin:$BUILD_PATH"
fi

PATH="$BUILD_PATH" \
CC="$NVFP4_CC" \
CXX="$NVFP4_CXX" \
TORCH_CUDA_ARCH_LIST="$NVFP4_TORCH_CUDA_ARCH_LIST" \
CUDAARCHS="$NVFP4_CUDAARCHS" \
    "$PY" setup.py build_ext --inplace

echo "[build] built artifacts:"
ls -lh nvfp4/_C*.so
echo
echo "[build] verifying import in $PY ..."
# The .so uses TORCH_LIBRARY (loaded via torch.ops.load_library in __init__.py),
# not pybind PYBIND11_MODULE, so it has no PyInit_ entry — `import nvfp4._C`
# would fail. Verify via the package and check that representative ops resolved.
"$PY" -c "
import nvfp4, torch
print(f'[build] OK — nvfp4 from {nvfp4.__file__}')
expected = [
    'scaled_fp4_quant_mext_r1',
    'scaled_fp4_quant',
    'official_scaled_fp4_quant_pack8',
    'mxfp4_residue_quant',
    'nvfp4_quantize_q_residue',
]
missing = [op for op in expected if not hasattr(torch.ops.nvfp4, op)]
if missing:
    raise SystemExit(f'[build] FAIL — missing torch.ops.nvfp4 ops: {missing}')
print(f'[build] torch.ops.nvfp4 has expected ops')
"
