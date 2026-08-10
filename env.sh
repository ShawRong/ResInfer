# Runtime: CUDA libraries for torch/vllm
# cu130 torch uses nvidia/cu13/lib for cudart; cu12 packages for cudnn/nccl/cusparselt/nvshmem
_VENV="$PWD/.venv/lib/python3.12/site-packages"
if [ -d /usr/local/cuda-13.0 ]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
    export PATH="$CUDA_HOME/bin:${PATH:-}"
fi
export LD_LIBRARY_PATH="\
${CUDA_HOME:+${CUDA_HOME}/lib64:}\
${_VENV}/nvidia/cu13/lib:\
${_VENV}/nvidia/cuda_runtime/lib:\
${_VENV}/nvidia/cudnn/lib:\
${_VENV}/nvidia/nccl/lib:\
${_VENV}/nvidia/cublas/lib:\
${_VENV}/nvidia/cusparselt/lib:\
${_VENV}/cusparselt/lib:\
${_VENV}/nvidia/nvshmem/lib:\
${LD_LIBRARY_PATH:-}"
unset _VENV

# Repo-local `sitecustomize.py` is used to auto-patch vLLM spawn workers when
# `RESIDUE_PATCH_VLLM=1` is set. Ensure repo root is importable.
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Torch CUDA extensions are sensitive to conda build flags. If this shell was
# launched from a conda/micromamba compiler environment, clear its injected
# flags and use the system C++ compiler that matches PyTorch Linux wheels.
if [ -n "${CONDA_PREFIX:-}" ]; then
    case "${CC:-}${CXX:-}${CFLAGS:-}${CPPFLAGS:-}${CXXFLAGS:-}${LDFLAGS:-}" in
        *"$CONDA_PREFIX"*|*x86_64-conda-linux-gnu*)
            unset CFLAGS CPPFLAGS CXXFLAGS LDFLAGS
            unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH CPATH LIBRARY_PATH
            ;;
    esac
fi
_RESINFER_GCC11="$(command -v gcc-11 2>/dev/null || true)"
_RESINFER_GXX11="$(command -v g++-11 2>/dev/null || true)"
_RESINFER_GCC="$(command -v gcc 2>/dev/null || true)"
_RESINFER_GXX="$(command -v g++ 2>/dev/null || true)"
if [ -z "${CC:-}" ] && [ -z "${CXX:-}" ] && [ -n "$_RESINFER_GCC11" ] && [ -n "$_RESINFER_GXX11" ]; then
    export CC="$_RESINFER_GCC11"
    export CXX="$_RESINFER_GXX11"
    export CUDAHOSTCXX="${CUDAHOSTCXX:-$_RESINFER_GXX11}"
elif [ -z "${CC:-}" ] && [ -z "${CXX:-}" ] && [ -n "$_RESINFER_GCC" ] && [ -n "$_RESINFER_GXX" ]; then
    export CC="$_RESINFER_GCC"
    export CXX="$_RESINFER_GXX"
    export CUDAHOSTCXX="${CUDAHOSTCXX:-$_RESINFER_GXX}"
fi
unset _RESINFER_GCC11 _RESINFER_GXX11 _RESINFER_GCC _RESINFER_GXX







if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/include" ]; then
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
fi
# NVIDIA conda packages use targets/x86_64-linux/{include,lib} layout
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/targets/x86_64-linux/include" ]; then
    export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
    export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
fi
# Fallback: system libcuda.so (driver library)
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
