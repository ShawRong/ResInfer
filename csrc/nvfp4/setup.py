import os
import subprocess
import torch
from setuptools import find_packages, setup
from torch.utils import cpp_extension

# Get CUDA version
def get_cuda_version():
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        for line in nvcc_output.split('\n'):
            if 'release' in line.lower():
                version = line.split('release')[-1].strip().split(',')[0].strip()
                return tuple(map(int, version.split('.')))
    except Exception:
        return (0, 0)
    return (0, 0)

# Check compute capability
cuda_version = get_cuda_version()
cuda_major, cuda_minor = cuda_version


def first_existing_path(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None

# Define compile flags
extra_compile_args = {
    'cxx': ['-std=c++17', '-O3'],
    'nvcc': [
        '-std=c++17',
        '-O3',
        '-DNDEBUG',
        '--use_fast_math',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT16_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT162_OPERATORS__',
        '-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
        '-diag-suppress', '177',  # Suppress unused variable warnings
        # Match vLLM's native CUTLASS/TMA path: call cuTensorMapEncodeTiled
        # directly instead of resolving it through cudaGetDriverEntryPoint* on
        # every GEMM initialize.
        '-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1',
        # '-DNVFP4_DEBUG_SWAP',  # Enable swap debug output (disabled for production)
    ]
}

if os.getenv("NVFP4_ENABLE_LINEINFO", "0") == "1":
    extra_compile_args["nvcc"].append("-lineinfo")
if os.getenv("NVFP4_ROWPAIR_SMOKE_ONLY", "0") == "1":
    extra_compile_args["nvcc"].append("-DNVFP4_ROWPAIR_SMOKE_ONLY=1")


# Enable NVFP4 for Blackwell (FP4 E2M1) architectures.
# By default we build a fatbin that includes:
#   - SM100a (B100/B200 datacenter Blackwell)
#   - SM103a (B300 Blackwell Ultra) -- "a" targets are single-arch: sm_100a
#     cubins do NOT load on sm_103, so B300 needs its own entry.
#   - SM120a (RTX 5090 / RTX PRO 6000 Blackwell)
# For remote one-off benchmarking, set NVFP4_BUILD_ARCHS=sm120 (etc.) to
# reduce nvcc build time without changing the default local build.
if cuda_major > 12 or (cuda_major == 12 and cuda_minor >= 8):
    default_archs = "sm100,sm120" if cuda_major < 13 else "sm100,sm103,sm120"
    requested_archs = {
        item.strip().lower().replace("_", "")
        for item in os.getenv("NVFP4_BUILD_ARCHS", default_archs).split(",")
        if item.strip()
    }
    if not requested_archs <= {"sm100", "sm103", "sm120"}:
        raise RuntimeError(
            "NVFP4_BUILD_ARCHS must be a comma-separated subset of: "
            "sm100,sm103,sm120"
        )
    if "sm103" in requested_archs and cuda_major < 13:
        raise RuntimeError("sm103 requires CUDA 13+")
    if "sm100" in requested_archs:
        extra_compile_args['nvcc'].extend([
            '-gencode', 'arch=compute_100a,code=sm_100a',
            '-DENABLE_NVFP4_SM100=1',
        ])
    if "sm103" in requested_archs:
        extra_compile_args['nvcc'].extend([
            '-gencode', 'arch=compute_103a,code=sm_103a',
            '-DENABLE_NVFP4_SM100=1',
        ])
    if "sm120" in requested_archs:
        extra_compile_args['nvcc'].extend([
            '-gencode', 'arch=compute_120a,code=sm_120a',
            '-DENABLE_NVFP4_SM120=1',
        ])
else:
    print(f"Warning: CUDA version {cuda_major}.{cuda_minor} < 12.8, NVFP4 kernels disabled")

# Source files for quantization plus the local CUTLASS FP4 GEMM/fold kernels.
sources = [
    'src/pybind.cpp',  # Python bindings
    'src/cuda_utils_kernels.cu',
    'src/fp4/nvfp4_quant_entry.cu',
    'src/fp4/nvfp4_quant_kernels.cu',
    'src/fp4/nvfp4_residue_pack8.cu',
    'src/fp4/nvfp4_residue_pack16.cu',
    'src/fp4/official_nvfp4_quant_kernels.cu',
    'src/attn/nvfp4_quantize_q.cu',
]

cutlass_include_dir = first_existing_path(
    os.getenv("CUTLASS_INCLUDE_DIR"),
    os.path.join(
        os.path.dirname(torch.__file__),
        '..',
        'flashinfer',
        'data',
        'cutlass',
        'include',
    ),
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        '.venv',
        'lib',
        'python3.12',
        'site-packages',
        'flashinfer',
        'data',
        'cutlass',
        'include',
    ),
)
cutlass_tools_include_dir = first_existing_path(
    os.getenv("CUTLASS_TOOLS_INCLUDE_DIR"),
    os.path.join(
        os.path.dirname(torch.__file__),
        '..',
        'flashinfer',
        'data',
        'cutlass',
        'tools',
        'util',
        'include',
    ),
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        '.venv',
        'lib',
        'python3.12',
        'site-packages',
        'flashinfer',
        'data',
        'cutlass',
        'tools',
        'util',
        'include',
    ),
)
if cutlass_include_dir is None:
    raise RuntimeError(
        "Could not locate a CUTLASS include directory for nvfp4 GEMM build. "
        "Install flashinfer with bundled CUTLASS data or set CUTLASS_INCLUDE_DIR."
    )
if cutlass_tools_include_dir is None:
    raise RuntimeError(
        "Could not locate a CUTLASS tools/util include directory for nvfp4 GEMM build. "
        "Install flashinfer with bundled CUTLASS data or set CUTLASS_TOOLS_INCLUDE_DIR."
    )

# Get PyTorch library path for RPATH
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

# Add RPATH to find PyTorch libraries at runtime
extra_link_args = [
    f'-Wl,-rpath,{torch_lib_path}',
    # Required by CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL.
    '-lcuda',
]

# Create extension module
ext_modules = [
    cpp_extension.CUDAExtension(
        name='nvfp4._C',
        sources=sources,
        include_dirs=[
            os.path.join(os.path.dirname(__file__), 'src'),
            cutlass_include_dir,
            cutlass_tools_include_dir,
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name='nvfp4',
    version='0.1.0',
    description='NVIDIA FP4 Quantization Kernels',
    # NOT ['nvfp4'] -- that ships the top level only and silently drops every
    # subpackage: cute_attn (ResAttn), cute_fold / cute_fold_sm120 (MExt fold),
    # experiment. Editable installs hide the bug because the source dir is on
    # sys.path directly; a real install of the same tree then fails at
    
    #
    # find_packages only sees directories with an __init__.py, so a new kernel
    # family ships only once it has one. Add the file when you add the family,
    # or a real install silently omits it exactly the way this bug did.
    packages=find_packages(include=['nvfp4', 'nvfp4.*']),
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    python_requires='>=3.10',
)
