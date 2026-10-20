# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import torch
from setuptools import find_packages, setup
from torch.utils import cpp_extension


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


cuda_version = get_cuda_version()
cuda_major, cuda_minor = cuda_version


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
        '-diag-suppress', '177',

    ]
}

if os.getenv("NVFP4_ENABLE_LINEINFO", "0") == "1":
    extra_compile_args["nvcc"].append("-lineinfo")


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


sources = [
    'bindings/pybind.cpp',
    'common/cuda/cuda_utils_kernels.cu',
    'nvfp4/cuda/nvfp4_quant_entry.cu',
    'nvfp4/cuda/nvfp4_quant_kernels.cu',
    'nvfp4/cuda/nvfp4_residue_pack8.cu',
    'nvfp4/cuda/nvfp4_residue_pack16.cu',
    'mxfp4/cuda/mxfp4_residue_quant.cu',
]


torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')


extra_link_args = [
    f'-Wl,-rpath,{torch_lib_path}',
]


ext_modules = [
    cpp_extension.CUDAExtension(
        name='nvfp4._C',
        sources=sources,
        include_dirs=[
            os.path.join(os.path.dirname(__file__), 'common', 'cuda'),
            os.path.join(os.path.dirname(__file__), 'common', 'fp4'),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

KERNEL_PACKAGES = {
    'nvfp4.mxfp4': 'mxfp4',
    'nvfp4.cute_attn': 'resattn',
    'nvfp4.cute_fold': 'common/fold/sm10x',
    'nvfp4.cute_fold_sm120': 'common/fold/sm120',
}

setup(
    name='nvfp4',
    version='0.1.0',
    description='NVIDIA FP4 Quantization Kernels',

    packages=(find_packages(include=['nvfp4', 'nvfp4.*']) + list(KERNEL_PACKAGES)),
    package_dir=KERNEL_PACKAGES,
    package_data={name: ['cuda/*.cu', 'cuda/*.cuh']
                  for name in ['nvfp4', 'nvfp4.mxfp4']},
    include_package_data=False,
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    python_requires='>=3.10',
)
