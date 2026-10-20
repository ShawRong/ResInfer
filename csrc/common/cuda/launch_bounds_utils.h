// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
// SPDX-License-Identifier: Apache-2.0
/*
 * Launch bounds utilities for NVFP4 quantization kernels.
 * Adapted from the vLLM project.
 */
#pragma once

#include <cuda_runtime_api.h>


#ifndef VLLM_LAUNCH_BLOCKS_CAP
  #define VLLM_LAUNCH_BLOCKS_CAP 4
#endif


#ifndef VLLM_MAX_THREADS_PER_SM
  #ifdef __CUDA_ARCH__


    #if (__CUDA_ARCH__ == 750)
      #define VLLM_MAX_THREADS_PER_SM 1024


    #elif (__CUDA_ARCH__ == 860) || (__CUDA_ARCH__ == 870) || \
        (__CUDA_ARCH__ == 890) || (__CUDA_ARCH__ == 1010) ||  \
        (__CUDA_ARCH__ == 1100) || (__CUDA_ARCH__ == 1200) || \
        (__CUDA_ARCH__ == 1210)
      #define VLLM_MAX_THREADS_PER_SM 1536


    #elif (__CUDA_ARCH__ == 700) || (__CUDA_ARCH__ == 720) || \
        (__CUDA_ARCH__ == 800) || (__CUDA_ARCH__ == 900) ||   \
        (__CUDA_ARCH__ == 1000) || (__CUDA_ARCH__ == 1030)
      #define VLLM_MAX_THREADS_PER_SM 2048


    #else
      #define VLLM_MAX_THREADS_PER_SM 2048
    #endif

  #else

    #define VLLM_MAX_THREADS_PER_SM 2048
  #endif
#endif


#define VLLM_BLOCKS_DIV(VAL) (VLLM_MAX_THREADS_PER_SM / (VAL))
#define VLLM_CLAMP_BLOCKS_PER_SM(VAL) \
  (((VAL) <= 0)                       \
       ? 1                            \
       : (((VAL) < VLLM_LAUNCH_BLOCKS_CAP) ? (VAL) : VLLM_LAUNCH_BLOCKS_CAP))
#define VLLM_BLOCKS_PER_SM(BLOCK_THREADS) \
  VLLM_CLAMP_BLOCKS_PER_SM(VLLM_BLOCKS_DIV(BLOCK_THREADS))


static inline int vllm_runtime_blocks_per_sm(int block_threads) {
  int device = -1;
  cudaGetDevice(&device);
  int max_threads_per_sm = VLLM_MAX_THREADS_PER_SM;
  cudaDeviceGetAttribute(&max_threads_per_sm,
                         cudaDevAttrMaxThreadsPerMultiProcessor, device);
  int blocks = (block_threads > 0) ? (max_threads_per_sm / block_threads) : 1;
  return VLLM_CLAMP_BLOCKS_PER_SM(blocks);
}
