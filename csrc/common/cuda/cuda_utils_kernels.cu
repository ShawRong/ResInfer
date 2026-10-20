// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
// SPDX-License-Identifier: Apache-2.0
/*
 * Adapted from
 * https://github.com/vllm-project/vllm/tree/main/vllm
 */
#include "cuda_utils.h"
#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>
#endif

int64_t get_device_attribute(int64_t attribute, int64_t device_id) {

  static int value = [=]() {
    int device = static_cast<int>(device_id);
    if (device < 0) {
      CUDA_CHECK(cudaGetDevice(&device));
    }
    int value;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &value, static_cast<cudaDeviceAttr>(attribute), device));
    return static_cast<int>(value);
  }();

  return value;
}

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id) {
  int64_t attribute;


#ifdef USE_ROCM
  attribute = hipDeviceAttributeMaxSharedMemoryPerBlock;
#else
  attribute = cudaDevAttrMaxSharedMemoryPerBlockOptin;
#endif

  return get_device_attribute(attribute, device_id);
}

