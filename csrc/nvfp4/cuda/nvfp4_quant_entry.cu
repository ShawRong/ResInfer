// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <cstdlib>
#include <cstring>

#include <cuda_runtime_api.h>

#if (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120) || \
    (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
void scaled_fp4_quant_with_residue_ext(torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf,
                             torch::Tensor const& channel_indices,
                             int64_t n_ext,
                             int64_t residue_per_8,
                             bool use_swapped_residue);
void scaled_fp4_quant_with_residue_ext_pack8_r0125_no_swap(
                             torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf,
                             torch::Tensor const& channel_indices,
                             int64_t n_ext,
                             int64_t residue_per_8);
void scaled_fp4_quant_with_residue_ext_pack8_r025_no_swap(
                             torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf,
                             torch::Tensor const& channel_indices,
                             int64_t n_ext,
                             int64_t residue_per_8);
void scaled_fp4_quant_with_residue_ext_pack8_r050_no_swap(
                             torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf,
                             torch::Tensor const& channel_indices,
                             int64_t n_ext,
                             int64_t residue_per_8);
void scaled_fp4_quant_mext_r1_impl(torch::Tensor const& output,
                                   torch::Tensor const& input,
                                   torch::Tensor const& output_sf,
                                   torch::Tensor const& input_sf,
                                   int64_t elts_mode,
                                   int64_t layout_mode,
                                   bool enable_pdl);
void scaled_fp4_quant_with_residue_ext_pack16_r0125_no_swap(
    torch::Tensor const& output, torch::Tensor const& input,
    torch::Tensor const& output_sf, torch::Tensor const& input_sf,
    torch::Tensor const& channel_indices, int64_t n_ext,
    int64_t residue_per_8);
void scaled_fp4_quant_with_residue_ext_pack16_r025_no_swap(
    torch::Tensor const& output, torch::Tensor const& input,
    torch::Tensor const& output_sf, torch::Tensor const& input_sf,
    torch::Tensor const& channel_indices, int64_t n_ext,
    int64_t residue_per_8);
void scaled_fp4_quant_with_residue_ext_pack16_r050_no_swap(
    torch::Tensor const& output, torch::Tensor const& input,
    torch::Tensor const& output_sf, torch::Tensor const& input_sf,
    torch::Tensor const& channel_indices, int64_t n_ext,
    int64_t residue_per_8);
namespace {


int resolve_residue_elts_mode() {
  static const int mode = [] {
    const char* env = getenv("NVFP4_RESIDUE_ELTS_PER_THREAD");
    if (env != nullptr) {
      if (strcmp(env, "8") == 0) return 8;
      if (strcmp(env, "16") == 0) return 16;
    }
    return 0;
  }();
  return mode;
}

int device_compute_capability() {
  static const int capability = [] {
    int device = 0;
    cudaGetDevice(&device);
    int cc_major = 0;
    int cc_minor = 0;
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor,
                           device);
    cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor,
                           device);
    return cc_major * 10 + cc_minor;
  }();
  return capability;
}

bool use_residue_pack16(torch::Tensor const& input, int64_t n_ext,
                        int64_t residue_per_8) {
  int mode = resolve_residue_elts_mode();
  if (mode == 8) return false;


  if (input.size(1) % 16 != 0 || n_ext % 16 != 0) return false;
  if (mode == 16) return true;
  int cc = device_compute_capability();
  if (cc / 10 != 10) return false;


  int64_t m = input.size(0);
  int64_t k = input.size(1);
  if (cc == 100 && m >= 1 && m <= 128) {
    if (residue_per_8 == 1 &&
        (k == 256 || k == 768 || (m >= 4 && (k == 512 || k == 1024)))) {
      return false;
    }
    if (residue_per_8 == 2 && (k == 8192 || k == 12288 || k == 16384)) {
      return true;
    }
    if (residue_per_8 == 4 && k == 16384) return true;
  }


  if (residue_per_8 == 1) return true;
  return input.size(0) >= 1024;
}

}
#endif

void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8, bool use_swapped_residue) {
#if (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120) || \
    (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
  if (!use_swapped_residue) {
    bool pack16 = use_residue_pack16(input, n_ext, residue_per_8);
    switch (residue_per_8) {
      case 1:
        return (pack16 ? scaled_fp4_quant_with_residue_ext_pack16_r0125_no_swap
                       : scaled_fp4_quant_with_residue_ext_pack8_r0125_no_swap)(
            output, input, output_sf, input_sf, channel_indices, n_ext,
            residue_per_8);
      case 2:
        return (pack16 ? scaled_fp4_quant_with_residue_ext_pack16_r025_no_swap
                       : scaled_fp4_quant_with_residue_ext_pack8_r025_no_swap)(
            output, input, output_sf, input_sf, channel_indices, n_ext,
            residue_per_8);
      case 4:
        return (pack16 ? scaled_fp4_quant_with_residue_ext_pack16_r050_no_swap
                       : scaled_fp4_quant_with_residue_ext_pack8_r050_no_swap)(
            output, input, output_sf, input_sf, channel_indices, n_ext,
            residue_per_8);
      default:
        break;
    }
  }


  TORCH_CHECK(residue_per_8 != 8,
              "residue_per_8 == 8 (ratio 1.0) K-extension is no longer "
              "supported; use scaled_fp4_quant_mext_r1 (M-extension).");

  return scaled_fp4_quant_with_residue_ext(
      output, input, output_sf, input_sf, channel_indices, n_ext,
      residue_per_8, use_swapped_residue);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled nvfp4 quantization with residue kernel for Blackwell (SM100a/SM120a).");
}

void scaled_fp4_quant_mext_r1(torch::Tensor& output, torch::Tensor const& input,
                              torch::Tensor& output_sf,
                              torch::Tensor const& input_sf,
                              int64_t elts_mode,
                              int64_t layout_mode,
                              bool enable_pdl) {
#if (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120) || \
    (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
  return scaled_fp4_quant_mext_r1_impl(output, input, output_sf, input_sf,
                                       elts_mode, layout_mode, enable_pdl);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled nvfp4 m-extension r1 quantization kernel for Blackwell (SM100a/SM120a).");
}
