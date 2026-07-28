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
void scaled_fp4_quant_with_residue_ext_pack8_r1000_no_swap(
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
                                   int64_t layout_mode);
#endif

void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8, bool use_swapped_residue) {
#if (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120) || \
    (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
  if (!use_swapped_residue) {
    switch (residue_per_8) {
      case 1:
        return scaled_fp4_quant_with_residue_ext_pack8_r0125_no_swap(
            output, input, output_sf, input_sf, channel_indices, n_ext,
            residue_per_8);
      case 2:
        return scaled_fp4_quant_with_residue_ext_pack8_r025_no_swap(
            output, input, output_sf, input_sf, channel_indices, n_ext,
            residue_per_8);
      case 4:
        return scaled_fp4_quant_with_residue_ext_pack8_r050_no_swap(
            output, input, output_sf, input_sf, channel_indices, n_ext,
            residue_per_8);
      case 8:
        return scaled_fp4_quant_with_residue_ext_pack8_r1000_no_swap(
            output, input, output_sf, input_sf, channel_indices, n_ext,
            residue_per_8);
      default:
        break;
    }
  }

  return scaled_fp4_quant_with_residue_ext(
      output, input, output_sf, input_sf, channel_indices, n_ext,
      residue_per_8, use_swapped_residue);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled nvfp4 quantization with residue kernel for Blackwell (SM100a/SM120a).");
}

void scaled_fp4_quant_pack8_r0125_no_swap(
                      torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8) {
#if (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120) || \
    (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
  return scaled_fp4_quant_with_residue_ext_pack8_r0125_no_swap(
      output, input, output_sf, input_sf, channel_indices, n_ext,
      residue_per_8);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled pack8 r0125 nvfp4 quantization with residue kernel for Blackwell (SM100a/SM120a).");
}

void scaled_fp4_quant_pack8_r025_no_swap(
                      torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8) {
#if (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120) || \
    (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
  return scaled_fp4_quant_with_residue_ext_pack8_r025_no_swap(
      output, input, output_sf, input_sf, channel_indices, n_ext,
      residue_per_8);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled pack8 r025 nvfp4 quantization with residue kernel for Blackwell (SM100a/SM120a).");
}

void scaled_fp4_quant_pack8_r050_no_swap(
                      torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8) {
#if (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120) || \
    (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
  return scaled_fp4_quant_with_residue_ext_pack8_r050_no_swap(
      output, input, output_sf, input_sf, channel_indices, n_ext,
      residue_per_8);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled pack8 r050 nvfp4 quantization with residue kernel for Blackwell (SM100a/SM120a).");
}

void scaled_fp4_quant_pack8_r1000_no_swap(
                      torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8) {
#if (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120) || \
    (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
  return scaled_fp4_quant_with_residue_ext_pack8_r1000_no_swap(
      output, input, output_sf, input_sf, channel_indices, n_ext,
      residue_per_8);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled pack8 r1000 nvfp4 quantization with residue kernel for Blackwell (SM100a/SM120a).");
}

void scaled_fp4_quant_mext_r1(torch::Tensor& output, torch::Tensor const& input,
                              torch::Tensor& output_sf,
                              torch::Tensor const& input_sf,
                              int64_t elts_mode,
                              int64_t layout_mode) {
#if (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120) || \
    (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
  return scaled_fp4_quant_mext_r1_impl(output, input, output_sf, input_sf,
                                       elts_mode, layout_mode);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled nvfp4 m-extension r1 quantization kernel for Blackwell (SM100a/SM120a).");
}
