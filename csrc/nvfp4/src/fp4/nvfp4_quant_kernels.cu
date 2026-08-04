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

 /*
 *  Main logic of the FP4 quantization kernel with residue.
 */

#include <torch/all.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp8.h>
#include "dispatch_utils.h"

#include "cuda_utils.h"
#include "launch_bounds_utils.h"
#include "nvfp4_utils.cuh"

namespace vllm {

// Use UE4M3 by default.

template <class Type, bool UE8M0_SF = false, int RESIDUE_PER_8_ELTS = 1, bool SWAP_RESIDUE = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numTotalCols, int32_t numMainCols, Type const* in,
                    float const* SFScale, uint32_t* out, uint32_t* SFout,
                    uint8_t const* channelIndices) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

	// Calculate where residue SFs start in SF buffer
	// Main SFs use the tiled layout, residue SFs start after them
	uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
	int32_t effectiveRows = computeEffectiveRows(numRows);
	int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
	if (colIdx >= numMainCols / CVT_FP4_ELTS_PER_THREAD) return;

	// Input tensor row/col loops.
	for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
	  bool validRow = rowIdx < numRows;
	      PackedVec in_vec{};
	      if (validRow) {
	        int64_t inOffset =
	            rowIdx * (numMainCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
	        in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
	      }

	      auto sf_out =
	          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
	                                             CVT_FP4_NUM_THREADS_PER_SF>(
	              rowIdx, colIdx, numTotalCols, SFout);

	      // Setup residue data write address
	      // For normal residue: threads 0-3 write
	      // For swap residue: threads 0, 8, 16, 24 (group first threads) write
	      uint32_t* out_pos = nullptr;
	      uint32_t* residue_data_out = nullptr;
	      if (validRow) {
	        int64_t outOffset =
	            rowIdx * (numTotalCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
	        out_pos = out + outOffset;
	        if constexpr (SWAP_RESIDUE) {
	          residue_data_out =
	              cvt_swap_residue_data_get_offset<uint32_t, RESIDUE_PER_8_ELTS>(
	                  rowIdx, colIdx, numMainCols, numTotalCols, out);
	        } else {
	          residue_data_out =
	              cvt_residue_data_get_offset<uint32_t, RESIDUE_PER_8_ELTS>(
	                  rowIdx, colIdx, numMainCols, numTotalCols, out);
	        }
	      }

	      // Setup residue scale factor write address
	      auto residue_sf_out =
	          cvt_residue_sf_get_offset<uint8_t, RESIDUE_PER_8_ELTS>(rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

      // Get mask (bitmap) for this thread's salient channels
      // Use only colIdx since the mask pattern is the same for all rows
      uint8_t mask = channelIndices[colIdx];

	      uint32_t packed;
	      if constexpr (SWAP_RESIDUE) {
	        packed = cvt_warp_fp16_to_fp4_swap<Type, UE8M0_SF, RESIDUE_PER_8_ELTS>(
	            in_vec, SFScaleVal, sf_out, mask, residue_data_out, residue_sf_out);
	      } else {
	        packed = cvt_warp_fp16_to_fp4<Type, UE8M0_SF, RESIDUE_PER_8_ELTS>(
	            in_vec, SFScaleVal, sf_out, mask, residue_data_out, residue_sf_out);
	      }
	      if (out_pos) {
	        *out_pos = packed;
	      }
	  }

}

template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4_mext_r1_pack8(int32_t inputRows, int32_t outputRows,
                                  int32_t numCols,
                                  Type const* in, float const* SFScale,
                                  uint32_t* out, uint32_t* SFout,
                                  int32_t layoutMode,
                                  int32_t numSFScales) {
  using PackedVec8 = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec8) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched for 8-element base helper.");

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  float const SFScaleResVal =
      (SFScale != nullptr && numSFScales > 1) ? SFScale[1] : SFScaleVal;

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int colGroups = numCols / CVT_FP4_ELTS_PER_THREAD;
  if (colIdx >= colGroups) return;

  for (int rowIdx = blockIdx.x; rowIdx < outputRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < inputRows;
    int32_t baseRow = compute_mext_r1_base_row(rowIdx, outputRows, layoutMode);
    int32_t residueRow =
        compute_mext_r1_residue_row(rowIdx, outputRows, layoutMode);

    PackedVec8 in_vec{};
    if (validRow) {
      int64_t inOffset =
          rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      in_vec = reinterpret_cast<PackedVec8 const*>(in)[inOffset];
    }

    // kConcatK (mode 3): one output row per token, [M, 2K] geometry. Base
    // lands at (row, col), residue at (row, col + K); both data and SF use
    // the canonical addressing of the (M, 2K) grid, so the doubled-K operand
    // is indistinguishable from a stock quantization of a 2K-wide input.
    int32_t colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
    bool kConcatK = layoutMode == 3;
    int64_t outOffset =
        kConcatK ? (int64_t)rowIdx * (2 * colsPerRow) + colIdx
                 : (int64_t)baseRow * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    auto sf_out =
        kConcatK
            ? cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                                 CVT_FP4_NUM_THREADS_PER_SF>(
                  rowIdx, colIdx, 2 * numCols, SFout)
            : cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                                 CVT_FP4_NUM_THREADS_PER_SF>(
                  baseRow, colIdx, numCols, SFout);

    int64_t residueOutOffset =
        kConcatK
            ? (int64_t)rowIdx * (2 * colsPerRow) + colsPerRow + colIdx
            : (int64_t)residueRow * colsPerRow + colIdx;
    uint32_t* residue_data_out = out + residueOutOffset;

    auto residue_sf_out =
        kConcatK
            ? cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                                 CVT_FP4_NUM_THREADS_PER_SF>(
                  rowIdx, colIdx + colsPerRow, 2 * numCols, SFout)
            : cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                                 CVT_FP4_NUM_THREADS_PER_SF>(
                  residueRow, colIdx, numCols, SFout);

    out_pos = cvt_warp_fp16_to_fp4_mext_r1_fast<Type, UE8M0_SF>(
        in_vec, SFScaleVal, SFScaleResVal, sf_out, residue_data_out, residue_sf_out);
  }
}

template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4_mext_r1_pack16(int32_t inputRows, int32_t outputRows,
                                   int32_t numCols,
                                   Type const* in, float const* SFScale,
                                   uint32_t* out, uint32_t* SFout,
                                   int32_t layoutMode,
                                   int32_t numSFScales) {
  using PackedVec8 = PackedVec<Type>;
  constexpr int CVT_FP4_NUM_THREADS_PER_SF = 1;

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  float const SFScaleResVal =
      (SFScale != nullptr && numSFScales > 1) ? SFScale[1] : SFScaleVal;

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int colGroups = numCols / 16;
  if (colIdx >= colGroups) return;

  for (int rowIdx = blockIdx.x; rowIdx < outputRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < inputRows;
    int32_t baseRow = compute_mext_r1_base_row(rowIdx, outputRows, layoutMode);
    int32_t residueRow =
        compute_mext_r1_residue_row(rowIdx, outputRows, layoutMode);

    PackedVec8 lo_vec{};
    PackedVec8 hi_vec{};
    if (validRow) {
      // One 256-bit load covers the thread's entire 16-element SF vector
      // (ld.global.v8.b32 on sm_100+, two 128-bit loads elsewhere).
      load_packed_vec16(
          in + static_cast<int64_t>(rowIdx) * numCols + colIdx * 16,
          lo_vec, hi_vec);
    }

    int64_t outOffset =
        baseRow * (numCols / CVT_FP4_ELTS_PER_UINT32) + colIdx * 2;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            baseRow, colIdx, numCols, SFout);

    int64_t residueOutOffset =
        residueRow * (numCols / CVT_FP4_ELTS_PER_UINT32) + colIdx * 2;
    uint32_t* residue_data_out = out + residueOutOffset;

    auto residue_sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            residueRow, colIdx, numCols, SFout);

    u32x2 packed = cvt_warp_fp16_to_fp4_mext_r1_fast16<Type, UE8M0_SF>(
        lo_vec, hi_vec, SFScaleVal, SFScaleResVal, sf_out, residue_data_out,
        residue_sf_out);
    // Single 64-bit store; outOffset is even because numCols % 16 == 0.
    *reinterpret_cast<uint2*>(out + outOffset) =
        make_uint2(packed.lo, packed.hi);
  }
}

// Helper template to dispatch kernel based on RESIDUE_PER_8_ELTS value
template <typename T, bool UE8M0_SF, bool SWAP_RESIDUE>
void launchKernelByResidue(int residue_per_8, dim3 grid, dim3 block, cudaStream_t stream,
                           int m, int n_ext, int n, T const* input, float const* SFScale,
                           uint32_t* output, uint32_t* SFOuput, uint8_t const* channelIndices) {
  switch (residue_per_8) {
    case 1:  
      cvt_fp16_to_fp4<T, UE8M0_SF, 1, SWAP_RESIDUE><<<grid, block, 0, stream>>>(
          m, n_ext, n, input, SFScale, output, SFOuput, channelIndices);
      break;
    case 2:  
      cvt_fp16_to_fp4<T, UE8M0_SF, 2, SWAP_RESIDUE><<<grid, block, 0, stream>>>(
          m, n_ext, n, input, SFScale, output, SFOuput, channelIndices);
      break;
    case 4:  
      cvt_fp16_to_fp4<T, UE8M0_SF, 4, SWAP_RESIDUE><<<grid, block, 0, stream>>>(
          m, n_ext, n, input, SFScale, output, SFOuput, channelIndices);
      break;
    // ratio=1.0 (residue_per_8 == 8) K-extension removed -- full-rank residue
    // is served by the M-extension path (scaled_fp4_quant_mext_r1). The entry
    // rejects residue_per_8 == 8 before reaching this dispatch.
    default:
      
      fprintf(stderr, "Warning: Unsupported residue_per_8 value %d. Supported values: 1 (12.5%%), 2 (25%%), 4 (50%%). Defaulting to 1 (12.5%%).\n", residue_per_8);
      cvt_fp16_to_fp4<T, UE8M0_SF, 1, SWAP_RESIDUE><<<grid, block, 0, stream>>>(
          m, n_ext, n, input, SFScale, output, SFOuput, channelIndices);
      break;
  }
}

template <typename T>
void invokeFP4Quantization(int m, int n_ext, int n, T const* input, float const* SFScale,
                           int64_t* output, int32_t* SFOuput, bool useUE8M0,
                           uint8_t const* channelIndices, int residue_per_8, bool useSwappedResidue,
                           int multiProcessorCount, cudaStream_t stream) {
  // Grid, Block size.
  // Each thread converts 8 values.
  // IMPORTANT: Need at least 32 threads (1 warp) for residue warp-level operations
	  int numThreads = n / ELTS_PER_THREAD;
	  numThreads = std::max(numThreads, 32);  // At least 1 warp
	  dim3 block(std::min(numThreads, 512));
	  // Get number of blocks per SM
	  int const numBlocksPerSM =
	      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
	  dim3 grid(std::min(computeEffectiveRows(m),
	                     multiProcessorCount * numBlocksPerSM),
	            cuda_utils::ceil_div(numThreads, static_cast<int>(block.x)));

  // Launch the cvt kernel with runtime dispatch based on residue_per_8
  if (useUE8M0) {
    if (useSwappedResidue) {
      launchKernelByResidue<T, true, true>(residue_per_8, grid, block, stream,
                                           m, n_ext, n, input, SFScale,
                                           reinterpret_cast<uint32_t*>(output),
                                           reinterpret_cast<uint32_t*>(SFOuput), channelIndices);
    } else {
      launchKernelByResidue<T, true, false>(residue_per_8, grid, block, stream,
                                            m, n_ext, n, input, SFScale,
                                            reinterpret_cast<uint32_t*>(output),
                                            reinterpret_cast<uint32_t*>(SFOuput), channelIndices);
    }
  } else {
    if (useSwappedResidue) {
      launchKernelByResidue<T, false, true>(residue_per_8, grid, block, stream,
                                            m, n_ext, n, input, SFScale,
                                            reinterpret_cast<uint32_t*>(output),
                                            reinterpret_cast<uint32_t*>(SFOuput), channelIndices);
    } else {
      launchKernelByResidue<T, false, false>(residue_per_8, grid, block, stream,
                                             m, n_ext, n, input, SFScale,
                                             reinterpret_cast<uint32_t*>(output),
                                             reinterpret_cast<uint32_t*>(SFOuput), channelIndices);
    }
  }
}

enum class MExtR1EltsMode : int {
  kAuto = 0,
  kElts8 = 8,
  kElts16 = 16,
};

enum class MExtR1LayoutMode : int {
  kConcat = 0,
  kRowPair = 1,
  kTile8 = 2,
  // K-concat: ONE output row per token, [M, 2K] geometry -- base in element
  // cols [0, K), residue in [K, 2K). Data and SF are both emitted in the
  // canonical order of the (M, 2K) grid, so a downstream GEMM consumes the
  // doubled-K activation as a perfectly ordinary operand (the wrapped-K-loop
  // kernel's input).
  kConcatK = 3,
};

inline bool use_pack16_for_mode(int elts_mode, int cc_major) {
  switch (static_cast<MExtR1EltsMode>(elts_mode)) {
    case MExtR1EltsMode::kElts8:
      return false;
    case MExtR1EltsMode::kElts16:
      // Explicit opt-in. The pack16 path owns one full 16-element SF vector
      // per thread (no cross-thread amax shuffles) and on Blackwell loads it
      // with a single 256-bit ld.global.v8.b32.
      return true;
    case MExtR1EltsMode::kAuto:
    default:
      
      
      
      
      return cc_major == 10;
  }
}

inline int normalize_mext_r1_layout_mode(int layout_mode) {
  switch (static_cast<MExtR1LayoutMode>(layout_mode)) {
    case MExtR1LayoutMode::kTile8:
      return static_cast<int>(MExtR1LayoutMode::kTile8);
    case MExtR1LayoutMode::kRowPair:
      return static_cast<int>(MExtR1LayoutMode::kRowPair);
    case MExtR1LayoutMode::kConcatK:
      return static_cast<int>(MExtR1LayoutMode::kConcatK);
    case MExtR1LayoutMode::kConcat:
    default:
      return static_cast<int>(MExtR1LayoutMode::kConcat);
  }
}

template <typename T>
void invokeFP4QuantizationMExtR1(int input_m, int output_m, int n,
                                 T const* input,
                                 float const* SFScale, int64_t* output,
                                 int32_t* SFOuput, bool useUE8M0,
                                 int elts_mode, int layout_mode,
                                 int numSFScales,
                                 int multiProcessorCount, int ccMajor,
                                 cudaStream_t stream) {
  bool pack16 = use_pack16_for_mode(elts_mode, ccMajor);
  int normalized_layout_mode = normalize_mext_r1_layout_mode(layout_mode);
  // kConcatK is implemented in the pack8 kernel only; pack8 is correct on
  // every arch (pack16 is a cc10.x throughput default, not a semantic).
  if (normalized_layout_mode == static_cast<int>(MExtR1LayoutMode::kConcatK)) {
    pack16 = false;
  }
  int colGroups = n / (pack16 ? 16 : CVT_FP4_ELTS_PER_THREAD);
  colGroups = std::max(colGroups, 1);

  // Decode M-extension is intentionally small-M. Split the K dimension into
  // multiple CTAs so M=1/tile8 does not collapse to only 8 row CTAs.
  int desiredGridY =
      std::max(1, cuda_utils::ceil_div(computeEffectiveRows(output_m),
                                       std::max(output_m, 1)));
  desiredGridY = std::min(desiredGridY, colGroups);
  int numThreads = cuda_utils::ceil_div(colGroups, desiredGridY);
  numThreads = std::max(round_up(numThreads, 32), 32);
  dim3 block(std::min(numThreads, 512));
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  dim3 grid(std::min(int(output_m), multiProcessorCount * numBlocksPerSM),
            cuda_utils::ceil_div(colGroups, static_cast<int>(block.x)));

  if (useUE8M0) {
    if (pack16) {
      cvt_fp16_to_fp4_mext_r1_pack16<T, true><<<grid, block, 0, stream>>>(
          input_m, output_m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(SFOuput), normalized_layout_mode,
          numSFScales);
    } else {
      cvt_fp16_to_fp4_mext_r1_pack8<T, true><<<grid, block, 0, stream>>>(
          input_m, output_m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(SFOuput), normalized_layout_mode,
          numSFScales);
    }
  } else {
    if (pack16) {
      cvt_fp16_to_fp4_mext_r1_pack16<T, false><<<grid, block, 0, stream>>>(
          input_m, output_m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(SFOuput), normalized_layout_mode,
          numSFScales);
    } else {
      cvt_fp16_to_fp4_mext_r1_pack8<T, false><<<grid, block, 0, stream>>>(
          input_m, output_m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(SFOuput), normalized_layout_mode,
          numSFScales);
    }
  }
}

// Instantiate the function.
template void invokeFP4Quantization(int m, int n_ext, int n, half const* input,
                                    float const* SFScale, int64_t* output,
                                    int32_t* SFOuput, bool useUE8M0,
                                    uint8_t const* channelIndices, int residue_per_8, bool useSwappedResidue,
                                    int multiProcessorCount,
                                    cudaStream_t stream);

template void invokeFP4Quantization(int m, int n_ext, int n, __nv_bfloat16 const* input,
                                    float const* SFScale, int64_t* output,
                                    int32_t* SFOuput, bool useUE8M0,
                                    uint8_t const* channelIndices, int residue_per_8, bool useSwappedResidue,
                                    int multiProcessorCount,
                                    cudaStream_t stream);

template void invokeFP4QuantizationMExtR1(
    int input_m, int output_m, int n, half const* input, float const* SFScale,
    int64_t* output, int32_t* SFOuput, bool useUE8M0, int elts_mode, int layout_mode,
    int numSFScales, int multiProcessorCount, int ccMajor, cudaStream_t stream);

template void invokeFP4QuantizationMExtR1(
    int input_m, int output_m, int n, __nv_bfloat16 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0,
    int elts_mode, int layout_mode, int numSFScales, int multiProcessorCount,
    int ccMajor, cudaStream_t stream);

}  // namespace vllm

void scaled_fp4_quant_with_residue_ext(torch::Tensor const& output,
                                       torch::Tensor const& input,
                                       torch::Tensor const& output_sf,
                                       torch::Tensor const& input_sf,
                                       torch::Tensor const& channel_indices,
                                       int64_t n_ext,
                                       int64_t residue_per_8,
                                       bool use_swapped_residue) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "Unsupported input data type for quantize_to_fp4.");

  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  // We don't support e8m0 scales at this moment.
  bool useUE8M0 = false;
  
  // Get channel indices pointer
  auto channel_indices_ptr = static_cast<uint8_t const*>(channel_indices.data_ptr());

  VLLM_DISPATCH_HALF_TYPES(input.scalar_type(), "nvfp4_quant_kernel", [&] {
    using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
    auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
    vllm::invokeFP4Quantization(m, n_ext, n, input_ptr, input_sf_ptr, output_ptr,
                                sf_out, useUE8M0, channel_indices_ptr,
                                static_cast<int>(residue_per_8), use_swapped_residue,
                                multiProcessorCount, stream);
  });
}

void scaled_fp4_quant_mext_r1_impl(torch::Tensor const& output,
                                   torch::Tensor const& input,
                                   torch::Tensor const& output_sf,
                                   torch::Tensor const& input_sf,
                                   int64_t elts_mode,
                                   int64_t layout_mode) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  // output_m (the padded M-extension row count) is derived HERE from the input
  // row count -- it is NOT an op argument. As an `int` op-arg, torch.compile
  // specialized it to a constant while the caller's output buffer stayed
  // symbolic in M, desyncing the two under non-eager / CUDA-graph capture.
  // Deriving it from input.size(0) keeps it in lockstep with that buffer
  // (which the caller also sizes from M). tile8 needs an 8-multiple row count;
  // other layouts use M directly.
  int64_t output_m =
      (static_cast<int>(layout_mode) ==
       static_cast<int>(vllm::MExtR1LayoutMode::kTile8))
          ? ((static_cast<int64_t>(m) + 7) / 8) * 8
          : static_cast<int64_t>(m);

  TORCH_CHECK(n % 16 == 0,
              "The N dimension must be multiple of 16.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "Unsupported input data type for quantize_to_fp4.");
  TORCH_CHECK(output.numel() >= 2 * output_m * n / 2,
              "M-extension output tensor is too small for the derived "
              "output_m (caller buffer undersized for the padded row count).");

  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  // NOT get_device_attribute(): its static cache keys on the first call's
  // attribute, so a second attribute would read back the SM count.
  static const int ccMajor = [] {
    int device = 0;
    cudaGetDevice(&device);
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, device);
    return v;
  }();

  TORCH_CHECK(input_sf.numel() >= 1 && input_sf.numel() <= 2,
              "input_sf must have 1 (single-alpha) or 2 (dual-alpha: "
              "[base, residue]) elements.");
  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  bool useUE8M0 = false;

  VLLM_DISPATCH_HALF_TYPES(input.scalar_type(), "nvfp4_quant_kernel_mext_r1", [&] {
    using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
    auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
    vllm::invokeFP4QuantizationMExtR1(
        m, static_cast<int>(output_m), n, input_ptr, input_sf_ptr, output_ptr, sf_out, useUE8M0,
        static_cast<int>(elts_mode), static_cast<int>(layout_mode),
        static_cast<int>(input_sf.numel()), multiProcessorCount, ccMajor,
        stream);
  });
}
