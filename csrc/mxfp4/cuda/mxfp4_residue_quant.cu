// SPDX-License-Identifier: Apache-2.0
// SM100 MXFP4: NVFP4 pack16/QDQ/mask/warp dataflow, block32 E8M0 ceil.
// FP16/BF16 inputs; quantization and residual arithmetic stay FP32.
// No global alpha or shared gather.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include "dispatch_utils.h"
#include "nvfp4_utils.cuh"

namespace {
using namespace vllm;

// The multiplier preserves the existing underflow policy: code zero zeros
// the quantized block. E8M0 code zero itself is a nonzero scale (2^-127).
__device__ __forceinline__ __nv_fp8_e8m0 compute_mxfp4_scale(
    float amax, float& quantMultiplier) {
  __nv_fp8_e8m0 scale;
  scale.__x = __nv_cvt_float_to_e8m0(
      __fmul_rn(amax, reciprocal_approximate_ftz(6.f)),
      __NV_SATFINITE, cudaRoundPosInf);
  quantMultiplier = scale.__x == 0
                        ? 0.f
                        : __uint_as_float(uint32_t(254 - scale.__x) << 23);
  return scale;
}

template <class Type>
__device__ __forceinline__ void load_vec16_as_float(
    Type const* p, int64_t stride, float (&v)[16]) {
  if (stride == 1 && (reinterpret_cast<uintptr_t>(p) & 31) == 0) {
    PackedVec<Type> lo, hi;
    load_packed_vec16(p, lo, hi);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 a = packed_vec_to_float2(lo, j);
      float2 b = packed_vec_to_float2(hi, j);
      v[2 * j] = a.x;
      v[2 * j + 1] = a.y;
      v[8 + 2 * j] = b.x;
      v[9 + 2 * j] = b.y;
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    v[j] = float(p[j * stride]);
  }
}

// The caller chooses the block32 E8M0 scale. Packing itself is shared
// with NVFP4; explicit FP32 multiplies retain MXFP4's rounding behavior.
template <bool WITH_DEQUANT>
__device__ __forceinline__ uint2 quantize_fp32_vec16(
    float (&values)[16], float quantMultiplier, __nv_fp8_e8m0 scale) {
  float2 loValues[4];
  float2 hiValues[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    loValues[i] = make_float2(
        __fmul_rn(values[2 * i], quantMultiplier),
        __fmul_rn(values[2 * i + 1], quantMultiplier));
    hiValues[i] = make_float2(
        __fmul_rn(values[8 + 2 * i], quantMultiplier),
        __fmul_rn(values[8 + 2 * i + 1], quantMultiplier));
  }
  u32x2 packed = fp32x16_to_e2m1<WITH_DEQUANT>(loValues, hiValues);
  if constexpr (WITH_DEQUANT) {
    float dequantScale = static_cast<float>(scale);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      values[2 * i] = __fmul_rn(loValues[i].x, dequantScale);
      values[2 * i + 1] = __fmul_rn(loValues[i].y, dequantScale);
      values[8 + 2 * i] = __fmul_rn(hiValues[i].x, dequantScale);
      values[8 + 2 * i + 1] = __fmul_rn(hiValues[i].y, dequantScale);
    }
  }
  return make_uint2(packed.lo, packed.hi);
}

// RESIDUE_PER_8=8: full-R1 row pairs,
// RESIDUE_PER_8=1/2/4: fixed selection per 8.
template <class Type, int RESIDUE_PER_8, bool SPLIT_K = false>
__global__ void cvt_to_mxfp4_residue(Type const* input, int64_t rowStride,
    int64_t colStride, uint8_t const* masks, uint8_t* output, uint8_t* outputSF,
    int numRows, int numCols, int numTotalCols, int outputRows, int paddedRows,
    bool swizzle, bool tile8, int pdlMode, int splitLog2 = 0) {
  // Trigger every CTA, including padding CTAs. The consumer must wait before
  // reading our outputs; this permits only its independent setup to overlap.
  if (pdlMode == 2 && threadIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
  constexpr bool SELECTIVE = RESIDUE_PER_8 < 8;
  constexpr int FACTOR = RESIDUE_PER_8 == 8 ? 2 : 1;
  int row = blockIdx.x, blockK = 0, kTiles = 1;
  if constexpr (SPLIT_K) {
    kTiles = 1 << splitLog2;
    int validBlocks = numRows << splitLog2;
    row = blockIdx.x < validBlocks ? blockIdx.x >> splitLog2
                                  : numRows + blockIdx.x - validBlocks;
    blockK = blockIdx.x < validBlocks ? blockIdx.x & (kTiles - 1) : 0;
  }
  int laneId = threadIdx.x % 32;
  int layoutMode = tile8 ? 2 : 1;
  int outputRow = RESIDUE_PER_8 == 8
                      ? compute_mext_r1_base_row(row, numRows, layoutMode)
                      : row;
  int residueRow = compute_mext_r1_residue_row(row, numRows, layoutMode);
  int numSfGroups = numTotalCols / 32,
      numSelected = SELECTIVE ? numCols / 8 * RESIDUE_PER_8 : 0;
  bool valid = row < numRows;
  int numWrittenSfGroups =
      numCols / 32 + (SELECTIVE ? (numSelected + 31) / 32 : 0);
  int numOutputSfGroups = swizzle ? (numSfGroups + 3) / 4 * 4 : numSfGroups;
  // Clear only padding/invalid rows, never overwrite valid producer stores.
  for (int g = threadIdx.x; g < numOutputSfGroups; g += blockDim.x) {
    if (blockK == 0 && (!valid || g >= numWrittenSfGroups) &&
        outputRow < (swizzle ? paddedRows : outputRows)) {
      outputSF[compute_tiled_sf_offset<uint8_t, 32>(
          outputRow, g, numTotalCols, swizzle)] = 0;
      if constexpr (FACTOR == 2) {
        outputSF[compute_tiled_sf_offset<uint8_t, 32>(
            residueRow, g, numTotalCols, swizzle)] = 0;
      }
    }
  }
  if (row >= numRows) {
    // A4/A6 read a padded tile8 activation as real GEMM rows. Zero its data
    // as well as the scales; the old row-pair path has no physical padding.
    if constexpr (RESIDUE_PER_8 == 8) {
      if (tile8 && row < outputRows / 2) {
        for (int j = threadIdx.x; j < numTotalCols / 2; j += blockDim.x) {
          output[int64_t(outputRow) * (numTotalCols / 2) + j] = 0;
          output[int64_t(residueRow) * (numTotalCols / 2) + j] = 0;
        }
      }
    }
    if (pdlMode == 1) {
      __syncthreads();
      if (threadIdx.x == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
      }
    }
    return;
  }
  for (int tile = blockK; tile < (numCols + blockDim.x * 16 - 1) / (blockDim.x * 16);
      tile += kTiles) {
    int g = tile * blockDim.x + threadIdx.x;
    bool hasInput = g < numCols / 16;
    float original[16] = {}, v[16];
    if (hasInput) {
      Type const* p =
          input + int64_t(row) * rowStride + int64_t(g * 16) * colStride;
      load_vec16_as_float(p, colStride, original);
    }
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      v[j] = original[j];
      amax = fmaxf(amax, fabsf(v[j]));
    }
    float quantMultiplier;
    auto base = compute_mxfp4_scale(
        warp_group_max<2>(amax, laneId), quantMultiplier);
    uint2 bits = quantize_fp32_vec16<true>(
        v, quantMultiplier, base);
    if (hasInput) {
      *reinterpret_cast<uint2*>(
          output + int64_t(outputRow) * (numTotalCols / 2) + g * 8) = bits;
      if ((g & 1) == 0) {
        outputSF[compute_tiled_sf_offset<uint8_t, 32>(
            outputRow, g / 2, numTotalCols, swizzle)] = base.__x;
      }
    }
    if constexpr (RESIDUE_PER_8 == 8) {
      float residueMax = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        v[j] = __fsub_rn(original[j], v[j]);
        residueMax = fmaxf(residueMax, fabsf(v[j]));
      }
      float residueQuantMultiplier;
      auto residueScale = compute_mxfp4_scale(
          warp_group_max<2>(residueMax, laneId), residueQuantMultiplier);
      uint2 residuePacked = quantize_fp32_vec16<false>(
          v, residueQuantMultiplier, residueScale);
      if (hasInput) {
        *reinterpret_cast<uint2*>(
            output + int64_t(residueRow) * (numTotalCols / 2) + g * 8) =
            residuePacked;
        if ((g & 1) == 0) {
          outputSF[compute_tiled_sf_offset<uint8_t, 32>(
              residueRow, g / 2, numTotalCols, swizzle)] = residueScale.__x;
        }
      }
    } else if constexpr (SELECTIVE) {
      constexpr int RESIDUE_PER_THREAD = 2 * RESIDUE_PER_8,
                    THREADS_PER_SF = 32 / RESIDUE_PER_THREAD,
                    THREADS_PER_PACK = 8 / RESIDUE_PER_THREAD;
      float residueValues[RESIDUE_PER_THREAD] = {};
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        unsigned mask = hasInput ? masks[g * 2 + half] : 0;
#pragma unroll
        for (int j = 0; j < RESIDUE_PER_8; ++j) {
          int index = __ffs(mask) - 1;
          float value = 0.f;
          // Compile-time register selects avoid dynamically indexing a local
          // array.
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            if (c == index) {
              value = __fsub_rn(original[half * 8 + c], v[half * 8 + c]);
            }
          }
          residueValues[half * RESIDUE_PER_8 + j] = value;
          mask &= mask - 1;
        }
      }
      float residueMax = 0.f;
#pragma unroll
      for (int j = 0; j < RESIDUE_PER_THREAD; ++j) {
        residueMax = fmaxf(residueMax, fabsf(residueValues[j]));
      }
      float residueQuantMultiplier;
      auto residueScale = compute_mxfp4_scale(
          warp_group_max<THREADS_PER_SF>(residueMax, laneId),
          residueQuantMultiplier);
      int outputStart = g * RESIDUE_PER_THREAD;
      if (threadIdx.x % THREADS_PER_SF == 0 && outputStart < numSelected) {
        outputSF[compute_tiled_sf_offset<uint8_t, 32>(
            row, numCols / 32 + outputStart / 32, numTotalCols, swizzle)] =
            residueScale.__x;
      }
      float values[8];
      int leader = (threadIdx.x % 32) / THREADS_PER_PACK * THREADS_PER_PACK;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        values[j] = __fmul_rn(
            __shfl_sync(0xffffffffu, residueValues[j % RESIDUE_PER_THREAD],
                leader + j / RESIDUE_PER_THREAD),
            residueQuantMultiplier);
      }
      uint32_t packed = fp32_vec_to_e2m1(values);
      if (threadIdx.x % THREADS_PER_PACK == 0 && outputStart < numSelected) {
        *reinterpret_cast<uint32_t*>(output +
                                     int64_t(row) * (numTotalCols / 2) +
                                     numCols / 2 + outputStart / 2) = packed;
      }
    }
  }
  if constexpr (SELECTIVE) {
    for (int j = numCols / 2 + ((numSelected + 7) / 8) * 4 + threadIdx.x;
        j < numTotalCols / 2; j += blockDim.x) {
      output[int64_t(row) * (numTotalCols / 2) + j] = 0;
    }
  }
  if (pdlMode == 1) {
    __syncthreads();
    if (threadIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
}

template <class Type>
void invokeMxfp4Quantization(torch::Tensor input, torch::Tensor output,
    torch::Tensor outputSF, uint8_t const* masks, int residuePer8,
    bool swizzle, bool tile8, int pdlMode, int smCount, cudaStream_t stream) {
  int numRows = input.size(0), numCols = input.size(1),
      numTotalCols = output.size(1) * 2, outputRows = output.size(0);
  int paddedRows = swizzle ? (outputRows + 127) / 128 * 128 : outputRows;
  dim3 grid(!masks ? paddedRows / 2 : paddedRows);
  
  
  
  int threads = (!masks && tile8 && numRows >= 64 &&
                    (numCols == 4096 || numCols == 8192))
                    ? numCols / 16
                    : 128;
  // Fill two waves of SMs when rows alone leave the GPU underoccupied.
  // Split only full residue; padding rows keep one CTA each. Powers of two
  // make the valid-row/K mapping a shift and mask instead of integer division.
  int splitLog2 = 0;
  int kIterations = (numCols + threads * 16 - 1) / (threads * 16);
  if (!masks) {
    while ((numRows << splitLog2) < 2 * smCount &&
           (1 << splitLog2) < kIterations) {
      ++splitLog2;
    }
  }
  grid.x += numRows * ((1 << splitLog2) - 1);
  auto p = reinterpret_cast<Type const*>(input.data_ptr());
#define LAUNCH(RESIDUE_PER_8, SPLIT_K)                                      \
  cvt_to_mxfp4_residue<Type, RESIDUE_PER_8, SPLIT_K>                        \
      <<<grid, threads, 0, stream>>>(p,                                     \
      input.stride(0), input.stride(1), masks, output.data_ptr<uint8_t>(),  \
      outputSF.data_ptr<uint8_t>(), numRows, numCols, numTotalCols,         \
      outputRows, paddedRows, swizzle, tile8, pdlMode, splitLog2)
  if (masks) {
    switch (residuePer8) {
      case 1:
        LAUNCH(1, false);
        break;
      case 2:
        LAUNCH(2, false);
        break;
      case 4:
        LAUNCH(4, false);
        break;
    }
  } else if (splitLog2) {
    LAUNCH(8, true);
  } else {
    LAUNCH(8, false);
  }
#undef LAUNCH
}
}  // namespace

void mxfp4_residue_quant(torch::Tensor input, torch::Tensor output,
    torch::Tensor outputSF, c10::optional<torch::Tensor> masks,
    int64_t residuePer8, bool swizzle, bool tile8,
    int64_t pdlMode) {
  TORCH_CHECK(pdlMode >= 0 && pdlMode <= 2,
      "pdl_mode must be 0/off, 1/tail or 2/early");
  TORCH_CHECK(input.is_cuda() && input.dim() == 2, "expected CUDA [M,K]");
  TORCH_CHECK(input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
      "fp16/bf16 required");
  c10::cuda::CUDAGuard guard(input.device());
  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 10 && prop->minor == 0, "SM100 required");
  TORCH_CHECK(output.is_cuda() && outputSF.is_cuda() &&
                  output.device() == input.device() &&
                  outputSF.device() == input.device(),
      "output device mismatch");
  TORCH_CHECK(output.scalar_type() == at::kByte &&
                  outputSF.scalar_type() == at::kByte &&
                  output.is_contiguous() && outputSF.is_contiguous(),
      "contiguous uint8 outputs required");
  int numRows = input.size(0), numCols = input.size(1);
  TORCH_CHECK(numCols > 0 && numCols % 32 == 0, "invalid MXFP4 extent");
  TORCH_CHECK(!masks || (residuePer8 == 1 || residuePer8 == 2 ||
                             residuePer8 == 4),
      "per8 must be 1/2/4");
  TORCH_CHECK(!tile8 || (!masks && swizzle),
      "tile8 requires full residue and 128x4 scales");
  if (masks) {
    TORCH_CHECK(masks->device() == input.device() &&
                    masks->scalar_type() == at::kByte &&
                    masks->is_contiguous() && masks->dim() == 1 &&
                    masks->numel() == numCols / 8,
        "prepared uint8 masks [K/8] required");
  }
  int outputRows = output.size(0),
      numSelected = masks ? numCols / 8 * residuePer8 : 0;
  TORCH_CHECK(output.dim() == 2 &&
                  (tile8 ? (outputRows >= 2 * numRows && outputRows % 16 == 0)
                         : outputRows == numRows * (!masks ? 2 : 1)),
      "invalid output rows");
  int numTotalCols = output.size(1) * 2;
  TORCH_CHECK(
      numTotalCols % 32 == 0 && (masks ? numTotalCols >= numCols + numSelected
                                       : numTotalCols == numCols),
      "invalid output K");
  int64_t numScales = swizzle ? int64_t((outputRows + 127) / 128) *
                                    ((numTotalCols + 127) / 128) * 512
                              : int64_t(outputRows) * (numTotalCols / 32);
  TORCH_CHECK(outputSF.numel() == numScales, "invalid SF extent");
  if (!numRows) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  auto maskPtr = masks ? masks->data_ptr<uint8_t>() : nullptr;
  VLLM_DISPATCH_HALF_TYPES(input.scalar_type(), "mxfp4_residue_quant", [&] {
    using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
    invokeMxfp4Quantization<cuda_type>(input, output, outputSF, maskPtr,
        residuePer8, swizzle, tile8, pdlMode, prop->multiProcessorCount, stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
