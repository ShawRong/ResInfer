// SPDX-License-Identifier: Apache-2.0
// SM100 MXFP4: NVFP4 pack16/QDQ/mask/warp dataflow, block32 E8M0 ceil.
// FP32 source and residual stay FP32. No global alpha or shared gather.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include "nvfp4_utils.cuh"

namespace {
using namespace vllm;

// E8M0 ceil scale; preserve the existing code-zero subnormal convention.
struct Mxfp4Scale {
  uint8_t code;
  float inverse;
  float value;
};

__device__ __forceinline__ Mxfp4Scale compute_mxfp4_scale(float amax) {
  auto scaleCode = __nv_cvt_float_to_e8m0(
      __fmul_rn(amax, reciprocal_approximate_ftz(6.f)),
      __NV_SATFINITE, cudaRoundPosInf);
  return {
      scaleCode,
      scaleCode == 0 ? 0.f : __uint_as_float(uint32_t(254 - scaleCode) << 23),
      __uint_as_float(scaleCode == 0 ? 0x00400000u : uint32_t(scaleCode) << 23),
  };
}

template <int GROUP_THREADS>
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int d = 1; d < GROUP_THREADS; d *= 2) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, d));
  }
  return v;
}

__device__ __forceinline__ int64_t mxfp4_sf_offset(
    int row, int g, int numSfGroups, bool swizzle) {
  if (!swizzle) {
    return int64_t(row) * numSfGroups + g;
  }
  return (int64_t(row / 128) * ((numSfGroups + 3) / 4) + g / 4) * 512 +
         (row % 32) * 16 + (row % 128 / 32) * 4 + g % 4;
}

template <class Type>
__device__ __forceinline__ void load_fp32_vec16(
    Type const* p, int64_t stride, float (&v)[16]) {
  if constexpr (!std::is_same_v<Type, float>) {
    if (stride == 1 && (reinterpret_cast<uintptr_t>(p) & 31) == 0) {
      PackedVec<Type> lo, hi;
      load_packed_vec16(p, lo, hi);  // production NVFP4 256-bit load
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
  } else if (stride == 1 && (reinterpret_cast<uintptr_t>(p) & 15) == 0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float4 a = reinterpret_cast<float4 const*>(p)[j];
      v[4 * j] = a.x;
      v[4 * j + 1] = a.y;
      v[4 * j + 2] = a.z;
      v[4 * j + 3] = a.w;
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
    float (&values)[16], Mxfp4Scale scale) {
  float2 loValues[4];
  float2 hiValues[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    loValues[i] = make_float2(
        __fmul_rn(values[2 * i], scale.inverse),
        __fmul_rn(values[2 * i + 1], scale.inverse));
    hiValues[i] = make_float2(
        __fmul_rn(values[8 + 2 * i], scale.inverse),
        __fmul_rn(values[8 + 2 * i + 1], scale.inverse));
  }
  u32x2 packed = fp32x16_to_e2m1<WITH_DEQUANT>(loValues, hiValues);
  if constexpr (WITH_DEQUANT) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      values[2 * i] = __fmul_rn(loValues[i].x, scale.value);
      values[2 * i + 1] = __fmul_rn(loValues[i].y, scale.value);
      values[8 + 2 * i] = __fmul_rn(hiValues[i].x, scale.value);
      values[8 + 2 * i + 1] = __fmul_rn(hiValues[i].y, scale.value);
    }
  }
  return make_uint2(packed.lo, packed.hi);
}

// RESIDUE_PER_8=0: standard, RESIDUE_PER_8=8: full-R1 row pairs,
// RESIDUE_PER_8=1/2/4: fixed selection per 8.
template <class Type, int RESIDUE_PER_8>
__global__ void cvt_to_mxfp4_residue(Type const* input, int64_t rowStride,
    int64_t colStride, uint8_t const* masks, uint8_t* output, uint8_t* outputSF,
    int numRows, int numCols, int numTotalCols, int outputRows, int paddedRows,
    bool swizzle, bool tile8, int pdlMode) {
  // Trigger every CTA, including padding CTAs. The consumer must wait before
  // reading our outputs; this permits only its independent setup to overlap.
  if (pdlMode == 2 && threadIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
  constexpr bool SELECTIVE = RESIDUE_PER_8 > 0 && RESIDUE_PER_8 < 8;
  constexpr int FACTOR = RESIDUE_PER_8 == 8 ? 2 : 1;
  int row = blockIdx.x;
  int outputRow =
      (RESIDUE_PER_8 == 8 && tile8) ? (row / 8) * 16 + row % 8 : row * FACTOR;
  int residueRow =
      (RESIDUE_PER_8 == 8 && tile8) ? outputRow + 8 : outputRow + 1;
  int numSfGroups = numTotalCols / 32,
      numSelected = SELECTIVE ? numCols / 8 * RESIDUE_PER_8 : 0;
  bool valid = row < numRows;
  int numWrittenSfGroups =
      numCols / 32 + (SELECTIVE ? (numSelected + 31) / 32 : 0);
  int numOutputSfGroups = swizzle ? (numSfGroups + 3) / 4 * 4 : numSfGroups;
  // Clear only padding/invalid rows, never overwrite valid producer stores.
  for (int g = threadIdx.x; g < numOutputSfGroups; g += blockDim.x) {
    if ((!valid || g >= numWrittenSfGroups) &&
        outputRow < (swizzle ? paddedRows : outputRows)) {
      outputSF[mxfp4_sf_offset(outputRow, g, numSfGroups, swizzle)] = 0;
      if constexpr (FACTOR == 2) {
        outputSF[mxfp4_sf_offset(residueRow, g, numSfGroups, swizzle)] = 0;
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
  for (int tile = 0; tile < (numCols + blockDim.x * 16 - 1) / (blockDim.x * 16);
      ++tile) {
    int g = tile * blockDim.x + threadIdx.x;
    bool hasInput = g < numCols / 16;
    float original[16] = {}, v[16];
    if (hasInput) {
      Type const* p =
          input + int64_t(row) * rowStride + int64_t(g * 16) * colStride;
      load_fp32_vec16(p, colStride, original);
    }
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      v[j] = original[j];
      amax = fmaxf(amax, fabsf(v[j]));
    }
    Mxfp4Scale base = compute_mxfp4_scale(warp_reduce_max<2>(amax));
    uint2 bits = quantize_fp32_vec16<(RESIDUE_PER_8 > 0)>(v, base);
    if (hasInput) {
      *reinterpret_cast<uint2*>(
          output + int64_t(outputRow) * (numTotalCols / 2) + g * 8) = bits;
      if ((g & 1) == 0) {
        outputSF[mxfp4_sf_offset(outputRow, g / 2, numSfGroups, swizzle)] =
            base.code;
      }
    }
    if constexpr (RESIDUE_PER_8 == 8) {
      float residueMax = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        v[j] = __fsub_rn(original[j], v[j]);
        residueMax = fmaxf(residueMax, fabsf(v[j]));
      }
      Mxfp4Scale residueScale =
          compute_mxfp4_scale(warp_reduce_max<2>(residueMax));
      uint2 residuePacked = quantize_fp32_vec16<false>(v, residueScale);
      if (hasInput) {
        *reinterpret_cast<uint2*>(
            output + int64_t(residueRow) * (numTotalCols / 2) + g * 8) =
            residuePacked;
        if ((g & 1) == 0) {
          outputSF[mxfp4_sf_offset(residueRow, g / 2, numSfGroups, swizzle)] =
              residueScale.code;
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
      Mxfp4Scale residueScale =
          compute_mxfp4_scale(warp_reduce_max<THREADS_PER_SF>(residueMax));
      int outputStart = g * RESIDUE_PER_THREAD;
      if (threadIdx.x % THREADS_PER_SF == 0 && outputStart < numSelected) {
        outputSF[mxfp4_sf_offset(row, numCols / 32 + outputStart / 32,
            numSfGroups, swizzle)] = residueScale.code;
      }
      float values[8];
      int leader = (threadIdx.x % 32) / THREADS_PER_PACK * THREADS_PER_PACK;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        values[j] = __fmul_rn(
            __shfl_sync(0xffffffffu, residueValues[j % RESIDUE_PER_THREAD],
                leader + j / RESIDUE_PER_THREAD),
            residueScale.inverse);
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
    torch::Tensor outputSF, uint8_t const* masks, int residuePer8, bool residue,
    bool swizzle, bool tile8, int pdlMode, cudaStream_t stream) {
  int numRows = input.size(0), numCols = input.size(1),
      numTotalCols = output.size(1) * 2, outputRows = output.size(0);
  int paddedRows = swizzle ? (outputRows + 127) / 128 * 128 : outputRows;
  dim3 grid(residue && !masks ? paddedRows / 2 : paddedRows);
  
  
  
  int threads = (residue && !masks && tile8 && numRows >= 64 &&
                    (numCols == 4096 || numCols == 8192))
                    ? numCols / 16
                    : 128;
  auto p = reinterpret_cast<Type const*>(input.data_ptr());
#define LAUNCH(RESIDUE_PER_8)                                                \
  cvt_to_mxfp4_residue<Type, RESIDUE_PER_8><<<grid, threads, 0, stream>>>(p, \
      input.stride(0), input.stride(1), masks, output.data_ptr<uint8_t>(),   \
      outputSF.data_ptr<uint8_t>(), numRows, numCols, numTotalCols,          \
      outputRows, paddedRows, swizzle, tile8, pdlMode)
  if (masks) {
    switch (residuePer8) {
      case 1:
        LAUNCH(1);
        break;
      case 2:
        LAUNCH(2);
        break;
      case 4:
        LAUNCH(4);
        break;
    }
  } else if (residue) {
    LAUNCH(8);
  } else {
    LAUNCH(0);
  }
#undef LAUNCH
}
}  // namespace

void mxfp4_residue_quant(torch::Tensor input, torch::Tensor output,
    torch::Tensor outputSF, c10::optional<torch::Tensor> masks,
    int64_t residuePer8, bool residue, bool swizzle, bool tile8,
    int64_t pdlMode) {
  TORCH_CHECK(pdlMode >= 0 && pdlMode <= 2,
      "pdl_mode must be 0/off, 1/tail or 2/early");
  TORCH_CHECK(input.is_cuda() && input.dim() == 2, "expected CUDA [M,K]");
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
  TORCH_CHECK(!masks || (residue && (residuePer8 == 1 || residuePer8 == 2 ||
                                        residuePer8 == 4)),
      "per8 must be 1/2/4");
  TORCH_CHECK(!tile8 || (residue && !masks && swizzle),
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
                         : outputRows == numRows * (residue && !masks ? 2 : 1)),
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
#define DTYPE(Type)                                                            \
  invokeMxfp4Quantization<Type>(input, output, outputSF, maskPtr, residuePer8, \
      residue, swizzle, tile8, pdlMode, stream)
  if (input.scalar_type() == at::kHalf) {
    DTYPE(half);
  } else if (input.scalar_type() == at::kBFloat16) {
    DTYPE(__nv_bfloat16);
  } else {
    TORCH_CHECK(input.scalar_type() == at::kFloat, "fp16/bf16/fp32 required");
    DTYPE(float);
  }
#undef DTYPE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
