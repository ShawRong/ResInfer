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

// Standalone vLLM-style NVFP4 activation quantization kernels for benchmarking.
//
// This file intentionally does not include the project's residue nvfp4_utils.cuh:
// that header is tied to the legacy 8-elements/thread residue layout.  Here we
// compile two explicit official-style variants in one extension:
//   - official_scaled_fp4_quant_pack8   :  8 elements/thread, 128-bit load
//   - official_scaled_fp4_quant_pack16  : 16 elements/thread, 256-bit load
//
// The implementation follows vLLM's swizzled-scale scaled_fp4_quant kernel, but
// uses a template parameter instead of the upstream CUDA_VERSION guard so both
// variants are available for raw-kernel latency comparisons.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <type_traits>

#include "cuda_utils.h"
#include "dispatch_utils.h"
#include "launch_bounds_utils.h"

namespace nvfp4_official_bench {

constexpr int kSfVecSize = 16;

struct u32x2 {
  uint32_t lo, hi;
};

struct alignas(32) u32x8_t {
  uint32_t d[8];
};

template <typename T>
struct CudaTypeConverter {
  using Type = T;
};

template <>
struct CudaTypeConverter<at::Half> {
  using Type = half;
};

template <>
struct CudaTypeConverter<at::BFloat16> {
  using Type = __nv_bfloat16;
};

template <typename T>
struct PackedTypeConverter;

template <>
struct PackedTypeConverter<half> {
  using Type = half2;
};

template <>
struct PackedTypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

template <class Type, bool Pack16>
struct alignas(Pack16 ? 32 : 16) PackedVec {
  typename PackedTypeConverter<Type>::Type elts[Pack16 ? 8 : 4];
};

template <typename Packed>
__device__ __forceinline__ float2 cast_to_float2(Packed const& val) {
  if constexpr (std::is_same_v<Packed, __nv_bfloat162>) {
    return __bfloat1622float2(val);
  } else {
    return __half22float2(val);
  }
}

template <typename Int>
__host__ __device__ __forceinline__ Int round_up(Int x, Int y) {
  return ((x + y - 1) / y) * y;
}

template <typename Int>
__host__ __device__ __forceinline__ Int div_round_up(Int x, Int y) {
  return (x + y - 1) / y;
}

__host__ __device__ __forceinline__ int compute_effective_rows(int m) {
  return round_up(m, 128);
}

__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
}

__device__ __forceinline__ u32x2 fp32_vec16_to_e2m1(float2 (&array)[8]) {
  u32x2 out;
  asm volatile(
      "{\n"
      ".reg .b8 b0;\n"
      ".reg .b8 b1;\n"
      ".reg .b8 b2;\n"
      ".reg .b8 b3;\n"
      ".reg .b8 b4;\n"
      ".reg .b8 b5;\n"
      ".reg .b8 b6;\n"
      ".reg .b8 b7;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b0,  %3,  %2;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b1,  %5,  %4;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b2,  %7,  %6;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b3,  %9,  %8;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b4, %11, %10;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b5, %13, %12;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b6, %15, %14;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b7, %17, %16;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "mov.b32 %1, {b4, b5, b6, b7};\n"
      "}\n"
      : "=r"(out.lo), "=r"(out.hi)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y),
        "f"(array[4].x), "f"(array[4].y), "f"(array[5].x), "f"(array[5].y),
        "f"(array[6].x), "f"(array[6].y), "f"(array[7].x), "f"(array[7].y));
  return out;
}

template <bool Pack16>
__device__ __forceinline__ auto pack_fp4(float2 (&vals)[Pack16 ? 8 : 4]) {
  if constexpr (Pack16) {
    return fp32_vec16_to_e2m1(vals);
  } else {
    return fp32_vec8_to_e2m1(vals);
  }
}

#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(CUDART_VERSION) && CUDART_VERSION >= 12090
#define NVFP4_OFFICIAL_BENCH_256B_PTX_ENABLED 1
#else
#define NVFP4_OFFICIAL_BENCH_256B_PTX_ENABLED 0
#endif

__device__ __forceinline__ void ld256_cg_or_zero(u32x8_t& val, const void* ptr,
                                                 bool pred) {
#if NVFP4_OFFICIAL_BENCH_256B_PTX_ENABLED
  asm volatile(
      "{\n"
      "  .reg .pred pr;\n"
      "  setp.ne.u32 pr, %8, 0;\n"
      "  mov.u32 %0, 0;\n"
      "  mov.u32 %1, 0;\n"
      "  mov.u32 %2, 0;\n"
      "  mov.u32 %3, 0;\n"
      "  mov.u32 %4, 0;\n"
      "  mov.u32 %5, 0;\n"
      "  mov.u32 %6, 0;\n"
      "  mov.u32 %7, 0;\n"
      "  @pr ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%9];\n"
      "}\n"
      : "=r"(val.d[0]), "=r"(val.d[1]), "=r"(val.d[2]), "=r"(val.d[3]),
        "=r"(val.d[4]), "=r"(val.d[5]), "=r"(val.d[6]), "=r"(val.d[7])
      : "r"((int)pred), "l"(ptr));
#else
  val = u32x8_t{};
#endif
}

__device__ __forceinline__ void ld128_cg_or_zero(uint4& val, const void* ptr,
                                                 bool pred) {
  uint32_t r0, r1, r2, r3;
  asm volatile(
      "{\n"
      "  .reg .pred pr;\n"
      "  setp.ne.u32 pr, %4, 0;\n"
      "  mov.u32 %0, 0;\n"
      "  mov.u32 %1, 0;\n"
      "  mov.u32 %2, 0;\n"
      "  mov.u32 %3, 0;\n"
      "  @pr ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%5];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"((int)pred), "l"(ptr));
  val = uint4{r0, r1, r2, r3};
}

template <int ThreadsPerSf>
__device__ __forceinline__ uint8_t* swizzled_sf_out_offset(
    int row_idx, int col_idx, int32_t num_k_tiles, uint32_t* sf_out) {
  static_assert(ThreadsPerSf == 1 || ThreadsPerSf == 2);
  if (threadIdx.x % ThreadsPerSf != 0) {
    return nullptr;
  }

  int32_t k_idx = col_idx / ThreadsPerSf;
  int32_t m_idx = row_idx;
  int32_t m_tile_idx = m_idx >> 7;
  int32_t outer_m_idx = m_idx & 31;
  int32_t inner_m_idx = (m_idx >> 5) & 3;
  int32_t k_tile_idx = k_idx >> 2;
  int32_t inner_k_idx = k_idx & 3;

  int64_t sf_offset = (static_cast<int64_t>(m_tile_idx) * num_k_tiles +
                       k_tile_idx)
                          << 9 |
                      (outer_m_idx << 4) | (inner_m_idx << 2) | inner_k_idx;
  return reinterpret_cast<uint8_t*>(sf_out) + sf_offset;
}

template <class Type, bool Pack16, bool UE8M0_SF = false>
__device__ __forceinline__ auto cvt_warp_fp16_to_fp4(
    PackedVec<Type, Pack16>& vec, float sf_scale_val, uint8_t* sf_out) {
  constexpr int EltsPerThread = Pack16 ? 16 : 8;
  constexpr int ThreadsPerSf = kSfVecSize / EltsPerThread;

  auto local_max = __habs2(vec.elts[0]);
#pragma unroll
  for (int i = 1; i < EltsPerThread / 2; ++i) {
    local_max = __hmax2(local_max, __habs2(vec.elts[i]));
  }

  if constexpr (ThreadsPerSf == 2) {
    local_max = __hmax2(__shfl_xor_sync(0xffffffffu, local_max, 1), local_max);
  }
  float vec_max = float(__hmax(local_max.x, local_max.y));

  float sf_value = sf_scale_val * (vec_max * reciprocal_approximate_ftz(6.0f));
  uint8_t fp8_sf_val;
  if constexpr (UE8M0_SF) {
    uint32_t tmp = reinterpret_cast<uint32_t&>(sf_value) >> 23;
    fp8_sf_val = tmp & 0xff;
    reinterpret_cast<uint32_t&>(sf_value) = tmp << 23;
  } else {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(sf_value);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8_sf_val) = tmp;
    sf_value = float(tmp);
  }
  if (sf_out) {
    *sf_out = fp8_sf_val;
  }

  float output_scale =
      sf_value != 0.0f
          ? reciprocal_approximate_ftz(sf_value *
                                       reciprocal_approximate_ftz(sf_scale_val))
          : 0.0f;

  float2 fp2_vals[EltsPerThread / 2];
#pragma unroll
  for (int i = 0; i < EltsPerThread / 2; ++i) {
    fp2_vals[i] = cast_to_float2(vec.elts[i]);
    fp2_vals[i].x *= output_scale;
    fp2_vals[i].y *= output_scale;
  }

  return pack_fp4<Pack16>(fp2_vals);
}

template <class Type, bool Pack16, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    official_cvt_fp16_to_fp4_swizzled(int32_t num_rows, int32_t num_cols,
                                      int32_t num_padded_cols,
                                      Type const* __restrict__ in,
                                      float const* __restrict__ sf_scale,
                                      uint32_t* __restrict__ out,
                                      uint32_t* __restrict__ sf_out) {
  constexpr int EltsPerThread = Pack16 ? 16 : 8;
  constexpr int ThreadsPerSf = kSfVecSize / EltsPerThread;
  using Vec = PackedVec<Type, Pack16>;
  static_assert(sizeof(Vec) == sizeof(Type) * EltsPerThread,
                "Packed vector size mismatch.");

  int32_t const num_k_tiles = (num_cols + 63) / 64;
  int32_t const sf_m = round_up(num_rows, 128);
  int32_t const col_idx = blockDim.x * blockIdx.y + threadIdx.x;
  int32_t const elem_idx = col_idx * EltsPerThread;
  float const global_scale = sf_scale == nullptr ? 1.0f : sf_scale[0];

  for (int row_idx = blockIdx.x; row_idx < sf_m; row_idx += gridDim.x) {
    if (col_idx < num_padded_cols) {
      Vec in_vec{};
      bool valid = row_idx < num_rows && elem_idx < num_cols;
      int64_t in_offset =
          static_cast<int64_t>(row_idx) * (num_cols / EltsPerThread) + col_idx;

      if constexpr (Pack16) {
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec),
                         &reinterpret_cast<uint32_t const*>(in)[in_offset * 8],
                         valid);
      } else {
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<uint32_t const*>(in)[in_offset * 4],
                         valid);
      }

      auto sf_pos = swizzled_sf_out_offset<ThreadsPerSf>(
          row_idx, col_idx, num_k_tiles, sf_out);
      auto out_val =
          cvt_warp_fp16_to_fp4<Type, Pack16, UE8M0_SF>(in_vec, global_scale,
                                                       sf_pos);

      if (valid) {
        if constexpr (Pack16) {
          int64_t out_offset =
              static_cast<int64_t>(row_idx) * (num_cols / 8) + col_idx * 2;
          uint64_t packed64 =
              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
          reinterpret_cast<uint64_t*>(out)[out_offset >> 1] = packed64;
        } else {
          out[in_offset] = out_val;
        }
      }
    }
  }
}

template <bool Pack16>
void launch_official_scaled_fp4_quant(torch::Tensor& output,
                                      torch::Tensor const& input,
                                      torch::Tensor& output_sf,
                                      torch::Tensor const& input_sf) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(output_sf.is_cuda(), "output_scale must be CUDA");
  TORCH_CHECK(input_sf.is_cuda(), "input_scale must be CUDA");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(output_sf.is_contiguous(), "output_scale must be contiguous");
  TORCH_CHECK(input_sf.numel() >= 1, "input_scale must contain at least one value");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Byte,
              "output must be uint8");
  TORCH_CHECK(output_sf.scalar_type() == at::ScalarType::Byte,
              "output_scale must be uint8");
  TORCH_CHECK(input_sf.scalar_type() == at::ScalarType::Float,
              "input_scale must be float32");

  int32_t m = static_cast<int32_t>(input.size(0));
  int32_t n = static_cast<int32_t>(input.size(1));
  TORCH_CHECK(n % 16 == 0, "K dimension must be a multiple of 16");
  TORCH_CHECK(output.numel() >= static_cast<int64_t>(m) * n / 2,
              "output buffer is too small");

  constexpr int EltsPerThread = Pack16 ? 16 : 8;
  int sf_n_unpadded = n / kSfVecSize;
  int sf_n_int = round_up(sf_n_unpadded, 4) / 4;
  int32_t num_padded_cols = sf_n_int * 4 * kSfVecSize / EltsPerThread;
  int64_t required_sf =
      static_cast<int64_t>(round_up(m, 128)) * round_up(sf_n_unpadded, 4);
  TORCH_CHECK(output_sf.numel() >= required_sf,
              "output_scale buffer is too small for swizzled layout");

  int multi_processor_count = get_device_attribute(cudaDevAttrMultiProcessorCount, -1);
  dim3 block(std::min(int(n / EltsPerThread), 512));
  int num_blocks_per_sm = vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  int grid_y = div_round_up(num_padded_cols, static_cast<int>(block.x));
  int grid_x = std::min(compute_effective_rows(m),
                        std::max(1, (multi_processor_count * num_blocks_per_sm) /
                                        std::max(1, grid_y)));
  dim3 grid(grid_x, grid_y);

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = reinterpret_cast<uint32_t*>(output_sf.data_ptr());
  auto output_ptr = reinterpret_cast<uint32_t*>(output.data_ptr());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  VLLM_DISPATCH_HALF_TYPES(input.scalar_type(), "official_nvfp4_quant_kernel", [&] {
    using cuda_type = typename CudaTypeConverter<scalar_t>::Type;
    auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
    official_cvt_fp16_to_fp4_swizzled<cuda_type, Pack16, false>
        <<<grid, block, 0, stream>>>(m, n, num_padded_cols, input_ptr,
                                     input_sf_ptr, output_ptr, sf_out);
  });
}

}  // namespace nvfp4_official_bench

void official_scaled_fp4_quant_pack8(torch::Tensor& output,
                                     torch::Tensor const& input,
                                     torch::Tensor& output_sf,
                                     torch::Tensor const& input_sf) {
  nvfp4_official_bench::launch_official_scaled_fp4_quant<false>(
      output, input, output_sf, input_sf);
}

void official_scaled_fp4_quant_pack16(torch::Tensor& output,
                                      torch::Tensor const& input,
                                      torch::Tensor& output_sf,
                                      torch::Tensor const& input_sf) {
  nvfp4_official_bench::launch_official_scaled_fp4_quant<true>(
      output, input, output_sf, input_sf);
}
