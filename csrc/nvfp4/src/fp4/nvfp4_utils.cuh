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

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>


#define ELTS_PER_THREAD 8
// 
constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;
constexpr int CVT_FP4_ELTS_PER_UINT32 = 8;

__host__ __device__ __forceinline__ int round_up(int x, int n) {
  return ((x + n - 1) / n) * n;
}

// Match upstream swizzled-scale quant launch behavior so small-M serving
// iterations still initialize the full tiled SF layout and avoid collapsing to
// a single active CTA.
__host__ __device__ __forceinline__ int computeEffectiveRows(int m) {
  constexpr int ROW_TILE = 128;
  return round_up(m, ROW_TILE);
}



template <int RESIDUE_PER_8_ELTS>
constexpr int get_residue_per_thread() {
    return RESIDUE_PER_8_ELTS * (CVT_FP4_ELTS_PER_THREAD / 8);
}

// Static assertions for validation
static_assert(CVT_FP4_ELTS_PER_THREAD == 8, "Current implementation assumes 8 elements per thread");

namespace vllm {

// Helper function to compute warp mask for a contiguous group of threads
// THREADS_PER_GROUP must be 1, 2, 4, 8, 16, or 32
template <int THREADS_PER_GROUP>
constexpr uint32_t compute_group_mask(int group_id) {
  return ((1u << THREADS_PER_GROUP) - 1u) << (group_id * THREADS_PER_GROUP);
}

// Warp group max reduction using shuffle
// Reduces max value within a fixed-size group of threads in a warp
template <int GROUP_THREADS>
__device__ __forceinline__ float warp_group_max(float v, int laneId) {
  int groupId = laneId / GROUP_THREADS;
  uint32_t mask = compute_group_mask<GROUP_THREADS>(groupId);
  #pragma unroll
  for (int off = 1; off < GROUP_THREADS; off <<= 1) {
    v = fmaxf(v, __shfl_xor_sync(mask, v, off));
  }
  return v;
}

// Extract values from float2 array using sparse bitmask iteration
// Input: fp2Vals[4] containing 8 float values (packed as float2)
//        mask: 8-bit mask indicating which values to extract
// Output: output[] array filled with extracted values (in mask bit order)
// Returns: number of values extracted
template <int MAX_OUTPUT>
__device__ __forceinline__ int extract_values_by_mask(
    const float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2],
    uint8_t mask,
    float output[MAX_OUTPUT]
) {
  int idx = 0;
  uint8_t extractMask = mask;
  while (extractMask) {
    int i = __ffs(extractMask) - 1;  // Find first set bit (0-indexed)
    int pairIdx = i >> 1;            // i / 2
    int offset = i & 1;              // i % 2
    output[idx++] = (offset == 0) ? fp2Vals[pairIdx].x : fp2Vals[pairIdx].y;
    extractMask &= extractMask - 1;  // Clear lowest set bit
  }
  return idx;
}

// Convert PyTorch cpp type to CUDA type
template <typename T>
struct CUDATypeConverter {
  using Type = T;
};

template <>
struct CUDATypeConverter<at::Half> {
  using Type = half;
};

template <>
struct CUDATypeConverter<at::BFloat16> {
  using Type = __nv_bfloat16;
};

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
  using Type = half2;
};  // keep for generality

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// Define a 16 bytes packed data type.
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};

template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
};


// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
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
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
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
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
}

// input and output using the same buffer:
//   in_out_buffer[i] initially holds scaled input (x,y)
//   after call, it holds dequant number
__device__ __forceinline__ uint32_t fp32x8_to_e2m1_with_dequant(
    float2 (&in_out_buffer)[4]) {

  uint32_t packed_u32;
  uint32_t f16x2_0, f16x2_1, f16x2_2, f16x2_3;

  asm volatile(
      "{\n"
      "  .reg .b8 b0, b1, b2, b3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b0, %6,  %5;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b1, %8,  %7;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b2, %10, %9;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b3, %12, %11;\n"
      "  cvt.rn.f16x2.e2m1x2           %0, b0;\n"
      "  cvt.rn.f16x2.e2m1x2           %1, b1;\n"
      "  cvt.rn.f16x2.e2m1x2           %2, b2;\n"
      "  cvt.rn.f16x2.e2m1x2           %3, b3;\n"
      "  mov.b32 %4, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(f16x2_0), "=r"(f16x2_1), "=r"(f16x2_2), "=r"(f16x2_3),
        "=r"(packed_u32)
      : "f"(in_out_buffer[0].x), "f"(in_out_buffer[0].y),
        "f"(in_out_buffer[1].x), "f"(in_out_buffer[1].y),
        "f"(in_out_buffer[2].x), "f"(in_out_buffer[2].y),
        "f"(in_out_buffer[3].x), "f"(in_out_buffer[3].y));

  // Avoid memcpy: bit-cast u32 -> half2
  union U32Half2 {
    uint32_t u;
    __half2  h2;
  };

  auto residual_inplace = [&](uint32_t f16x2_u32, float2 &inout) {
    U32Half2 v;
    v.u = f16x2_u32;

    // Reconstructed values (lane order follows your original unpack style)
    __half hx = __low2half(v.h2);
    __half hy = __high2half(v.h2);

    float hat_x = __half2float(hx);
    float hat_y = __half2float(hy);

    inout.x = hat_x;
    inout.y = hat_y;
  };

  residual_inplace(f16x2_0, in_out_buffer[0]);
  residual_inplace(f16x2_1, in_out_buffer[1]);
  residual_inplace(f16x2_2, in_out_buffer[2]);
  residual_inplace(f16x2_3, in_out_buffer[3]);

  return packed_u32;
}

// Separate input and output buffers:
//   input_buffer[i] holds normalized input (x,y) - preserved
//   dequant_buffer[i] receives dequantized values
__device__ __forceinline__ uint32_t fp32x8_to_e2m1_separate(
    const float2 (&input_buffer)[4],
    float2 (&dequant_buffer)[4]) {

  uint32_t packed_u32;
  uint32_t f16x2_0, f16x2_1, f16x2_2, f16x2_3;

  asm volatile(
      "{\n"
      "  .reg .b8 b0, b1, b2, b3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b0, %6,  %5;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b1, %8,  %7;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b2, %10, %9;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b3, %12, %11;\n"
      "  cvt.rn.f16x2.e2m1x2           %0, b0;\n"
      "  cvt.rn.f16x2.e2m1x2           %1, b1;\n"
      "  cvt.rn.f16x2.e2m1x2           %2, b2;\n"
      "  cvt.rn.f16x2.e2m1x2           %3, b3;\n"
      "  mov.b32 %4, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(f16x2_0), "=r"(f16x2_1), "=r"(f16x2_2), "=r"(f16x2_3),
        "=r"(packed_u32)
      : "f"(input_buffer[0].x), "f"(input_buffer[0].y),
        "f"(input_buffer[1].x), "f"(input_buffer[1].y),
        "f"(input_buffer[2].x), "f"(input_buffer[2].y),
        "f"(input_buffer[3].x), "f"(input_buffer[3].y));

  // Avoid memcpy: bit-cast u32 -> half2
  union U32Half2 {
    uint32_t u;
    __half2  h2;
  };

  auto write_dequant = [&](uint32_t f16x2_u32, float2 &out) {
    U32Half2 v;
    v.u = f16x2_u32;

    __half hx = __low2half(v.h2);
    __half hy = __high2half(v.h2);

    out.x = __half2float(hx);
    out.y = __half2float(hy);
  };

  write_dequant(f16x2_0, dequant_buffer[0]);
  write_dequant(f16x2_1, dequant_buffer[1]);
  write_dequant(f16x2_2, dequant_buffer[2]);
  write_dequant(f16x2_3, dequant_buffer[3]);

  return packed_u32;
}

__device__ __forceinline__ uint8_t fp32x2_to_e2m1_qdq_residue(
    float x, float y,
    float &x_hat, float &y_hat,
    float &x_res, float &y_res)
{
  uint32_t f16x2_u32;   // reconstructed half2 packed in 32b
  uint32_t qbyte_u32;   // store qb in low 8 bits

  asm volatile(
    "{\n"
    "  .reg .b8 qb;\n"
    // Keep the same operand order style as your original (y, x).
    // If your lane mapping differs, swap %1/%2 here.
    "  cvt.rn.satfinite.e2m1x2.f32   qb, %2, %1;\n"
    "  cvt.rn.f16x2.e2m1x2           %0, qb;\n"
    "  mov.b32                       %3, qb;\n"
    "}\n"
    : "=r"(f16x2_u32), "=r"(qbyte_u32)
    : "f"(x), "f"(y)
  );

  union U32Half2 { uint32_t u; __half2 h2; } v;
  v.u = f16x2_u32;

  __half hx = __low2half(v.h2);
  __half hy = __high2half(v.h2);

  x_hat = __half2float(hx);
  y_hat = __half2float(hy);

  x_res = x - x_hat;
  y_res = y - y_hat;

  return (uint8_t)(qbyte_u32 & 0xFFu);
}

__device__ __forceinline__ void e2m1x2_byte_to_f32x2(
    uint8_t qbyte,
    float &x_hat, float &y_hat)
{
  uint32_t f16x2_u32;

  asm volatile(
    "{\n"
    "  .reg .b8 qb;\n"
    "  mov.b32 qb, %1;\n"
    "  cvt.rn.f16x2.e2m1x2 %0, qb;\n"
    "}\n"
    : "=r"(f16x2_u32)
    : "r"((uint32_t)qbyte)
  );

  union U32Half2 { uint32_t u; __half2 h2; } v;
  v.u = f16x2_u32;

  // Assumption: low half corresponds to low nibble (x), high half to high nibble (y)
  x_hat = __half2float(__low2half(v.h2));
  y_hat = __half2float(__high2half(v.h2));
}

__device__ __forceinline__ uint8_t fp32_to_e2m1_qdq_residue(
    float x,
    float &x_hat,
    float &x_res)
{
  float y_hat, y_res;
  uint8_t qbyte = fp32x2_to_e2m1_qdq_residue(x, 0.0f, x_hat, y_hat, x_res, y_res);

  // lane-to-nibble mapping assumption:
  // low nibble holds x, high nibble holds y
  return (uint8_t)(qbyte & 0x0Fu);
}

__device__ __forceinline__ uint8_t fp32x2_to_e2m1_qdq_residue(
    const float in[2],
    float hat[2],
    float res[2])
{
  return fp32x2_to_e2m1_qdq_residue(
      in[0], in[1],
      hat[0], hat[1],
      res[0], res[1]);
}

__device__ __forceinline__ uint16_t fp32x4_to_e2m1_qdq_residue(
    const float in[4],
    float hat[4],
    float res[4])
{
  uint8_t b0 = fp32x2_to_e2m1_qdq_residue(
      in[0], in[1],
      hat[0], hat[1],
      res[0], res[1]);

  uint8_t b1 = fp32x2_to_e2m1_qdq_residue(
      in[2], in[3],
      hat[2], hat[3],
      res[2], res[3]);

  return (uint16_t)( (uint16_t)b0 | ((uint16_t)b1 << 8) );
}


// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Compute scale factor value from max absolute value.
// Formula: SF = SFScaleVal * (maxVal / 6.0)
// The division by 6.0 normalizes the max value to FP4 e2m1 representable range.
__device__ __forceinline__
float compute_sf_from_max(float maxVal, float SFScaleVal) {
  return SFScaleVal * (maxVal * reciprocal_approximate_ftz(6.0f));
}

// Encode scale factor to FP8 format and return the quantized value as float.
// Template parameter UE8M0_SF controls encoding format:
//   - true: UE8M0 format (exponent only, 8-bit unsigned exponent)
//   - false: E4M3 format (standard FP8 with 4-bit exponent, 3-bit mantissa)
// Returns: The quantized SF value converted back to float32
// Side effect: Writes the FP8 byte to *out_fp8 if out_fp8 is non-null
template <bool UE8M0_SF>
__device__ __forceinline__
float encode_sf_to_fp8(float SFValue, uint8_t* out_fp8) {
  uint8_t fp8;
  if constexpr (UE8M0_SF) {
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8 = tmp & 0xff;
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23; // SFValue becomes "rounded" exponent-only float
  } else {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8) = tmp;
    SFValue = float(tmp);
  }
  if (out_fp8) *out_fp8 = fp8;
  return SFValue; // Return quantized SF as float
}

// Compute output scale with precise numerical ordering for FP8->FP4 quantization path.
// DO NOT rewrite: this ordering is numerically chosen for accuracy.
// This function ensures consistent precision across main and residue quantization paths.
__device__ __forceinline__
float compute_output_scale_precise(float SFValue, float SFScaleVal) {
  return (SFValue != 0.0f) ? SFScaleVal * reciprocal_approximate_ftz(SFValue) : 0.0f;
}

// Shuffle-based collection of values within a group.
// Each thread in a group collects all values from its group members into outputChunk[4].
// Used for both salient (swap version) and residue (non-swap version) quantization.
//
// Template parameters:
//   RESIDUE_PER_THREAD: number of values per thread (1, 2, 4, or 8)
//
// Parameters:
//   scaledValues: input float2 array containing scaled values to shuffle
//   groupFirstThread: first thread index of this group
//   outputChunk: output float2[4] array to store collected values
template <int RESIDUE_PER_THREAD>
__device__ __forceinline__ void shuffle_collect_group_float2(
    const float2 (&scaledValues)[(RESIDUE_PER_THREAD + 1) / 2],
    int groupFirstThread,
    float2 (&outputChunk)[4]) {
  if constexpr (RESIDUE_PER_THREAD == 1) {
    // Special case: 4 groups of 8 threads each
    // All threads in a group collect the same 8 values from their group members
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int sourceThread0 = groupFirstThread + i * 2;
      int sourceThread1 = groupFirstThread + i * 2 + 1;

      // Each source thread's scaledValues[0].x contains its value
      float val0 = __shfl_sync(__activemask(), scaledValues[0].x, sourceThread0);
      float val1 = __shfl_sync(__activemask(), scaledValues[0].x, sourceThread1);
      outputChunk[i] = make_float2(val0, val1);
    }
  } else {
    // General case: RESIDUE_PER_THREAD >= 2
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int sourceThread = groupFirstThread + i / (RESIDUE_PER_THREAD / 2);
      int float2Offset = i % (RESIDUE_PER_THREAD / 2);

      // Shuffle float2 as unsigned long long
      unsigned long long tmp = __shfl_sync(__activemask(),
        reinterpret_cast<const unsigned long long&>(scaledValues[float2Offset]),
        sourceThread);
      outputChunk[i] = reinterpret_cast<float2&>(tmp);
    }
  }
}

// Specialized fast path for the M-extension ratio=1.0 layout.
//
// This path intentionally does not reuse the generic residue machinery:
// - no sparse mask extraction
// - no warp-level residue gather
// - no extended-K residue address logic
//
// Instead it performs:
// 1. standard main quantization + QDQ
// 2. residue = original - dequant(main)
// 3. a second simple quantization pass for the residue row
//
// With the current 8-elements-per-thread implementation, one scale factor
// still covers 16 elements, so the residue amax uses the same 2-thread
// reduction as the main tensor.
template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ uint32_t cvt_warp_fp16_to_fp4_mext_r1_fast(
    PackedVec<Type>& vec,
    float SFScaleVal,
    float SFScaleResVal,
    uint8_t* SFout,
    uint32_t* residueDataOut,
    uint8_t* residueSFOut) {
  constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);

  // ============================================================
  // Main quantization (standard FP4 path)
  // ============================================================
  auto localMax = __habs2(vec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  }
  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = compute_sf_from_max(vecMax, SFScaleVal);
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 fp2ValsOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 mainScaled[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2ValsOriginal[i] = __half22float2(vec.elts[i]);
    } else {
      fp2ValsOriginal[i] = __bfloat1622float2(vec.elts[i]);
    }
    mainScaled[i].x = fp2ValsOriginal[i].x * outputScale;
    mainScaled[i].y = fp2ValsOriginal[i].y * outputScale;
  }

  // QDQ the main tensor so we can form the exact runtime residue.
  uint32_t e2m1Vec = fp32x8_to_e2m1_with_dequant(mainScaled);

  // ============================================================
  // Residue quantization (simple dense ratio=1.0 path)
  // ============================================================
  float inverseOutputScale =
      outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;

  float2 residueVals[CVT_FP4_ELTS_PER_THREAD / 2];
  float localMaxResidue = 0.0f;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    float deq_x = mainScaled[i].x * inverseOutputScale;
    float deq_y = mainScaled[i].y * inverseOutputScale;

    residueVals[i].x = fp2ValsOriginal[i].x - deq_x;
    residueVals[i].y = fp2ValsOriginal[i].y - deq_y;

    localMaxResidue = fmaxf(
        localMaxResidue,
        fmaxf(fabsf(residueVals[i].x), fabsf(residueVals[i].y)));
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMaxResidue = fmaxf(
        localMaxResidue,
        __shfl_xor_sync(uint32_t(-1), localMaxResidue, 1));
  }

  // Dual-alpha: the residue half encodes its SF against its OWN global scale
  // (SFScaleResVal == SFScaleVal in the legacy single-scale path).
  float residueSFValue = compute_sf_from_max(localMaxResidue, SFScaleResVal);
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale =
      compute_output_scale_precise(residueSFValue, SFScaleResVal);

  float2 residueScaled[CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    residueScaled[i].x = residueVals[i].x * residueOutputScale;
    residueScaled[i].y = residueVals[i].y * residueOutputScale;
  }

  if (residueDataOut) {
    *residueDataOut = fp32_vec_to_e2m1(residueScaled);
  }

  return e2m1Vec;
}

struct u32x2 {
  uint32_t lo, hi;
};

// Load 16 fp16/bf16 elements (one full SF vector, 32 bytes) in a single
// 256-bit global load. Blackwell (sm_100+) exposes ld.global.v8.b32
// (PTX ISA 8.8); older targets fall back to two 128-bit loads. `ptr` must be
// 32-byte aligned -- guaranteed by K % 16 == 0 plus torch's base alignment.
template <class Type>
__device__ __forceinline__ void load_packed_vec16(Type const* ptr,
                                                  PackedVec<Type>& lo,
                                                  PackedVec<Type>& hi) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  union {
    uint32_t u[8];
    struct {
      PackedVec<Type> lo, hi;
    } v;
  } tmp;
  asm volatile(
      "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(tmp.u[0]), "=r"(tmp.u[1]), "=r"(tmp.u[2]), "=r"(tmp.u[3]),
        "=r"(tmp.u[4]), "=r"(tmp.u[5]), "=r"(tmp.u[6]), "=r"(tmp.u[7])
      : "l"(ptr));
  lo = tmp.v.lo;
  hi = tmp.v.hi;
#else
  lo = reinterpret_cast<PackedVec<Type> const*>(ptr)[0];
  hi = reinterpret_cast<PackedVec<Type> const*>(ptr)[1];
#endif
}

// Transitional pack16 path for mext_r1:
// - one thread owns 16 elements
// - main/residue SF are each computed once over all 16 elements
// - quantization still reuses the validated 8-element QDQ helpers on the
//   low/high 8-element half-chunks
template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ u32x2 cvt_warp_fp16_to_fp4_mext_r1_fast16(
    PackedVec<Type>& loVec,
    PackedVec<Type>& hiVec,
    float SFScaleVal,
    float SFScaleResVal,
    uint8_t* SFout,
    uint32_t* residueDataOut,
    uint8_t* residueSFOut) {
  static_assert(CVT_FP4_ELTS_PER_THREAD == 8,
                "pack16 transitional path assumes 8-element base helper.");

  auto localMax = __habs2(loVec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(loVec.elts[i]));
  }

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(hiVec.elts[i]));
  }

  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = compute_sf_from_max(vecMax, SFScaleVal);
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 loOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 loScaled[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiScaled[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      loOriginal[i] = __half22float2(loVec.elts[i]);
      hiOriginal[i] = __half22float2(hiVec.elts[i]);
    } else {
      loOriginal[i] = __bfloat1622float2(loVec.elts[i]);
      hiOriginal[i] = __bfloat1622float2(hiVec.elts[i]);
    }
    loScaled[i].x = loOriginal[i].x * outputScale;
    loScaled[i].y = loOriginal[i].y * outputScale;
    hiScaled[i].x = hiOriginal[i].x * outputScale;
    hiScaled[i].y = hiOriginal[i].y * outputScale;
  }

  u32x2 mainPacked;
  mainPacked.lo = fp32x8_to_e2m1_with_dequant(loScaled);
  mainPacked.hi = fp32x8_to_e2m1_with_dequant(hiScaled);

  float inverseOutputScale =
      outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;

  float2 loResidue[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiResidue[CVT_FP4_ELTS_PER_THREAD / 2];
  float localMaxResidue = 0.0f;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    float loDeqX = loScaled[i].x * inverseOutputScale;
    float loDeqY = loScaled[i].y * inverseOutputScale;
    float hiDeqX = hiScaled[i].x * inverseOutputScale;
    float hiDeqY = hiScaled[i].y * inverseOutputScale;

    loResidue[i].x = loOriginal[i].x - loDeqX;
    loResidue[i].y = loOriginal[i].y - loDeqY;
    hiResidue[i].x = hiOriginal[i].x - hiDeqX;
    hiResidue[i].y = hiOriginal[i].y - hiDeqY;

    localMaxResidue = fmaxf(
        localMaxResidue,
        fmaxf(fabsf(loResidue[i].x), fabsf(loResidue[i].y)));
    localMaxResidue = fmaxf(
        localMaxResidue,
        fmaxf(fabsf(hiResidue[i].x), fabsf(hiResidue[i].y)));
  }

  float residueSFValue = compute_sf_from_max(localMaxResidue, SFScaleResVal);
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale =
      compute_output_scale_precise(residueSFValue, SFScaleResVal);

  float2 loResidueScaled[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiResidueScaled[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    loResidueScaled[i].x = loResidue[i].x * residueOutputScale;
    loResidueScaled[i].y = loResidue[i].y * residueOutputScale;
    hiResidueScaled[i].x = hiResidue[i].x * residueOutputScale;
    hiResidueScaled[i].y = hiResidue[i].y * residueOutputScale;
  }

  if (residueDataOut) {
    // Single 64-bit store (st.global.v2.b32); the residue offset is always
    // an even uint32 index because numCols % 16 == 0.
    *reinterpret_cast<uint2*>(residueDataOut) =
        make_uint2(fp32_vec_to_e2m1(loResidueScaled),
                   fp32_vec_to_e2m1(hiResidueScaled));
  }

  return mainPacked;
}

// Version with swap
template <class Type, bool UE8M0_SF = false, int RESIDUE_PER_8_ELTS = 1>
__device__ uint32_t cvt_warp_fp16_to_fp4_swap(PackedVec<Type>& vec, float SFScaleVal,
                                         uint8_t* SFout, uint8_t mask,
                                         uint32_t* residueDataOut, uint8_t* salientSFOut) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();

  // ============================================================
  // Quantization for salient
  // ============================================================
  
  // Transfer vec to float2
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
  }

  // Extract salient from salient channel mask (using sparse extraction helper)
  float salient[RESIDUE_PER_THREAD > 0 ? RESIDUE_PER_THREAD : 1];
  extract_values_by_mask<RESIDUE_PER_THREAD>(fp2Vals, mask, salient);

  int laneId = threadIdx.x % 32;


  // ============================================================
  // Phase 1: Compute shared SF for every 16 residues
  // ============================================================

  // Find group max
  float localMaxSalient = 0.0f;
  #pragma unroll
  for (int i = 0; i < RESIDUE_PER_THREAD; i++) {
    localMaxSalient = fmaxf(localMaxSalient, fabsf(salient[i]));
  }

  // Reduce within SF group (16 elements = 16 / RESIDUE_PER_THREAD threads)
  constexpr int THREADS_PER_SF_GROUP = 16 / RESIDUE_PER_THREAD;
  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int sfGroupThreadId = laneId % THREADS_PER_SF_GROUP;

  // All threads in SF group get the same max via warp shuffle
  float sfGroupMaxSalient = warp_group_max<THREADS_PER_SF_GROUP>(localMaxSalient, laneId);
  float salientSFValue = SFScaleVal * (sfGroupMaxSalient * reciprocal_approximate_ftz(6.0f));
  // Convert to FP8 and get quantized value
  salientSFValue = encode_sf_to_fp8<UE8M0_SF>(salientSFValue, salientSFOut);

  float salientOutputScale = compute_output_scale_precise(salientSFValue, SFScaleVal);


  // Store salient as float2 (handle odd RESIDUE_PER_THREAD)
  // Prepare for shuffle
  float2 shuffleSalient[(RESIDUE_PER_THREAD + 1) / 2];
  #pragma unroll
  for (int i = 0; i < (RESIDUE_PER_THREAD + 1) / 2; i++) {
    float val0 = (i * 2 < RESIDUE_PER_THREAD) ? salient[i * 2]: 0.0f;
    float val1 = (i * 2 + 1 < RESIDUE_PER_THREAD) ? salient[i * 2 + 1]: 0.0f;
    shuffleSalient[i] = make_float2(val0, val1);
  }

  // Shuffle float2 directly
  // New design: All threads in a group perform full shuffle within the group
  // No backward shuffle needed - each thread can directly extract its residue
  float2 residue[(RESIDUE_PER_THREAD + 1) / 2];
  float2 salientChunk[4];

  // Determine group membership
  // For RESIDUE_PER_THREAD == 1: 4 groups of 8 threads
  // For RESIDUE_PER_THREAD == 2: 4 groups of 4 threads
  // For RESIDUE_PER_THREAD == 4: 4 groups of 2 threads
  // For RESIDUE_PER_THREAD == 8: 4 groups of 1 thread (no shuffle needed)
  constexpr int numThreadsPerGroup = 8 / RESIDUE_PER_THREAD;
  int groupId = laneId / numThreadsPerGroup;           // Which group (0-3)
  int threadInGroup = laneId % numThreadsPerGroup;     // Position within group (0-7 for RESIDUE_PER_THREAD==1)
  int groupFirstThread = groupId * numThreadsPerGroup; // First thread in this group

  // NEW: Collect UNSCALED salient values via shuffle (for residue computation without round-trip error)
  float2 unscaledSalientChunk[4];
  shuffle_collect_group_float2<RESIDUE_PER_THREAD>(shuffleSalient, groupFirstThread, unscaledSalientChunk);
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    salientChunk[i] = make_float2(
      unscaledSalientChunk[i].x * salientOutputScale,
      unscaledSalientChunk[i].y * salientOutputScale);
  }

  // TODO: Do the quantization using 1, 2, 4, 8 element versions based on RESIDUE_PER_THREAD
  // The current implementation always uses 8-element version for simplicity
  // But if we use less elements versio, we need 
  // 1. Do the quantization in each thread
  // 2. pass only the quantized bytes, nibble to the leader for writing to global memory for a uint32.
  // 3. Each thread naturally get their own residue from their own quantization step. no need for shuffle.

  // Quantize salient with separate buffers to preserve original values
  float2 dequantChunk[4];
  uint32_t quantizedChunk = fp32x8_to_e2m1_separate(salientChunk, dequantChunk);

  if (residueDataOut) {
    *residueDataOut = quantizedChunk;
  }
  // Compute residue: original - dequant
  // dequantChunk contains scaled dequantized values, unscaledSalientChunk contains original unscaled values
  float inverseSalientOutputScale = salientOutputScale != 0 ? reciprocal_approximate_ftz(salientOutputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // Scale dequantized values back to original space
    float deq_x = dequantChunk[i].x * inverseSalientOutputScale;
    float deq_y = dequantChunk[i].y * inverseSalientOutputScale;
    // Use UNSCALED originals directly (no round-trip error from scale * inverse_scale)
    float orig_x = unscaledSalientChunk[i].x;
    float orig_y = unscaledSalientChunk[i].y;
    // Store residue back in salientChunk for shuffle back
    salientChunk[i].x = orig_x - deq_x;
    salientChunk[i].y = orig_y - deq_y;
  }



  // No backward shuffle needed. Each thread directly extracts its residue from salientChunk
  // Since all threads in the group have the same salientChunk after quantization,
  // each thread just needs to extract the value at its position within the group
  if constexpr (RESIDUE_PER_THREAD == 1) {
    // threadInGroup ranges from 0-7 within each group
    // Extract the corresponding residue value from salientChunk
    // salientChunk[0] = {residue[0], residue[1]}
    // salientChunk[1] = {residue[2], residue[3]}
    // salientChunk[2] = {residue[4], residue[5]}
    // salientChunk[3] = {residue[6], residue[7]}
    int pairIdx = threadInGroup / 2;  // Which float2 pair (0-3)
    int elemIdx = threadInGroup % 2;   // Which element within the pair: 0=x, 1=y
    float val = (elemIdx == 0) ? salientChunk[pairIdx].x : salientChunk[pairIdx].y;
    residue[0] = make_float2(val, 0.0f);
  } else {
    // General case: RESIDUE_PER_THREAD >= 2
    // Each thread extracts its portion of residues from salientChunk
    constexpr int float2PerThread = RESIDUE_PER_THREAD / 2;
    #pragma unroll
    for (int i = 0; i < float2PerThread; i++) {
      int idx = threadInGroup * float2PerThread + i;
      residue[i] = salientChunk[idx];
    }
  }


  // ============================================================
  // Main quantization
  // ============================================================


  // we get residue in the scaledResidue2, we put the residue to the vec.elts here according to the mark.
  // Use sparse iteration for better performance when mask is sparse
  int residueWriteIdx = 0;
  uint8_t tempMask = mask;
  while (tempMask) {
    int i = __ffs(tempMask) - 1;  // Find first set bit (0-indexed)
    int pairIdx = i / 2;
    int offset = i % 2;
    float val;
    if constexpr (RESIDUE_PER_THREAD == 1) {
      // Special case: each thread has only 1 residue value stored in residue[0].x
      val = residue[0].x;
    } else {
      int f2Idx = residueWriteIdx / 2;
      int f2Off = residueWriteIdx % 2;
      val = (f2Off == 0) ? residue[f2Idx].x : residue[f2Idx].y;
    }
    if (offset == 0) {
      fp2Vals[pairIdx].x = val;
    } else {
      fp2Vals[pairIdx].y = val;
    }
    residueWriteIdx++;
    tempMask &= tempMask - 1;  // Clear lowest set bit
  }
  // ============================================================
  // Main SF computation - use FLOAT precision (same as non-swap residue SF)
  // Compute max BEFORE converting to half/bfloat16 to avoid precision loss
  // ============================================================
  float localMaxFloat = 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMaxFloat = fmaxf(localMaxFloat, fmaxf(fabsf(fp2Vals[i].x), fabsf(fp2Vals[i].y)));
  }
  // 2-thread reduction (same grouping as before)
  localMaxFloat = fmaxf(localMaxFloat, __shfl_xor_sync(uint32_t(-1), localMaxFloat, 1));
  float vecMax = localMaxFloat;

  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));

  // Convert to FP8 and get quantized value
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);

  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  // Scale fp2Vals directly in float (avoid half round-trip for precision)
  // This matches non-swap mode which scales residues directly in float
  float2 mainVals[CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    mainVals[i].x = fp2Vals[i].x * outputScale;
    mainVals[i].y = fp2Vals[i].y * outputScale;
  }

  uint32_t e2m1Vec = fp32_vec_to_e2m1(mainVals);

  // Store to vec.elts for any downstream use (after quantization to avoid affecting precision)
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      vec.elts[i] = __float22half2_rn(fp2Vals[i]);
    } else {
      vec.elts[i] = __float22bfloat162_rn(fp2Vals[i]);
    }
  }

  return e2m1Vec;
}

// Quantizes the provided PackedVec into the uint32_t output
// Use provided index of the channel to preserve. using channelIdx to indicate
template <class Type, bool UE8M0_SF = false, int RESIDUE_PER_8_ELTS = 1>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal,
                                         uint8_t* SFout, uint8_t mask,
                                         uint32_t* residueDataOut, uint8_t* residueSFOut) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  // ============================================================
  // Main quantization
  // ============================================================
  auto localMax = __habs2(vec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // TODO: use half as compute data type.
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // Convert to FP8 and get quantized value
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);

  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 fp2ValsOriginal[CVT_FP4_ELTS_PER_THREAD / 2];  // Preserve float originals for residue

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2ValsOriginal[i] = fp2Vals[i];  // Copy BEFORE scaling
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Quantize and Dequantize
  uint32_t e2m1Vec = fp32x8_to_e2m1_with_dequant(fp2Vals);

  // get residue
  float inverseOutputScale = outputScale != 0 ? reciprocal_approximate_ftz(outputScale) : 0.0f;
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // Scale dequantized values back to original space using inverse of outputScale
    fp2Vals[i].x *= inverseOutputScale;
    fp2Vals[i].y *= inverseOutputScale;

    // Compute residue: original - dequantized
    // Use preserved float originals (avoids half→float precision loss)
    fp2Vals[i].x = fp2ValsOriginal[i].x - fp2Vals[i].x;
    fp2Vals[i].y = fp2ValsOriginal[i].y - fp2Vals[i].y;
  }

  // ============================================================
  // Residue quantization
  // ============================================================

  // Extract residue from salient channel mask (using sparse extraction helper)
  float residues[RESIDUE_PER_THREAD > 0 ? RESIDUE_PER_THREAD : 1];
  extract_values_by_mask<RESIDUE_PER_THREAD>(fp2Vals, mask, residues);

  int laneId = threadIdx.x % 32;
  // ============================================================
  // Phase 1: Compute shared SF for every 16 residues
  // ============================================================

  // Find group max residue
  float localMaxResidue = 0.0f;
  #pragma unroll
  for (int i = 0; i < RESIDUE_PER_THREAD; i++) {
    localMaxResidue = fmaxf(localMaxResidue, fabsf(residues[i]));
  }

  // Reduce within SF group (16 elements = 16 / RESIDUE_PER_THREAD threads)
  constexpr int THREADS_PER_SF_GROUP = 16 / RESIDUE_PER_THREAD;
  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int sfGroupThreadId = laneId % THREADS_PER_SF_GROUP;

  // All threads in SF group get the same max via warp shuffle
  float sfGroupMaxResidue = warp_group_max<THREADS_PER_SF_GROUP>(localMaxResidue, laneId);
  float residueSFValue = SFScaleVal * (sfGroupMaxResidue * reciprocal_approximate_ftz(6.0f));
  // Convert to FP8 and get quantized value
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);

  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

  // Store residues as float2 (handle odd RESIDUE_PER_THREAD)
  float2 scaledResidues2[(RESIDUE_PER_THREAD + 1) / 2];
  #pragma unroll
  for (int i = 0; i < (RESIDUE_PER_THREAD + 1) / 2; i++) {
    float val0 = (i * 2 < RESIDUE_PER_THREAD) ? residues[i * 2] * residueOutputScale : 0.0f;
    float val1 = (i * 2 + 1 < RESIDUE_PER_THREAD) ? residues[i * 2 + 1] * residueOutputScale : 0.0f;
    scaledResidues2[i] = make_float2(val0, val1);
  }

  // Two-stage shuffle for normal residue extension:
  // Stage 1: Group-level shuffle - each group collects its residues
  // Stage 2: Inter-group shuffle - gather from group leaders to writer threads (0-3)

  // Determine group membership
  constexpr int numThreadsPerGroup = 8 / RESIDUE_PER_THREAD;
  int groupId = laneId / numThreadsPerGroup;           // Which group (0-3)
  int threadInGroup = laneId % numThreadsPerGroup;     // Position within group
  int groupFirstThread = groupId * numThreadsPerGroup; // First thread in this group

  // Stage 1: Group-level shuffle - collect residues from group members
  float2 groupResidueChunk[4];
  shuffle_collect_group_float2<RESIDUE_PER_THREAD>(scaledResidues2, groupFirstThread, groupResidueChunk);

  // Now each group's threads have collected their group's residues in groupResidueChunk
  // Stage 2: Gather from group leaders (threads 0, 8, 16, 24) to writer threads (0-3)
  // Writer thread 0 gets ALL 4 float2s from group 0 leader (thread 0)
  // Writer thread 1 gets ALL 4 float2s from group 1 leader (thread 8)
  // Writer thread 2 gets ALL 4 float2s from group 2 leader (thread 16)
  // Writer thread 3 gets ALL 4 float2s from group 3 leader (thread 24)

  float2 residueChunk2[4];

  // Determine which group leader this thread should read from
  // For writer threads 0-3: read from threads 0, 8, 16, 24 respectively
  // For other threads: doesn't matter, they won't write
  int mySourceGroupLeader = laneId * numThreadsPerGroup;  // 0→0, 1→8, 2→16, 3→24, 4→32(invalid), ...

  // Shuffle all 4 float2s from the source group leader to this thread
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    // Shuffle groupResidueChunk[i] from mySourceGroupLeader to this thread
    unsigned long long tmp = __shfl_sync(__activemask(),
      reinterpret_cast<unsigned long long&>(groupResidueChunk[i]),
      mySourceGroupLeader);
    residueChunk2[i] = reinterpret_cast<float2&>(tmp);
  }

  // Quantize residues (no need for residual-of-residual)
  uint32_t quantizedChunk = fp32_vec_to_e2m1(residueChunk2);

  // Write residue data (only writer threads 0-3)
  if (residueDataOut) {
    *residueDataOut = quantizedChunk;
  }

  return e2m1Vec;
}

// Function to get residue data write address (for normal residue extension)
// Parameters:
//   rowIdx: current row index
//   colIdx: thread index (for warp/lane calculation)
//   numMainCols: number of base channels (K)
//   numTotalCols: extended dimension including residue (K + salient)
//   outputStart: pointer to start of output buffer
template <class ResidueType, int RESIDUE_PER_8_ELTS>
__device__ ResidueType* cvt_residue_data_get_offset(int rowIdx, int colIdx, int numMainCols, int numTotalCols, ResidueType* outputStart) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  constexpr int RESIDUES_PER_WARP = 32 * RESIDUE_PER_THREAD;
  constexpr int UINT32_PER_WARP = (RESIDUES_PER_WARP + 7) / 8;

  // Warp level consecutive write
  int warpId = colIdx / 32;
  int laneId = colIdx % 32;

  if (laneId < UINT32_PER_WARP) {
    // Row-interleaved layout: [Row 0: base|residue][Row 1: base|residue]...
    // Each row has (numTotalCols / CVT_FP4_ELTS_PER_UINT32) uint32s
    // Residue section starts at (numMainCols / CVT_FP4_ELTS_PER_UINT32) uint32s into each row
    int64_t uint32s_per_row = numTotalCols / CVT_FP4_ELTS_PER_UINT32;
    int64_t base_uint32s_per_row = numMainCols / CVT_FP4_ELTS_PER_UINT32;

    int64_t residueDataIdx = rowIdx * uint32s_per_row          // Start of this row
                           + base_uint32s_per_row              // Skip base section
                           + warpId * UINT32_PER_WARP          // This warp's residue section
                           + laneId;                           // This thread's uint32
    return outputStart + residueDataIdx;
  }
  return nullptr;
}

// Function to get residue data write address for SWAP extension
// In swap mode, each group's first thread writes the quantized salient chunk
// Group structure: 4 groups per warp, with (8/RESIDUE_PER_THREAD) threads per group
// Only threads 0, 8, 16, 24 (for RESIDUE_PER_THREAD==1) write
template <class ResidueType, int RESIDUE_PER_8_ELTS>
__device__ ResidueType* cvt_swap_residue_data_get_offset(int rowIdx, int colIdx, int numMainCols, int numTotalCols, ResidueType* outputStart) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  constexpr int RESIDUES_PER_WARP = 32 * RESIDUE_PER_THREAD;
  constexpr int UINT32_PER_WARP = (RESIDUES_PER_WARP + 7) / 8;

  // Warp level consecutive write
  int warpId = colIdx / 32;
  int laneId = colIdx % 32;

  // Group structure for swap extension
  constexpr int numThreadsPerGroup = 8 / RESIDUE_PER_THREAD;
  int groupId = laneId / numThreadsPerGroup;           // Which group (0-3)
  int threadInGroup = laneId % numThreadsPerGroup;     // Position within group

  // Only the first thread in each group writes (threads 0, 8, 16, 24 for RESIDUE_PER_THREAD==1)
  if (threadInGroup == 0 && groupId < UINT32_PER_WARP) {
    // Row-interleaved layout: [Row 0: base|residue][Row 1: base|residue]...
    // Each row has (numTotalCols / CVT_FP4_ELTS_PER_UINT32) uint32s
    // Residue section starts at (numMainCols / CVT_FP4_ELTS_PER_UINT32) uint32s into each row
    int64_t uint32s_per_row = numTotalCols / CVT_FP4_ELTS_PER_UINT32;
    int64_t base_uint32s_per_row = numMainCols / CVT_FP4_ELTS_PER_UINT32;

    int64_t residueDataIdx = rowIdx * uint32s_per_row          // Start of this row
                           + base_uint32s_per_row              // Skip base section
                           + warpId * UINT32_PER_WARP          // This warp's residue section
                           + groupId;                          // This group's uint32 (based on groupId, not laneId)
    return outputStart + residueDataIdx;
  }
  return nullptr;
}

// helper function for tiled offset calculation
template <class SFType>
__device__ inline int64_t compute_tiled_sf_offset(
  int32_t mIdx, int32_t kIdx, int32_t numTotalCols) {
  int32_t mTileIdx = mIdx / (32 * 4);
  constexpr int factor = CVT_FP4_SF_VEC_SIZE * 4; // 64
  int32_t numKTiles = (numTotalCols + factor - 1) / factor;
  int64_t mTileStride = numKTiles * 32 * 4 * 4;

  int32_t kTileIdx = (kIdx / 4); // kIdx here represents scale factor index, and 
  // 4 SFs x 16 elements = 64 elements
  int64_t kTileStride = 32 * 4 * 4;

  int32_t outerMIdx = (mIdx % 32);
  int64_t outerMStride = 4 * 4;

  int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
  int64_t innerMStride = 4;

  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;

  return mTileIdx * mTileStride + kTileIdx * kTileStride + 
  outerMIdx * outerMStride + innerMIdx * innerMStride + 
  innerKIdx * innerKStride;
}

__device__ __forceinline__ int32_t compute_mext_r1_base_row(
    int32_t rowIdx, int32_t numRows, int32_t layoutMode) {
  (void)numRows;
  if (layoutMode == 2 || layoutMode == 4) {
    constexpr int kTile8Rows = 8;
    int32_t chunk = rowIdx / kTile8Rows;
    int32_t local = rowIdx % kTile8Rows;
    return chunk * (2 * kTile8Rows) + local;
  }
  if (layoutMode == 1) {
    return rowIdx * 2;
  }
  return rowIdx;
}

__device__ __forceinline__ int32_t compute_mext_r1_residue_row(int32_t rowIdx,
                                                               int32_t numRows,
                                                               int32_t layoutMode) {
  if (layoutMode == 2 || layoutMode == 4) {
    constexpr int kTile8Rows = 8;
    int32_t chunk = rowIdx / kTile8Rows;
    int32_t local = rowIdx % kTile8Rows;
    return chunk * (2 * kTile8Rows) + kTile8Rows + local;
  }
  if (layoutMode == 1) {
    return rowIdx * 2 + 1;
  }
  return rowIdx + numRows;
}

// ResAttn launches one KQ CTA per (batch, kv_head). Its Q-scale descriptor
// therefore gives every logical 8-head group an independent 128-row SF atom,
// even though the packed Q data uses only 16 compact base+residue rows. Other
// MExt layouts use the data row as the SF row.
__device__ __forceinline__ int32_t compute_mext_r1_base_sf_row(
    int32_t rowIdx, int32_t numRows, int32_t layoutMode) {
  if (layoutMode != 4) {
    return compute_mext_r1_base_row(rowIdx, numRows, layoutMode);
  }
  constexpr int kTile8Rows = 8;
  int32_t chunk = rowIdx / kTile8Rows;
  int32_t local = rowIdx % kTile8Rows;
  return chunk * 128 + local;
}

__device__ __forceinline__ int32_t compute_mext_r1_residue_sf_row(
    int32_t rowIdx, int32_t numRows, int32_t layoutMode) {
  if (layoutMode != 4) {
    return compute_mext_r1_residue_row(rowIdx, numRows, layoutMode);
  }
  constexpr int kTile8Rows = 8;
  int32_t chunk = rowIdx / kTile8Rows;
  int32_t local = rowIdx % kTile8Rows;
  return chunk * 128 + kTile8Rows + local;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
  int rowIdx, int colIdx,
  int numTotalCols, SFType* SFout) {
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 ||
                CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: To decide how to write the sf in the end? stg.8 is not that good.
  // TODO: follow the residue way to write the SF.
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF vector index (16 elements share one SF in the K dimension).
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    int64_t SFOffset = compute_tiled_sf_offset<SFType>(
      mIdx, kIdx, numTotalCols);

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
  return nullptr;
}


// Function to get residue scale factor write address
template <class ResidueSFType, int RESIDUE_PER_8_ELTS>
__device__ ResidueSFType* cvt_residue_sf_get_offset(
  int rowIdx, int colIdx,
  int numTotalCols,
  int numMainCols,
  ResidueSFType* residueSFOut) {

  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  // Warp-based SF group logic
  // Every 16 residues share 1 SF
  // For ratio 1:8: 16 threads share 1 SF → THREADS_PER_SF_GROUP = 16
  // For ratio 2:8: 8 threads share 1 SF  → THREADS_PER_SF_GROUP = 8
  // For ratio 4:8: 4 threads share 1 SF  → THREADS_PER_SF_GROUP = 4
  // For ratio 8:8: 2 threads share 1 SF  → THREADS_PER_SF_GROUP = 2
  constexpr int THREADS_PER_SF_GROUP = 16 / RESIDUE_PER_THREAD;

  int warpId = colIdx / 32;
  int laneId = colIdx % 32;

  // Determine which SF group within the warp (0, 1, 2, ...)
  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int sfGroupThreadId = laneId % THREADS_PER_SF_GROUP;

  // Only the first thread in each SF group writes
  if (sfGroupThreadId != 0) return nullptr;

  // Calculate the residue element index for this SF group
  // Each warp has 2 groups of 16 threads (for ratio 1:8)
  // warpId * 32 threads * RESIDUE_PER_THREAD residues per thread
  int warpBaseResidueIdx = warpId * 32 * RESIDUE_PER_THREAD;
  int sfGroupResidueIdx = sfGroupId * 16;  // Each SF covers 16 residues
  int residueElementIdx = numMainCols + warpBaseResidueIdx + sfGroupResidueIdx;

  int32_t kIdx = residueElementIdx / CVT_FP4_SF_VEC_SIZE;  // 16 elements per SF
  int32_t mIdx = rowIdx;
  int64_t SFOffset = compute_tiled_sf_offset<ResidueSFType>(
    mIdx, kIdx, numTotalCols);

  return residueSFOut + SFOffset;
}

}  // namespace vllm
