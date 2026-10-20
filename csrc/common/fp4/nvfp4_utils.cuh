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

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>


#define ELTS_PER_THREAD 8

constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;
constexpr int CVT_FP4_ELTS_PER_UINT32 = 8;

__host__ __device__ __forceinline__ int round_up(int x, int n) {
  return ((x + n - 1) / n) * n;
}


__host__ __device__ __forceinline__ int computeEffectiveRows(int m) {
  constexpr int ROW_TILE = 128;
  return round_up(m, ROW_TILE);
}


template <int RESIDUE_PER_8_ELTS>
constexpr int get_residue_per_thread() {
    return RESIDUE_PER_8_ELTS * (CVT_FP4_ELTS_PER_THREAD / 8);
}


static_assert(CVT_FP4_ELTS_PER_THREAD == 8, "Current implementation assumes 8 elements per thread");

namespace vllm {


template <int THREADS_PER_GROUP>
constexpr uint32_t compute_group_mask(int group_id) {
  return ((1u << THREADS_PER_GROUP) - 1u) << (group_id * THREADS_PER_GROUP);
}


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


template <int MAX_OUTPUT>
__device__ __forceinline__ int extract_values_by_mask(
    const float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2],
    uint8_t mask,
    float output[MAX_OUTPUT]
) {
  int idx = 0;
  uint8_t extractMask = mask;
  while (extractMask) {
    int i = __ffs(extractMask) - 1;
    int pairIdx = i >> 1;
    int offset = i & 1;
    output[idx++] = (offset == 0) ? fp2Vals[pairIdx].x : fp2Vals[pairIdx].y;
    extractMask &= extractMask - 1;
  }
  return idx;
}


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


template <typename T>
struct TypeConverter {
  using Type = half2;
};

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


template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};

template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
};


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


  union U32Half2 {
    uint32_t u;
    __half2  h2;
  };

  auto residual_inplace = [&](uint32_t f16x2_u32, float2 &inout) {
    U32Half2 v;
    v.u = f16x2_u32;


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
  uint32_t f16x2_u32;
  uint32_t qbyte_u32;

  asm volatile(
    "{\n"
    "  .reg .b8 qb;\n"


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


inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}


__device__ __forceinline__
float compute_sf_from_max(float maxVal, float SFScaleVal) {
  return SFScaleVal * (maxVal * reciprocal_approximate_ftz(6.0f));
}


template <bool UE8M0_SF>
__device__ __forceinline__
float encode_sf_to_fp8(float SFValue, uint8_t* out_fp8) {
  uint8_t fp8;
  if constexpr (UE8M0_SF) {
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8 = tmp & 0xff;
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8) = tmp;
    SFValue = float(tmp);
  }
  if (out_fp8) *out_fp8 = fp8;
  return SFValue;
}


__device__ __forceinline__
float compute_output_scale_precise(float SFValue, float SFScaleVal) {
  return (SFValue != 0.0f) ? SFScaleVal * reciprocal_approximate_ftz(SFValue) : 0.0f;
}


template <int RESIDUE_PER_THREAD>
__device__ __forceinline__ void shuffle_collect_group_float2(
    const float2 (&scaledValues)[(RESIDUE_PER_THREAD + 1) / 2],
    int groupFirstThread,
    float2 (&outputChunk)[4]) {
  if constexpr (RESIDUE_PER_THREAD == 1) {


    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int sourceThread0 = groupFirstThread + i * 2;
      int sourceThread1 = groupFirstThread + i * 2 + 1;


      float val0 = __shfl_sync(__activemask(), scaledValues[0].x, sourceThread0);
      float val1 = __shfl_sync(__activemask(), scaledValues[0].x, sourceThread1);
      outputChunk[i] = make_float2(val0, val1);
    }
  } else {

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int sourceThread = groupFirstThread + i / (RESIDUE_PER_THREAD / 2);
      int float2Offset = i % (RESIDUE_PER_THREAD / 2);


      unsigned long long tmp = __shfl_sync(__activemask(),
        reinterpret_cast<const unsigned long long&>(scaledValues[float2Offset]),
        sourceThread);
      outputChunk[i] = reinterpret_cast<float2&>(tmp);
    }
  }
}


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


  uint32_t e2m1Vec = fp32x8_to_e2m1_with_dequant(mainScaled);


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


template <class Type>
__device__ __forceinline__ float2 packed_vec_to_float2(
    PackedVec<Type> const& vec, int index) {
  if constexpr (std::is_same_v<Type, half>) {
    return __half22float2(vec.elts[index]);
  } else {
    static_assert(std::is_same_v<Type, __nv_bfloat16>);
    return __bfloat1622float2(vec.elts[index]);
  }
}

template <bool WITH_DEQUANT>
__device__ __forceinline__ u32x2 fp32x16_to_e2m1(
    float2 (&loValues)[4], float2 (&hiValues)[4]) {
  u32x2 packed;
  if constexpr (WITH_DEQUANT) {
    packed.lo = fp32x8_to_e2m1_with_dequant(loValues);
    packed.hi = fp32x8_to_e2m1_with_dequant(hiValues);
  } else {
    packed.lo = fp32_vec_to_e2m1(loValues);
    packed.hi = fp32_vec_to_e2m1(hiValues);
  }
  return packed;
}


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
    loOriginal[i] = packed_vec_to_float2(loVec, i);
    hiOriginal[i] = packed_vec_to_float2(hiVec, i);
    loScaled[i].x = loOriginal[i].x * outputScale;
    loScaled[i].y = loOriginal[i].y * outputScale;
    hiScaled[i].x = hiOriginal[i].x * outputScale;
    hiScaled[i].y = hiOriginal[i].y * outputScale;
  }

  u32x2 mainPacked = fp32x16_to_e2m1<true>(loScaled, hiScaled);

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


    *reinterpret_cast<uint2*>(residueDataOut) =
        make_uint2(fp32_vec_to_e2m1(loResidueScaled),
                   fp32_vec_to_e2m1(hiResidueScaled));
  }

  return mainPacked;
}


template <class Type, bool UE8M0_SF = false, int RESIDUE_PER_8_ELTS = 1>
__device__ uint32_t cvt_warp_fp16_to_fp4_swap(PackedVec<Type>& vec, float SFScaleVal,
                                         uint8_t* SFout, uint8_t mask,
                                         uint32_t* residueDataOut, uint8_t* salientSFOut) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();


  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
  }


  float salient[RESIDUE_PER_THREAD > 0 ? RESIDUE_PER_THREAD : 1];
  extract_values_by_mask<RESIDUE_PER_THREAD>(fp2Vals, mask, salient);

  int laneId = threadIdx.x % 32;


  float localMaxSalient = 0.0f;
  #pragma unroll
  for (int i = 0; i < RESIDUE_PER_THREAD; i++) {
    localMaxSalient = fmaxf(localMaxSalient, fabsf(salient[i]));
  }


  constexpr int THREADS_PER_SF_GROUP = 16 / RESIDUE_PER_THREAD;
  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int sfGroupThreadId = laneId % THREADS_PER_SF_GROUP;


  float sfGroupMaxSalient = warp_group_max<THREADS_PER_SF_GROUP>(localMaxSalient, laneId);
  float salientSFValue = SFScaleVal * (sfGroupMaxSalient * reciprocal_approximate_ftz(6.0f));

  salientSFValue = encode_sf_to_fp8<UE8M0_SF>(salientSFValue, salientSFOut);

  float salientOutputScale = compute_output_scale_precise(salientSFValue, SFScaleVal);


  float2 shuffleSalient[(RESIDUE_PER_THREAD + 1) / 2];
  #pragma unroll
  for (int i = 0; i < (RESIDUE_PER_THREAD + 1) / 2; i++) {
    float val0 = (i * 2 < RESIDUE_PER_THREAD) ? salient[i * 2]: 0.0f;
    float val1 = (i * 2 + 1 < RESIDUE_PER_THREAD) ? salient[i * 2 + 1]: 0.0f;
    shuffleSalient[i] = make_float2(val0, val1);
  }


  float2 residue[(RESIDUE_PER_THREAD + 1) / 2];
  float2 salientChunk[4];


  constexpr int numThreadsPerGroup = 8 / RESIDUE_PER_THREAD;
  int groupId = laneId / numThreadsPerGroup;
  int threadInGroup = laneId % numThreadsPerGroup;
  int groupFirstThread = groupId * numThreadsPerGroup;


  float2 unscaledSalientChunk[4];
  shuffle_collect_group_float2<RESIDUE_PER_THREAD>(shuffleSalient, groupFirstThread, unscaledSalientChunk);
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    salientChunk[i] = make_float2(
      unscaledSalientChunk[i].x * salientOutputScale,
      unscaledSalientChunk[i].y * salientOutputScale);
  }


  float2 dequantChunk[4];
  uint32_t quantizedChunk = fp32x8_to_e2m1_separate(salientChunk, dequantChunk);

  if (residueDataOut) {
    *residueDataOut = quantizedChunk;
  }


  float inverseSalientOutputScale = salientOutputScale != 0 ? reciprocal_approximate_ftz(salientOutputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {

    float deq_x = dequantChunk[i].x * inverseSalientOutputScale;
    float deq_y = dequantChunk[i].y * inverseSalientOutputScale;

    float orig_x = unscaledSalientChunk[i].x;
    float orig_y = unscaledSalientChunk[i].y;

    salientChunk[i].x = orig_x - deq_x;
    salientChunk[i].y = orig_y - deq_y;
  }


  if constexpr (RESIDUE_PER_THREAD == 1) {

    int pairIdx = threadInGroup / 2;
    int elemIdx = threadInGroup % 2;
    float val = (elemIdx == 0) ? salientChunk[pairIdx].x : salientChunk[pairIdx].y;
    residue[0] = make_float2(val, 0.0f);
  } else {


    constexpr int float2PerThread = RESIDUE_PER_THREAD / 2;
    #pragma unroll
    for (int i = 0; i < float2PerThread; i++) {
      int idx = threadInGroup * float2PerThread + i;
      residue[i] = salientChunk[idx];
    }
  }


  int residueWriteIdx = 0;
  uint8_t tempMask = mask;
  while (tempMask) {
    int i = __ffs(tempMask) - 1;
    int pairIdx = i / 2;
    int offset = i % 2;
    float val;
    if constexpr (RESIDUE_PER_THREAD == 1) {

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
    tempMask &= tempMask - 1;
  }

  float localMaxFloat = 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMaxFloat = fmaxf(localMaxFloat, fmaxf(fabsf(fp2Vals[i].x), fabsf(fp2Vals[i].y)));
  }

  localMaxFloat = fmaxf(localMaxFloat, __shfl_xor_sync(uint32_t(-1), localMaxFloat, 1));
  float vecMax = localMaxFloat;

  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));


  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);

  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);


  float2 mainVals[CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    mainVals[i].x = fp2Vals[i].x * outputScale;
    mainVals[i].y = fp2Vals[i].y * outputScale;
  }

  uint32_t e2m1Vec = fp32_vec_to_e2m1(mainVals);


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


template <class Type, bool UE8M0_SF = false, int RESIDUE_PER_8_ELTS = 1>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal,
                                         uint8_t* SFout, uint8_t mask,
                                         uint32_t* residueDataOut, uint8_t* residueSFOut) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();


  auto localMax = __habs2(vec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));


  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));

  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);

  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 fp2ValsOriginal[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2ValsOriginal[i] = fp2Vals[i];
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }


  uint32_t e2m1Vec = fp32x8_to_e2m1_with_dequant(fp2Vals);


  float inverseOutputScale = outputScale != 0 ? reciprocal_approximate_ftz(outputScale) : 0.0f;
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {

    fp2Vals[i].x *= inverseOutputScale;
    fp2Vals[i].y *= inverseOutputScale;


    fp2Vals[i].x = fp2ValsOriginal[i].x - fp2Vals[i].x;
    fp2Vals[i].y = fp2ValsOriginal[i].y - fp2Vals[i].y;
  }


  float residues[RESIDUE_PER_THREAD > 0 ? RESIDUE_PER_THREAD : 1];
  extract_values_by_mask<RESIDUE_PER_THREAD>(fp2Vals, mask, residues);

  int laneId = threadIdx.x % 32;


  float localMaxResidue = 0.0f;
  #pragma unroll
  for (int i = 0; i < RESIDUE_PER_THREAD; i++) {
    localMaxResidue = fmaxf(localMaxResidue, fabsf(residues[i]));
  }


  constexpr int THREADS_PER_SF_GROUP = 16 / RESIDUE_PER_THREAD;
  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int sfGroupThreadId = laneId % THREADS_PER_SF_GROUP;


  float sfGroupMaxResidue = warp_group_max<THREADS_PER_SF_GROUP>(localMaxResidue, laneId);
  float residueSFValue = SFScaleVal * (sfGroupMaxResidue * reciprocal_approximate_ftz(6.0f));

  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);

  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);


  float2 scaledResidues2[(RESIDUE_PER_THREAD + 1) / 2];
  #pragma unroll
  for (int i = 0; i < (RESIDUE_PER_THREAD + 1) / 2; i++) {
    float val0 = (i * 2 < RESIDUE_PER_THREAD) ? residues[i * 2] * residueOutputScale : 0.0f;
    float val1 = (i * 2 + 1 < RESIDUE_PER_THREAD) ? residues[i * 2 + 1] * residueOutputScale : 0.0f;
    scaledResidues2[i] = make_float2(val0, val1);
  }


  constexpr int numThreadsPerGroup = 8 / RESIDUE_PER_THREAD;
  int groupId = laneId / numThreadsPerGroup;
  int threadInGroup = laneId % numThreadsPerGroup;
  int groupFirstThread = groupId * numThreadsPerGroup;


  float2 groupResidueChunk[4];
  shuffle_collect_group_float2<RESIDUE_PER_THREAD>(scaledResidues2, groupFirstThread, groupResidueChunk);


  float2 residueChunk2[4];


  int mySourceGroupLeader = laneId * numThreadsPerGroup;


  #pragma unroll
  for (int i = 0; i < 4; i++) {

    unsigned long long tmp = __shfl_sync(__activemask(),
      reinterpret_cast<unsigned long long&>(groupResidueChunk[i]),
      mySourceGroupLeader);
    residueChunk2[i] = reinterpret_cast<float2&>(tmp);
  }


  uint32_t quantizedChunk = fp32_vec_to_e2m1(residueChunk2);


  if (residueDataOut) {
    *residueDataOut = quantizedChunk;
  }

  return e2m1Vec;
}


template <class ResidueType, int RESIDUE_PER_8_ELTS>
__device__ ResidueType* cvt_residue_data_get_offset(int rowIdx, int colIdx, int numMainCols, int numTotalCols, ResidueType* outputStart) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  constexpr int RESIDUES_PER_WARP = 32 * RESIDUE_PER_THREAD;
  constexpr int UINT32_PER_WARP = (RESIDUES_PER_WARP + 7) / 8;


  int warpId = colIdx / 32;
  int laneId = colIdx % 32;

  if (laneId < UINT32_PER_WARP) {


    int64_t uint32s_per_row = numTotalCols / CVT_FP4_ELTS_PER_UINT32;
    int64_t base_uint32s_per_row = numMainCols / CVT_FP4_ELTS_PER_UINT32;

    int64_t residueDataIdx = rowIdx * uint32s_per_row
                           + base_uint32s_per_row
                           + warpId * UINT32_PER_WARP
                           + laneId;
    return outputStart + residueDataIdx;
  }
  return nullptr;
}


template <class ResidueType, int RESIDUE_PER_8_ELTS>
__device__ ResidueType* cvt_swap_residue_data_get_offset(int rowIdx, int colIdx, int numMainCols, int numTotalCols, ResidueType* outputStart) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  constexpr int RESIDUES_PER_WARP = 32 * RESIDUE_PER_THREAD;
  constexpr int UINT32_PER_WARP = (RESIDUES_PER_WARP + 7) / 8;


  int warpId = colIdx / 32;
  int laneId = colIdx % 32;


  constexpr int numThreadsPerGroup = 8 / RESIDUE_PER_THREAD;
  int groupId = laneId / numThreadsPerGroup;
  int threadInGroup = laneId % numThreadsPerGroup;


  if (threadInGroup == 0 && groupId < UINT32_PER_WARP) {


    int64_t uint32s_per_row = numTotalCols / CVT_FP4_ELTS_PER_UINT32;
    int64_t base_uint32s_per_row = numMainCols / CVT_FP4_ELTS_PER_UINT32;

    int64_t residueDataIdx = rowIdx * uint32s_per_row
                           + base_uint32s_per_row
                           + warpId * UINT32_PER_WARP
                           + groupId;
    return outputStart + residueDataIdx;
  }
  return nullptr;
}


template <class SFType, int SF_VEC_SIZE = CVT_FP4_SF_VEC_SIZE>
__device__ inline int64_t compute_tiled_sf_offset(
  int32_t mIdx, int32_t kIdx, int32_t numTotalCols, bool swizzle = true) {
  if (!swizzle) {
    return int64_t(mIdx) * (numTotalCols / SF_VEC_SIZE) + kIdx;
  }
  int32_t mTileIdx = mIdx / (32 * 4);
  constexpr int factor = SF_VEC_SIZE * 4;
  int32_t numKTiles = (numTotalCols + factor - 1) / factor;
  int64_t mTileStride = numKTiles * 32 * 4 * 4;

  int32_t kTileIdx = (kIdx / 4);

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


  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {

    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;


    int64_t SFOffset = compute_tiled_sf_offset<SFType>(
      mIdx, kIdx, numTotalCols);

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
  return nullptr;
}


template <class ResidueSFType, int RESIDUE_PER_8_ELTS>
__device__ ResidueSFType* cvt_residue_sf_get_offset(
  int rowIdx, int colIdx,
  int numTotalCols,
  int numMainCols,
  ResidueSFType* residueSFOut) {

  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();

  constexpr int THREADS_PER_SF_GROUP = 16 / RESIDUE_PER_THREAD;

  int warpId = colIdx / 32;
  int laneId = colIdx % 32;


  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int sfGroupThreadId = laneId % THREADS_PER_SF_GROUP;


  if (sfGroupThreadId != 0) return nullptr;


  int warpBaseResidueIdx = warpId * 32 * RESIDUE_PER_THREAD;
  int sfGroupResidueIdx = sfGroupId * 16;
  int residueElementIdx = numMainCols + warpBaseResidueIdx + sfGroupResidueIdx;

  int32_t kIdx = residueElementIdx / CVT_FP4_SF_VEC_SIZE;
  int32_t mIdx = rowIdx;
  int64_t SFOffset = compute_tiled_sf_offset<ResidueSFType>(
    mIdx, kIdx, numTotalCols);

  return residueSFOut + SFOffset;
}

}
