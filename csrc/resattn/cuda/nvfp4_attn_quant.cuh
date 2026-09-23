// SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
// SPDX-License-Identifier: Apache-2.0


















#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/torch.h>

#include <cstdint>

namespace nvfp4_attn {

constexpr int kPageSize = 128;  // MMA-native profile(设计文档 D1,冻结)
constexpr int kSfVec = 16;      // nvfp4 block-scale 组大小(硬件定死)
constexpr int kQSfRows = 128;   // Q-SF tile 垫高行数(UTCCP 按 N=128 粒度)

// 池张量的步长:**一律从张量自带的 stride 取**,不假设稠密 —— vLLM 把 KV 池
// 切成 [data|scale] 分侧视图后,页步长是 2*H*ps*full_dim 而非 H*ps*(D/2)。
struct S4 {
  long long s0, s1, s2, s3;
};
struct S3 {
  long long s0, s1, s2;
};

inline S4 strides4(const torch::Tensor& t) {
  return {t.stride(0), t.stride(1), t.stride(2), t.stride(3)};
}
inline S3 strides3(const torch::Tensor& t) {
  return {t.stride(0), t.stride(1), t.stride(2)};
}

__device__ __forceinline__ int sf_atom_off(int mn, int kg, int KG) {
  return (mn % 32) * 16 + ((mn / 32) % 4) * 4 + (kg % 4) + (kg / 4) * 512 +
         (mn / 128) * 512 * (KG / 4);
}

// torch.bucketize(right=False) 复刻:平局(恰在中点)取下值
__device__ __forceinline__ uint32_t e2m1_nibble(float y) {
  float a = fabsf(y);
  uint32_t idx = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) +
                 (a > 2.5f) + (a > 3.5f) + (a > 5.0f);
  return idx + (y < 0.f ? 8u : 0u);
}

__device__ __forceinline__ uint8_t f32_to_e4m3_byte(float x) {
  __nv_fp8_e4m3 v(x);  // cvt.rn.satfinite —— 与 torch .to(float8_e4m3fn) 一致
  return *reinterpret_cast<uint8_t*>(&v);
}

// 量化一个 16 元组:in[16](已取 float)→ 8 个打包字节 + 1 个 SF 字节。
// n_valid < 16 时无效位置写 0 nibble(读侧会被掩码,零值最干净)。
// perm != nullptr 时按 perm 重排组内 nibble(V 的 π 预排)。
__device__ __forceinline__ void quant_group16(const float* in, int n_valid,
                                              const int* perm, uint8_t* out8,
                                              uint8_t* sf_out) {
  float amax = 0.f;
  for (int i = 0; i < n_valid; ++i) amax = fmaxf(amax, fabsf(in[i]));
  uint8_t sf8 = f32_to_e4m3_byte(__fdiv_rn(amax, 6.0f));
  __nv_fp8_e4m3 sfv;
  *reinterpret_cast<uint8_t*>(&sfv) = sf8;
  float sf_f = fmaxf(float(sfv), 1e-12f);
  uint32_t nib[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    if (i < n_valid) {
      float y = __fdiv_rn(in[i], sf_f);
      y = fminf(fmaxf(y, -6.f), 6.f);
      nib[i] = e2m1_nibble(y);
    } else {
      nib[i] = 0;
    }
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    uint32_t lo = perm ? nib[perm[2 * k]] : nib[2 * k];
    uint32_t hi = perm ? nib[perm[2 * k + 1]] : nib[2 * k + 1];
    out8[k] = static_cast<uint8_t>(lo | (hi << 4));
  }
  *sf_out = sf8;
}

}  // namespace nvfp4_attn
