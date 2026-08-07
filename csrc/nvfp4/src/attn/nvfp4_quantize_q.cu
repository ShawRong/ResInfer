// SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
// SPDX-License-Identifier: Apache-2.0
//
// ResAttn decode / context 的 **Q 量化**(融合版)。
//
// 语义基准 = decode 的 host._quantize_q / context 的
// host_prefill._quantize_k_major torch 实现,逐字节一致
// (公用量化数学见 nvfp4_attn_quant.cuh)。
//
// 为什么值得融合:decode 的 q 很小([B, H_q, D]),torch 版那串算子的
// **launch 开销与注意力 kernel 本身同量级**,融合成一发 kernel 收益明显。
// context 的 ragged Q 原先还要先逐请求左垫零、再逐头发 torch 量化 kernel；
// context 入口直接读 [T,H,D] + query_indptr,一发完成 padding + pack + SF。
//
// 布局:输入 [B, H_kv, group_pad, D] bf16(GQA 比 <16 时 host 已把每 KV 头的
// Q 行垫到 16,多出的行填零);输出打包 [B, H_kv, group_pad, D/2],SF 为每
// (b, h_kv) 一个 kQSfRows×KG 的 atom 序 tile —— 只写前 group_pad 行,其余靠
// host 侧 `torch.zeros` 保证为 0。

#include <ATen/cuda/CUDAContext.h>

#include "nvfp4_attn_quant.cuh"

namespace {

using namespace nvfp4_attn;

// 一 block 一 (batch, kv_head);线程遍历 (行, 组) 工作项。
__global__ void quantize_q_kernel(const __nv_bfloat16* __restrict__ q,  // [B,H,gp,D]
                                  uint8_t* __restrict__ q_pack,
                                  uint8_t* __restrict__ q_sf,
                                  __nv_bfloat16* __restrict__ out_to_zero,
                                  int H, int gp, int D) {
  int bh = blockIdx.x;
  int b = bh / H, h = bh % H;
  int KG = D / kSfVec;
  long long q_base = ((long long)b * H + h) * gp * D;
  long long p_base = ((long long)b * H + h) * gp * (D / 2);
  long long s_base = ((long long)b * H + h) * kQSfRows * KG;
  for (int t = threadIdx.x; t < gp * KG; t += blockDim.x) {
    int r = t / KG, grp = t % KG;
    float in[16];
#pragma unroll
    for (int e = 0; e < 16; ++e)
      in[e] = float(q[q_base + (long long)r * D + grp * kSfVec + e]);
    uint8_t out8[8], sf8;
    quant_group16(in, 16, nullptr, out8, &sf8);
    uint8_t* dst = q_pack + p_base + (long long)r * (D / 2) + grp * 8;
#pragma unroll
    for (int k = 0; k < 8; ++k) dst[k] = out8[k];
    q_sf[s_base + sf_atom_off(r, grp, KG)] = sf8;
    // The decode output has the same logical [B,H,gp,D] shape as Q.  Each
    // quantization work item already owns one contiguous 16-element group, so
    // it can initialize the atomic-reduction destination without another
    // kernel launch or any cross-block synchronization.
    if (out_to_zero != nullptr) {
#pragma unroll
      for (int e = 0; e < 16; ++e)
        out_to_zero[q_base + (long long)r * D + grp * kSfVec + e] =
            __float2bfloat16(0.0f);
    }
  }
}

// 一 block 一 (batch, q_head, 128-row tile)。query_indptr 的值索引原始
// q_source [T,H,D],可以不是从 0 开始(vLLM 直接传全局 query_start_loc 切片)。
// 每个请求在共同的 S_pad 桶内**左对齐补零**:真实 query 保持 end-aligned
// causal 位置,与 context attention kernel 的掩码约定一致。
__global__ void quantize_context_q_kernel(
    const __nv_bfloat16* __restrict__ q_source,  // [T,H,D]
    const int32_t* __restrict__ query_indptr,    // [B+1]
    uint8_t* __restrict__ q_pack,                // [B,H,S_pad,D/2]
    uint8_t* __restrict__ q_sf,                  // [B,H,S_pad*(D/16)]
    int T, int B, int H, int S_pad, int D) {
  int q_tiles = S_pad / kQSfRows;
  int tile_bh = blockIdx.x / q_tiles;
  int q_tile = blockIdx.x % q_tiles;
  int b = tile_bh / H, h = tile_bh % H;
  int KG = D / kSfVec;
  int start = query_indptr[b], end = query_indptr[b + 1];
  int q_len = end - start;
  int pad = S_pad - q_len;
  // Metadata normally comes straight from vLLM. Keep malformed metadata from
  // becoming an out-of-bounds global read without synchronizing it to host.
  bool span_ok = start >= 0 && end >= start && end <= T && q_len <= S_pad;
  long long p_base = ((long long)b * H + h) * S_pad * (D / 2);
  long long s_base = ((long long)b * H + h) * S_pad * KG;

  for (int t = threadIdx.x; t < kQSfRows * KG; t += blockDim.x) {
    int tile_row = t / KG, grp = t % KG;
    int r = q_tile * kQSfRows + tile_row;
    bool row_valid = span_ok && r >= pad;
    float in[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      long long src = ((long long)start + r - pad) * H * D +
                      (long long)h * D + grp * kSfVec + e;
      in[e] = row_valid ? float(q_source[src]) : 0.f;
    }
    uint8_t out8[8], sf8;
    quant_group16(in, 16, nullptr, out8, &sf8);
    uint8_t* dst = q_pack + p_base + (long long)r * (D / 2) + grp * 8;
#pragma unroll
    for (int k = 0; k < 8; ++k) dst[k] = out8[k];
    q_sf[s_base + sf_atom_off(r, grp, KG)] = sf8;
  }
}

}  // namespace

void nvfp4_quantize_q(torch::Tensor const& q, torch::Tensor& q_pack,
                      torch::Tensor& q_sf) {
  using namespace nvfp4_attn;
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.is_contiguous(),
              "q must be contiguous bf16");
  TORCH_CHECK(q.dim() == 4, "q must be [B, H_kv, group_pad, D]");
  TORCH_CHECK(q_pack.scalar_type() == torch::kUInt8 &&
                  q_sf.scalar_type() == torch::kUInt8,
              "q_pack / q_sf must be uint8");
  TORCH_CHECK(q_pack.is_contiguous() && q_sf.is_contiguous(),
              "q_pack / q_sf must be contiguous (freshly allocated)");
  int B = q.size(0), H = q.size(1), gp = q.size(2), D = q.size(3);
  TORCH_CHECK(D % kSfVec == 0, "head_dim must be divisible by 16");
  TORCH_CHECK(gp <= kQSfRows, "padded group size must be <= 128");
  TORCH_CHECK(q_sf.numel() >= (int64_t)B * H * kQSfRows * (D / kSfVec),
              "q_sf buffer too small");
  if (B == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  quantize_q_kernel<<<dim3(B * H), dim3(128), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      q_pack.data_ptr<uint8_t>(), q_sf.data_ptr<uint8_t>(), nullptr, H, gp, D);
}

void nvfp4_quantize_q_and_zero(torch::Tensor const& q,
                               torch::Tensor& q_pack,
                               torch::Tensor& q_sf,
                               torch::Tensor& out) {
  using namespace nvfp4_attn;
  TORCH_CHECK(q.is_cuda() && q_pack.is_cuda() && q_sf.is_cuda() && out.is_cuda(),
              "q/q_pack/q_sf/out must be CUDA tensors");
  TORCH_CHECK(q.device() == q_pack.device() && q.device() == q_sf.device() &&
                  q.device() == out.device(),
              "q/q_pack/q_sf/out must be on the same CUDA device");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.is_contiguous(),
              "q must be contiguous bf16");
  TORCH_CHECK(q.dim() == 4, "q must be [B, H_kv, group_pad, D]");
  TORCH_CHECK(q_pack.scalar_type() == torch::kUInt8 &&
                  q_sf.scalar_type() == torch::kUInt8,
              "q_pack / q_sf must be uint8");
  TORCH_CHECK(q_pack.is_contiguous() && q_sf.is_contiguous(),
              "q_pack / q_sf must be contiguous (freshly allocated)");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.is_contiguous(),
              "out must be contiguous bf16");
  int B = q.size(0), H = q.size(1), gp = q.size(2), D = q.size(3);
  TORCH_CHECK(D % kSfVec == 0, "head_dim must be divisible by 16");
  TORCH_CHECK(gp <= kQSfRows, "padded group size must be <= 128");
  TORCH_CHECK(q_pack.numel() >= (int64_t)B * H * gp * (D / 2),
              "q_pack buffer too small");
  TORCH_CHECK(q_sf.numel() >= (int64_t)B * H * kQSfRows * (D / kSfVec),
              "q_sf buffer too small");
  TORCH_CHECK(out.numel() >= (int64_t)B * H * gp * D,
              "out buffer too small");
  if (B == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  quantize_q_kernel<<<dim3(B * H), dim3(128), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      q_pack.data_ptr<uint8_t>(), q_sf.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, gp, D);
}

void nvfp4_quantize_context_q(torch::Tensor const& q_source,
                              torch::Tensor const& query_indptr,
                              torch::Tensor& q_pack,
                              torch::Tensor& q_sf) {
  using namespace nvfp4_attn;
  TORCH_CHECK(q_source.is_cuda() && query_indptr.is_cuda() &&
                  q_pack.is_cuda() && q_sf.is_cuda(),
              "context Q tensors must be CUDA tensors");
  TORCH_CHECK(q_source.device() == query_indptr.device() &&
                  q_source.device() == q_pack.device() &&
                  q_source.device() == q_sf.device(),
              "context Q tensors must be on the same CUDA device");
  TORCH_CHECK(q_source.scalar_type() == torch::kBFloat16 &&
                  q_source.is_contiguous() && q_source.dim() == 3,
              "q_source must be contiguous bf16 [T, H_q, D]");
  TORCH_CHECK(query_indptr.scalar_type() == torch::kInt32 &&
                  query_indptr.is_contiguous() && query_indptr.dim() == 1,
              "query_indptr must be contiguous int32 [B+1]");
  TORCH_CHECK(q_pack.scalar_type() == torch::kUInt8 &&
                  q_sf.scalar_type() == torch::kUInt8 &&
                  q_pack.is_contiguous() && q_sf.is_contiguous(),
              "q_pack / q_sf must be contiguous uint8 outputs");
  TORCH_CHECK(q_pack.dim() == 4,
              "q_pack must be [B, H_q, S_pad, D/2]");

  int T = q_source.size(0), H = q_source.size(1), D = q_source.size(2);
  int B = query_indptr.numel() - 1;
  TORCH_CHECK(B >= 0, "query_indptr must contain at least one entry");
  TORCH_CHECK(D % kSfVec == 0, "head_dim must be divisible by 16");
  TORCH_CHECK(q_pack.size(0) == B && q_pack.size(1) == H &&
                  q_pack.size(3) == D / 2,
              "q_pack shape must match query_indptr and q_source");
  int S_pad = q_pack.size(2);
  TORCH_CHECK(S_pad > 0 && S_pad % kQSfRows == 0,
              "S_pad must be a positive multiple of 128");
  TORCH_CHECK(q_sf.numel() == (int64_t)B * H * S_pad * (D / kSfVec),
              "q_sf must contain B*H*S_pad*(D/16) bytes");
  if (B == 0) return;

  int q_tiles = S_pad / kQSfRows;
  auto stream = at::cuda::getCurrentCUDAStream();
  quantize_context_q_kernel<<<dim3(B * H * q_tiles), dim3(128), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_source.data_ptr()),
      query_indptr.data_ptr<int32_t>(), q_pack.data_ptr<uint8_t>(),
      q_sf.data_ptr<uint8_t>(), T, B, H, S_pad, D);
}
