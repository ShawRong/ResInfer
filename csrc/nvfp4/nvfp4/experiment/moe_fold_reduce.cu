// SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
// SPDX-License-Identifier: Apache-2.0
//
// Row-pair fold as a standalone reduction, for the MExt-on-MoE experiment.
//
// The MExt fold does not have to live in a GEMM epilogue:
//
//     out[i] = q(base_i) . W + q(res_i) . W
//            = (stock GEMM over the row-doubled activation)[2i] + [2i+1]
//
// so the whole fold is: given a (2M, R) tensor, produce (M, R) with
// out[m] = in[2m] + in[2m+1]. That is what this file does, and nothing else.
//
// Why a kernel at all, when `t[0::2] + t[1::2]` already works: the slice form
// launches an elementwise add over two strided views, so it reads the full
// 2M-row tensor and writes M rows -- but the strided access pattern gives up
// the 128-bit loads a contiguous pair walk can use. This does the same
// arithmetic with 16-byte vector loads and a grid-stride loop.
//
// It does NOT recover what an epilogue fold gets. In the epilogue the two
// halves are already in adjacent accumulator registers, so the add is free
// and only M rows are ever written. Here the GEMM still writes all 2M rows to
// global memory and we read them back. This kernel removes the strided-access
// penalty, not the extra memory round trip.
//
// Accumulation is in fp32 with a single rounding on store, which is what
// PyTorch's bf16/fp16 elementwise add does, so results match the torch
// reference exactly rather than approximately.
//
// Built by torch.utils.cpp_extension.load (see moe_fold.py) into its own
// module, deliberately NOT into nvfp4._C: a stale shared .so silently
// deciding which ops exist has already cost two sessions a debugging round,
// and an experiment must not be able to do that to the production extension.

#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

// 16-byte carrier so each thread issues one 128-bit load per operand.
template <typename T>
struct alignas(16) Vec16 {
  static constexpr int N = 16 / sizeof(T);
  T x[N];
};

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ void from_f(float v, float* o) { *o = v; }
__device__ __forceinline__ void from_f(float v, __half* o) {
  *o = __float2half(v);
}
__device__ __forceinline__ void from_f(float v, __nv_bfloat16* o) {
  *o = __float2bfloat16(v);
}

template <typename T>
__global__ void fold_pairs_vec_kernel(const T* __restrict__ in,
                                      T* __restrict__ out, int64_t rows_out,
                                      int64_t cols) {
  using V = Vec16<T>;
  constexpr int N = V::N;
  const int64_t cols_v = cols / N;
  const int64_t total = rows_out * cols_v;

  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = i / cols_v;
    const int64_t cv = i - row * cols_v;

    const V* a = reinterpret_cast<const V*>(in + (2 * row) * cols) + cv;
    const V* b = reinterpret_cast<const V*>(in + (2 * row + 1) * cols) + cv;
    V* o = reinterpret_cast<V*>(out + row * cols) + cv;

    V va = *a;
    V vb = *b;
    V vo;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      from_f(to_f(va.x[j]) + to_f(vb.x[j]), &vo.x[j]);
    }
    *o = vo;
  }
}

// Fallback for a row length that is not a multiple of the vector width. The
// production case (N a multiple of 16 for NVFP4) always takes the vector path;
// this exists so a test can use an awkward shape and still be correct.
template <typename T>
__global__ void fold_pairs_scalar_kernel(const T* __restrict__ in,
                                         T* __restrict__ out, int64_t rows_out,
                                         int64_t cols) {
  const int64_t total = rows_out * cols;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = i / cols;
    const int64_t col = i - row * cols;
    const float a = to_f(in[(2 * row) * cols + col]);
    const float b = to_f(in[(2 * row + 1) * cols + col]);
    from_f(a + b, &out[row * cols + col]);
  }
}

template <typename T>
void launch(const torch::Tensor& in, torch::Tensor& out, int64_t rows_out,
            int64_t cols, cudaStream_t stream) {
  constexpr int N = Vec16<T>::N;
  const int threads = 256;
  const auto* ip = reinterpret_cast<const T*>(in.data_ptr());
  auto* op = reinterpret_cast<T*>(out.data_ptr());

  if (cols % N == 0) {
    const int64_t work = rows_out * (cols / N);
    // Cap the grid; the loop is grid-stride so any grid is correct.
    const int64_t blocks =
        std::min<int64_t>((work + threads - 1) / threads, 65535);
    if (blocks > 0) {
      fold_pairs_vec_kernel<T><<<blocks, threads, 0, stream>>>(ip, op, rows_out,
                                                               cols);
    }
  } else {
    const int64_t work = rows_out * cols;
    const int64_t blocks =
        std::min<int64_t>((work + threads - 1) / threads, 65535);
    if (blocks > 0) {
      fold_pairs_scalar_kernel<T><<<blocks, threads, 0, stream>>>(
          ip, op, rows_out, cols);
    }
  }
}

}  // namespace

// in : (2M, ...) contiguous. Trailing dims are flattened; only dim 0 matters.
// out: (M, ...)  same trailing shape.
torch::Tensor moe_fold_pairs(torch::Tensor in) {
  TORCH_CHECK(in.is_cuda(), "moe_fold_pairs: input must be a CUDA tensor");
  TORCH_CHECK(in.dim() >= 1, "moe_fold_pairs: input must have >= 1 dim");
  TORCH_CHECK(in.size(0) % 2 == 0,
              "moe_fold_pairs: leading dim must be even (got ", in.size(0),
              ") -- the row-pair contract is [base_0, res_0, base_1, ...]");
  in = in.contiguous();

  const int64_t rows_out = in.size(0) / 2;
  const int64_t cols = in.numel() / in.size(0);

  auto out_shape = in.sizes().vec();
  out_shape[0] = rows_out;
  auto out = torch::empty(out_shape, in.options());
  if (rows_out == 0 || cols == 0) return out;

  const at::cuda::OptionalCUDAGuard guard(device_of(in));
  auto stream = at::cuda::getCurrentCUDAStream();

  if (in.scalar_type() == at::ScalarType::BFloat16) {
    launch<__nv_bfloat16>(in, out, rows_out, cols, stream);
  } else if (in.scalar_type() == at::ScalarType::Half) {
    launch<__half>(in, out, rows_out, cols, stream);
  } else if (in.scalar_type() == at::ScalarType::Float) {
    launch<float>(in, out, rows_out, cols, stream);
  } else {
    TORCH_CHECK(false, "moe_fold_pairs: unsupported dtype ", in.scalar_type(),
                " (expected bfloat16, float16 or float32)");
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_fold_pairs", &moe_fold_pairs,
        "Sum adjacent row pairs: (2M, ...) -> (M, ...)");
}