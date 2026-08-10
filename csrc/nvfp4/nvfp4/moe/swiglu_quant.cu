// SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
// SPDX-License-Identifier: Apache-2.0




















































#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

#include "fp4/nvfp4_utils.cuh"

// PackedVec, the cvt_* helpers and compute_tiled_sf_offset all live in
// namespace vllm (the CVT_FP4_* constants do not -- they precede it).
using namespace vllm;

namespace {

__device__ __forceinline__ float silu_f(float x) {
  return x / (1.0f + __expf(-x));
}

// [gate | up] -> rPost * (rPre*up) * silu(rPre*gate), one thread's chunk.
//
// rPre is the per-expert GEMM1 dequant ratio and must hit gate AND up
// BEFORE the nonlinearity -- silu(r*g) has no post-hoc equivalent. rPost is
// the per-expert factor of the NEXT GEMM's scale chain, applied to h before
// it is rounded and quantised. Both default to 1.0f, and multiplying by
// 1.0f in fp32 is exact, so the no-prescale path is bit-identical.
//
// Rounded back to the input dtype rather than kept in fp32: the quantiser
// consumes a PackedVec<Type> and, more to the point, the unfused chain also
// materialises bf16 here. Keeping extra precision would make the fused and
// unfused arms disagree for a reason unrelated to the fusion.
template <class Type>
__device__ __forceinline__ PackedVec<Type> swiglu_packed(
    PackedVec<Type> const& g, PackedVec<Type> const& u, float rPre,
    float rPost) {
  PackedVec<Type> r;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    float2 gf, uf;
    if constexpr (std::is_same_v<Type, half>) {
      gf = __half22float2(g.elts[i]);
      uf = __half22float2(u.elts[i]);
    } else {
      gf = __bfloat1622float2(g.elts[i]);
      uf = __bfloat1622float2(u.elts[i]);
    }
    float2 hf;
    hf.x = rPost * ((rPre * uf.x) * silu_f(rPre * gf.x));
    hf.y = rPost * ((rPre * uf.y) * silu_f(rPre * gf.y));
    if constexpr (std::is_same_v<Type, half>) {
      r.elts[i] = __float22half2_rn(hf);
    } else {
      r.elts[i] = __float22bfloat162_rn(hf);
    }
  }
  return r;
}

// Non-fused input scaling: pre and post collapse to one factor (there is no
// nonlinearity between them here). r == 1.0f must be exact: dtype -> fp32 is
// lossless, *1.0f is exact, and rounding an unchanged value back is the
// identity -- asserted bitwise by tests/test_swiglu_quant_prescale.py.
template <class Type>
__device__ __forceinline__ PackedVec<Type> scale_packed(
    PackedVec<Type> const& v, float r) {
  PackedVec<Type> o;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    float2 f;
    if constexpr (std::is_same_v<Type, half>) {
      f = __half22float2(v.elts[i]);
    } else {
      f = __bfloat1622float2(v.elts[i]);
    }
    f.x *= r;
    f.y *= r;
    if constexpr (std::is_same_v<Type, half>) {
      o.elts[i] = __float22half2_rn(f);
    } else {
      o.elts[i] = __float22bfloat162_rn(f);
    }
  }
  return o;
}

// FUSE_SWIGLU=false is the same kernel reading h directly. It exists so a
// benchmark can charge the fusion honestly: quantise-only vs
// swiglu+quantise, same code, same launch geometry, one template argument
// apart. Without it the comparison is against a different kernel.
template <class Type, bool FUSE_SWIGLU, bool UE8M0_SF = false>
__global__ void __launch_bounds__(256)
    swiglu_quant_mext_r1_pack8(int32_t mTok, int32_t numCols,
                               Type const* __restrict__ in,
                               float const* __restrict__ SFScale,
                               uint32_t* __restrict__ out,
                               uint32_t* __restrict__ SFout,
                               int32_t sfWordsPerExpert,
                               int32_t numSFScales, int32_t layoutMode,
                               // Optional (E*mTok,) map slot -> source row.
                               // With it this kernel also does the PERMUTE:
                               // the expert's slots read scattered tokens out
                               // of the unsorted activation instead of a
                               // contiguous block, so no separate reorder pass
                               // and no permuted copy in HBM. Same trick as
                               // flashinfer's token_id_mapping.
                               int32_t const* __restrict__ rowIdx,
                               // Optional (experts,) per-SLOT f32 factors:
                               // preScale multiplies gate and up BEFORE the
                               // activation (per-expert GEMM1 dequant ratio),
                               // postScale multiplies h after it (per-expert
                               // part of GEMM2's scale chain). Indexed by
                               // blockIdx.z, so with a gather table the
                               // caller passes vectors already gathered per
                               // slot (e.g. ratio[expert_ids]).
                               float const* __restrict__ preScale,
                               float const* __restrict__ postScale,
                               
                               
                               
                               
                               
                               
                               bool linearSF) {
  using PackedVec8 = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec8) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched for 8-element base helper.");

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  float const SFScaleResVal =
      (SFScale != nullptr && numSFScales > 1) ? SFScale[1] : SFScaleVal;

  int const expert = blockIdx.z;
  int const colGroups = numCols / CVT_FP4_ELTS_PER_THREAD;
  int const colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= colGroups) return;

  // Uniform per block (expert == blockIdx.z): one load, no divergence.
  float const rPre = preScale == nullptr ? 1.0f : preScale[expert];
  float const rPost = postScale == nullptr ? 1.0f : postScale[expert];

  // colIdx parity == threadIdx.x parity because blockDim.x is even, so the
  // 2-thread SF reduction inside the quantiser (shfl_xor lane^1) still pairs
  // the two 8-element halves of one 16-element SF vector, and the
  // even-thread SF write still lands once per vector.
  int const inGroupsPerRow = FUSE_SWIGLU ? 2 * colGroups : colGroups;
  int const outGroupsPerRow = colGroups;

  auto const* inVec = reinterpret_cast<PackedVec8 const*>(in);
  uint32_t* sfExpert = SFout + static_cast<int64_t>(expert) * sfWordsPerExpert;
  int64_t const outRowBase = static_cast<int64_t>(expert) * 2 * mTok;

  for (int r = blockIdx.x; r < mTok; r += gridDim.x) {
    // A padding slot must still read something in range. The caller points it
    // at a row it owns (conventionally a zero row appended to the input), not
    // at -1: a branch here would diverge across the whole block for the sake
    // of rows whose output the fold GEMM's m_tok padding discards anyway.
    int64_t const inRow =
        rowIdx == nullptr
            ? static_cast<int64_t>(expert) * mTok + r
            : static_cast<int64_t>(rowIdx[expert * mTok + r]);

    PackedVec8 v;
    if constexpr (FUSE_SWIGLU) {
      PackedVec8 g = inVec[inRow * inGroupsPerRow + colIdx];
      PackedVec8 u = inVec[inRow * inGroupsPerRow + colGroups + colIdx];
      v = swiglu_packed<Type>(g, u, rPre, rPost);
    } else {
      v = inVec[inRow * inGroupsPerRow + colIdx];
      // No nonlinearity in this path, so pre and post are one factor.
      // Skipped entirely when neither table exists: the fp32 round trip is
      // exact, but not free.
      if (preScale != nullptr || postScale != nullptr) {
        v = scale_packed<Type>(v, rPre * rPost);
      }
    }

    
    
    
    
    
    
    int32_t const baseRow = compute_mext_r1_base_row(r, mTok, layoutMode);
    int32_t const resRow = compute_mext_r1_residue_row(r, mTok, layoutMode);

    uint32_t* out_pos =
        out + (outRowBase + baseRow) * outGroupsPerRow + colIdx;
    uint32_t* residue_data_out =
        out + (outRowBase + resRow) * outGroupsPerRow + colIdx;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            baseRow, colIdx, numCols, sfExpert);
    auto residue_sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            resRow, colIdx, numCols, sfExpert);
    if (linearSF) {
      // Row-major SF: byte offset = row * (numCols/16) + group. Same
      // even-thread-writes convention as the swizzled helper.
      uint8_t* sfBase = reinterpret_cast<uint8_t*>(sfExpert);
      int32_t const sfCols = numCols / 16;
      bool const writer = (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF) == 0;
      sf_out = writer ? sfBase + static_cast<int64_t>(baseRow) * sfCols
                            + colIdx / CVT_FP4_NUM_THREADS_PER_SF
                      : nullptr;
      residue_sf_out = writer
          ? sfBase + static_cast<int64_t>(resRow) * sfCols
                + colIdx / CVT_FP4_NUM_THREADS_PER_SF
          : nullptr;
    }

    *out_pos = cvt_warp_fp16_to_fp4_mext_r1_fast<Type, UE8M0_SF>(
        v, SFScaleVal, SFScaleResVal, sf_out, residue_data_out,
        residue_sf_out);
  }
}

}  // namespace

// out:     uint8 (E*2*mTok, I/2)      -- FP4 base/residue rows, l-outermost,
//                                        interleaved per layout_mode
// out_sf:  uint8 (E * sfBytesPerExpert,) -- one padded SF block per expert
// input:   bf16/fp16 (E*mTok, 2I) when fuse_swiglu else (E*mTok, I)
// sf_scale: float32 [1] (single global scale) or [2] ([base, residue])
// row_idx: optional int32 (experts*m_tok,) slot -> source row of `input`.
//          Empty means the identity, i.e. expert e owns rows [e*m_tok, ...).
//          With a table `input` is the UNSORTED activation and m_tok can no
//          longer be read off its row count, so pass m_tok explicitly.
// pre_scale / post_scale: optional float32 (experts,) per-SLOT factors.
//          pre multiplies gate and up BEFORE SwiGLU (per-expert GEMM1
//          dequant ratio -- the nonlinearity is why it cannot wait), post
//          multiplies h after it (per-expert part of GEMM2's chain). With a
//          gather table the caller passes them already gathered per slot.
void swiglu_quant_mext_r1(torch::Tensor out, torch::Tensor out_sf,
                          torch::Tensor input, torch::Tensor sf_scale,
                          int64_t experts, bool fuse_swiglu,
                          int64_t layout_mode, torch::Tensor row_idx,
                          int64_t m_tok, torch::Tensor pre_scale,
                          torch::Tensor post_scale,
                          bool linear_sf) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2, "input must be 2-D CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kUInt8,
              "out must be uint8 CUDA");
  TORCH_CHECK(out_sf.is_cuda() && out_sf.scalar_type() == torch::kUInt8,
              "out_sf must be uint8 CUDA");
  TORCH_CHECK(sf_scale.scalar_type() == torch::kFloat32,
              "sf_scale must be float32");
  TORCH_CHECK(experts > 0, "experts must be positive");

  bool const gather = row_idx.defined() && row_idx.numel() > 0;
  if (gather) {
    TORCH_CHECK(row_idx.is_cuda() && row_idx.scalar_type() == torch::kInt,
                "row_idx must be int32 CUDA");
    TORCH_CHECK(row_idx.is_contiguous(), "row_idx must be contiguous");
    TORCH_CHECK(m_tok > 0, "a row_idx table needs an explicit m_tok");
    TORCH_CHECK(row_idx.numel() == experts * m_tok, "row_idx has ",
                row_idx.numel(), " entries, expected experts*m_tok = ",
                experts * m_tok);
  } else {
    TORCH_CHECK(input.size(0) % experts == 0,
                "input rows must divide by experts");
    TORCH_CHECK(m_tok <= 0 || m_tok == input.size(0) / experts,
                "m_tok disagrees with the input's row count");
  }
  int32_t const mTok = static_cast<int32_t>(
      gather ? m_tok : input.size(0) / experts);
  int64_t const inCols = input.size(1);
  TORCH_CHECK(!fuse_swiglu || inCols % 2 == 0,
              "fused SwiGLU needs an even input width ([gate | up])");
  int32_t const numCols =
      static_cast<int32_t>(fuse_swiglu ? inCols / 2 : inCols);
  TORCH_CHECK(numCols % CVT_FP4_ELTS_PER_THREAD == 0,
              "output width must be a multiple of 8, got ", numCols);
  TORCH_CHECK(numCols % CVT_FP4_SF_VEC_SIZE == 0,
              "output width must be a multiple of 16, got ", numCols);
  TORCH_CHECK(layout_mode == 1 || layout_mode == 2,
              "layout_mode must be 1 (row_pair, N-ext) or 2 (tile8, M-ext), "
              "got ", layout_mode);
  // tile8 groups tokens into chunks of 8 occupying 16 rows each; a partial
  // chunk would leave holes the GEMM still reads.
  TORCH_CHECK(layout_mode != 2 || mTok % 8 == 0,
              "layout_mode=tile8 needs m_tok divisible by 8, got ", mTok);

  int64_t const expectedOut =
      static_cast<int64_t>(experts) * 2 * mTok * (numCols / 2);
  TORCH_CHECK(out.numel() == expectedOut, "out has ", out.numel(),
              " bytes, expected ", expectedOut);

  // Same padded block the fold GEMM strides by: ceil(2*mTok/128) m-tiles.
  // linear_sf: plain row-major (2*mTok, numCols/16) bytes per expert.
  int64_t const sfBytesPerExpert =
      linear_sf ? static_cast<int64_t>(2 * mTok) * (numCols / 16)
                : static_cast<int64_t>((2 * mTok + 127) / 128) *
                      ((numCols + 63) / 64) * 512;
  TORCH_CHECK(out_sf.numel() == experts * sfBytesPerExpert,
              "out_sf has ", out_sf.numel(), " bytes, expected ",
              experts * sfBytesPerExpert);
  TORCH_CHECK(sfBytesPerExpert % 4 == 0, "SF block must be word-aligned");

  auto check_slot_scale = [&](torch::Tensor const& t, char const* name) {
    if (!t.defined() || t.numel() == 0) return (float const*)nullptr;
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32,
                name, " must be float32 CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.numel() == experts, name, " has ", t.numel(),
                " entries, expected one per slot = ", experts);
    return (float const*)t.data_ptr<float>();
  };
  float const* prePtr = check_slot_scale(pre_scale, "pre_scale");
  float const* postPtr = check_slot_scale(post_scale, "post_scale");

  int const colGroups = numCols / CVT_FP4_ELTS_PER_THREAD;
  int threads = colGroups < 256 ? ((colGroups + 31) / 32) * 32 : 256;
  // RESIDUE-GRAFT: at tiny mTok*experts (decode M=1: grid was (1,1,1) --
  // one CTA on a 148-SM part) shrink the block so the column dimension
  // spreads across more CTAs. 32 stays even, so the 2-thread SF pairing
  // (shfl_xor lane^1) and the even-thread SF write are preserved.
  while (threads > 32 &&
         static_cast<int64_t>(mTok) * experts *
                 ((colGroups + threads - 1) / threads) <
             64) {
    threads /= 2;
  }
  dim3 const block(threads);
  dim3 const grid(mTok, (colGroups + threads - 1) / threads,
                  static_cast<unsigned>(experts));

  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  auto* outPtr = reinterpret_cast<uint32_t*>(out.data_ptr<uint8_t>());
  auto* sfPtr = reinterpret_cast<uint32_t*>(out_sf.data_ptr<uint8_t>());
  float const* scalePtr = sf_scale.data_ptr<float>();
  int32_t const numSFScales = static_cast<int32_t>(sf_scale.numel());
  int32_t const sfWords = static_cast<int32_t>(sfBytesPerExpert / 4);
  int32_t const* rowPtr = gather ? row_idx.data_ptr<int32_t>() : nullptr;

#define LAUNCH(TYPE, FUSE)                                                  \
  swiglu_quant_mext_r1_pack8<TYPE, FUSE><<<grid, block, 0, stream>>>(       \
      mTok, numCols, reinterpret_cast<TYPE const*>(input.data_ptr()),       \
      scalePtr, outPtr, sfPtr, sfWords, numSFScales,                        \
      static_cast<int32_t>(layout_mode), rowPtr, prePtr, postPtr,       \
      linear_sf)

  if (input.scalar_type() == torch::kBFloat16) {
    if (fuse_swiglu) {
      LAUNCH(__nv_bfloat16, true);
    } else {
      LAUNCH(__nv_bfloat16, false);
    }
  } else if (input.scalar_type() == torch::kHalf) {
    if (fuse_swiglu) {
      LAUNCH(half, true);
    } else {
      LAUNCH(half, false);
    }
  } else {
    TORCH_CHECK(false, "input must be bfloat16 or float16");
  }
#undef LAUNCH

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swiglu_quant_mext_r1", &swiglu_quant_mext_r1,
        "Fused SwiGLU + MExt-R1 row-pair FP4 quantisation (experiment)",
        py::arg("out"), py::arg("out_sf"), py::arg("input"),
        py::arg("sf_scale"), py::arg("experts"), py::arg("fuse_swiglu"),
        py::arg("layout_mode"), py::arg("row_idx"), py::arg("m_tok"),
        py::arg("pre_scale"), py::arg("post_scale"),
        py::arg("linear_sf"));
}
