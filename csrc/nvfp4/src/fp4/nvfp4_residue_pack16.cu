/*
 * Pack16 (16-elements-per-thread) residue quantization specializations.
 *
 * The elt16 mapping gives each thread one full 16-element main SF vector, so
 * the main amax reduction needs no cross-thread shuffle and the input row
 * slice is a single 256-bit load on Blackwell (load_packed_vec16). The
 * residue side shrinks every warp structure by 2x relative to pack8:
 *
 *   ratio   residues/thread   residue-SF group   residue data gather
 *   r0125         2               8 threads      4 source lanes / chunk
 *   r025          4               4 threads      2 source lanes / chunk
 *   r050          8               2 threads      none (1 chunk per thread)
 *
 * (ratio=1.0 has no K-ext kernel: full-rank residue is served by the
 *  M-extension path, scaled_fp4_quant_mext_r1.)
 *
 * Output format (data, residue packing order, SF placement) is bit-identical
 * to the pack8 family -- only the thread mapping changes. Kept in a separate
 * file mirroring nvfp4_residue_pack8.cu so the two families stay comparable.
 *
 * Assumes: exactly RESIDUE_PER_8_ELTS mask bits per channel byte, full warps
 * (launcher rounds block size to 32), and numTotalCols % 16 == 0 so the
 * 64-bit main-data stores stay aligned (the dispatcher falls back to pack8
 * otherwise).
 */

#include <torch/all.h>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_utils.h"
#include "dispatch_utils.h"
#include "launch_bounds_utils.h"
#include "nvfp4_utils.cuh"

namespace vllm {

namespace pack16 {

__device__ __forceinline__ float pick_residue_value(
    float2 const (&residue)[4], int idx) {
  int pairIdx = idx >> 1;
  return (idx & 1) == 0 ? residue[pairIdx].x : residue[pairIdx].y;
}

// Residue data chunk address for the elt16 mapping. Chunks are the same
// element-ordered consecutive uint32s the pack8 helpers emit; an elt16 warp
// spans two pack8 warp sections, i.e. CHUNKS_PER_WARP = 32 * R2 / 8 chunks
// starting at warpId * CHUNKS_PER_WARP. R2 = residues per thread.
template <int R2>
__device__ __forceinline__ uint32_t* residue_data_offset(
    int rowIdx, int colIdx, int numMainCols, int numTotalCols,
    uint32_t* out) {
  constexpr int CHUNKS_PER_WARP = 32 * R2 / 8;
  int64_t u32PerRow = numTotalCols / CVT_FP4_ELTS_PER_UINT32;
  int64_t baseU32 = numMainCols / CVT_FP4_ELTS_PER_UINT32;
  int64_t rowBase = static_cast<int64_t>(rowIdx) * u32PerRow + baseU32;
  if constexpr (R2 == 8) {
    // One full chunk per thread; CHUNKS_PER_WARP == 32 so chunk == colIdx.
    return out + rowBase + colIdx;
  } else {
    int warpId = colIdx / 32;
    int laneId = colIdx % 32;
    if (laneId >= CHUNKS_PER_WARP) return nullptr;
    return out + rowBase + warpId * CHUNKS_PER_WARP + laneId;
  }
}

// Residue SF address for the elt16 mapping: every 16 element-ordered
// residues share one SF in the same tiled layout as the main SFs, indexed by
// extended-K position -- identical placement to cvt_residue_sf_get_offset,
// re-derived for 16-element threads.
template <class SFType, int R2>
__device__ __forceinline__ SFType* residue_sf_offset(
    int rowIdx, int colIdx, int numTotalCols, int numMainCols, SFType* base) {
  constexpr int THREADS_PER_SF_GROUP = 16 / R2;
  int warpId = colIdx / 32;
  int laneId = colIdx % 32;
  if (laneId % THREADS_PER_SF_GROUP != 0) return nullptr;
  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int residueElementIdx =
      numMainCols + warpId * 32 * R2 + sfGroupId * 16;
  int32_t kIdx = residueElementIdx / CVT_FP4_SF_VEC_SIZE;
  int64_t SFOffset =
      compute_tiled_sf_offset<SFType>(rowIdx, kIdx, numTotalCols);
  return base + SFOffset;
}

// Main quantization for one 16-element SF vector (thread-local, no
// shuffles). Math follows the pack8 ratio helpers line-for-line on each
// 8-element half so the output stays bit-identical. On return lo/hi hold the
// unscaled residues (original - dequant) for all 16 positions.
template <class Type, bool UE8M0_SF>
__device__ __forceinline__ u32x2 main_quant16(
    PackedVec<Type>& loVec, PackedVec<Type>& hiVec, float SFScaleVal,
    uint8_t* SFout, float2 (&loResidue)[4], float2 (&hiResidue)[4]) {
  auto localMax = __habs2(loVec.elts[0]);
#pragma unroll
  for (int i = 1; i < 4; ++i) {
    localMax = __hmax2(localMax, __habs2(loVec.elts[i]));
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    localMax = __hmax2(localMax, __habs2(hiVec.elts[i]));
  }
  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = compute_sf_from_max(vecMax, SFScaleVal);
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 loOriginal[4];
  float2 hiOriginal[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      loOriginal[i] = __half22float2(loVec.elts[i]);
      hiOriginal[i] = __half22float2(hiVec.elts[i]);
    } else {
      loOriginal[i] = __bfloat1622float2(loVec.elts[i]);
      hiOriginal[i] = __bfloat1622float2(hiVec.elts[i]);
    }
    loResidue[i].x = loOriginal[i].x * outputScale;
    loResidue[i].y = loOriginal[i].y * outputScale;
    hiResidue[i].x = hiOriginal[i].x * outputScale;
    hiResidue[i].y = hiOriginal[i].y * outputScale;
  }

  u32x2 packed;
  packed.lo = fp32x8_to_e2m1_with_dequant(loResidue);
  packed.hi = fp32x8_to_e2m1_with_dequant(hiResidue);

  float inverseOutputScale =
      outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    loResidue[i].x = loOriginal[i].x - loResidue[i].x * inverseOutputScale;
    loResidue[i].y = loOriginal[i].y - loResidue[i].y * inverseOutputScale;
    hiResidue[i].x = hiOriginal[i].x - hiResidue[i].x * inverseOutputScale;
    hiResidue[i].y = hiOriginal[i].y - hiResidue[i].y * inverseOutputScale;
  }
  return packed;
}

}  // namespace pack16

// ── r0125: 2 residues/thread, 8-thread SF groups, 4 source lanes/chunk ──────
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, 2)
    cvt_fp16_to_fp4_residue_pack16_r0125_no_swap(
        int32_t numRows, int32_t numTotalCols, int32_t numMainCols,
        Type const* in, float const* SFScale, uint32_t* out, uint32_t* SFout,
        uint8_t const* channelIndices) {
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= numMainCols / 16) {
    return;
  }
  uint8_t mask0 = channelIndices[colIdx * 2];
  uint8_t mask1 = channelIndices[colIdx * 2 + 1];
  int laneId = threadIdx.x % 32;

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVec<Type> loVec{};
    PackedVec<Type> hiVec{};
    if (validRow) {
      load_packed_vec16(
          in + static_cast<int64_t>(rowIdx) * numMainCols + colIdx * 16,
          loVec, hiVec);
    }

    auto sfOut = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 1>(
        rowIdx, colIdx, numTotalCols, SFout);
    auto residueSFOut = pack16::residue_sf_offset<uint8_t, 2>(
        rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

    float2 loResidue[4];
    float2 hiResidue[4];
    u32x2 packed = pack16::main_quant16<Type, UE8M0_SF>(
        loVec, hiVec, SFScaleVal, sfOut, loResidue, hiResidue);

    int i0 = __ffs(mask0) - 1;
    int i1 = __ffs(mask1) - 1;
    float rv0 = pack16::pick_residue_value(loResidue, i0);
    float rv1 = pack16::pick_residue_value(hiResidue, i1);

    float localMaxResidue = fmaxf(fabsf(rv0), fabsf(rv1));
    float sfGroupMaxResidue = warp_group_max<8>(localMaxResidue, laneId);
    float residueSFValue =
        compute_sf_from_max(sfGroupMaxResidue, SFScaleVal);
    residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
    float residueOutputScale =
        compute_output_scale_precise(residueSFValue, SFScaleVal);
    rv0 *= residueOutputScale;
    rv1 *= residueOutputScale;

    // Writer lane L packs residues from source lanes 4L..4L+3 (2 each).
    float2 residueChunk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int sourceThread = (laneId < 8 ? laneId : 0) * 4 + i;
      residueChunk[i] =
          make_float2(__shfl_sync(uint32_t(-1), rv0, sourceThread),
                      __shfl_sync(uint32_t(-1), rv1, sourceThread));
    }

    if (validRow) {
      int64_t outOffset = static_cast<int64_t>(rowIdx) *
                              (numTotalCols / CVT_FP4_ELTS_PER_UINT32) +
                          colIdx * 2;
      *reinterpret_cast<uint2*>(out + outOffset) =
          make_uint2(packed.lo, packed.hi);
      uint32_t* residueDataOut = pack16::residue_data_offset<2>(
          rowIdx, colIdx, numMainCols, numTotalCols, out);
      if (residueDataOut) {
        *residueDataOut = fp32_vec_to_e2m1(residueChunk);
      }
    }
  }
}

// ── r025: 4 residues/thread, 4-thread SF groups, 2 source lanes/chunk ───────
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, 2)
    cvt_fp16_to_fp4_residue_pack16_r025_no_swap(
        int32_t numRows, int32_t numTotalCols, int32_t numMainCols,
        Type const* in, float const* SFScale, uint32_t* out, uint32_t* SFout,
        uint8_t const* channelIndices) {
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= numMainCols / 16) {
    return;
  }
  uint8_t mask0 = channelIndices[colIdx * 2];
  uint8_t mask1 = channelIndices[colIdx * 2 + 1];
  int laneId = threadIdx.x % 32;

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVec<Type> loVec{};
    PackedVec<Type> hiVec{};
    if (validRow) {
      load_packed_vec16(
          in + static_cast<int64_t>(rowIdx) * numMainCols + colIdx * 16,
          loVec, hiVec);
    }

    auto sfOut = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 1>(
        rowIdx, colIdx, numTotalCols, SFout);
    auto residueSFOut = pack16::residue_sf_offset<uint8_t, 4>(
        rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

    float2 loResidue[4];
    float2 hiResidue[4];
    u32x2 packed = pack16::main_quant16<Type, UE8M0_SF>(
        loVec, hiVec, SFScaleVal, sfOut, loResidue, hiResidue);

    int i0 = __ffs(mask0) - 1;
    int i1 = __ffs(mask0 & (mask0 - 1)) - 1;
    int i2 = __ffs(mask1) - 1;
    int i3 = __ffs(mask1 & (mask1 - 1)) - 1;
    float rv0 = pack16::pick_residue_value(loResidue, i0);
    float rv1 = pack16::pick_residue_value(loResidue, i1);
    float rv2 = pack16::pick_residue_value(hiResidue, i2);
    float rv3 = pack16::pick_residue_value(hiResidue, i3);

    float localMaxResidue =
        fmaxf(fmaxf(fabsf(rv0), fabsf(rv1)), fmaxf(fabsf(rv2), fabsf(rv3)));
    float sfGroupMaxResidue = warp_group_max<4>(localMaxResidue, laneId);
    float residueSFValue =
        compute_sf_from_max(sfGroupMaxResidue, SFScaleVal);
    residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
    float residueOutputScale =
        compute_output_scale_precise(residueSFValue, SFScaleVal);
    rv0 *= residueOutputScale;
    rv1 *= residueOutputScale;
    rv2 *= residueOutputScale;
    rv3 *= residueOutputScale;

    // Writer lane L packs residues from source lanes 2L, 2L+1 (4 each).
    int writerGroup = laneId < 16 ? laneId : 0;
    float2 residueChunk[4];
    residueChunk[0] =
        make_float2(__shfl_sync(uint32_t(-1), rv0, writerGroup * 2),
                    __shfl_sync(uint32_t(-1), rv1, writerGroup * 2));
    residueChunk[1] =
        make_float2(__shfl_sync(uint32_t(-1), rv2, writerGroup * 2),
                    __shfl_sync(uint32_t(-1), rv3, writerGroup * 2));
    residueChunk[2] =
        make_float2(__shfl_sync(uint32_t(-1), rv0, writerGroup * 2 + 1),
                    __shfl_sync(uint32_t(-1), rv1, writerGroup * 2 + 1));
    residueChunk[3] =
        make_float2(__shfl_sync(uint32_t(-1), rv2, writerGroup * 2 + 1),
                    __shfl_sync(uint32_t(-1), rv3, writerGroup * 2 + 1));

    if (validRow) {
      int64_t outOffset = static_cast<int64_t>(rowIdx) *
                              (numTotalCols / CVT_FP4_ELTS_PER_UINT32) +
                          colIdx * 2;
      *reinterpret_cast<uint2*>(out + outOffset) =
          make_uint2(packed.lo, packed.hi);
      uint32_t* residueDataOut = pack16::residue_data_offset<4>(
          rowIdx, colIdx, numMainCols, numTotalCols, out);
      if (residueDataOut) {
        *residueDataOut = fp32_vec_to_e2m1(residueChunk);
      }
    }
  }
}

// ── r050: 8 residues/thread = exactly one chunk, 2-thread SF groups, no
// data gather at all ────────────────────────────────────────────────────────
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, 2)
    cvt_fp16_to_fp4_residue_pack16_r050_no_swap(
        int32_t numRows, int32_t numTotalCols, int32_t numMainCols,
        Type const* in, float const* SFScale, uint32_t* out, uint32_t* SFout,
        uint8_t const* channelIndices) {
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= numMainCols / 16) {
    return;
  }
  uint8_t mask0 = channelIndices[colIdx * 2];
  uint8_t mask1 = channelIndices[colIdx * 2 + 1];
  int laneId = threadIdx.x % 32;

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVec<Type> loVec{};
    PackedVec<Type> hiVec{};
    if (validRow) {
      load_packed_vec16(
          in + static_cast<int64_t>(rowIdx) * numMainCols + colIdx * 16,
          loVec, hiVec);
    }

    auto sfOut = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 1>(
        rowIdx, colIdx, numTotalCols, SFout);
    auto residueSFOut = pack16::residue_sf_offset<uint8_t, 8>(
        rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

    float2 loResidue[4];
    float2 hiResidue[4];
    u32x2 packed = pack16::main_quant16<Type, UE8M0_SF>(
        loVec, hiVec, SFScaleVal, sfOut, loResidue, hiResidue);

    int i0 = __ffs(mask0) - 1;
    uint8_t rem = mask0 & (mask0 - 1);
    int i1 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int i2 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int i3 = __ffs(rem) - 1;
    int j0 = __ffs(mask1) - 1;
    rem = mask1 & (mask1 - 1);
    int j1 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int j2 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int j3 = __ffs(rem) - 1;

    float rv0 = pack16::pick_residue_value(loResidue, i0);
    float rv1 = pack16::pick_residue_value(loResidue, i1);
    float rv2 = pack16::pick_residue_value(loResidue, i2);
    float rv3 = pack16::pick_residue_value(loResidue, i3);
    float rv4 = pack16::pick_residue_value(hiResidue, j0);
    float rv5 = pack16::pick_residue_value(hiResidue, j1);
    float rv6 = pack16::pick_residue_value(hiResidue, j2);
    float rv7 = pack16::pick_residue_value(hiResidue, j3);

    float localMaxResidue =
        fmaxf(fmaxf(fmaxf(fabsf(rv0), fabsf(rv1)), fmaxf(fabsf(rv2), fabsf(rv3))),
              fmaxf(fmaxf(fabsf(rv4), fabsf(rv5)), fmaxf(fabsf(rv6), fabsf(rv7))));
    float sfGroupMaxResidue = warp_group_max<2>(localMaxResidue, laneId);
    float residueSFValue =
        compute_sf_from_max(sfGroupMaxResidue, SFScaleVal);
    residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
    float residueOutputScale =
        compute_output_scale_precise(residueSFValue, SFScaleVal);

    float2 residueChunk[4];
    residueChunk[0] =
        make_float2(rv0 * residueOutputScale, rv1 * residueOutputScale);
    residueChunk[1] =
        make_float2(rv2 * residueOutputScale, rv3 * residueOutputScale);
    residueChunk[2] =
        make_float2(rv4 * residueOutputScale, rv5 * residueOutputScale);
    residueChunk[3] =
        make_float2(rv6 * residueOutputScale, rv7 * residueOutputScale);

    if (validRow) {
      int64_t outOffset = static_cast<int64_t>(rowIdx) *
                              (numTotalCols / CVT_FP4_ELTS_PER_UINT32) +
                          colIdx * 2;
      *reinterpret_cast<uint2*>(out + outOffset) =
          make_uint2(packed.lo, packed.hi);
      uint32_t* residueDataOut = pack16::residue_data_offset<8>(
          rowIdx, colIdx, numMainCols, numTotalCols, out);
      *residueDataOut = fp32_vec_to_e2m1(residueChunk);
    }
  }
}

// Grid/block for a pack16 launch: 16 elements per thread, full warps (the
// residue gathers use a constant full shuffle mask), one row per grid.x.
static void pack16_launch_config(int m, int n, int multiProcessorCount,
                                 dim3& grid, dim3& block) {
  int numThreads = std::max(n / 16, 32);
  int blockSize = std::min(((std::min(numThreads, 512) + 31) / 32) * 32, 512);
  block = dim3(blockSize);
  int const numBlocksPerSM = vllm_runtime_blocks_per_sm(blockSize);
  grid = dim3(std::min(computeEffectiveRows(m),
                       multiProcessorCount * numBlocksPerSM),
              cuda_utils::ceil_div(numThreads, blockSize));
}

#define PACK16_LAUNCHER(NAME, KERNEL)                                         \
  template <typename T>                                                       \
  void NAME(int m, int n_ext, int n, T const* input, float const* SFScale,    \
            int64_t* output, int32_t* SFOuput, bool useUE8M0,                 \
            uint8_t const* channelIndices, int multiProcessorCount,           \
            cudaStream_t stream) {                                            \
    dim3 grid, block;                                                         \
    pack16_launch_config(m, n, multiProcessorCount, grid, block);             \
    if (useUE8M0) {                                                           \
      KERNEL<T, true><<<grid, block, 0, stream>>>(                            \
          m, n_ext, n, input, SFScale, reinterpret_cast<uint32_t*>(output),   \
          reinterpret_cast<uint32_t*>(SFOuput), channelIndices);              \
    } else {                                                                  \
      KERNEL<T, false><<<grid, block, 0, stream>>>(                           \
          m, n_ext, n, input, SFScale, reinterpret_cast<uint32_t*>(output),   \
          reinterpret_cast<uint32_t*>(SFOuput), channelIndices);              \
    }                                                                         \
  }

PACK16_LAUNCHER(invokeFP4QuantizationPack16R0125NoSwap,
                cvt_fp16_to_fp4_residue_pack16_r0125_no_swap)
PACK16_LAUNCHER(invokeFP4QuantizationPack16R025NoSwap,
                cvt_fp16_to_fp4_residue_pack16_r025_no_swap)
PACK16_LAUNCHER(invokeFP4QuantizationPack16R050NoSwap,
                cvt_fp16_to_fp4_residue_pack16_r050_no_swap)

#undef PACK16_LAUNCHER

}  // namespace vllm

namespace {

void pack16_entry_common_checks(torch::Tensor const& input, int64_t n_ext,
                                int64_t residue_per_8,
                                int64_t expected_residue_per_8) {
  int32_t n = input.size(1);
  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  TORCH_CHECK(n_ext % 16 == 0,
              "pack16 residue path requires the extended K to be a multiple "
              "of 16 (64-bit store alignment); use the pack8 path otherwise.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "Unsupported input data type for pack16 quantize_to_fp4.");
  TORCH_CHECK(residue_per_8 == expected_residue_per_8,
              "pack16 residue entry called with mismatched residue_per_8.");
}

}  // namespace

#define PACK16_ENTRY(NAME, INVOKER, EXPECTED_R)                               \
  void NAME(torch::Tensor const& output, torch::Tensor const& input,          \
            torch::Tensor const& output_sf, torch::Tensor const& input_sf,    \
            torch::Tensor const& channel_indices, int64_t n_ext,              \
            int64_t residue_per_8) {                                          \
    int32_t m = input.size(0);                                                \
    int32_t n = input.size(1);                                                \
    pack16_entry_common_checks(input, n_ext, residue_per_8, EXPECTED_R);      \
    int multiProcessorCount =                                                 \
        get_device_attribute(cudaDevAttrMultiProcessorCount, -1);             \
    auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());       \
    auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());                \
    auto output_ptr = static_cast<int64_t*>(output.data_ptr());               \
    auto channel_indices_ptr =                                                \
        static_cast<uint8_t const*>(channel_indices.data_ptr());              \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));         \
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());         \
    bool useUE8M0 = false;                                                    \
    VLLM_DISPATCH_HALF_TYPES(                                                 \
        input.scalar_type(), "nvfp4_quant_kernel_pack16", [&] {               \
          using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;          \
          auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());   \
          vllm::INVOKER(m, static_cast<int>(n_ext), n, input_ptr,             \
                        input_sf_ptr, output_ptr, sf_out, useUE8M0,           \
                        channel_indices_ptr, multiProcessorCount, stream);    \
        });                                                                   \
  }

PACK16_ENTRY(scaled_fp4_quant_with_residue_ext_pack16_r0125_no_swap,
             invokeFP4QuantizationPack16R0125NoSwap, 1)
PACK16_ENTRY(scaled_fp4_quant_with_residue_ext_pack16_r025_no_swap,
             invokeFP4QuantizationPack16R025NoSwap, 2)
PACK16_ENTRY(scaled_fp4_quant_with_residue_ext_pack16_r050_no_swap,
             invokeFP4QuantizationPack16R050NoSwap, 4)

#undef PACK16_ENTRY
