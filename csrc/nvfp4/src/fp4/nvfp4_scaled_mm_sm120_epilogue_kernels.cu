/*
 * SM120 CUTLASS FP4 GEMM epilogues for MExt R1 fold.
 *
 * This file intentionally keeps only the clean vec-store path.  The development
 * branch also has atomic/debug/TMA/store-mode experiments; those are excluded so
 * the production migration has a single, reviewable contract:
 *
 *   tile8 path:
 *     A: physical [2M, K] tile8 row-pair layout
 *     D: compact  [ M, N] output
 *     fold: D[i, j] = GEMM_A[base_tile8_row(i), j]
 *                  + GEMM_A[residue_tile8_row(i), j]
 *
 *   row-pair col-swap path:
 *     A: physical [2M, K] interleaved [base0,res0,base1,res1,...] layout
 *     D: compact  [ M, N] output
 *     fold: D[i, j] = GEMM_A[2*i, j] + GEMM_A[2*i+1, j]
 */

#include <algorithm>
#include <cstdint>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define NVFP4_CUTLASS_CHECK(status)                 \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                __FILE__, ":", __LINE__, ": ",      \
                cutlassGetStatusString(error));     \
  }

#define CHECK_TYPE(x, st, m) \
  TORCH_CHECK(x.scalar_type() == st, ": Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) \
  TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

namespace cutlass::epilogue::fusion {

template <class GmemLayoutTagAux_, class ElementOutput_,
          class ElementCompute_, class ElementScalar_ = ElementCompute_,
          int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementOutput_>,
          FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest>
struct MextPairRowPairFoldTile8VecOnly : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementScalar_;
  using ElementAux = ElementOutput_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  static constexpr int AlignmentScalar = 1;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr auto RoundStyle = RoundStyle_;
  static constexpr bool IsAuxOutSupported = true;
};

template <class GmemLayoutTagAux_, class ElementOutput_,
          class ElementCompute_, class ElementScalar_ = ElementCompute_,
          int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementOutput_>,
          FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest>
struct MextPairColPairFoldRowPairFixed : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementScalar_;
  using ElementAux = ElementOutput_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  static constexpr int AlignmentScalar = 1;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr auto RoundStyle = RoundStyle_;
  static constexpr bool IsAuxOutSupported = true;
};

namespace detail {

template <class ElementOutput, class ElementCompute, FloatRoundStyle RoundStyle,
          class StrideAux>
struct Sm120MextPairRowPairFoldTile8VecOnlyStore {
  using ElementAux = ElementOutput;

  struct SharedStorage {};

  struct Arguments {
    ElementOutput* ptr_aux = nullptr;
    StrideAux dAux = {};
  };

  struct Params {
    ElementOutput* ptr_aux = nullptr;
    StrideAux dAux = {};
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&,
                                                  Arguments const& args,
                                                  void*) {
    return {args.ptr_aux, args.dAux};
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&,
                                              Arguments const&, void*,
                                              cudaStream_t,
                                              CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm120MextPairRowPairFoldTile8VecOnlyStore() {}

  CUTLASS_HOST_DEVICE
  Sm120MextPairRowPairFoldTile8VecOnlyStore(Params const& params,
                                            SharedStorage const&)
      : params(params) {}

  Params params;

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }

  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class ProblemShapeMNKL, class PhysCoordTensor,
            class ProblemShapeMNL>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(ProblemShapeMNKL problem_shape_mnkl,
                           PhysCoordTensor&& tCcPhys,
                           ProblemShapeMNL problem_shape_mnl,
                           Params const& params)
        : problem_shape_mnkl(problem_shape_mnkl),
          tCcPhys(cute::forward<PhysCoordTensor>(tCcPhys)),
          problem_shape_mnl(problem_shape_mnl),
          params(params) {}

    ProblemShapeMNKL problem_shape_mnkl;
    PhysCoordTensor tCcPhys;
    ProblemShapeMNL problem_shape_mnl;
    Params params;

    template <typename ElementAccumulator, typename ElementInput,
              int FragmentSize>
    CUTLASS_DEVICE auto visit(
        Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v,
        int epi_m, int epi_n,
        Array<ElementInput, FragmentSize> const& frg_input) {
      (void)frg_acc;
      (void)epi_v;
      (void)epi_m;
      (void)epi_n;
      return frg_input;
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(STensor&& reduction_buffer,
                               SyncFn const& sync_fn, int epi_m, int epi_n,
                               bool is_last_iteration,
                               VTensor visit_results) {
      (void)reduction_buffer;
      (void)sync_fn;
      (void)epi_m;
      (void)epi_n;
      (void)is_last_iteration;
      if (params.ptr_aux == nullptr) {
        return;
      }

      auto [problem_m2, problem_n, problem_k, problem_l] = problem_shape_mnkl;
      (void)problem_n;
      (void)problem_k;
      (void)problem_l;

      constexpr int kThreadFoldRows = 8;
      constexpr int kThreadFoldChunkRows = 2 * kThreadFoldRows;
      static_assert(sizeof(ElementOutput) == 2,
                    "tile8 vec-store assumes 16-bit output elements");

      Tensor tCcPhys_flat = coalesce(tCcPhys(_, _, _, epi_m, epi_n));
      Tensor visit_flat = recast<ElementCompute>(coalesce(visit_results));
      int compact_m_extent = int(problem_m2) / 2;
      auto stride_m = get<0>(params.dAux);
      auto stride_l = get<2>(params.dAux);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i + 3 < size(tCcPhys_flat); i += 4) {
        auto coord = tCcPhys_flat(i);
        bool valid = elem_less(coord, problem_shape_mnl);
        int global_m = valid ? int(get<0>(coord)) : 0;
        int global_n = valid ? int(get<1>(coord)) : 0;
        int global_l = valid ? int(get<2>(coord)) : 0;
        int chunk_local_m = global_m % kThreadFoldChunkRows;
        if (!valid || chunk_local_m >= kThreadFoldRows ||
            (global_n & 1) != 0) {
          continue;
        }

        int compact_global_m =
            (global_m / kThreadFoldChunkRows) * kThreadFoldRows +
            chunk_local_m;
        if (compact_global_m >= compact_m_extent) {
          continue;
        }

        // The tile8 quant layout makes each thread own:
        //   i/i+1   -> base row, adjacent columns
        //   i+2/i+3 -> residue row, same adjacent columns
        // So the row-pair fold can stay in registers and store two compact
        // output elements as one packed 32-bit value.
        //
        // Precision contract: fold in ElementCompute, then cast once to
        // ElementOutput. This matches the extended-K path more closely than
        // casting each physical row before the row-pair fold.
        ElementOutput* dst =
            params.ptr_aux + compact_global_m * stride_m + global_n +
            global_l * stride_l;
        Array<ElementOutput, 2> values;
        values[0] =
            ElementOutput(float(visit_flat(i)) + float(visit_flat(i + 2)));
        values[1] =
            ElementOutput(float(visit_flat(i + 1)) + float(visit_flat(i + 3)));
        uint32_t bits = *reinterpret_cast<uint32_t const*>(&values);
        *reinterpret_cast<uint32_t*>(dst) = bits;
      }
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    (void)K;
    auto problem_shape_mnl = make_shape(M, N, L);

    Tensor coordAux = make_identity_tensor(problem_shape_mnl);
    Tensor tCcPhys = sm90_partition_for_epilogue<ReferenceSrc>(
        coordAux, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile,
        args.tiled_copy, args.thread_idx);

    return ConsumerStoreCallbacks<decltype(args.problem_shape_mnkl),
                                  decltype(tCcPhys),
                                  decltype(problem_shape_mnl)>(
        args.problem_shape_mnkl, cute::move(tCcPhys), problem_shape_mnl,
        params);
  }
};

template <class ElementOutput, class ElementCompute, FloatRoundStyle RoundStyle,
          class StrideAux>
struct Sm120MextPairColPairFoldRowPairFixedStore {
  using ElementAux = ElementOutput;

  struct SharedStorage {};

  struct Arguments {
    ElementOutput* ptr_aux = nullptr;
    StrideAux dAux = {};
  };

  struct Params {
    ElementOutput* ptr_aux = nullptr;
    StrideAux dAux = {};
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&,
                                                  Arguments const& args,
                                                  void*) {
    return {args.ptr_aux, args.dAux};
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&,
                                              Arguments const&, void*,
                                              cudaStream_t,
                                              CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm120MextPairColPairFoldRowPairFixedStore() {}

  CUTLASS_HOST_DEVICE
  Sm120MextPairColPairFoldRowPairFixedStore(Params const& params,
                                            SharedStorage const&)
      : params(params) {}

  Params params;

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }

  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class ProblemShapeMNKL, class PhysCoordTensor,
            class ProblemShapeMNL>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(ProblemShapeMNKL problem_shape_mnkl,
                           PhysCoordTensor&& tCcPhys,
                           ProblemShapeMNL problem_shape_mnl,
                           Params const& params)
        : problem_shape_mnkl(problem_shape_mnkl),
          tCcPhys(cute::forward<PhysCoordTensor>(tCcPhys)),
          problem_shape_mnl(problem_shape_mnl),
          params(params) {}

    ProblemShapeMNKL problem_shape_mnkl;
    PhysCoordTensor tCcPhys;
    ProblemShapeMNL problem_shape_mnl;
    Params params;

    template <typename ElementAccumulator, typename ElementInput,
              int FragmentSize>
    CUTLASS_DEVICE auto visit(
        Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v,
        int epi_m, int epi_n,
        Array<ElementInput, FragmentSize> const& frg_input) {
      (void)frg_acc;
      (void)epi_v;
      (void)epi_m;
      (void)epi_n;
      return frg_input;
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(STensor&& reduction_buffer,
                               SyncFn const& sync_fn, int epi_m, int epi_n,
                               bool is_last_iteration,
                               VTensor visit_results) {
      (void)reduction_buffer;
      (void)sync_fn;
      (void)epi_m;
      (void)epi_n;
      (void)is_last_iteration;
      if (params.ptr_aux == nullptr) {
        return;
      }

      auto [problem_n_original, problem_m2, problem_k, problem_l] =
          problem_shape_mnkl;
      (void)problem_n_original;
      (void)problem_k;
      (void)problem_l;

      Tensor tCcPhys_flat = coalesce(tCcPhys(_, _, _, epi_m, epi_n));
      Tensor visit_flat = recast<ElementCompute>(coalesce(visit_results));
      int compact_m_extent = int(problem_m2) / 2;
      auto stride_m = get<0>(params.dAux);
      auto stride_l = get<2>(params.dAux);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i + 1 < size(tCcPhys_flat); i += 2) {
        auto base_coord = tCcPhys_flat(i);
        bool base_valid = elem_less(base_coord, problem_shape_mnl);
        int base_n = base_valid ? int(get<0>(base_coord)) : 0;
        int base_m = base_valid ? int(get<1>(base_coord)) : 0;
        int base_l = base_valid ? int(get<2>(base_coord)) : 0;
        if (!base_valid) {
          continue;
        }

        // Row-pair col-swap tactics are only emitted for quant layouts where
        // adjacent epilogue registers map to [base, residue] at the same
        // (n, l). Correctness tests compare every row-pair tactic against the
        // tile8 reference, so avoid re-validating the pair in the hot path.
        int compact_global_m = base_m / 2;

        ElementOutput* dst =
            params.ptr_aux + compact_global_m * stride_m + base_n +
            base_l * stride_l;
        *dst = ElementOutput(float(visit_flat(i)) +
                             float(visit_flat(i + 1)));
      }
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    (void)K;
    auto problem_shape_mnl = make_shape(M, N, L);

    Tensor coordAux = make_identity_tensor(problem_shape_mnl);
    Tensor tCcPhys = sm90_partition_for_epilogue<ReferenceSrc>(
        coordAux, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile,
        args.tiled_copy, args.thread_idx);

    return ConsumerStoreCallbacks<decltype(args.problem_shape_mnkl),
                                  decltype(tCcPhys),
                                  decltype(problem_shape_mnl)>(
        args.problem_shape_mnkl, cute::move(tCcPhys), problem_shape_mnl,
        params);
  }
};

}  // namespace detail

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC,
          bool DelayTmaStore, class GmemLayoutTagAux, class ElementOutput,
          class ElementCompute, class ElementScalar, int AlignmentAux,
          FloatRoundStyle RoundStyle, class CtaTileShapeMNK,
          class EpilogueTile, class SmemLayoutAtom, class CopyOpR2S>
struct FusionCallbacks<
    epilogue::Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                     ReuseSmemC, DelayTmaStore>,
    fusion::MextPairRowPairFoldTile8VecOnly<
        GmemLayoutTagAux, ElementOutput, ElementCompute, ElementScalar,
        AlignmentAux, RoundStyle>,
    CtaTileShapeMNK, EpilogueTile, SmemLayoutAtom, CopyOpR2S>
    : Sm90EVT<
          detail::Sm120MextPairRowPairFoldTile8VecOnlyStore<
              ElementOutput, ElementCompute, RoundStyle,
              cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>>,
          Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute,
                              RoundStyle>,
                  Sm90ScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,
                  Sm90AccFetch>> {
  using Impl = Sm90EVT<
      detail::Sm120MextPairRowPairFoldTile8VecOnlyStore<
          ElementOutput, ElementCompute, RoundStyle,
          cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>>,
      Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute,
                          RoundStyle>,
              Sm90ScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,
              Sm90AccFetch>>;
  using Operation = fusion::MextPairRowPairFoldTile8VecOnly<
      GmemLayoutTagAux, ElementOutput, ElementCompute, ElementScalar,
      AlignmentAux, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar const* alpha_ptr = nullptr;
    using StrideAlpha = Stride<_0, _0, int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementOutput* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      return {
          {
              {{alpha}, {alpha_ptr}, {dAlpha}},
              {},
              {},
          },
          {aux_ptr, dAux},
      };
    }
  };

  using Impl::Impl;
};

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC,
          bool DelayTmaStore, class GmemLayoutTagAux, class ElementOutput,
          class ElementCompute, class ElementScalar, int AlignmentAux,
          FloatRoundStyle RoundStyle, class CtaTileShapeMNK,
          class EpilogueTile, class SmemLayoutAtom, class CopyOpR2S>
struct FusionCallbacks<
    epilogue::Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                     ReuseSmemC, DelayTmaStore>,
    fusion::MextPairColPairFoldRowPairFixed<
        GmemLayoutTagAux, ElementOutput, ElementCompute, ElementScalar,
        AlignmentAux, RoundStyle>,
    CtaTileShapeMNK, EpilogueTile, SmemLayoutAtom, CopyOpR2S>
    : Sm90EVT<
          detail::Sm120MextPairColPairFoldRowPairFixedStore<
              ElementOutput, ElementCompute, RoundStyle,
              cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>>,
          Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute,
                              RoundStyle>,
                  Sm90ScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,
                  Sm90AccFetch>> {
  using Impl = Sm90EVT<
      detail::Sm120MextPairColPairFoldRowPairFixedStore<
          ElementOutput, ElementCompute, RoundStyle,
          cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>>,
      Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute,
                          RoundStyle>,
              Sm90ScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,
              Sm90AccFetch>>;
  using Operation = fusion::MextPairColPairFoldRowPairFixed<
      GmemLayoutTagAux, ElementOutput, ElementCompute, ElementScalar,
      AlignmentAux, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar const* alpha_ptr = nullptr;
    using StrideAlpha = Stride<_0, _0, int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementOutput* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      return {
          {
              {{alpha}, {alpha_ptr}, {dAlpha}},
              {},
              {},
          },
          {aux_ptr, dAux},
      };
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion

namespace {

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

uint32_t next_power_of_2(uint32_t num) {
  if (num <= 1) {
    return num;
  }
  return 1u << (32 - __builtin_clz(num - 1));
}

int round_up(int x, int y) {
  return (x + y - 1) / y * y;
}

struct Sm120Fp4ConfigM256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

struct Sm120Fp4Config128x128x256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _256>;
  using PerSmTileShape_MNK = Shape<_128, _128, _256>;
};

struct Sm120Fp4Config128x32x128 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _32, _128>;
  using PerSmTileShape_MNK = Shape<_128, _32, _128>;
};

struct Sm120Fp4Config128x32x256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _32, _256>;
  using PerSmTileShape_MNK = Shape<_128, _32, _256>;
};

struct Sm120Fp4Config128x64x128 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _64, _128>;
  using PerSmTileShape_MNK = Shape<_128, _64, _128>;
};

struct Sm120Fp4Config128x64x256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _64, _256>;
  using PerSmTileShape_MNK = Shape<_128, _64, _256>;
};

struct Sm120Fp4Config128x256x128 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _256, _128>;
  using PerSmTileShape_MNK = Shape<_128, _256, _128>;
};

struct Sm120Fp4ConfigDefault {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

template <typename Config, typename OutType, typename Scheduler = void>
struct Fp4GemmSm120MextFoldTile8VecOnly {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = void;
  using ElementC = void;
  using ElementAux = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementAux>::value;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using FusionOp = cutlass::epilogue::fusion::MextPairRowPairFoldTile8VecOnly<
      LayoutDTag, ElementAux, ElementCompute>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          FusionOp>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      Scheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Config, typename OutType, typename Scheduler = void>
struct Fp4GemmSm120MextFoldRowPairSwapAB {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = void;
  using ElementC = void;
  using ElementAux = OutType;
  using LayoutCTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::ColumnMajor;
  using LayoutAuxTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementAux>::value;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using FusionOp =
      cutlass::epilogue::fusion::MextPairColPairFoldRowPairFixed<
          LayoutAuxTag, ElementAux, ElementCompute>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          FusionOp>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using SchedulerTag =
      std::conditional_t<std::is_void_v<Scheduler>,
                         cutlass::gemm::StaticPersistentScheduler, Scheduler>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      SchedulerTag>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Config, typename ElementAux, typename Scheduler = void>
using MextRowPairSwapGemmT =
    typename Fp4GemmSm120MextFoldRowPairSwapAB<Config, ElementAux,
                                               Scheduler>::Gemm;

template <typename Gemm, typename ElementAux>
typename Gemm::Arguments args_from_options_mext_fold_tile8_vec_only(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B_ext,
    at::Tensor const& A_sf, at::Tensor const& B_sf, torch::Tensor const& alpha,
    int M2, int N, int K, int physical_k_ext) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int M = M2 / 2;
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M2, K, 1});
  auto stride_B = cute::make_int_tuple_from<StrideB>(physical_k_ext, 0);
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M2, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M2, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M2, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B_ext.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {{},
       nullptr,
       stride_D,
       nullptr,
       stride_D}};

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  fusion_args.aux_ptr = static_cast<ElementAux*>(D.data_ptr());
  fusion_args.dAux = stride_D;

  if constexpr (!std::is_const_v<decltype(arguments.scheduler.max_swizzle_size)>) {
    arguments.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(arguments.scheduler.raster_order)>) {
    using RasterOrder = decltype(arguments.scheduler.raster_order);
    arguments.scheduler.raster_order = RasterOrder::Heuristic;
  }
  // Keep the SM120 launch contract aligned with FlashInfer's FP4 StreamK path.
  arguments.hw_info.cluster_shape = dim3(1, 1, 1);
  arguments.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  return arguments;
}

template <typename Gemm, typename ElementAux>
typename Gemm::Arguments args_from_options_mext_fold_row_pair_swap_ab(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B_ext,
    at::Tensor const& A_sf, at::Tensor const& B_sf, torch::Tensor const& alpha,
    int M2, int N, int K, int physical_k_ext) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int M = M2 / 2;
  auto stride_A = cute::make_int_tuple_from<StrideA>(physical_k_ext, 0);
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {M2, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {N, M2, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(N, M2, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(N, M2, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M2, K, 1},
      {static_cast<ElementA const*>(B_ext.data_ptr()), stride_A,
       static_cast<ElementB const*>(A.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(B_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(A_sf.data_ptr()), layout_SFB},
      {{},
       nullptr,
       stride_D,
       nullptr,
       stride_D}};

  auto& fusion_args = arguments.epilogue.thread;
  using StrideAux = std::decay_t<decltype(fusion_args.dAux)>;
  auto stride_aux = cutlass::make_cute_packed_stride(StrideAux{}, {M, N, 1});
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  fusion_args.aux_ptr = static_cast<ElementAux*>(D.data_ptr());
  fusion_args.dAux = stride_aux;

  if constexpr (!std::is_const_v<decltype(arguments.scheduler.max_swizzle_size)>) {
    arguments.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(arguments.scheduler.raster_order)>) {
    using RasterOrder = decltype(arguments.scheduler.raster_order);
    arguments.scheduler.raster_order = RasterOrder::Heuristic;
  }
  arguments.hw_info.cluster_shape = dim3(1, 1, 1);
  arguments.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  return arguments;
}

template <typename Gemm, typename ElementAux>
void run_gemm_mext_fold_tile8_vec_only(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B_ext,
    at::Tensor const& A_sf, at::Tensor const& B_sf,
    torch::Tensor const& alpha, int M2, int N, int K, int physical_k_ext,
    cudaStream_t stream) {
  Gemm gemm;

  auto arguments =
      args_from_options_mext_fold_tile8_vec_only<Gemm, ElementAux>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  NVFP4_CUTLASS_CHECK(gemm.can_implement(arguments));
  NVFP4_CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  NVFP4_CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

template <typename Gemm, typename ElementAux>
void run_gemm_mext_fold_row_pair_swap_ab(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B_ext,
    at::Tensor const& A_sf, at::Tensor const& B_sf,
    torch::Tensor const& alpha, int M2, int N, int K, int physical_k_ext,
    cudaStream_t stream) {
  Gemm gemm;

  auto arguments =
      args_from_options_mext_fold_row_pair_swap_ab<Gemm, ElementAux>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  NVFP4_CUTLASS_CHECK(gemm.can_implement(arguments));
  NVFP4_CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  NVFP4_CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

void dispatch_bf16_mext_fold_tile8_vec_only(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B_ext,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, int M2, int N, int K, int physical_k_ext,
    cudaStream_t stream) {
#if defined(NVFP4_ROWPAIR_SMOKE_ONLY)
  (void)D;
  (void)A;
  (void)B_ext;
  (void)A_sf;
  (void)B_sf;
  (void)alpha;
  (void)M2;
  (void)N;
  (void)K;
  (void)physical_k_ext;
  (void)stream;
  TORCH_CHECK(false,
              "NVFP4_ROWPAIR_SMOKE_ONLY only builds tactic row-pair paths");
#else
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_power_of_2(M2));
  if (mp2 <= 256) {
    run_gemm_mext_fold_tile8_vec_only<
        Fp4GemmSm120MextFoldTile8VecOnly<
            Sm120Fp4ConfigM256, cutlass::bfloat16_t>::Gemm,
        cutlass::bfloat16_t>(
        D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    return;
  }

  run_gemm_mext_fold_tile8_vec_only<
      Fp4GemmSm120MextFoldTile8VecOnly<
          Sm120Fp4ConfigDefault, cutlass::bfloat16_t>::Gemm,
      cutlass::bfloat16_t>(
      D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
#endif
}

#define RUN_ROWPAIR_SWAP_TACTIC(CONFIG, SCHEDULER)                         \
  return run_gemm_mext_fold_row_pair_swap_ab<                             \
      MextRowPairSwapGemmT<CONFIG, ElementAux, SCHEDULER>, ElementAux>(    \
      D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream)

template <typename ElementAux>
void dispatch_mext_fold_row_pair_swap_ab_tactic(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B_ext,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, int M2, int N, int K, int physical_k_ext,
    int tactic, cudaStream_t stream) {
  switch (tactic) {
    case 0:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4ConfigM256, void);
    case 1:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4ConfigM256,
                              cutlass::gemm::StreamKScheduler);
    case 2:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x128x256, void);
    case 3:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x128x256,
                              cutlass::gemm::StreamKScheduler);
    case 4:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4ConfigDefault, void);
    case 5:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4ConfigDefault,
                              cutlass::gemm::StreamKScheduler);
    case 6:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x32x128, void);
    case 7:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x32x128,
                              cutlass::gemm::StreamKScheduler);
    case 8:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x32x256, void);
    case 9:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x32x256,
                              cutlass::gemm::StreamKScheduler);
    case 10:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x64x128, void);
    case 11:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x64x128,
                              cutlass::gemm::StreamKScheduler);
    case 12:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x64x256, void);
    case 13:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x64x256,
                              cutlass::gemm::StreamKScheduler);
    case 14:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x256x128, void);
    case 15:
      RUN_ROWPAIR_SWAP_TACTIC(Sm120Fp4Config128x256x128,
                              cutlass::gemm::StreamKScheduler);
    default:
      TORCH_CHECK(false,
                  "mext row-pair swap-ab tactic must be in [0, 15], got ",
                  tactic);
  }
}

#undef RUN_ROWPAIR_SWAP_TACTIC

void dispatch_bf16_mext_fold_tile8_vec_only_tactic(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B_ext,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, int M2, int N, int K, int physical_k_ext,
    int tactic, cudaStream_t stream) {
  if (tactic >= 16 && tactic <= 31) {
    return dispatch_mext_fold_row_pair_swap_ab_tactic<cutlass::bfloat16_t>(
        D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, tactic - 16,
        stream);
  }
#if defined(NVFP4_ROWPAIR_SMOKE_ONLY)
  switch (tactic) {
    case 0:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4ConfigM256, cutlass::bfloat16_t>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 6:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x32x128, cutlass::bfloat16_t>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 7:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x32x128, cutlass::bfloat16_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 15:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x256x128, cutlass::bfloat16_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
  }
  TORCH_CHECK(false,
              "NVFP4_ROWPAIR_SMOKE_ONLY only builds row-pair tactics 16..31 "
              "and tile8 reference tactics 0/6/7/15, "
              "got ",
              tactic);
#else
  switch (tactic) {
    case 0:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4ConfigM256, cutlass::bfloat16_t>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 1:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4ConfigM256, cutlass::bfloat16_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 2:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x128x256, cutlass::bfloat16_t>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 3:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x128x256, cutlass::bfloat16_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 4:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4ConfigDefault, cutlass::bfloat16_t>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 5:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4ConfigDefault, cutlass::bfloat16_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 6:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x32x128, cutlass::bfloat16_t>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 7:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x32x128, cutlass::bfloat16_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 8:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x32x256, cutlass::bfloat16_t>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 9:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x32x256, cutlass::bfloat16_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 10:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x64x128, cutlass::bfloat16_t>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 11:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x64x128, cutlass::bfloat16_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 12:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x64x256, cutlass::bfloat16_t>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 13:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x64x256, cutlass::bfloat16_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 14:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x256x128, cutlass::bfloat16_t>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 15:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x256x128, cutlass::bfloat16_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::bfloat16_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    default:
      TORCH_CHECK(false, "mext encoded tactic must be in [0, 31], got ",
                  tactic);
  }
#endif
}

void dispatch_f16_mext_fold_tile8_vec_only(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B_ext,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, int M2, int N, int K, int physical_k_ext,
    cudaStream_t stream) {
#if defined(NVFP4_ROWPAIR_SMOKE_ONLY)
  (void)D;
  (void)A;
  (void)B_ext;
  (void)A_sf;
  (void)B_sf;
  (void)alpha;
  (void)M2;
  (void)N;
  (void)K;
  (void)physical_k_ext;
  (void)stream;
  TORCH_CHECK(false,
              "NVFP4_ROWPAIR_SMOKE_ONLY only builds tactic row-pair paths");
#else
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_power_of_2(M2));
  if (mp2 <= 256) {
    run_gemm_mext_fold_tile8_vec_only<
        Fp4GemmSm120MextFoldTile8VecOnly<
            Sm120Fp4ConfigM256, cutlass::half_t>::Gemm,
        cutlass::half_t>(
        D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    return;
  }

  run_gemm_mext_fold_tile8_vec_only<
      Fp4GemmSm120MextFoldTile8VecOnly<
          Sm120Fp4ConfigDefault, cutlass::half_t>::Gemm,
      cutlass::half_t>(
      D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
#endif
}

void dispatch_f16_mext_fold_tile8_vec_only_tactic(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B_ext,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, int M2, int N, int K, int physical_k_ext,
    int tactic, cudaStream_t stream) {
  if (tactic >= 16 && tactic <= 31) {
    return dispatch_mext_fold_row_pair_swap_ab_tactic<cutlass::half_t>(
        D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, tactic - 16,
        stream);
  }
#if defined(NVFP4_ROWPAIR_SMOKE_ONLY)
  TORCH_CHECK(false,
              "NVFP4_ROWPAIR_SMOKE_ONLY only builds row-pair tactics 16..31 "
              "for fp16, got ",
              tactic);
#else
  switch (tactic) {
    case 0:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4ConfigM256, cutlass::half_t>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 1:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4ConfigM256, cutlass::half_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 2:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x128x256, cutlass::half_t>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 3:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x128x256, cutlass::half_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 4:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4ConfigDefault, cutlass::half_t>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 5:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4ConfigDefault, cutlass::half_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 6:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x32x128, cutlass::half_t>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 7:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x32x128, cutlass::half_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 8:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x32x256, cutlass::half_t>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 9:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x32x256, cutlass::half_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 10:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x64x128, cutlass::half_t>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 11:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x64x128, cutlass::half_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 12:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x64x256, cutlass::half_t>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 13:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x64x256, cutlass::half_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 14:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x256x128, cutlass::half_t>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    case 15:
      return run_gemm_mext_fold_tile8_vec_only<
          Fp4GemmSm120MextFoldTile8VecOnly<
              Sm120Fp4Config128x256x128, cutlass::half_t,
              cutlass::gemm::StreamKScheduler>::Gemm,
          cutlass::half_t>(
          D, A, B_ext, A_sf, B_sf, alpha, M2, N, K, physical_k_ext, stream);
    default:
      TORCH_CHECK(false, "mext encoded tactic must be in [0, 31], got ",
                  tactic);
  }
#endif
}

template <typename Config, typename OutType, typename Scheduler = void>
struct Fp4GemmSm120Prefix {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      Scheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Config, typename OutType, typename Scheduler = void>
struct Fp4GemmSm120PrefixSwapAB {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using SchedulerTag =
      std::conditional_t<std::is_void_v<Scheduler>,
                         cutlass::gemm::StaticPersistentScheduler, Scheduler>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      SchedulerTag>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
typename Gemm::Arguments args_from_options_prefix(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B_ext,
    at::Tensor const& A_sf, at::Tensor const& B_sf, torch::Tensor const& alpha,
    int M, int N, int K, int physical_k_ext) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cute::make_int_tuple_from<StrideB>(physical_k_ext, 0);
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B_ext.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {{},
       static_cast<ElementD const*>(D.data_ptr()),
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());

  if constexpr (!std::is_const_v<decltype(arguments.scheduler.max_swizzle_size)>) {
    arguments.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(arguments.scheduler.raster_order)>) {
    using RasterOrder = decltype(arguments.scheduler.raster_order);
    arguments.scheduler.raster_order = RasterOrder::Heuristic;
  }
  // Match the launch contract used by the MExt and FlashInfer FP4 paths.
  arguments.hw_info.cluster_shape = dim3(1, 1, 1);
  arguments.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  return arguments;
}

template <typename Gemm>
typename Gemm::Arguments args_from_options_prefix_swap_ab(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B_ext,
    at::Tensor const& A_sf, at::Tensor const& B_sf, torch::Tensor const& alpha,
    int M, int N, int K, int physical_k_ext) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cute::make_int_tuple_from<StrideA>(physical_k_ext, 0);
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {M, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {N, M, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(N, M, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(N, M, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K, 1},
      {static_cast<ElementA const*>(B_ext.data_ptr()), stride_A,
       static_cast<ElementB const*>(A.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(B_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(A_sf.data_ptr()), layout_SFB},
      {{},
       nullptr,
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 0.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());

  if constexpr (!std::is_const_v<decltype(arguments.scheduler.max_swizzle_size)>) {
    arguments.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(arguments.scheduler.raster_order)>) {
    using RasterOrder = decltype(arguments.scheduler.raster_order);
    arguments.scheduler.raster_order = RasterOrder::Heuristic;
  }
  arguments.hw_info.cluster_shape = dim3(1, 1, 1);
  arguments.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  return arguments;
}

template <typename Gemm>
void run_gemm_prefix(at::Tensor& D, at::Tensor const& A,
                     at::Tensor const& B_ext, at::Tensor const& A_sf,
                     at::Tensor const& B_sf, torch::Tensor const& alpha,
                     int M, int N, int K, int physical_k_ext,
                     cudaStream_t stream) {
  Gemm gemm;

  auto arguments = args_from_options_prefix<Gemm>(
      D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  NVFP4_CUTLASS_CHECK(gemm.can_implement(arguments));
  NVFP4_CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  NVFP4_CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

template <typename Gemm>
void run_gemm_prefix_swap_ab(at::Tensor& D, at::Tensor const& A,
                             at::Tensor const& B_ext,
                             at::Tensor const& A_sf,
                             at::Tensor const& B_sf,
                             torch::Tensor const& alpha, int M, int N, int K,
                             int physical_k_ext, cudaStream_t stream) {
  Gemm gemm;

  auto arguments = args_from_options_prefix_swap_ab<Gemm>(
      D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  NVFP4_CUTLASS_CHECK(gemm.can_implement(arguments));
  NVFP4_CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  NVFP4_CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

void dispatch_bf16_prefix(torch::Tensor& D, torch::Tensor const& A,
                          torch::Tensor const& B_ext,
                          torch::Tensor const& A_sf,
                          torch::Tensor const& B_sf,
                          torch::Tensor const& alpha, int M, int N, int K,
                          int physical_k_ext, cudaStream_t stream) {
#if defined(NVFP4_ROWPAIR_SMOKE_ONLY)
  (void)D;
  (void)A;
  (void)B_ext;
  (void)A_sf;
  (void)B_sf;
  (void)alpha;
  (void)M;
  (void)N;
  (void)K;
  (void)physical_k_ext;
  (void)stream;
  TORCH_CHECK(false,
              "NVFP4_ROWPAIR_SMOKE_ONLY does not build prefix GEMM paths");
#else
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_power_of_2(M));
  if (mp2 <= 256) {
    run_gemm_prefix<Fp4GemmSm120Prefix<
        Sm120Fp4ConfigM256, cutlass::bfloat16_t>::Gemm>(
        D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    return;
  }

  run_gemm_prefix<Fp4GemmSm120Prefix<
      Sm120Fp4ConfigDefault, cutlass::bfloat16_t>::Gemm>(
      D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
#endif
}

void dispatch_f16_prefix(torch::Tensor& D, torch::Tensor const& A,
                         torch::Tensor const& B_ext,
                         torch::Tensor const& A_sf,
                         torch::Tensor const& B_sf,
                         torch::Tensor const& alpha, int M, int N, int K,
                         int physical_k_ext, cudaStream_t stream) {
#if defined(NVFP4_ROWPAIR_SMOKE_ONLY)
  (void)D;
  (void)A;
  (void)B_ext;
  (void)A_sf;
  (void)B_sf;
  (void)alpha;
  (void)M;
  (void)N;
  (void)K;
  (void)physical_k_ext;
  (void)stream;
  TORCH_CHECK(false,
              "NVFP4_ROWPAIR_SMOKE_ONLY does not build prefix GEMM paths");
#else
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_power_of_2(M));
  if (mp2 <= 256) {
    run_gemm_prefix<Fp4GemmSm120Prefix<
        Sm120Fp4ConfigM256, cutlass::half_t>::Gemm>(
        D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    return;
  }

  run_gemm_prefix<Fp4GemmSm120Prefix<
      Sm120Fp4ConfigDefault, cutlass::half_t>::Gemm>(
      D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
#endif
}

template <typename ElementD>
void dispatch_prefix_tactic(torch::Tensor& D, torch::Tensor const& A,
                            torch::Tensor const& B_ext,
                            torch::Tensor const& A_sf,
                            torch::Tensor const& B_sf,
                            torch::Tensor const& alpha, int M, int N, int K,
                            int physical_k_ext, int tactic,
                            cudaStream_t stream) {
#if defined(NVFP4_ROWPAIR_SMOKE_ONLY)
  (void)D;
  (void)A;
  (void)B_ext;
  (void)A_sf;
  (void)B_sf;
  (void)alpha;
  (void)M;
  (void)N;
  (void)K;
  (void)physical_k_ext;
  (void)tactic;
  (void)stream;
  TORCH_CHECK(false,
              "NVFP4_ROWPAIR_SMOKE_ONLY does not build prefix tactic paths");
#else
  switch (tactic) {
    case 0:
      return run_gemm_prefix<typename Fp4GemmSm120Prefix<
          Sm120Fp4ConfigM256, ElementD>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 1:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4ConfigM256, ElementD,
                             cutlass::gemm::StreamKScheduler>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 2:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x128x256, ElementD>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 3:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x128x256, ElementD,
                             cutlass::gemm::StreamKScheduler>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 4:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4ConfigDefault, ElementD>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 5:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4ConfigDefault, ElementD,
                             cutlass::gemm::StreamKScheduler>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 6:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x32x128, ElementD>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 7:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x32x128, ElementD,
                             cutlass::gemm::StreamKScheduler>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 8:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x32x256, ElementD>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 9:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x32x256, ElementD,
                             cutlass::gemm::StreamKScheduler>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 10:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x64x128, ElementD>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 11:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x64x128, ElementD,
                             cutlass::gemm::StreamKScheduler>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 12:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x64x256, ElementD>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 13:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x64x256, ElementD,
                             cutlass::gemm::StreamKScheduler>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 14:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x256x128, ElementD>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    case 15:
      return run_gemm_prefix<typename
          Fp4GemmSm120Prefix<Sm120Fp4Config128x256x128, ElementD,
                             cutlass::gemm::StreamKScheduler>::Gemm>(
          D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream);
    default:
      TORCH_CHECK(false, "prefix tactic must be in [0, 15], got ", tactic);
  }
#endif
}

template <typename ElementD>
void dispatch_prefix_swap_ab_tactic(torch::Tensor& D, torch::Tensor const& A,
                                    torch::Tensor const& B_ext,
                                    torch::Tensor const& A_sf,
                                    torch::Tensor const& B_sf,
                                    torch::Tensor const& alpha, int M, int N,
                                    int K, int physical_k_ext, int tactic,
                                    cudaStream_t stream) {
#if defined(NVFP4_ROWPAIR_SMOKE_ONLY)
  (void)D;
  (void)A;
  (void)B_ext;
  (void)A_sf;
  (void)B_sf;
  (void)alpha;
  (void)M;
  (void)N;
  (void)K;
  (void)physical_k_ext;
  (void)tactic;
  (void)stream;
  TORCH_CHECK(false,
              "NVFP4_ROWPAIR_SMOKE_ONLY does not build prefix swap paths");
#else
#define RUN_PREFIX_SWAP_TACTIC(Config, Scheduler)                         \
  return run_gemm_prefix_swap_ab<typename Fp4GemmSm120PrefixSwapAB<       \
      Config, ElementD, Scheduler>::Gemm>(                                \
      D, A, B_ext, A_sf, B_sf, alpha, M, N, K, physical_k_ext, stream)

  switch (tactic) {
    case 0:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4ConfigM256, void);
    case 1:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4ConfigM256,
                             cutlass::gemm::StreamKScheduler);
    case 2:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x128x256, void);
    case 3:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x128x256,
                             cutlass::gemm::StreamKScheduler);
    case 4:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4ConfigDefault, void);
    case 5:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4ConfigDefault,
                             cutlass::gemm::StreamKScheduler);
    case 6:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x32x128, void);
    case 7:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x32x128,
                             cutlass::gemm::StreamKScheduler);
    case 8:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x32x256, void);
    case 9:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x32x256,
                             cutlass::gemm::StreamKScheduler);
    case 10:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x64x128, void);
    case 11:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x64x128,
                             cutlass::gemm::StreamKScheduler);
    case 12:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x64x256, void);
    case 13:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x64x256,
                             cutlass::gemm::StreamKScheduler);
    case 14:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x256x128, void);
    case 15:
      RUN_PREFIX_SWAP_TACTIC(Sm120Fp4Config128x256x128,
                             cutlass::gemm::StreamKScheduler);
    default:
      TORCH_CHECK(false, "prefix swap-ab tactic must be in [0, 15], got ",
                  tactic);
  }

#undef RUN_PREFIX_SWAP_TACTIC
#endif
}

}  // namespace

void cutlass_scaled_fp4_mm_ext_b_prefix_sm120a(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B_ext,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, int64_t k_base) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_TH_CUDA(D, "output");
  CHECK_CONTIGUOUS(D, "output");
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B_ext, FLOAT4_E2M1X2, "b_ext");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(D.dim() == 2, "output must be a matrix");
  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B_ext.dim() == 2, "b_ext must be a matrix");

  auto const m = static_cast<int>(A.sizes()[0]);
  auto const n = static_cast<int>(B_ext.sizes()[0]);
  auto const k = static_cast<int>(k_base);
  auto const a_k = static_cast<int>(A.sizes()[1] * 2);
  auto const physical_k_ext = static_cast<int>(B_ext.sizes()[1] * 2);

  TORCH_CHECK(D.sizes()[0] == m,
              "output rows must match A rows, got output M=", D.sizes()[0],
              " and A rows=", m);
  TORCH_CHECK(D.sizes()[1] == n,
              "output N must match b_ext rows, got output N=", D.sizes()[1],
              " and b_ext rows=", n);
  TORCH_CHECK(k > 0, "k_base must be positive, got ", k);
  TORCH_CHECK(a_k == k,
              "a and logical k_base mismatch: a has unpacked K=", a_k,
              ", k_base=", k);
  TORCH_CHECK(physical_k_ext >= k,
              "b_ext unpacked K=", physical_k_ext,
              " must cover logical k_base=", k);

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k_base to be divisible by ",
              alignment, ", but got ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got b_ext shape: (", B_ext.sizes()[0], "x",
              B_ext.sizes()[1], ").");

  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(A_sf.sizes()[1] == B_sf.sizes()[1],
              "scale_a and scale_b shapes cannot be multiplied (",
              A_sf.sizes()[0], "x", A_sf.sizes()[1], " and ",
              B_sf.sizes()[0], "x", B_sf.sizes()[1], ")");
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to a shape (", rounded_m,
              "x", rounded_k, "), but got a shape (", A_sf.sizes()[0], "x",
              A_sf.sizes()[1], ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to a shape (", rounded_n,
              "x", rounded_k, "), but got a shape (", B_sf.sizes()[0], "x",
              B_sf.sizes()[1], ")");

  auto out_dtype = D.dtype();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  if (out_dtype == at::ScalarType::BFloat16) {
    dispatch_bf16_prefix(D, A, B_ext, A_sf, B_sf, alpha, m, n, k,
                         physical_k_ext, stream);
    return;
  }
  if (out_dtype == at::ScalarType::Half) {
    dispatch_f16_prefix(D, A, B_ext, A_sf, B_sf, alpha, m, n, k,
                        physical_k_ext, stream);
    return;
  }

  TORCH_CHECK(false,
              "Unsupported output data type of nvfp4 prefix-B GEMM sm120 (",
              out_dtype, ")");
#else
  TORCH_CHECK(false,
              "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
              "a CUTLASS 3.8 source directory to enable support.");
#endif
}

void cutlass_scaled_fp4_mm_ext_b_prefix_tactic_sm120a(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B_ext,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, int64_t k_base, int64_t tactic) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_TH_CUDA(D, "output");
  CHECK_CONTIGUOUS(D, "output");
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B_ext, FLOAT4_E2M1X2, "b_ext");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(D.dim() == 2, "output must be a matrix");
  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B_ext.dim() == 2, "b_ext must be a matrix");

  auto const m = static_cast<int>(A.sizes()[0]);
  auto const n = static_cast<int>(B_ext.sizes()[0]);
  auto const k = static_cast<int>(k_base);
  auto const tactic_int = static_cast<int>(tactic);
  auto const a_k = static_cast<int>(A.sizes()[1] * 2);
  auto const physical_k_ext = static_cast<int>(B_ext.sizes()[1] * 2);

  TORCH_CHECK(tactic_int >= 0 && tactic_int <= 15,
              "prefix tactic must be in [0, 15], got ", tactic);
  TORCH_CHECK(D.sizes()[0] == m,
              "output rows must match A rows, got output M=", D.sizes()[0],
              " and A rows=", m);
  TORCH_CHECK(D.sizes()[1] == n,
              "output N must match b_ext rows, got output N=", D.sizes()[1],
              " and b_ext rows=", n);
  TORCH_CHECK(k > 0, "k_base must be positive, got ", k);
  TORCH_CHECK(a_k == k,
              "a and logical k_base mismatch: a has unpacked K=", a_k,
              ", k_base=", k);
  TORCH_CHECK(physical_k_ext >= k,
              "b_ext unpacked K=", physical_k_ext,
              " must cover logical k_base=", k);

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k_base to be divisible by ",
              alignment, ", but got ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got b_ext shape: (", B_ext.sizes()[0], "x",
              B_ext.sizes()[1], ").");

  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(A_sf.sizes()[1] == B_sf.sizes()[1],
              "scale_a and scale_b shapes cannot be multiplied (",
              A_sf.sizes()[0], "x", A_sf.sizes()[1], " and ",
              B_sf.sizes()[0], "x", B_sf.sizes()[1], ")");
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to a shape (", rounded_m,
              "x", rounded_k, "), but got a shape (", A_sf.sizes()[0], "x",
              A_sf.sizes()[1], ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to a shape (", rounded_n,
              "x", rounded_k, "), but got a shape (", B_sf.sizes()[0], "x",
              B_sf.sizes()[1], ")");

  auto out_dtype = D.dtype();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  if (out_dtype == at::ScalarType::BFloat16) {
    dispatch_prefix_tactic<cutlass::bfloat16_t>(
        D, A, B_ext, A_sf, B_sf, alpha, m, n, k, physical_k_ext,
        tactic_int, stream);
    return;
  }
  if (out_dtype == at::ScalarType::Half) {
    dispatch_prefix_tactic<cutlass::half_t>(
        D, A, B_ext, A_sf, B_sf, alpha, m, n, k, physical_k_ext,
        tactic_int, stream);
    return;
  }

  TORCH_CHECK(false,
              "Unsupported output data type of nvfp4 prefix-B tactic GEMM "
              "sm120 (",
              out_dtype, ")");
#else
  TORCH_CHECK(false,
              "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
              "a CUTLASS 3.8 source directory to enable support.");
#endif
}

void cutlass_scaled_fp4_mm_ext_b_prefix_swap_tactic_sm120a(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B_ext,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, int64_t k_base, int64_t tactic) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_TH_CUDA(D, "output");
  CHECK_CONTIGUOUS(D, "output");
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B_ext, FLOAT4_E2M1X2, "b_ext");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(D.dim() == 2, "output must be a matrix");
  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B_ext.dim() == 2, "b_ext must be a matrix");

  auto const m = static_cast<int>(A.sizes()[0]);
  auto const n = static_cast<int>(B_ext.sizes()[0]);
  auto const k = static_cast<int>(k_base);
  auto const tactic_int = static_cast<int>(tactic);
  auto const a_k = static_cast<int>(A.sizes()[1] * 2);
  auto const physical_k_ext = static_cast<int>(B_ext.sizes()[1] * 2);

  TORCH_CHECK(tactic_int >= 0 && tactic_int <= 15,
              "prefix swap-ab tactic must be in [0, 15], got ", tactic);
  TORCH_CHECK(D.sizes()[0] == m,
              "output rows must match A rows, got output M=", D.sizes()[0],
              " and A rows=", m);
  TORCH_CHECK(D.sizes()[1] == n,
              "output N must match b_ext rows, got output N=", D.sizes()[1],
              " and b_ext rows=", n);
  TORCH_CHECK(k > 0, "k_base must be positive, got ", k);
  TORCH_CHECK(a_k == k,
              "a and logical k_base mismatch: a has unpacked K=", a_k,
              ", k_base=", k);
  TORCH_CHECK(physical_k_ext >= k,
              "b_ext unpacked K=", physical_k_ext,
              " must cover logical k_base=", k);

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k_base to be divisible by ",
              alignment, ", but got ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got b_ext shape: (", B_ext.sizes()[0], "x",
              B_ext.sizes()[1], ").");

  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(A_sf.sizes()[1] == B_sf.sizes()[1],
              "scale_a and scale_b shapes cannot be multiplied (",
              A_sf.sizes()[0], "x", A_sf.sizes()[1], " and ",
              B_sf.sizes()[0], "x", B_sf.sizes()[1], ")");
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to a shape (", rounded_m,
              "x", rounded_k, "), but got a shape (", A_sf.sizes()[0], "x",
              A_sf.sizes()[1], ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to a shape (", rounded_n,
              "x", rounded_k, "), but got a shape (", B_sf.sizes()[0], "x",
              B_sf.sizes()[1], ")");

  auto out_dtype = D.dtype();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  if (out_dtype == at::ScalarType::BFloat16) {
    dispatch_prefix_swap_ab_tactic<cutlass::bfloat16_t>(
        D, A, B_ext, A_sf, B_sf, alpha, m, n, k, physical_k_ext,
        tactic_int, stream);
    return;
  }
  if (out_dtype == at::ScalarType::Half) {
    dispatch_prefix_swap_ab_tactic<cutlass::half_t>(
        D, A, B_ext, A_sf, B_sf, alpha, m, n, k, physical_k_ext,
        tactic_int, stream);
    return;
  }

  TORCH_CHECK(false,
              "Unsupported output data type of nvfp4 prefix-B swap-ab tactic "
              "GEMM sm120 (",
              out_dtype, ")");
#else
  TORCH_CHECK(false,
              "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
              "a CUTLASS 3.8 source directory to enable support.");
#endif
}

void cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean_sm120a(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B_ext,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, int64_t k_base) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_TH_CUDA(D, "output");
  CHECK_CONTIGUOUS(D, "output");
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B_ext, FLOAT4_E2M1X2, "b_ext");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(D.dim() == 2, "output must be a matrix");
  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B_ext.dim() == 2, "b_ext must be a matrix");

  auto const m2 = static_cast<int>(A.sizes()[0]);
  auto const m = static_cast<int>(D.sizes()[0]);
  auto const n = static_cast<int>(B_ext.sizes()[0]);
  auto const k = static_cast<int>(k_base);
  auto const a_k = static_cast<int>(A.sizes()[1] * 2);
  auto const physical_k_ext = static_cast<int>(B_ext.sizes()[1] * 2);

  TORCH_CHECK(m2 % 16 == 0,
              "mext tile8 fold requires A rows divisible by 16, got ", m2);
  TORCH_CHECK(m * 2 == m2,
              "output rows must be half of A rows for mext fold, got output M=",
              m, " and A rows=", m2);
  TORCH_CHECK(D.sizes()[1] == n,
              "output N must match b_ext rows, got output N=", D.sizes()[1],
              " and b_ext rows=", n);
  TORCH_CHECK(k > 0, "k_base must be positive, got ", k);
  TORCH_CHECK(a_k == k,
              "a and logical k_base mismatch: a has unpacked K=", a_k,
              ", k_base=", k);
  TORCH_CHECK(physical_k_ext >= k,
              "b_ext unpacked K=", physical_k_ext,
              " must cover logical k_base=", k);

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k_base to be divisible by ",
              alignment, ", but got ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got b_ext shape: (", B_ext.sizes()[0], "x",
              B_ext.sizes()[1], ").");

  int rounded_m2 = round_up(m2, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(A_sf.sizes()[1] == B_sf.sizes()[1],
              "scale_a and scale_b shapes cannot be multiplied (",
              A_sf.sizes()[0], "x", A_sf.sizes()[1], " and ",
              B_sf.sizes()[0], "x", B_sf.sizes()[1], ")");
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m2 && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to a shape (", rounded_m2,
              "x", rounded_k, "), but got a shape (", A_sf.sizes()[0], "x",
              A_sf.sizes()[1], ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to a shape (", rounded_n,
              "x", rounded_k, "), but got a shape (", B_sf.sizes()[0], "x",
              B_sf.sizes()[1], ")");

  auto out_dtype = D.dtype();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  if (out_dtype == at::ScalarType::BFloat16) {
    dispatch_bf16_mext_fold_tile8_vec_only(
        D, A, B_ext, A_sf, B_sf, alpha, m2, n, k, physical_k_ext, stream);
    return;
  }
  if (out_dtype == at::ScalarType::Half) {
    dispatch_f16_mext_fold_tile8_vec_only(
        D, A, B_ext, A_sf, B_sf, alpha, m2, n, k, physical_k_ext, stream);
    return;
  }

  TORCH_CHECK(false,
              "Unsupported output data type of nvfp4 mext tile8 clean "
              "vec-store GEMM sm120 (",
              out_dtype, ")");
#else
  TORCH_CHECK(false,
              "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
              "a CUTLASS 3.8 source directory to enable support.");
#endif
}

void cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean_tactic_sm120a(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B_ext,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, int64_t k_base, int64_t tactic) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_TH_CUDA(D, "output");
  CHECK_CONTIGUOUS(D, "output");
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B_ext, FLOAT4_E2M1X2, "b_ext");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(D.dim() == 2, "output must be a matrix");
  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B_ext.dim() == 2, "b_ext must be a matrix");

  auto const m2 = static_cast<int>(A.sizes()[0]);
  auto const m = static_cast<int>(D.sizes()[0]);
  auto const n = static_cast<int>(B_ext.sizes()[0]);
  auto const k = static_cast<int>(k_base);
  auto const tactic_int = static_cast<int>(tactic);
  auto const a_k = static_cast<int>(A.sizes()[1] * 2);
  auto const physical_k_ext = static_cast<int>(B_ext.sizes()[1] * 2);

  TORCH_CHECK(tactic_int >= 0 && tactic_int <= 31,
              "mext fold encoded tactic must be in [0, 31], got ", tactic);
  if (tactic_int >= 16) {
    TORCH_CHECK(m2 % 2 == 0,
                "mext row-pair swap fold requires A rows divisible by 2, got ",
                m2);
  } else {
    TORCH_CHECK(m2 % 16 == 0,
                "mext tile8 fold requires A rows divisible by 16, got ", m2);
  }
  TORCH_CHECK(m * 2 == m2,
              "output rows must be half of A rows for mext fold, got output M=",
              m, " and A rows=", m2);
  TORCH_CHECK(D.sizes()[1] == n,
              "output N must match b_ext rows, got output N=", D.sizes()[1],
              " and b_ext rows=", n);
  TORCH_CHECK(k > 0, "k_base must be positive, got ", k);
  TORCH_CHECK(a_k == k,
              "a and logical k_base mismatch: a has unpacked K=", a_k,
              ", k_base=", k);
  TORCH_CHECK(physical_k_ext >= k,
              "b_ext unpacked K=", physical_k_ext,
              " must cover logical k_base=", k);

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k_base to be divisible by ",
              alignment, ", but got ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got b_ext shape: (", B_ext.sizes()[0], "x",
              B_ext.sizes()[1], ").");

  int rounded_m2 = round_up(m2, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(A_sf.sizes()[1] == B_sf.sizes()[1],
              "scale_a and scale_b shapes cannot be multiplied (",
              A_sf.sizes()[0], "x", A_sf.sizes()[1], " and ",
              B_sf.sizes()[0], "x", B_sf.sizes()[1], ")");
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m2 && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to a shape (", rounded_m2,
              "x", rounded_k, "), but got a shape (", A_sf.sizes()[0], "x",
              A_sf.sizes()[1], ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to a shape (", rounded_n,
              "x", rounded_k, "), but got a shape (", B_sf.sizes()[0], "x",
              B_sf.sizes()[1], ")");

  auto out_dtype = D.dtype();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  if (out_dtype == at::ScalarType::BFloat16) {
    dispatch_bf16_mext_fold_tile8_vec_only_tactic(
        D, A, B_ext, A_sf, B_sf, alpha, m2, n, k, physical_k_ext,
        tactic_int, stream);
    return;
  }
  if (out_dtype == at::ScalarType::Half) {
    dispatch_f16_mext_fold_tile8_vec_only_tactic(
        D, A, B_ext, A_sf, B_sf, alpha, m2, n, k, physical_k_ext,
        tactic_int, stream);
    return;
  }

  TORCH_CHECK(false,
              "Unsupported output data type of nvfp4 mext tile8 clean "
              "vec-store tactic GEMM sm120 (",
              out_dtype, ")");
#else
  TORCH_CHECK(false,
              "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
              "a CUTLASS 3.8 source directory to enable support.");
#endif
}
