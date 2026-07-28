# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file is ported from TensorRT-LLM's dense_blockscaled_gemm_persistent.py
# with modifications for FlashInfer integration.
# Original: https://github.com/NVIDIA/TensorRT-LLM

from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

# ResInfer fork shim: cutlass-dsl 4.6 removed the deprecated make_fragment*
# aliases this kernel still uses (identical signatures, verified in 4.5.2).
if not hasattr(cute, "make_fragment"):
    cute.make_fragment = cute.make_rmem_tensor
if not hasattr(cute, "make_fragment_like") and hasattr(cute, "make_rmem_tensor_like"):
    cute.make_fragment_like = cute.make_rmem_tensor_like
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass._mlir.dialects import nvvm as _nvvm_dialect
from cutlass.pipeline import PipelineTmaUmma, PipelineUmmaAsync


class Sm100BlockScaledPersistentDenseGemmKernel:
    """Implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and Blackwell GPU architectural features, including persistent tile scheduling and warp specialization.

    Notes:
        - In the current version, tensors A and B must have the same data type.
          For example, using Float8E4M3FN for A and Float8E5M2 for B is not supported.
        - Supported combinations of A/B data types, scale factor (SF) data types, and SF vector size:
            * MXF8:   A/B: Float8E5M2/Float8E4M3FN, SF: Float8E8M0FNU, sf_vec_size: 32
            * MXF4:   A/B: Float4E2M1FN,            SF: Float8E8M0FNU, sf_vec_size: 32
            * NVF4:   A/B: Float4E2M1FN,            SF: Float8E8M0FNU/Float8E4M3FN, sf_vec_size: 16
        - Supported accumulator data types:
            * Float32
        - Supported C data types:
            * Float32
            * Float16/BFloat16
            * Float8E4M3FN/Float8E5M2
        - Constraints:
            * MMA tiler M must be 128 or 256 (use_2cta_instrs).
            * MMA tiler N must be 64/128/192/256
            * Cluster shape M must be a multiple of 2 if MMA tiler M is 256.
            * Cluster shape M/N must be positive and a power of 2, with total cluster size <= 16.
            * Cluster shape M/N must be <= 4 for scale factor multicasts due to limited scale factor size.

    Example:
        >>> gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 128),
        ...     cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
        fold_pairs_tma: bool = False,
        fold_on_m: bool = False,
        acc_stage_override: int | None = None,
        ab_stage_override: int | None = None,
        mext_direct_store: bool = False,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        Args:
            sf_vec_size (int): Scalefactor vector size.
            mma_tiler_mn (Tuple[int, int]): Shape of the Matrix Multiply-Accumulate (MMA) tile (M, N).
            cluster_shape_mn (Tuple[int, int]): Cluster dimensions (M, N) for parallel processing.
        """

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        # ResInfer MExt: TMA-store fold mode (see cute_fold/fold_epilogue.py).
        # Accumulator/SF side keeps stock m2-wide geometry; only the C/TMA
        # side switches to half-width tiles over compact row-major D.
        self.fold_pairs_tma = fold_pairs_tma
        # ResInfer MExt "non-swap" orientation: tokens on M (128 datapath lanes),
        # weight on N. The (base,res) pair is then adjacent M-lanes, so the fold
        # sums along the M value-mode of the m-major (.16x256b) TMEM-load atom and
        # the output D[m_tok, n_w] is written WITHOUT the N-ext transpose. Gated:
        # every fold_on_m branch is inert unless this is set (N-ext path untouched).
        self.fold_on_m = fold_on_m
        
        
        
        
        self.acc_stage_override = acc_stage_override
        self.ab_stage_override = ab_stage_override
        # (2,) tensor; the fold applies alpha_b*even + alpha_r*odd (one extra
        # f32 FMA). Default False keeps the single-alpha expression untouched
        # (bitwise-identical to the legacy path).
        # A/B knob: force the strict-serial accumulator scheme at N=256 (single
        # 256-col buffer, forward epilogue order, release after full drain)
        # instead of the inherited overlapping-accum (two pseudo-buffers at
        # stride 256-SF_cols, reversed drain on phase 0, early release).
        # Exists to MEASURE what overlapping-accum is worth; never a prod path.
        # A6 (M-ext direct store): fold in registers, then store each thread's
        # folded fragment straight to gmem. Requires the WEIGHT rows pre-permuted
        # by pi (per 32-N block: kernel position p = va+2*ta+8*vc holds logical
        # row f(p) = va+8*ta+2*vc) so the pi-composed gC alias makes every
        # thread's 8 folded values land CONTIGUOUS (16B) in standard-order D.
        # Geometry verified by atom introspection: dst_tv of the .16x256b op at
        # epi (128,32) = ((4,8),(2,2,4)):((2,32),(1,256,8)) for both 1SM/2SM.
        assert not mext_direct_store or fold_on_m, (
            "mext_direct_store requires the M-ext fold path")
        self.mext_direct_store = mext_direct_store
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        self.tma_cluster_shape_mn = cluster_shape_mn
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_prefetch = use_prefetch
        self.enable_pdl = enable_pdl
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Compute mma instruction shapes
        mma_inst_bits_k = 256
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_bits_k // self.a_dtype.width,
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mnk_sfb[0],
            self.mma_inst_shape_mnk_sfb[1],
            self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1
        self.tma_cluster_layout_vmnk = self.cluster_layout_vmnk
        self.tma_cluster_layout_sfb_vmnk = self.cluster_layout_sfb_vmnk

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        self.epi_tile_n = cute.size(self.epi_tile[1])
        # ResInfer MExt: half-width C-side epi tile for the TMA fold path.
        # epi_tile stays untouched for the acc/SF side (num_sf_tmem_cols is
        # budgeted with epi_tile_n -- do not couple it to the C side).
        # N-ext folds the token pair on N -> halve N; M-ext folds it on M -> halve M.
        if self.fold_on_m:
            self.epi_tile_c = (
                cute.size(self.epi_tile[0]) // 2,
                self.epi_tile_n,
            )
        else:
            self.epi_tile_c = (
                cute.size(self.epi_tile[0]),
                self.epi_tile_n // 2,
            )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.a_major_mode,
            self.b_dtype,
            self.b_major_mode,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )
        # Exp2 depth knobs. acc>1 requires the non-overlapping accum scheme,
        # so N=256 tiles (heuristic picks 1) reject an override. ab clamps
        # down only -- the heuristic is already smem-greedy-max.
        if self.acc_stage_override is not None:
            if self.mma_tiler[1] == 256 and self.acc_stage_override > 1:
                raise ValueError("acc_stage_override>1 unsupported for N=256 tiles")
            self.num_acc_stage = self.acc_stage_override
        if self.ab_stage_override is not None:
            if self.ab_stage_override > self.num_ab_stage:
                raise ValueError(
                    f"ab_stage_override {self.ab_stage_override} exceeds smem-max "
                    f"{self.num_ab_stage}")
            self.num_ab_stage = self.ab_stage_override

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        if self.fold_pairs_tma and self.fold_on_m:
            # M-ext (A4 coalesced store): compact half-M staging. Register fold
            # collapses the token pair on M -> stage the folded base rows compact
            # [epi_m/2, epi_n] (N-contiguous), then one coalesced TMA lands them
            # on row-major D[m_tok, n_w]. Half-M frees smem -> 2x stages.
            self.num_c_stage *= 2
            _epi_m2 = cute.size(self.epi_tile[0]) // 2
            _epi_n = self.epi_tile_n
            self.c_smem_layout_staged = cute.make_layout(
                (_epi_m2, _epi_n, self.num_c_stage),
                stride=(_epi_n, 1, _epi_m2 * _epi_n),
            )
        elif self.fold_pairs_tma:
            # N-ext: transposed half-width plain staging (n_w contiguous): TMA box
            # lands directly on row-major D[m_tok, n_w]; half-width frees
            # smem -> spend it on 2x stages (fewer TMA-drain waits). Threads
            # write one strided row each; swizzle buys nothing here.
            self.num_c_stage *= 2
            _epi_m = cute.size(self.epi_tile[0])
            _epi_n2 = self.epi_tile_n // 2
            self.c_smem_layout_staged = cute.make_layout(
                (_epi_m, _epi_n2, self.num_c_stage),
                stride=(1, _epi_m, _epi_m * _epi_n2),
            )
        else:
            self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                self.epi_tile,
                self.num_c_stage,
            )

        self.overlapping_accum = self.num_acc_stage == 1
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )
        if (self.num_accumulator_tmem_cols + self.num_sf_tmem_cols) > 512:
            raise ValueError(
                f"TMEM over budget: acc {self.num_accumulator_tmem_cols} + sf "
                f"{self.num_sf_tmem_cols} > 512 cols "
                f"(num_acc_stage={self.num_acc_stage})")

        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols // self.epi_tile_n
        )

        # TODO: [alel] Currently set prefetch dist to num_ab_stage, we may have more options for prefetch dist auto tuning
        self.prefetch_dist = self.num_ab_stage

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        alpha: cute.Tensor,  # Single-element tensor containing alpha value
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        c_alias_tensor: cute.Tensor = None,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        Args:
            a_tensor (cute.Tensor): Input tensor A
            b_tensor (cute.Tensor): Input tensor B
            sfa_tensor (cute.Tensor): Scale factor tensor A
            sfb_tensor (cute.Tensor): Scale factor tensor B
            c_tensor (cute.Tensor): Output tensor C
            max_active_clusters (cutlass.Constexpr): Maximum number of active clusters
            stream (cuda.CUstream): CUDA stream for asynchronous execution
            epilogue_op (cutlass.Constexpr): Optional elementwise lambda function to apply to the output tensor

        Raises:
            TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        if cutlass.const_expr(self.fold_pairs_tma and self.fold_on_m):
            # M-ext non-swap: tokens on M. Pin COL_MAJOR (m-major) so
            # get_tmem_load_op selects the .16x256b atom whose value mode runs
            # along M -> the adjacent-M-lane (base,res) pair is thread-local for
            # the in-register fold, and D[m_tok, n_w] is written with no transpose.
            self.c_layout = utils.LayoutEnum.COL_MAJOR
        elif cutlass.const_expr(self.fold_pairs_tma):
            # N-ext: physical C is the transposed D view (m-major); pin the enum
            # to the n-major fiction so the TMEM-load atom keeps the
            # pair-in-value-mode fragment orientation the fold relies on
            # (this also feeds compute_epilogue_tile_shape upstream).
            self.c_layout = utils.LayoutEnum.ROW_MAJOR
        else:
            self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.tma_cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.tma_cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.tma_cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.tma_cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.tma_cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.tma_cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.tma_cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.tma_cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C.
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile_c if self.fold_pairs_tma else self.epi_tile,
        )
        # Compute grid size. Fold mode: compact C is half the GEMM's logical
        # N, so tiling must come from the m2-congruent alias view. split-K
        # widens the scheduler's L axis (real L stays 1; the kernel maps the
        # extra coordinate to a K range and clamps tensor slices to l=0).
        # A6: grid counting only needs the m2-congruent SHAPE; rebuild a flat
        # view so the pi-nested N mode never enters zipped_divide.
        if cutlass.const_expr(self.mext_direct_store):
            _grid_c = cute.make_tensor(
                c_alias_tensor.iterator,
                layout=cute.make_layout(
                    (
                        cute.size(c_alias_tensor, mode=[0]),
                        cute.size(c_alias_tensor, mode=[1]),
                        cute.size(c_alias_tensor, mode=[2]),
                    )
                ),
            )
        else:
            _grid_c = c_alias_tensor if self.fold_pairs_tma else c_tensor
        self.tile_sched_params, grid = self._compute_grid(
            _grid_c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    (
                        cute.cosize(self.c_smem_layout_staged)
                        if self.fold_pairs_tma
                        else cute.cosize(self.c_smem_layout_staged.outer)
                    ),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.tma_cluster_layout_vmnk,
            self.tma_cluster_layout_sfb_vmnk,
            self.tma_cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            alpha,
            # Always a real tensor: the DSL cannot marshal None kernel args.
            # Non-fold mode passes a harmless stand-in (read only under
            # const_expr(self.fold_pairs_tma)).
            c_alias_tensor if self.fold_pairs_tma else tma_tensor_c,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.enable_pdl,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        pipe_cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
        mC_fold_alias: cute.Tensor,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        # Keep alpha in FP32 for precision: the accumulator is in FP32 and alpha
        # may be a very small scaling factor. Converting to c_dtype (e.g., FP16)
        # before multiplication could cause overflow when acc values are large.
        alpha_value = alpha[0].to(cutlass.Float32)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        # For TMA we want the rank within the TMA cluster, where the two split
        # partners are both rank 0 -- they load the same full tile, differing
        # only in the K window. Identical to the raw rank whenever the two
        _tma_rank = cta_rank_in_cluster % cute.size(cluster_layout_vmnk)
        _tma_rank_sfb = cta_rank_in_cluster % cute.size(cluster_layout_sfb_vmnk)
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            _tma_rank
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            _tma_rank_sfb
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.ptr
        tmem_holding_buf = storage.tmem_holding_buf.ptr

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=pipe_cluster_layout_vmnk,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=pipe_cluster_layout_vmnk,
        )


        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )
        cute.arch.mbarrier_init_fence()

        # so the receiver arms with expect_tx -- AFTER the init fence (unusable
        # before it) and BEFORE the cluster barrier, which is the only free
        # cross-CTA edge and therefore the only thing that can keep a peer's
        # st.async from landing on an unarmed barrier. (Our protocol is
        # ASYMMETRIC -- follower sends all, owner receives all -- so unlike the
        # symmetric CUTLASS example there is no in-CTA edge that orders this.)

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed(aligned=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        if cutlass.const_expr(self.fold_pairs_tma):
            # Plain (non-composed) half-width fold staging: no swizzle.
            sC = storage.sC.get_tensor(c_smem_layout_staged)
        else:
            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        if cutlass.const_expr(self.fold_pairs_tma):
            # t2r/partition geometry from the m2-congruent alias (metadata
            # only, never stored through); TMA targets the compact D tiled
            # by the half-width CTA tile, indexed by CTA-tile coords.
            gC_mnl = cute.local_tile(
                mC_fold_alias,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            # Compact D tiled by the half-width CTA tile. N-ext folds the token
            # pair on N -> halve N; M-ext folds it on M -> halve M.
            if cutlass.const_expr(self.fold_on_m):
                _fold_cta_tile = (
                    self.cta_tile_shape_mnk[0] // 2,
                    self.cta_tile_shape_mnk[1],
                )
            else:
                _fold_cta_tile = (
                    self.cta_tile_shape_mnk[0],
                    self.cta_tile_shape_mnk[1] // 2,
                )
            gC_fold_mnl = cute.local_tile(
                mC_mnl, _fold_cta_tile, (None, None, None),
            )
        else:
            gC_mnl = cute.local_tile(
                mC_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            gC_fold_mnl = None
        k_block_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))
        # k_split_idx * k_win_cnt (k_split_idx = the hijacked L coordinate).
        k_win_cnt = k_block_cnt

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )

        # PDL bookend: griddepcontrol_wait is always emitted; the actual PDL
        # behavior is controlled by use_pdl= in .launch(). The griddepcontrol
        # instructions are effectively no-ops when PDL is not enabled at
        # the driver level.
        griddepcontrol_wait()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                # is always 0. k_lo offsets every K-block index below.
                k_lo = cutlass.Int32(0)
                _m_coord = cur_tile_coord[0] // cute.size(
                    tiled_mma.thr_id.shape)
                _l_coord = cur_tile_coord[2]
                mma_tile_coord_mnl = (_m_coord, cur_tile_coord[1], _l_coord)

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                if cutlass.const_expr(self.use_prefetch):
                    # Prefetch both A and B (default behavior)
                    for pf_k_block in cutlass.range(
                        0, min(self.prefetch_dist, k_win_cnt), unroll=1
                    ):
                        cute.prefetch(
                            tma_atom_a,
                            tAgA_slice[(None, k_lo + pf_k_block)],
                        )
                        cute.prefetch(
                            tma_atom_b,
                            tBgB_slice[(None, k_lo + pf_k_block)],
                        )
                        cute.prefetch(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, k_lo + pf_k_block)],
                        )
                        cute.prefetch(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, k_lo + pf_k_block)],
                        )

                # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_win_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # Tma load loop
                #
                for k_block in cutlass.range(0, k_win_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, k_lo + ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, k_lo + ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, k_lo + ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, k_lo + ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Prefetch: Rolling prefetch for next tiles
                    if cutlass.const_expr(self.use_prefetch):
                        if k_block < k_win_cnt - self.prefetch_dist:
                            future_k_block = (
                                k_lo + ab_producer_state.count + self.prefetch_dist
                            )
                            # Prefetch both A and B (default behavior)
                            cute.prefetch(
                                tma_atom_a,
                                tAgA_slice[(None, future_k_block)],
                            )
                            cute.prefetch(
                                tma_atom_b,
                                tBgB_slice[(None, future_k_block)],
                            )
                            cute.prefetch(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, future_k_block)],
                            )
                            cute.prefetch(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, future_k_block)],
                            )

                    # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt + k_block + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_win_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            # Make accumulator tmem tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                _m_coord = cur_tile_coord[0] // cute.size(
                    tiled_mma.thr_id.shape)
                _l_coord = cur_tile_coord[2]
                mma_tile_coord_mnl = (_m_coord, cur_tile_coord[1], _l_coord)

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_block = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_win_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for cta_tile_shape_n=192 case by two words (ignores first 64 columns of SFB)
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for _k_block in range(k_win_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (
                                None,
                                None,
                                kphase_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kphase_coord = (None, None, kphase_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kphase_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kphase_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kphase_coord],
                                tCrB[kphase_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kphase
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_block = k_block + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_win_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
            )

            if cutlass.const_expr(self.fold_pairs_tma):
                # ResInfer fold: no r2s tiled-copy machinery. Each thread writes
                # its folded fragment into compact smem (N-ext: one strided row;
                # M-ext A4: via the (8,2,nb) m2-alias), then a single coalesced
                # TMA drains the compact epi tile to D[m_tok, n_w]. TMA partition
                # is over the compact D tiled by the half-width epi_tile_c.
                # The (base,res) pair is the innermost stride-1 submode of the
                # TMEM-load value mode. N-ext (.32x32b) nests it 2 deep
                # ((2,X)); M-ext (.16x256b) nests it 3 deep ((2,2,4)) so the
                # pair coord needs an extra None to stay weakly congruent.
                if cutlass.const_expr(self.fold_on_m):
                    
                    
                    
                    _PAIR_EVEN = (((None, 0, None), None), None, None)
                    _PAIR_ODD = (((None, 1, None), None), None, None)
                else:
                    _PAIR_EVEN = (((0, None), None), None, None)
                    _PAIR_ODD = (((1, None), None), None, None)
                tTR_rC_half = cute.make_fragment(
                    tTR_rAcc[_PAIR_EVEN].shape, self.c_dtype
                )
                # TMA partition over compact D -- IDENTICAL for M-ext and N-ext:
                # gC_fold_mnl is the compact D tiled by the half CTA tile, sC is
                # the compact [epi_m/2|epi_n/2] staging. One coalesced TMA drains
                # each folded epi subtile to D. A6 direct store bypasses TMA
                # entirely (and its partition must not trace over the pi alias).
                if cutlass.const_expr(self.mext_direct_store):
                    bSG_sC, bSG_gC_partitioned = None, None
                else:
                    gC_fold_epi = cute.flat_divide(gC_fold_mnl, self.epi_tile_c)
                    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        cute.group_modes(gC_fold_epi, 0, 2),
                    )
                _thr_t2r = tiled_copy_t2r.get_slice(epi_tidx)
                if cutlass.const_expr(self.mext_direct_store):
                    # A6: partition output addresses through the pi-composed
                    # m2-alias (Option-B skeleton). Each thread's folded 8
                    # values land contiguous (16B) because the alias N mode
                    # maps partition coordinate p -> pre-permuted-weight
                    # address f(p). 8-mode result mirrors
                    # epilog_tmem_copy_and_partition's internal gC partition.
                    _gC_epi = cute.flat_divide(
                        tCgC[((None, None), 0, 0, None, None, None)], epi_tile
                    )
                    tTR_gC_part = _thr_t2r.partition_D(_gC_epi)
                    _sC_alias_layout = None
                elif cutlass.const_expr(self.fold_on_m):
                    # M-ext (A4): the register fold collapses the pair on M, so
                    # the folded base rows are written into the COMPACT smem via
                    # the (8,2,nb) m2-alias partition (same aliasing as the gmem
                    # c_alias, but into smem) -- then the TMA above lands them
                    # coalesced.
                    tTR_gC_part = None
                    _epi_m_sz = cute.size(epi_tile[0])
                    _epi_n_sz = cute.size(epi_tile[1])
                    _nb_epi = _epi_m_sz // 16
                    _sC_alias_layout = cute.make_layout(
                        ((8, 2, _nb_epi), _epi_n_sz),
                        stride=((_epi_n_sz, 0, 8 * _epi_n_sz), 1),
                    )
                else:
                    tTR_gC_part = None
                    _sC_alias_layout = None
                tiled_copy_r2s, tRS_rC, tRS_sC = None, None, None
            else:
                tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = (
                    self.epilog_smem_copy_and_partition(
                        tiled_copy_t2r, tTR_rC, epi_tidx, sC
                    )
                )
                tma_atom_c, bSG_sC, bSG_gC_partitioned = (
                    self.epilog_gmem_copy_and_partition(
                        epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                    )
                )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            # scheduler's tile counter is WRONG -- it is not in lockstep with the
            # barrier's phase flips, and mbarrier_wait on a mismatched phase returns
            # IMMEDIATELY (silently no sync at all).
            _red_phase = cutlass.Int32(0)
            # Separate counter: the credit barrier flips on its OWN schedule
            _empty_phase = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                # (runs the real epilogue after folding in the followers'
                # partials), splits >0 = followers (drain partials to the
                # r2g workspace). Data L for gC is always 0 under split.
                k_split_idx = cutlass.Int32(0)
                _m_coord = cur_tile_coord[0] // cute.size(
                    tiled_mma.thr_id.shape)
                _l_coord = cur_tile_coord[2]
                mma_tile_coord_mnl = (_m_coord, cur_tile_coord[1], _l_coord)

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                if cutlass.const_expr(self.mext_direct_store):
                    # A6: slice the pi-alias partition_D output-address tensor
                    # to this mma tile (frag modes kept, mma coords fixed), then
                    # group the trailing modes into a per-subtile axis.
                    tTR_gC = tTR_gC_part[
                        (None, None, None, None, None, *mma_tile_coord_mnl)
                    ]
                    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
                    bSG_gC = None
                elif cutlass.const_expr(self.fold_pairs_tma):
                    # fold gC tiles are CTA-tile sized (compact D, half CTA tile):
                    # slice with CTA-level coords (2-CTA MMA: mma-level coords skip
                    # peer halves). Same for M-ext and N-ext -- both coalesced-TMA.
                    _cta_m_coord = cur_tile_coord[0]
                    bSG_gC = bSG_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            _cta_m_coord,
                            cur_tile_coord[1],
                            _l_coord,
                        )
                    ]
                else:
                    bSG_gC = bSG_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True)
                        if acc_stage_index == 0
                        else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                # A4: M-ext now also drives the TMA store, so both fold paths need
                # bSG_gC grouped into a per-subtile axis. A6 stores direct (no TMA).
                if cutlass.const_expr(not self.mext_direct_store):
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Store accumulator to global memory in sub-tiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                # the fragment-native fp32 workspace, publish a flag). The
                # owner is the LAST L plane so every follower work item has a
                # smaller linear index than every owner item; per-CTA work
                # sequences increase, so followers always run first and the
                # owner spin cannot deadlock the persistent schedule.
                _skl_do_epi = None

                # Run the real epilogue only for the owner (followers drain
                # below). The guard sits OUTSIDE the loop so the trip count
                # stays STATIC -- a dynamic bound turns the whole epilogue
                # into a dynamic scf.for, killing unrolling and static
                # smem/TMA addressing. Python's `or` short-circuits at trace
                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n
                                - 1
                                - subtile_idx
                            )
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    #
                    # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            # Fence for TMEM load
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()
                    #
                    # Convert to C type (fold mode: sum (base,residue) pairs
                    # in f32, alpha in f32, single rounding; write the half
                    # fragment as one strided smem row per thread)
                    #
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    if cutlass.const_expr(self.fold_pairs_tma):
                        _fold_sum = (
                            tTR_rAcc[_PAIR_EVEN].load().to(cutlass.Float32)
                            + tTR_rAcc[_PAIR_ODD].load().to(cutlass.Float32)
                        )
                        folded_half = epilogue_op(
                            (alpha_value * _fold_sum).to(self.c_dtype)
                        )
                        tTR_rC_half.store(folded_half)
                        if cutlass.const_expr(self.mext_direct_store):
                            # A6: store the folded fragment STRAIGHT to gmem.
                            # The pi-composed alias makes each thread's 8 values
                            # contiguous; runtime alignment is 16B by
                            # construction (n%32==0, block/ta offsets 16B), but
                            # n is a dynamic sym so autovec cannot PROVE it and
                            # falls back to scalar st.global.b16 (PTX-verified).
                            # Force one 128-bit store via an explicit atom over
                            # the coalesced 8-element views.
                            tTR_gC_subtile = tTR_gC[
                                (None, None, None, real_subtile_idx)
                            ]
                            _dst = tTR_gC_subtile[_PAIR_EVEN]
                            # Per thread the fragment is one-or-more rows of 8
                            # contiguous values each (pi construction; PTX-
                            # verified literal +0..+14B offsets). Emit one
                            # 128-bit store per row: slice the trailing
                            # (T2R_M, T2R_N) modes, flatten each 8-value run.
                            _stg_atom = cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                self.c_dtype,
                                num_bits_per_copy=128,
                            )
                            # Post-slice the view is FLAT rank-5:
                            # (va=2, vc=4, 1, T2R_M, T2R_N) with strides
                            # (1, 2, 0, dyn, dyn) -- modes 0+1 form the 8
                            # contiguous values; iterate the trailing rows.
                            for _im in cutlass.range_constexpr(
                                cute.size(_dst, mode=[3])
                            ):
                                for _in in cutlass.range_constexpr(
                                    cute.size(_dst, mode=[4])
                                ):
                                    _d1 = _dst[(None, None, 0, _im, _in)]
                                    _s1 = tTR_rC_half[
                                        (None, None, 0, _im, _in)
                                    ]
                                    cute.copy(
                                        _stg_atom,
                                        cute.make_tensor(
                                            _s1.iterator,
                                            cute.make_layout(cute.size(_s1)),
                                        ),
                                        cute.make_tensor(
                                            _d1.iterator,
                                            cute.make_layout(cute.size(_d1)),
                                        ),
                                    )
                        elif cutlass.const_expr(self.fold_on_m):
                            # M-ext A4: write the folded base rows into the COMPACT
                            # smem buffer via the (8,2,nb) m2-alias partition (base
                            # sub-mode -> real compact row; res sub-mode is stride-0
                            # and dropped by _PAIR_EVEN). The coalesced TMA below
                            # then lands the whole compact epi tile on D.
                            sC_stage = sC[(None, None, c_buffer)]
                            sC_alias = cute.make_tensor(
                                sC_stage.iterator, _sC_alias_layout
                            )
                            sC_alias_epi = cute.flat_divide(sC_alias, epi_tile)
                            tTR_sC = _thr_t2r.partition_D(sC_alias_epi)
                            tTR_sC_sub = tTR_sC[(None, None, None, 0, 0)]
                            cute.autovec_copy(
                                tTR_rC_half, tTR_sC_sub[_PAIR_EVEN]
                            )
                        else:
                            sC_row = sC[(epi_tidx, None, c_buffer)]
                            for _i in range(cute.size(sC_row.shape)):
                                sC_row[_i] = tTR_rC_half[_i]
                    else:
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        # Multiply alpha in FP32 before converting to c_dtype to
                        # avoid overflow when c_dtype is FP16 and acc values are large.
                        acc_vec = epilogue_op((alpha_value * acc_vec).to(self.c_dtype))
                        tRS_rC.store(acc_vec)
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                    # A4: both M-ext and N-ext stage the folded epi tile into
                    # compact smem, then a single coalesced TMA drains it to D.
                    # Fence + barrier so the smem store is visible to the TMA.
                    # A6 wrote gmem directly above: no smem, no fence, no TMA.
                    if cutlass.const_expr(not self.mext_direct_store):
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        epilog_threads = 32 * len(self.epilog_warp_id)
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                        #
                        # TMA store C to global memory
                        #
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, real_subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                # OWNER's smem (peer CTA of this cluster), arrive on the owner's
                # mbarrier. No gmem, no gpu-scope fence, no spin -- same-cluster
                # CTAs are co-resident by construction.

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Dealloc the tensor memory buffer
            #
            # SYMMETRIC exit rendezvous. A CGA has NO exit barrier: CTAs leave
            # independently, and a remote access into a departed peer is a hard
            # fault -- "CGA Exception: CTA Not Present", straight from the
            # hardware. Our credit is a remote write into the follower's smem,
            # so the follower must not leave while the owner may still credit it.
            #
            # No roles, no phases, no dynamic branch: each CTA arrives on its
            # PEER's barrier and waits on its OWN. Neither leaves until both are
            # done. Role-asymmetric versions of this block silently became
            # no-ops three separate times.

            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        Args:
            sSF (cute.Tensor): The scale factor tensor in smem
            tSF (cute.Tensor): The scale factor tensor in tmem

        Returns:
            A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
                - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
                - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
                - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        Args:
            tidx (cutlass.Int32): The thread index in epilogue warp groups
            tAcc (cute.Tensor): The accumulator tensor to be copied and partitioned
            gC_mnl (cute.Tensor): The global tensor C
            epi_tile (cute.Tile): The epilogue tiler
            use_2cta_instrs (bool): Whether use_2cta_instrs is enabled

        Returns:
            A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
                - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
                - tTR_tAcc: The partitioned accumulator tensor
                - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_fragment(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        Args:
            tiled_copy_t2r (cute.TiledCopy): The tiled copy operation for tmem to register copy(t2r)
            tTR_rC (cute.Tensor): The partitioned accumulator tensor
            tidx (cutlass.Int32): The thread index in epilogue warp groups
            sC (cute.Tensor): The shared memory tensor to be copied and partitioned
            sepi (cute.Tensor):

        Returns:
            A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
                - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
                - tRS_rC: The partitioned tensor C (register source)
                - tRS_sC: The partitioned tensor C (smem destination)
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        Args:
            tidx (cutlass.Int32): The thread index in epilogue warp groups
            atom (Union[cute.CopyAtom, cute.TiledCopy]): The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
            gC_mnl (cute.Tensor): The global tensor C
            epi_tile (cute.Tile): The epilogue tiler
            sC (cute.Tensor): The shared memory tensor to be copied and partitioned

        Returns:
            A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        a_major_mode: tcgen05.OperandMajorMode,
        b_dtype: Type[cutlass.Numeric],
        b_major_mode: tcgen05.OperandMajorMode,
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the optimal number of pipeline stages for operands.

        This method uses heuristics to determine the number of stages for the
        accumulator (ACC), A/B operands, and C operand based on the kernel
        configuration and available shared memory.

        Args:
            tiled_mma (cute.TiledMma): The tiled MMA object.
            mma_tiler_mnk (Tuple[int, int, int]): The shape (M, N, K) of the
                MMA tiler.
            a_dtype (Type[cutlass.Numeric]): Data type of operand A.
            a_major_mode (tcgen05.OperandMajorMode): Layout of operand A.
            b_dtype (Type[cutlass.Numeric]): Data type of operand B.
            b_major_mode (tcgen05.OperandMajorMode): Layout of operand B.
            epi_tile (cute.Tile): The epilogue tile shape.
            c_dtype (Type[cutlass.Numeric]): Data type of operand C.
            c_layout (utils.LayoutEnum): Layout of operand C.
            sf_dtype (Type[cutlass.Numeric]): Data type of the scale factors.
            sf_vec_size (int): Vector size of the scale factors.
            smem_capacity (int): Total available shared memory in bytes.
            occupancy (int): Target number of CTAs per SM.

        Returns:
            Tuple[int, int, int]: A tuple containing the number of stages for
                (ACC, A/B, C).
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Computes the grid size for the kernel launch.

        This method uses a persistent tile scheduler to determine the grid
        dimensions based on the output tensor shape and hardware constraints.

        Args:
            c (cute.Tensor): The output tensor C.
            cta_tile_shape_mnk (Tuple[int, int, int]): The shape (M, N, K) of the
                CTA tile.
            cluster_shape_mn (Tuple[int, int]): The shape of a CTA cluster.
            max_active_clusters (cutlass.Constexpr): The maximum number of
                active clusters supported by the hardware.

        Returns:
            A tuple containing the tile scheduler parameters and the computed
            grid shape.
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Checks if the data types and scale factor vector size are a valid combination.

        Args:
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            sf_dtype (Type[cutlass.Numeric]): Data type of the scale factors.
            sf_vec_size (int): Vector size of the scale factors.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.

        Returns:
            bool: True if the combination is valid, False otherwise.
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Checks if the tensor layouts are valid for the given data types.

        Args:
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.
            a_major (str): The major layout of tensor A ('k' or 'm').
            b_major (str): The major layout of tensor B ('k' or 'n').
            c_major (str): The major layout of tensor C ('n' or 'm').

        Returns:
            bool: True if the layouts are valid, False otherwise.
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Checks if the MMA tiler and cluster shape are a valid combination.

        Args:
            mma_tiler_mn (Tuple[int, int]): The (M, N) shape of the MMA tiler.
            cluster_shape_mn (Tuple[int, int]): The (M, N) shape of the CTA cluster.

        Returns:
            bool: True if the combination is valid, False otherwise.
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [64, 128, 192, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        _is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not _is_power_of_2(cluster_shape_mn[0])
            or not _is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Checks if the tensor dimensions are valid for memory alignment.

        Args:
            m (cutlass.Int64): The M dimension of the GEMM problem.
            n (cutlass.Int64): The N dimension of the GEMM problem.
            k (cutlass.Int64): The K dimension of the GEMM problem.
            l (cutlass.Int64): The batch dimension (L) of the GEMM problem.
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.
            a_major (str): The major layout of tensor A ('k' or 'm').
            b_major (str): The major layout of tensor B ('k' or 'n').
            c_major (str): The major layout of tensor C ('n' or 'm').

        Returns:
            bool: True if the tensor alignment is valid, False otherwise.
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @classmethod
    def can_implement(
        cls,
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Checks if the kernel can implement the given GEMM problem.

        This method aggregates checks for data types, layouts, tile shapes,
        and alignment to determine if the configuration is supported.

        Args:
            ab_dtype (Type[cutlass.Numeric]): Data type of operands A and B.
            sf_dtype (Type[cutlass.Numeric]): Data type of the scale factors.
            sf_vec_size (int): Vector size of the scale factors.
            c_dtype (Type[cutlass.Numeric]): Data type of the output tensor C.
            mma_tiler_mn (Tuple[int, int]): The (M, N) shape of the MMA tiler.
            cluster_shape_mn (Tuple[int, int]): The (M, N) shape of the CTA
                cluster.
            m (cutlass.Int64): The M dimension of the GEMM problem.
            n (cutlass.Int64): The N dimension of the GEMM problem.
            k (cutlass.Int64): The K dimension of the GEMM problem.
            l (cutlass.Int64): The batch dimension (L) of the GEMM problem.
            a_major (str): The major layout of tensor A ('k' or 'm').
            b_major (str): The major layout of tensor B ('k' or 'n').
            c_major (str): The major layout of tensor C ('n' or 'm').

        Returns:
            bool: True if the configuration is supported, False otherwise.
        """
        can_implement = True
        # Skip unsupported types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not cls.is_valid_layouts(ab_dtype, c_dtype, a_major, b_major, c_major):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(mma_tiler_mn, cluster_shape_mn):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not cls.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement

    # fully dynamic shape
    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Executes the wrapped GEMM kernel with dynamically shaped tensors.

        Uses TVM-FFI for efficient tensor passing: A, B, C, and alpha are passed
        as cute.Tensor directly (torch tensors at runtime via TVM-FFI's C-level
        dlpack, with negligible conversion cost). Scale factor tensors remain as
        pointers because their 6D BlockScaledBasicChunk layout cannot be expressed
        as a torch tensor.

        Args:
            mA (cute.Tensor): Input tensor A.
                FP4 path: shape (m, k_packed), dtype Uint8 (2xFP4 packed per byte).
                MXFP8 path: shape (m, k), dtype Float8.
            mB (cute.Tensor): Input tensor B.
                FP4 path: shape (n, k_packed), dtype Uint8.
                MXFP8 path: shape (n, k), dtype Float8.
            mC (cute.Tensor): Output tensor C, shape (m, n).
            sf_m (cutlass.Int64): Scale factor M dim (ceil(m/128)).
            sf_n (cutlass.Int64): Scale factor N dim (ceil(n/128)).
            sf_k (cutlass.Int64): Scale factor K dim (ceil(k/sf_vec_size/4)).
            l (cutlass.Constexpr): Batch dimension (L).
            a_sf_ptr (cute.Pointer): Pointer to scale factor tensor for A.
            b_sf_ptr (cute.Pointer): Pointer to scale factor tensor for B.
            alpha_tensor (cute.Tensor): Alpha scaling factor, shape (1,), float32.
            max_active_clusters (cutlass.Constexpr): Max active clusters.
            current_stream: CUDA stream (managed by TVM-FFI fake stream).
            swap_ab (cutlass.Constexpr): Whether A/B are swapped (controls C layout).
            epilogue_op (cutlass.Constexpr): Elementwise epilogue function.
        """
        # A, B, C are passed as cute.Tensor via TVM-FFI.
        # Support two input encodings:
        # 1) FP4 packed as uint8 (recast to Float4E2M1FN, k = k_packed * 2)
        # 2) MXFP8 as float8 (k = k_raw, no recast)
        m = cute.size(mA, mode=[0])
        k_raw = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])

        if cutlass.const_expr(
            mA.element_type == cutlass.Uint8 and mB.element_type == cutlass.Uint8
        ):
            # FP4 packed path: 2 FP4 values per uint8 byte
            k = k_raw * 2
            a_ptr = cute.recast_ptr(mA.iterator, dtype=cutlass.Float4E2M1FN)
            b_ptr = cute.recast_ptr(mB.iterator, dtype=cutlass.Float4E2M1FN)
        elif cutlass.const_expr(mA.element_type != mB.element_type):
            raise TypeError(
                "Unsupported mixed input dtypes for block-scaled GEMM: "
                "mA and mB must have matching element_type "
                "(both Uint8 for FP4 path, or both FP8 for MXFP8 path)."
            )
        else:
            # MXFP8 path: input tensors are already FP8.
            k = k_raw
            a_ptr = mA.iterator
            b_ptr = mB.iterator

        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)),
        )
        # Reshape C to (m, n, l) -- swap_ab is constexpr, determines layout at compile time
        c_alias_tensor = None
        if cutlass.const_expr(self.fold_pairs_tma and self.fold_on_m):
            # M-ext non-swap: tokens on M (m = 2*m_tok), weight on N (n = n_w).
            # mC is row-major D[m_tok, n_w] -- NATURAL, no transpose. Kernel-facing
            # C is the half-height view over m_tok (N-contiguous). The alias puts
            # the (base,res) pair on M (mode 0) with stride 0 so both members
            # alias one output row (partition-geometry metadata; never deref'd).
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_layout(
                    (m // 2, n, l), stride=(n, 1, (m // 2) * n)
                ),
            )
            # tile8 activation layout: physical rows [base0..7, res0..7, base8..,
            # res8.., ...]. .16x256b co-locates accumulator M-rows {r, r+8} in one
            # thread, and tile8 makes those = base_j/res_j. Physical M-index
            # m -> output row (m//16)*8 + m%8. Decompose M as (local=8, baseres=2,
            # block=m//16), col-major so m = local + 8*baseres + 16*block; output
            # strides: local->n, baseres->0 (alias), block->8n.
            _nb = m // 16
            if cutlass.const_expr(self.mext_direct_store):
                # A6: compose pi into the N mode. The .16x256b atom hands thread
                # ta the folded columns {2ta+va+8vc} of each 32-N block; with the
                # weight rows pre-permuted by f(p)=va+8ta+2vc, mapping partition
                # coordinate p -> address f(p) (layout (2,4,4):(1,8,2) per block)
                # makes every thread's 8 values contiguous in standard-order D.
                c_alias_tensor = cute.make_tensor(
                    mC.iterator,
                    layout=cute.make_layout(
                        ((8, 2, _nb), ((2, 4, 4), n // 32), l),
                        stride=((n, 0, 8 * n), ((1, 8, 2), 32), (m // 2) * n),
                    ),
                )
            else:
                c_alias_tensor = cute.make_tensor(
                    mC.iterator,
                    layout=cute.make_layout(
                        ((8, 2, _nb), n, l),
                        stride=((n, 0, 8 * n), 1, (m // 2) * n),
                    ),
                )
        elif cutlass.const_expr(self.fold_pairs_tma):
            # N-ext: mC is row-major D[m_tok, n_w]; kernel-facing C = transposed
            # affine view (n_w contiguous) so the TMA box lands on D
            # directly. The alias keeps the m2-contiguous stride fiction
            # (partition-geometry metadata only, never dereferenced).
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_layout(
                    (m, n // 2, l), stride=(1, m, m * (n // 2))
                ),
            )
            c_alias_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_layout(
                    (m, (2, n // 2), l),
                    stride=(n // 2, (0, 1), m * (n // 2)),
                ),
            )
        elif cutlass.const_expr(swap_ab):
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_ordered_layout((m, n, l), order=(0, 1, 2)),
            )
        else:
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
            )
        # Scale factor tensors: 6D BlockScaledBasicChunk layout from pointers
        # (32, 4, sf_m, 4, sf_k, l) with order (2, 1, 4, 0, 3, 5)
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_m, 4, sf_k, l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_n, 4, sf_k, l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )

        if cutlass.const_expr(self.fold_pairs_tma):
            self(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha_tensor,
                max_active_clusters,
                current_stream,
                epilogue_op,
                c_alias_tensor,
            )
        else:
            self(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha_tensor,
                max_active_clusters,
                current_stream,
                epilogue_op,
            )
