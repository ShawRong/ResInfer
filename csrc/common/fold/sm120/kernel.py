# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'CuTeDSL NVFP4 col-pair fold GEMM for SM120 (GeForce Blackwell / RTX 5090).\n\nDerived from the CUTLASS CuTeDSL example\n`blackwell_geforce/.../dense_blockscaled_gemm_persistent_cooperative.py`\n(BSD-3-Clause, NVIDIA CORPORATION).\n'

from typing import Tuple, Type, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.blackwell_helpers as sm120_utils

from ._dsl_helpers import (
    FP4_SHIFT_BITS,
    make_ldmatrix_atom,
    make_sm120_blockscaled_mma_op,
)


def _sfb_padN(ts):
    """SFB is always laid out at N>=128 (C++ sm120_blockscaled_mma_builder.inl:209)."""
    return (ts[0], max(ts[1], 128), ts[2])


def _sfa_padM(ts):
    """SFA twin of _sfb_padN: the SF block is always 128 rows."""
    return (max(ts[0], 128), ts[1], ts[2])


class Sm120BlockScaledGemmKernel:
    def __init__(
        self,
        acc_dtype,
        sf_vec_size,
        tile_shape_mnk,
        epi_tile,
        k_loop: bool = False,
    ):
        
        # B is the K-concat activation [m_tok, 2K]; A (the weight) keeps K and
        # is read twice by wrapping the producer's k coordinate at the
        # midpoint, so the MMA's own K-reduction performs the base+residue
        # fold and the epilogue is a plain predicated store (one output row
        # per token, no pair-sum). Weight bytes in memory: 1x, unchanged.
        self.k_loop = k_loop
        self.acc_dtype = acc_dtype
        self.sf_vec_size = sf_vec_size
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.epi_tile = tuple(epi_tile)
        self.tiled_mma = None

        self.occupancy = 1
        self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        mma_op, use_mxf8f6f4 = make_sm120_blockscaled_mma_op(
            self.a_dtype,
            self.b_dtype,
            self.acc_dtype,
            self.sf_dtype,
            self.sf_vec_size,
        )
        self.mixed_mode = self.a_dtype != self.b_dtype
        # a_fp4_in_mixed / b_fp4_in_mixed: this side carries an FP4 operand in the
        # mixed FP4 x FP8 mode, so SMEM/TMA see Int8 storage and the mma.sync
        # consumer needs the LDSM b4x16_p64 unpack + register `<< FP4_SHIFT_BITS`.
        a_fp4_in_mixed = self.mixed_mode and self.a_dtype.width < 8
        b_fp4_in_mixed = self.mixed_mode and self.b_dtype.width < 8
        self.smem_alloc_a_dtype = cutlass.Int8 if a_fp4_in_mixed else self.a_dtype
        self.smem_alloc_b_dtype = cutlass.Int8 if b_fp4_in_mixed else self.b_dtype
        # `internal_type` for `_make_tma_atoms_and_tensors` — None when the dtype
        # already matches (TMA sees the native dtype), Int8 when we recast for FP4.
        self.tma_internal_a_dtype = cutlass.Int8 if a_fp4_in_mixed else None
        self.tma_internal_b_dtype = cutlass.Int8 if b_fp4_in_mixed else None
        atom_shape = (4, 2, 1)
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, use_mxf8f6f4
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        # Compute the smem size of SFA/SFB
        sfa_smem_layout_per_stage = blockscaled_utils.sm120_make_smem_layout_sfa(
            self.tiled_mma,
            _sfa_padM(self.tile_shape_mnk),
            self.sf_vec_size,
            1,
        )

        sfb_smem_layout_per_stage = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma,
            _sfb_padN(self.tile_shape_mnk),
            self.sf_vec_size,
            1,
        )

        # Compute stage before compute smem layout. No epilogue smem at all:
        # the compact fold store is direct-STG from registers (C++ Family B
        # style), so ALL smem goes to the mainloop A/B/SF stages.
        self.ab_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.smem_alloc_a_dtype,
            self.smem_alloc_b_dtype,
            self.sf_dtype,
            sfa_smem_layout_per_stage,
            sfb_smem_layout_per_stage,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.smem_alloc_a_dtype,
            self.a_layout,
            self.smem_alloc_b_dtype,
            self.b_layout,
            self.ab_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        d: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = d.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)
        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
            internal_type=self.tma_internal_a_dtype,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
            internal_type=self.tma_internal_b_dtype,
        )

        tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            (max(self.tile_shape_mnk[0], 128), self.tile_shape_mnk[2]),
            1,
            internal_type=cutlass.Int16,
        )

        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            (max(self.tile_shape_mnk[1], 128), self.tile_shape_mnk[2]),
            1,
            internal_type=cutlass.Int16,
        )

        # Grid from the full (unfolded) problem shape, recovered from D:
        # M = d.shape[1] (weight dim), N = 2 * d.shape[0] (extended token dim).
        # k_loop: tokens are NOT row-doubled -- N comes from B's (possibly
        # padded) row count so every activation row is covered; the store is
        # predicated on D's real extent.
        _grid_n = b.shape[0] if self.k_loop else 2 * d.shape[0]
        tile_sched_params, grid = self._compute_grid(
            (d.shape[1], _grid_n, d.shape[2]),
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            d,
            alpha,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mD_nml: cute.Tensor,
        mAlpha: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        # bidx, bidy, bidz = cute.arch.block_idx()
        # bdimx, bdimy, bdimz = cute.arch.grid_dim()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        #  Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # Unified real-N view of the 128-padded sSFB for the MMA/copy path (TMA
        # keeps the 128 view). Same iterator; first real-N positions via stock strides.
        _realN = self.tile_shape_mnk[1]
        if cutlass.const_expr(_realN < 128):
            _nsub = _realN // 32
            _mma_nsf = self.tiled_mma.shape_mnk[2] // self.sf_vec_size
            _kout = self.tile_shape_mnk[2] // self.sf_vec_size // 4
            _nstage = cute.size(sSFB, mode=[2])
            _stage_stride = cute.cosize(cute.filter_zeros(
                cute.slice_(sfb_smem_layout_staged, (None, None, 0))))
            if cutlass.const_expr(_nsub == 1):
                _mn_shape = ((32, 1), 1)
                _mn_stride = ((16, 0), 512)
            else:
                _mn_shape = ((32, _nsub), 1)
                _mn_stride = ((16, 4), 512)
            _sSFB_real = cute.make_tensor(sSFB.iterator, cute.make_layout(
                (_mn_shape, ((self.sf_vec_size, _mma_nsf), 1, _kout), _nstage),
                stride=(_mn_stride, ((0, 1), _mma_nsf, 512), _stage_stride)))
        else:
            _sSFB_real = sSFB

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # (tM, tK, loopM, loopK, loopL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(_sfa_padM(self.tile_shape_mnk), (None, 0, None)),
            (None, None, None),
        )
        # (tN, tK, loopN, loopK, loopL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(_sfb_padN(self.tile_shape_mnk), (0, None, None)),
            (None, None, None),
        )
        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA_mkl, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_nkl, 0, 2),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #  Make frangments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA = sm120_utils.partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
        tCrSFB_full = sm120_utils.partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)
        tCrSFA = cute.group_modes(tCrSFA, 2, cute.rank(tCrSFA))
        tCrSFB_full = cute.group_modes(tCrSFB_full, 2, cute.rank(tCrSFB_full))
        _sfb_tiles_per_block = max(128, self.tile_shape_mnk[1]) // self.tile_shape_mnk[1]
        _sfa_tiles_per_block = max(128, self.tile_shape_mnk[0]) // self.tile_shape_mnk[0]

        # Physical-coord tensor for the compact fold store (C++ Family B idiom):
        # identity over one CTA tile, partitioned exactly like the accumulators,
        # so flat index e -> in-tile (m, n) of acc[e]. Its shape IS the
        # accumulator fragment shape — no gmem C tensor needed at all.
        cC = cute.make_identity_tensor(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
        )
        tCcC = thr_mma.partition_C(cC)
        accumulators = cute.make_rmem_tensor(tCcC.shape, self.acc_dtype)

        # cluster wait for barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        # Loop trip count from B: identical to A's for the fold (same K), and
        # 2x A's for k_loop (B carries the K-concat activation at 2K while A,
        # the weight, wraps). The consumer only indexes staged smem buffers,
        # so it needs nothing beyond this count.
        k_tile_cnt = cute.size(gB_nkl, mode=[3])
        k_base_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Create the tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Create the pipeline states for producer and consumer
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # ///////////////////////////////////////////////////////////////////////////////
            # Copy Atom A/B retiling for TMA load A/B
            # ///////////////////////////////////////////////////////////////////////////////
            atom_copy_ldmatrix_A = make_ldmatrix_atom(
                self.a_dtype,
                transpose=self.a_layout.is_m_major_a(),
                num_matrices=4,
                mixed_mode=self.mixed_mode,
            )
            atom_copy_ldmatrix_B = make_ldmatrix_atom(
                self.b_dtype,
                transpose=self.b_layout.is_n_major_b(),
                num_matrices=4,
                mixed_mode=self.mixed_mode,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            atom_copy_ldmatrix_SF = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.sf_dtype,
            )
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                sm120_utils.get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                sm120_utils.get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)
            tCsSFA_copy_view = thr_copy_ldmatrix_SFA.partition_S(sSFA)
            tCrSFA_copy_view = thr_copy_ldmatrix_SFA.retile(tCrSFA)
            tCsSFB_copy_view_full = thr_copy_ldmatrix_SFB.partition_S(sSFB)
            tCrSFB_copy_view_full = thr_copy_ldmatrix_SFB.retile(tCrSFB_full)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                if cutlass.const_expr(self.tile_shape_mnk[1] == 32):
                    # N=32: local_tile partition is degenerate (bad_variant) -> reg-slice
                    _sub_k = tile_coord_mnl[1] % _sfb_tiles_per_block
                    tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                    tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                    tCrSFB_tile = tCrSFB_full
                elif cutlass.const_expr(_sfb_tiles_per_block > 1):
                    _off = tile_coord_mnl[1] % _sfb_tiles_per_block
                    _sSFB_tile = cute.local_tile(
                        sSFB, cute.slice_(self.tile_shape_mnk, (0, None, None)),
                        (_off, 0, None))
                    tCrSFB_tile = sm120_utils.partition_fragment_SFB(
                        _sSFB_tile[None, None, 0], thr_mma, tidx)
                    if cutlass.const_expr(cute.rank(tCrSFB_tile) > 2):
                        tCrSFB_tile = cute.group_modes(tCrSFB_tile, 2, cute.rank(tCrSFB_tile))
                    tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(_sSFB_tile)
                    tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(tCrSFB_tile)
                else:
                    tCrSFB_tile = tCrSFB_full
                    tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                    tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                # SFA sub-128 tile_M: slice the 128-padded SFA block down to the
                # real tile_M rows of this work tile (same port as SFB N>=64).
                if cutlass.const_expr(_sfa_tiles_per_block > 1):
                    _offm = tile_coord_mnl[0] % _sfa_tiles_per_block
                    _sSFA_tile = cute.local_tile(
                        sSFA, cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                        (_offm, 0, None))
                    tCrSFA_tile = sm120_utils.partition_fragment_SFA(
                        _sSFA_tile[None, None, 0], thr_mma, tidx)
                    if cutlass.const_expr(cute.rank(tCrSFA_tile) > 2):
                        tCrSFA_tile = cute.group_modes(tCrSFA_tile, 2, cute.rank(tCrSFA_tile))
                    tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(_sSFA_tile)
                    tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(tCrSFA_tile)
                else:
                    tCrSFA_tile = tCrSFA
                    tCsSFA_tile_copy_view = tCsSFA_copy_view
                    tCrSFA_tile_copy_view = tCrSFA_copy_view
                # Clear the accumulator
                accumulators.fill(0.0)

                # /////////////////////////////////////////////////////////////////////////////
                #  Pipelined MAINLOOP
                # /////////////////////////////////////////////////////////////////////////////

                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                #  Wait for TMA copies to complete
                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsA_p: (MMA, (4, MMA_N / 4), MMA_K)
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsSFA_p = tCsSFA_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                tCsSFB_p = tCsSFB_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)

                cute.copy(
                    smem_tiled_copy_SFA,
                    tCsSFA_p_filtered[None, None, 0],
                    tCrSFA_copy_view_filtered[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_SFB,
                    tCsSFB_p_filtered[None, None, 0],
                    tCrSFB_copy_view_filtered[None, None, 0],
                )

                for k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    # unroll the loop
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFA_p = tCsSFA_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFB_p = tCsSFB_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        # Mixed FP4 x FP8 register-side bit shift before mma.sync
                        # to move the FP4 nibble (in low half of each byte after
                        # b4x16_p64 ldmatrix) into the middle as mxf8f6f4 expects.
                        if cutlass.const_expr(
                            self.mixed_mode and self.a_dtype.width < 8
                        ):
                            a_view = cute.recast_tensor(
                                tCrA[None, None, k_block_idx], cutlass.Int8
                            )
                            for _i in cutlass.range_constexpr(cute.size(a_view)):
                                a_view[_i] = cutlass.Int8(
                                    a_view[_i] << FP4_SHIFT_BITS
                                )
                        if cutlass.const_expr(
                            self.mixed_mode and self.b_dtype.width < 8
                        ):
                            b_view = cute.recast_tensor(
                                tCrB[None, None, k_block_idx], cutlass.Int8
                            )
                            for _i in cutlass.range_constexpr(cute.size(b_view)):
                                b_view[_i] = cutlass.Int8(
                                    b_view[_i] << FP4_SHIFT_BITS
                                )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            [
                                tCrA[None, None, k_block_idx],
                                tCrSFA_tile[None, None, k_block_idx],
                            ],
                            [
                                tCrB[None, None, k_block_idx],
                                (tCrSFB_full[None, (None, _sub_k), None][None, None, k_block_idx] if cutlass.const_expr(self.tile_shape_mnk[1] == 32) else tCrSFB_tile[None, None, k_block_idx]),
                            ],
                            accumulators,
                        )
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )

                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                        tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)
                        cute.copy(
                            smem_tiled_copy_SFA,
                            tCsSFA_p_filtered[None, None, k_block_next],
                            tCrSFA_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered[None, None, k_block_next],
                            tCrSFB_copy_view_filtered[None, None, k_block_next],
                        )

                        # end of for loop
                # end of for loop
                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
                        cute.arch.fence_proxy("async.shared", space="cta")
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                        tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)
                        cute.copy(
                            smem_tiled_copy_SFA,
                            tCsSFA_p_filtered[None, None, k_block_next],
                            tCrSFA_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered[None, None, k_block_next],
                            tCrSFB_copy_view_filtered[None, None, k_block_next],
                        )
                    # Mixed FP4 x FP8 register-side bit shift before mma.sync (hoisted tail).
                    if cutlass.const_expr(
                        self.mixed_mode and self.a_dtype.width < 8
                    ):
                        a_view_h = cute.recast_tensor(
                            tCrA[None, None, k_block_idx], cutlass.Int8
                        )
                        for _i in cutlass.range_constexpr(cute.size(a_view_h)):
                            a_view_h[_i] = cutlass.Int8(
                                a_view_h[_i] << FP4_SHIFT_BITS
                            )
                    if cutlass.const_expr(
                        self.mixed_mode and self.b_dtype.width < 8
                    ):
                        b_view_h = cute.recast_tensor(
                            tCrB[None, None, k_block_idx], cutlass.Int8
                        )
                        for _i in cutlass.range_constexpr(cute.size(b_view_h)):
                            b_view_h[_i] = cutlass.Int8(
                                b_view_h[_i] << FP4_SHIFT_BITS
                            )
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        [
                            tCrA[None, None, k_block_idx],
                            tCrSFA_tile[None, None, k_block_idx],
                        ],
                        [
                            tCrB[None, None, k_block_idx],
                            (tCrSFB_full[None, (None, _sub_k), None][None, None, k_block_idx] if cutlass.const_expr(self.tile_shape_mnk[1] == 32) else tCrSFB_tile[None, None, k_block_idx]),
                        ],
                        accumulators,
                    )

                # /////////////////////////////////////////////////////////////////////////////
                #  EPILOG
                # /////////////////////////////////////////////////////////////////////////////

                # ==== col-pair fold + compact direct-STG store (C++ Family B
                # port). Consecutive accumulator elements (2e, 2e+1) are the
                # adjacent-N pair (base, residue) at the SAME m. Fold in fp32,
                # store D[tok, m] with tok = global_n // 2 - no smem, no TMA.
                _m_lim = mD_nml.shape[1]
                _tok_lim = mD_nml.shape[0]
                _m_base = tile_coord_mnl[0] * self.tile_shape_mnk[0]
                _n_base = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                _l_crd = tile_coord_mnl[2]
                # Dequant scale, applied in the C++ EVT order (alpha*a + alpha*b:
                # ScalarBroadcast(alpha) * AccFetch feeds the fold node) so the
                # output stays bit-identical to the production C++ kernel.
                _alpha = mAlpha[0]
                _nacc = cute.size(accumulators)
                if cutlass.const_expr(self.k_loop):
                    # k_loop: the fold already happened inside the MMA's
                    # K-reduction; every accumulator element IS a finished
                    # output. Plain predicated store, one row per token.
                    for _e in cutlass.range_constexpr(_nacc):
                        _crd = tCcC[_e]
                        _gm = _m_base + _crd[0]
                        _tok = _n_base + _crd[1]
                        if cute.elem_less((_gm, _tok), (_m_lim, _tok_lim)):
                            _v = accumulators[_e] * _alpha
                            mD_nml[(_tok, _gm, _l_crd)] = _v.to(self.c_dtype)
                else:
                    for _e in cutlass.range_constexpr(_nacc // 2):
                        _crd = tCcC[2 * _e]
                        _gm = _m_base + _crd[0]
                        _tok = (_n_base + _crd[1]) // 2
                        if cute.elem_less((_gm, _tok), (_m_lim, _tok_lim)):
                            _v = (
                                accumulators[2 * _e] * _alpha
                                + accumulators[2 * _e + 1] * _alpha
                            )
                            mD_nml[(_tok, _gm, _l_crd)] = _v.to(self.c_dtype)
                # ==== end fold + store ====

                # Advance to the next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                # End of for k_tile loop
            # End of while loop
        # End of MMA warp group
        # Start of DMA warp group
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                _ratio_m = max(1, 128 // self.tile_shape_mnk[0])
                tAgSFA_mkl = tAgSFA[(None, tile_coord_mnl[0] // _ratio_m, None, tile_coord_mnl[2])]
                _ratio = max(1, 128 // self.tile_shape_mnk[1])
                tBgSFB_nkl = tBgSFB[(None, tile_coord_mnl[1] // _ratio, None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                for k_tile in range(0, k_tile_cnt, 1, unroll=1):
                    # /////////////////////////////////////////////////////////////////////////////
                    #  Wait for A/B buffers to be empty before loading into them
                    #  Also sets the transaction barrier for the A/B buffers
                    # /////////////////////////////////////////////////////////////////////////////
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    # /////////////////////////////////////////////////////////////////////////////
                    #  Slice to global/shared memref to current k_tile
                    # /////////////////////////////////////////////////////////////////////////////
                    # k_loop: the weight (A) and its SF wrap at the K midpoint
                    # -- one compare+subtract in the producer warp, off the
                    # critical path. An operand and its SF are tiled by the
                    # same tiler, so one wrapped index cannot desynchronise
                    # them (kext-via-wrapped-kloop.md).
                    _a_k = mainloop_producer_state.count
                    if cutlass.const_expr(self.k_loop):
                        _a_k = cutlass.Int32(
                            _a_k if _a_k < k_base_tile_cnt
                            else _a_k - k_base_tile_cnt)
                    tAgA_k = tAgA_mkl[(None, _a_k)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    tAgSFA_k = tAgSFA_mkl[(None, _a_k)]
                    tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                    tBgSFB_k = tBgSFB_nkl[(None, mainloop_producer_state.count)]
                    tBsSFB_pipe = tBsSFB[(None, mainloop_producer_state.index)]

                    # /////////////////////////////////////////////////////////////////////////////
                    #  TMA load A/B
                    # /////////////////////////////////////////////////////////////////////////////
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_k,
                        tAsSFA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    # Mainloop pipeline's producer commit is a NOP
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # end of while loop

            # Wait A/B buffer empty
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        sf_dtype: type[cutlass.Numeric],
        sfa_smem_layout: cute.Layout,
        sfb_smem_layout: cute.Layout,
        smem_capacity: int,
        occupancy: int,
    ) -> int:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        # Direct-STG compact epilogue: zero epilogue smem.
        epi_bytes = 0

        # A/B shared memory size
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        # SFA/SFB shared memory size
        sf_bytes_per_stage = (
            cute.size(cute.filter_zeros(sfa_smem_layout).shape) * sf_dtype.width // 8
            + cute.size(cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
        )
        # Mbar helpers size
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // (ab_bytes_per_stage + sf_bytes_per_stage)
        return ab_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: cute.Layout,
        b_dtype: type[cutlass.Numeric],
        b_layout: cute.Layout,
        ab_stage: int,
        sf_vec_size: int,
        tiled_mma: cute.TiledMma,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.Layout, cute.Layout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout for matrix A
        :type a_layout: Layout
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout for matrix B
        :type b_layout: Layout
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: leading dimension of the output matrix C
        :type c_layout: Layout
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]

        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        sfa_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfa(
            tiled_mma,
            _sfa_padM(tile_shape_mnk),
            sf_vec_size,
            ab_stage,
        )

        sfb_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
            tiled_mma,
            _sfb_padN(tile_shape_mnk),
            sf_vec_size,
            ab_stage,
        )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
        )

    @staticmethod
    def _compute_grid(
        problem_mnl: tuple,
        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[int, int, int]:
        """Compute grid shape from the full (unfolded) problem shape (M, N, L).

        :param problem_mnl: Full GEMM output shape (M, N, L)
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        num_ctas_mnl = (
            cute.ceil_div(problem_mnl[0], tile_shape_mnk[0]),
            cute.ceil_div(problem_mnl[1], tile_shape_mnk[1]),
            problem_mnl[2],
        )
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
        internal_type: Optional[Type[cutlass.Numeric]] = None,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
            internal_type=internal_type,
        )

        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
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
