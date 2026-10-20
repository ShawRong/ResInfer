# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
"""SM120 register-fold GEMM with direct compact output stores.
Derived from CUTLASS blackwell_geforce/dense_blockscaled_gemm_persistent_cooperative.py
(NVIDIA, BSD-3-Clause); residue folding is a ResInfer extension."""

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


def _sfb_padN(ts):
    """Compute the padded B-scale shape."""
    return (ts[0], max(ts[1], 128), ts[2])


def _sfa_padM(ts):
    """Compute the padded A-scale shape."""
    return (max(ts[0], 128), ts[1], ts[2])


class Sm120BlockScaledGemmKernel:
    def __init__(
        self,
        acc_dtype,
        sf_vec_size,
        tile_shape_mnk,
        k_loop: bool = False,
    ):
        self.k_loop = k_loop
        self.acc_dtype = acc_dtype
        self.sf_vec_size = sf_vec_size
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.tiled_mma = None

        self.occupancy = 1
        self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1
        ) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        if (
            self.a_dtype != cutlass.Float4E2M1FN
            or self.b_dtype != cutlass.Float4E2M1FN
            or self.sf_dtype != cutlass.Float8E4M3FN
            or self.sf_vec_size != 16
        ):
            raise ValueError("SM120 Fold requires NVFP4 operands with block size 16")
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype, self.acc_dtype, self.sf_dtype
        )
        atom_shape = (4, 2, 1)
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, False
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

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

        self.ab_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
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
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
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
        """Build TMA descriptors and launch the Fold kernel."""

        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = d.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        self._setup_attributes()

        self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)

        self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
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
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
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
        """Pipeline TMA loads and warp MMA, then fold into compact output."""

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

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

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

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

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

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

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )

        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )

        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(_sfa_padM(self.tile_shape_mnk), (None, 0, None)),
            (None, None, None),
        )

        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(_sfb_padN(self.tile_shape_mnk), (0, None, None)),
            (None, None, None),
        )

        thr_mma = tiled_mma.get_slice(tidx)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

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

        cC = cute.make_identity_tensor(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
        )
        tCcC = thr_mma.partition_C(cC)
        accumulators = cute.make_rmem_tensor(tCcC.shape, self.acc_dtype)

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        k_tile_cnt = cute.size(gB_nkl, mode=[3])
        k_base_tile_cnt = cute.size(gA_mkl, mode=[3])

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    transpose=self.a_layout.is_m_major_a(),
                    num_matrices=4,
                ),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    transpose=self.b_layout.is_n_major_b(),
                    num_matrices=4,
                ),
                self.b_dtype,
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
                    _sub_k = tile_coord_mnl[1] % _sfb_tiles_per_block
                    tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                    tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                    tCrSFB_tile = tCrSFB_full[None, (None, _sub_k), None]
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
                accumulators.fill(0.0)

                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )

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

                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            [
                                tCrA[None, None, k_block_idx],
                                tCrSFA_tile[None, None, k_block_idx],
                            ],
                            [
                                tCrB[None, None, k_block_idx],
                                tCrSFB_tile[None, None, k_block_idx],
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

                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        [
                            tCrA[None, None, k_block_idx],
                            tCrSFA_tile[None, None, k_block_idx],
                        ],
                        [
                            tCrB[None, None, k_block_idx],
                            tCrSFB_tile[None, None, k_block_idx],
                        ],
                        accumulators,
                    )

                _m_lim = mD_nml.shape[1]
                _tok_lim = mD_nml.shape[0]
                _m_base = tile_coord_mnl[0] * self.tile_shape_mnk[0]
                _n_base = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                _l_crd = tile_coord_mnl[2]

                _alpha = mAlpha[0]
                _nacc = cute.size(accumulators)
                if cutlass.const_expr(self.k_loop):
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

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

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
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

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

                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

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
        """Choose the GEMM pipeline stage count."""

        epi_bytes = 0

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        sf_bytes_per_stage = (
            cute.size(cute.filter_zeros(sfa_smem_layout).shape) * sf_dtype.width // 8
            + cute.size(cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
        )
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
        """Create operand and scale-factor shared-memory layouts."""
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
        """Compute grid shape from the full (unfolded) problem shape (M, N, L)."""

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
        """Return a TMA atom and its tensor view for an operand."""
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
