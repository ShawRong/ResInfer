# SPDX-License-Identifier: BSD-3-Clause
"""SM120 block-scaled MMA helpers.

Vendored verbatim from the CUTLASS CuTeDSL example tree
(examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/
blockscaled_gemm_dispatch.py, BSD-3-Clause, NVIDIA CORPORATION) so the
production fold kernel does not depend on a CUTLASS source checkout.
Only the three symbols the kernel needs are kept; the argparse validation
helper is intentionally omitted.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.warp.mma import MXF8F6F4_SUPPORTED_PAIRS

# Number of bits to shift the FP4 byte left by before mma.sync.kind::mxf8f6f4.
# ldsm.b4x16_p64 places the FP4 nibble in the LOW half of the 8-bit register
# byte; mma.sync.kind::mxf8f6f4 reads the FP4 from the MIDDLE of the byte. The
# fixed `<< 2` shift moves it into position. See the C++ reference
# implementation `cute::fp4_shift_A/B` in
# cutlass/include/cute/atom/mma_traits_sm120.hpp.
FP4_SHIFT_BITS = 2


def make_ldmatrix_atom(operand_dtype, transpose, num_matrices=4, mixed_mode=False):
    """Build a warp-level ldmatrix copy atom for the SM120 block-scaled GEMM
    A/B SMEM->RMEM path.
    """
    if mixed_mode and operand_dtype.width == 4:
        assert not transpose, "LdMatrix8x16x8bOp does not support transpose"
        return cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x16x8bOp(
                transpose=False,
                num_matrices=num_matrices,
                unpack_bits=4,
            ),
            cutlass.Int8,
        )
    atom_element_type = operand_dtype
    if mixed_mode and operand_dtype.width < 8:
        atom_element_type = cutlass.Int8
    return cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=transpose,
            num_matrices=num_matrices,
        ),
        atom_element_type,
    )


def make_sm120_blockscaled_mma_op(a_dtype, b_dtype, acc_dtype, sf_dtype, sf_vec_size):
    """Dispatch the SM120 block-scaled MMA op for the given operand / scale
    factor combination.

    Returns
    -------
    Tuple[MmaOp, bool]
        (mma_op, use_mxf8f6f4) where use_mxf8f6f4 is True for FP8 / mixed
        paths (atom shape (16,8,32)) and False for FP4 same-dtype paths
        (atom shape (16,8,64)). `use_mxf8f6f4` is consumed by
        `cutlass.utils.blackwell_helpers.get_permutation_mnk`.

    Raises
    ------
    ValueError
        On any unsupported combination, with a diagnostic that names the
        offending combination and the supported alternatives.
    """
    # Same-dtype paths use the FP4 / FP8 same-dtype atoms.
    if a_dtype == b_dtype:
        if a_dtype == cutlass.Float4E2M1FN:
            if sf_vec_size == 16:
                if sf_dtype != cutlass.Float8E4M3FN:
                    raise ValueError(
                        f"Float4E2M1FN + sf_vec_size=16 requires sf_dtype=Float8E4M3FN, "
                        f"got sf_dtype={sf_dtype}"
                    )
                return (
                    cute.nvgpu.warp.MmaMXF4NVF4Op(a_dtype, acc_dtype, sf_dtype),
                    False,
                )
            if sf_vec_size == 32:
                if sf_dtype != cutlass.Float8E8M0FNU:
                    raise ValueError(
                        f"Float4E2M1FN + sf_vec_size=32 requires sf_dtype=Float8E8M0FNU, "
                        f"got sf_dtype={sf_dtype}"
                    )
                return (
                    cute.nvgpu.warp.MmaMXF4Op(a_dtype, acc_dtype, sf_dtype),
                    False,
                )
            raise ValueError(
                f"Float4E2M1FN requires sf_vec_size in (16, 32), got {sf_vec_size}"
            )
        if a_dtype in _FP8_DTYPES:
            if sf_vec_size != 32:
                raise ValueError(
                    f"FP8 ab_dtype ({a_dtype}) requires sf_vec_size=32, "
                    f"got {sf_vec_size}"
                )
            if sf_dtype != cutlass.Float8E8M0FNU:
                raise ValueError(
                    f"FP8 ab_dtype + sf_vec_size=32 requires sf_dtype=Float8E8M0FNU, "
                    f"got sf_dtype={sf_dtype}"
                )
            return (
                cute.nvgpu.warp.MmaMXF8Op(a_dtype, acc_dtype, sf_dtype),
                True,
            )
        raise ValueError(
            f"Unsupported same-dtype ab_dtype={a_dtype} for SM120 block-scaled GEMM. "
            f"Supported same-dtype: Float4E2M1FN, Float8E4M3FN, Float8E5M2."
        )
    # Mixed-precision FP4 x FP8 paths.
    is_a_fp4_b_fp8 = a_dtype == cutlass.Float4E2M1FN and b_dtype in _FP8_DTYPES
    is_a_fp8_b_fp4 = a_dtype in _FP8_DTYPES and b_dtype == cutlass.Float4E2M1FN
    if is_a_fp4_b_fp8 or is_a_fp8_b_fp4:
        if sf_vec_size != 32:
            raise ValueError(
                f"FP4 x FP8 mixed-precision requires sf_vec_size=32, got {sf_vec_size}"
            )
        if sf_dtype != cutlass.Float8E8M0FNU:
            raise ValueError(
                f"FP4 x FP8 mixed-precision requires sf_dtype=Float8E8M0FNU, "
                f"got sf_dtype={sf_dtype}"
            )
        return (
            cute.nvgpu.warp.MmaMXF8F6F4Op(a_dtype, b_dtype, acc_dtype, sf_dtype),
            True,
        )
    # Reject same-width mixed-FP8 (E4M3 + E5M2) and FP6 with named diagnostics.
    if a_dtype in _FP8_DTYPES and b_dtype in _FP8_DTYPES:
        raise ValueError(
            f"same-width mixed-FP8 (a_dtype={a_dtype}, b_dtype={b_dtype}) is not supported. "
            f"Supported FP4 x FP8 mixed pairs: (Float4E2M1FN, Float8E4M3FN), "
            f"(Float4E2M1FN, Float8E5M2), (Float8E4M3FN, Float4E2M1FN), "
            f"(Float8E5M2, Float4E2M1FN). Same-dtype FP8 uses MmaMXF8Op."
        )
    raise ValueError(
        f"Unsupported (a_dtype={a_dtype}, b_dtype={b_dtype}, sf_dtype={sf_dtype}, "
        f"sf_vec_size={sf_vec_size}) for SM120 block-scaled GEMM. FP6 mixed pairs "
        f"are not supported; supported mixed pairs are FP4 x FP8 only."
    )


