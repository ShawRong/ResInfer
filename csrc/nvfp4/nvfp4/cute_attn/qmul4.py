# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""CUDA 13.4 packed NVFP4 block-scale multiply for CuTe DSL kernels.

PTX ISA 9.4 exposes the instruction that Blackwell SASS names ``QMUL4``.
One instruction multiplies four packed E2M1 values by four E4M3 values and
returns four packed E4M3 values.  NVFP4 has one E4M3 scale per 16 values, so
the scale byte is broadcast to all four lanes before the multiply.

The instruction is architecture-specific and is currently documented only
for ``sm_100a`` and ``sm_103a``.  Callers must keep a non-QMUL4 path for other
targets and must compile this helper with CUDA 13.4 / PTX ISA 9.4 or newer.

CUTLASS DSL 4.6.1 still embeds PTX ISA 9.3.  On that compiler this module
installs the temporary SM100 cubin-loader bridge before any kernel is compiled;
the device primitive emits exact ``FMUL.RM``/``FMUL.RP`` markers for it.  A
CUDA 13.4-based DSL takes the native public-PTX branch automatically.
"""

import os
from typing import Optional, Type

import cutlass
import cutlass.cute as cute
import cutlass.cutlass_dsl as cutlass_dsl
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cute.typing import Float32, Int8, Int32


if (
    not cutlass_dsl.target_version(min_version="13.4")
    and (
        os.environ.get("RESATTN_QMUL4_MODE", "auto") != "portable"
        # ``portable`` selects the scalar fallback used by qmul4_d.  The
        # direct LDSM -> QMUL4 -> TMEM path has no scalar substitute and
        # still emits its N6420 marker, so opting into that path must install
        # the loader bridge regardless of the qmul4_d conversion mode.
        or os.environ.get("RESATTN_QMUL4_N6420_PATCH") == "1"
    )
):
    # This import must happen before cute.compile materializes/loads the cubin.
    # Keeping it here makes it impossible to accidentally execute the marker
    # FMULs as real arithmetic in a kernel that imports this primitive.
    from .qmul4_cubin_patch import install_qmul4_loader_patch

    install_qmul4_loader_patch()


def _ir_value(value, *, loc=None, ip=None) -> ir.Value:
    """Return the MLIR value carried by either a DSL Numeric or an ir.Value."""
    if isinstance(value, ir.Value):
        return value
    return value.ir_value(loc=loc, ip=ip)


@dsl_user_op
def mul_nvfp4x8_e4m3_scale(
    src_vec8: ir.Value,
    scale,
    dst_type: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Dequantize eight packed E2M1 values to eight E4M3 values.

    ``src_vec8`` is a ``vector<8xf4E2M1>`` (32 packed bits). ``scale`` is one
    E4M3 scalar whose raw byte is broadcast to the four lanes of each packed
    multiply.  The returned value is a ``vector<8xE4M3>``.

    The explicit ``.rn`` and ``.satfinite`` modifiers match the NVFP4
    dequantization semantics used by TensorRT-LLM's Blackwell kernels.
    """
    assert dst_type == cutlass.Float8E4M3FN, dst_type

    src_i32 = llvm.bitcast(Int32.mlir_type, src_vec8, loc=loc, ip=ip)
    scale_ir = _ir_value(scale, loc=loc, ip=ip)
    # Alternate FP types are not LLVM scalar types.  Bitcast through a
    # one-element vector, which is the representation CuTe DSL uses for raw
    # FP8 byte conversions elsewhere in its numeric helpers.
    scale_vec_type = ir.VectorType.get([1], scale_ir.type, loc=loc)
    scale_vec = vector.from_elements(scale_vec_type, [scale_ir], loc=loc, ip=ip)
    scale_i8_vec_type = ir.VectorType.get([1], Int8.mlir_type, loc=loc)
    scale_i8_vec = vector.bitcast(scale_i8_vec_type, scale_vec, loc=loc, ip=ip)
    scale_i8 = vector.extract(
        scale_i8_vec, dynamic_position=[], static_position=[0], loc=loc, ip=ip
    )
    scale_i32 = arith.extui(Int32.mlir_type, scale_i8, loc=loc, ip=ip)

    if cutlass_dsl.target_version(min_version="13.4"):
        # Native public PTX ISA 9.4 path.  The scale byte is replicated four
        # times because the packed multiply has one E4M3 scale per lane while
        # NVFP4 has one scale for the whole 16-value block.
        result_pair = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32(), T.i32()]),
            [src_i32, scale_i32],
            """{\n\t
                .reg .b16 f4<2>;\n\t
                .reg .b32 sf4;\n\t
                mov.b32 {f40, f41}, $2;\n\t
                prmt.b32 sf4, $3, 0, 0x0000;\n\t
                mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite $0, f40, sf4;\n\t
                mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite $1, f41, sf4;\n\t
            }""",
            "=r,=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        lo = llvm.extractvalue(T.i32(), result_pair, [0], loc=loc, ip=ip)
        hi = llvm.extractvalue(T.i32(), result_pair, [1], loc=loc, ip=ip)
    else:
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        scale_bcast = llvm.inline_asm(
            T.i32(),
            [scale_i32],
            "prmt.b32 $0, $1, 0, 0x0000;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        src_f32 = llvm.bitcast(Float32.mlir_type, src_i32, loc=loc, ip=ip)
        scale_f32 = llvm.bitcast(Float32.mlir_type, scale_bcast, loc=loc, ip=ip)
        result_pair_f32 = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32(), T.f32()]),
            [src_f32, scale_f32],
            """{\n\t
                mul.rm.f32 $0, $2, $3;\n\t
                mul.rp.f32 $1, $2, $3;\n\t
            }""",
            "=f,=f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        lo_f32 = llvm.extractvalue(T.f32(), result_pair_f32, [0], loc=loc, ip=ip)
        hi_f32 = llvm.extractvalue(T.f32(), result_pair_f32, [1], loc=loc, ip=ip)
        lo = llvm.bitcast(Int32.mlir_type, lo_f32, loc=loc, ip=ip)
        hi = llvm.bitcast(Int32.mlir_type, hi_f32, loc=loc, ip=ip)
    packed_type = ir.VectorType.get([2], Int32.mlir_type, loc=loc)
    packed = vector.from_elements(packed_type, [lo, hi], loc=loc, ip=ip)
    result_type = ir.VectorType.get([8], dst_type.mlir_type, loc=loc)
    return vector.bitcast(result_type, packed, loc=loc, ip=ip)


@dsl_user_op
def mul_nvfp4x4_ldsm_e4m3_scales(
    src_vec4: ir.Value,
    scales_vec4: ir.Value,
    dst_type: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Transform one TRT-style ``LDSM.U4`` register to four E4M3 values.

    ``LDSM.U4x16P64TO8`` leaves four FP4 values in nibbles 0/2/4/6 of a
    32-bit register.  In the transposed V path those values belong to four
    different tokens, hence they also carry four independently packed E4M3
    scales.  TRT-LLM consumes precisely this pair as ``QMUL4 ... .N6420,
    ... .B3210`` before storing the result directly to TMEM.

    This is deliberately separate from :func:`mul_nvfp4x8_e4m3_scale`:
    that helper owns a contiguous 16-D NVFP4 block and broadcasts one scale,
    while this helper owns the post-LDSM register geometry used by decode.
    """
    assert dst_type == cutlass.Float8E4M3FN, dst_type
    assert isinstance(src_vec4.type, ir.VectorType)
    assert isinstance(scales_vec4.type, ir.VectorType)
    assert src_vec4.type.shape == [4], src_vec4.type
    assert scales_vec4.type.shape == [4], scales_vec4.type

    src_i32 = llvm.bitcast(Int32.mlir_type, src_vec4, loc=loc, ip=ip)
    scales_i32 = llvm.bitcast(Int32.mlir_type, scales_vec4, loc=loc, ip=ip)

    if cutlass_dsl.target_version(min_version="13.4"):
        # Public PTX exposes packed QMUL4 but not the SASS-only N6420 source
        # selector.  Pack the four even nibbles into a legal f4x4 operand;
        # the current DSL-4.6 path below is the exact one-instruction TRT
        # form, while this branch is the forward-compatible correctness path.
        result_i32 = llvm.inline_asm(
            T.i32(),
            [src_i32, scales_i32],
            """{\n\t
                .reg .b16 f4, pad;\n\t
                .reg .b32 p0, p1, p2, p3, packed;\n\t
                and.b32 p0, $1, 0x0000000f;\n\t
                and.b32 p1, $1, 0x00000f00;\n\t
                and.b32 p2, $1, 0x000f0000;\n\t
                and.b32 p3, $1, 0x0f000000;\n\t
                shr.u32 p1, p1, 4;\n\t
                shr.u32 p2, p2, 8;\n\t
                shr.u32 p3, p3, 12;\n\t
                or.b32 packed, p0, p1;\n\t
                or.b32 packed, packed, p2;\n\t
                or.b32 packed, packed, p3;\n\t
                mov.b32 {f4, pad}, packed;\n\t
                mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite $0, f4, $2;\n\t
            }""",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    else:
        if os.environ.get("RESATTN_QMUL4_N6420_PATCH") != "1":
            raise RuntimeError(
                "the experimental LDSM/QMUL4/TMEM path on CuTe DSL PTX 9.3 "
                "requires RESATTN_QMUL4_N6420_PATCH=1"
            )
        # Opt-in marker rewritten to QMUL4.N6420.B3210 by the cubin bridge.
        # Both operands are bit containers, so the placeholder's numerical
        # result is never observed.
        src_f32 = llvm.bitcast(Float32.mlir_type, src_i32, loc=loc, ip=ip)
        scales_f32 = llvm.bitcast(
            Float32.mlir_type, scales_i32, loc=loc, ip=ip
        )
        marker_asm = """{
                .reg .b32 lane, scale_gpr;
                // ptxas otherwise promotes some per-fragment scales to a
                // uniform-register QMUL4 form.  N6420+UR does not share the
                // format bits inferred from public N3210+UR (B200 Q0 drops
                // from 0.9997 to -0.0317).  This bit-preserving round trip
                // forces the proven ordinary-GPR N6420 encoding.
                mov.u32 lane, %laneid;
                xor.b32 scale_gpr, $2, lane;
                xor.b32 scale_gpr, scale_gpr, lane;
                mul.rz.f32 $0, $1, scale_gpr;
            }"""
        result_f32 = llvm.inline_asm(
            T.f32(),
            [src_f32, scales_f32],
            marker_asm,
            # In-place packed QMUL4 is legal (TRT-LLM emits both Rd==Ra and
            # Rd==Rb forms), so do not retain operands with post-marker
            # no-ops.  The marker's FMUL data dependency gives ptxas the
            # corresponding read/write scoreboard contract.
            "=f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        result_i32 = llvm.bitcast(
            Int32.mlir_type, result_f32, loc=loc, ip=ip
        )

    # ``vector.bitcast`` requires vector operands on both sides.  Alternate
    # FP8 scalar types also cannot be reached with ``llvm.bitcast`` directly,
    # so use the same one-element-vector bridge as the scale handling above.
    result_i32_type = ir.VectorType.get([1], Int32.mlir_type, loc=loc)
    result_i32_vec = vector.from_elements(
        result_i32_type, [result_i32], loc=loc, ip=ip
    )
    result_type = ir.VectorType.get([4], dst_type.mlir_type, loc=loc)
    return vector.bitcast(result_type, result_i32_vec, loc=loc, ip=ip)


def dequantize_nvfp4x8_ldsm(
    src: cute.Tensor,
    scales: cute.Tensor,
    dst_type: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
) -> cute.TensorSSA:
    """Apply two ``QMUL4.N6420.B3210`` ops to one 8-value LDSM fragment."""
    src_fragment = src.load()
    scale_fragment = scales.load()
    assert cute.size(src_fragment.shape) == 8, src_fragment.shape
    assert cute.size(scale_fragment.shape) == 8, scale_fragment.shape

    src_half_type = ir.VectorType.get([4], src.dtype.mlir_type)
    scale_half_type = ir.VectorType.get([4], scales.dtype.mlir_type)
    src_lo = vector.extract_strided_slice(src_half_type, src_fragment, [0], [4], [1])
    src_hi = vector.extract_strided_slice(src_half_type, src_fragment, [4], [4], [1])
    scale_lo = vector.extract_strided_slice(
        scale_half_type, scale_fragment, [0], [4], [1]
    )
    scale_hi = vector.extract_strided_slice(
        scale_half_type, scale_fragment, [4], [4], [1]
    )
    dst_lo = mul_nvfp4x4_ldsm_e4m3_scales(src_lo, scale_lo, dst_type)
    dst_hi = mul_nvfp4x4_ldsm_e4m3_scales(src_hi, scale_hi, dst_type)
    combined = vector.shuffle(dst_lo, dst_hi, list(range(8)))
    return cute.TensorSSA.from_vector(
        combined, shape=src_fragment.shape, dtype=dst_type
    )


def dequantize_nvfp4x8(
    src: cute.Tensor,
    scale,
    dst_type: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
) -> cute.TensorSSA:
    """Load an eight-element register fragment and apply one NVFP4 scale."""
    fragment = src.load()
    assert cute.size(fragment.shape) == 8, fragment.shape
    return cute.TensorSSA.from_vector(
        mul_nvfp4x8_e4m3_scale(fragment, scale, dst_type),
        shape=fragment.shape,
        dtype=dst_type,
    )


#: ``portable`` 强制走不依赖 QMUL4 的转换(cvt + 乘法)。
#:
#: 两个用途:(a) **二分** —— QMUL4 路线在某个 kernel 里 trap 时,换成它就能
#: 判定问题是出在指令本身还是流水线别处;(b) **兜底** —— 统一 KV 布局这条
#: 主线不该被一条指令的可用性卡住,QMUL4 是转换步骤的优化而非前提。
#: 代价:每 16 值 16 次 cvt + 16 次乘,而 QMUL4 只要 4 条指令。
_CONV_MODE = os.environ.get("RESATTN_QMUL4_MODE", "auto")


def dequantize_nvfp4x16_portable(
    src: cute.Tensor,
    scale,
    dst_type: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
) -> cute.TensorSSA:
    """QMUL4 的可移植等价物:纯 DSL 转换,无 inline asm、无 cubin patch。

    数值上应与 QMUL4 路径一致(同为 e2m1 精确值 × e4m3 scale,再 RNE 到
    e4m3);隔离冒烟门对两条路径用同一份参考,所以差异会被立刻抓到。
    """
    fragment = src.load()
    assert cute.size(fragment.shape) == 16, fragment.shape
    vals = fragment.to(Float32) * Float32(scale)
    return vals.to(dst_type)


def dequantize_nvfp4x16(
    src: cute.Tensor,
    scale,
    dst_type: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
) -> cute.TensorSSA:
    """Dequantize one complete 16-value NVFP4 block.

    A standard KV-cache scale along ``D`` covers exactly these 16 adjacent
    values.  The helper emits four packed multiplies (two per eight-value
    register fragment) and is the unit used by the V transform warps.
    """
    if cutlass.const_expr(_CONV_MODE == "portable"):
        return dequantize_nvfp4x16_portable(src, scale, dst_type)
    fragment = src.load()
    assert cute.size(fragment.shape) == 16, fragment.shape

    # Sub-byte Tensor scalars cannot be dereferenced by CuTe DSL.  Keep the
    # FP4 values packed in registers: slice the loaded vector into its two
    # eight-value (32-bit) halves, transform each half, then concatenate the
    # two FP8 vectors in registers.
    src_half_type = ir.VectorType.get([8], src.dtype.mlir_type)
    src_lo = vector.extract_strided_slice(
        src_half_type, fragment, [0], [8], [1]
    )
    src_hi = vector.extract_strided_slice(
        src_half_type, fragment, [8], [8], [1]
    )
    dst_lo = mul_nvfp4x8_e4m3_scale(src_lo, scale, dst_type)
    dst_hi = mul_nvfp4x8_e4m3_scale(src_hi, scale, dst_type)
    combined = vector.shuffle(dst_lo, dst_hi, list(range(16)))
    return cute.TensorSSA.from_vector(
        combined,
        shape=fragment.shape,
        dtype=dst_type,
    )


@cute.jit
def dequantize_nvfp4_tile_d(
    src: cute.Tensor,
    scales: cute.Tensor,
    dst: cute.Tensor,
    thread_idx: Int32,
    num_threads: cutlass.Constexpr,
):
    """Cooperatively transform a ``(D, tokens)`` tile from FP4 to FP8.

    ``D`` must be the contiguous/vectorized mode in ``src`` and scales have
    shape ``(D // 16, tokens)``.  Threads own complete 16-value D blocks, so
    every load uses one scale and every loop iteration maps to four QMUL4
    instructions.  ``dst`` may use a different logical layout (including an
    MMA-compatible shared-memory swizzle) as long as the corresponding
    16-value D slice supports a vector store.
    """
    assert cute.rank(src) == cute.rank(dst) == cute.rank(scales) == 2
    # 按**模大小**比较,不比字面 shape:dst 可能是从真实 SMEM 布局组合出来的
    # 嵌套视图(如 (128,(32,4))),索引语义与扁平 (128,128) 一致 —— CuTe 按
    # colex 拆扁平坐标 —— 但 tuple 不相等。scales 同理(interleave 视图)。
    assert cute.size(src, mode=[0]) == cute.size(dst, mode=[0]), (
        src.shape, dst.shape)
    assert cute.size(src, mode=[1]) == cute.size(dst, mode=[1]), (
        src.shape, dst.shape)
    assert src.dtype == cutlass.Float4E2M1FN
    assert dst.dtype == cutlass.Float8E4M3FN
    d = cutlass.const_expr(src.shape[0])
    tokens = cutlass.const_expr(src.shape[1])
    assert d % 16 == 0
    # 按**模大小**比较,不比字面 shape:trtllm 的 4-token interleave 视图是
    # 嵌套的 (S, (4, ps/4)),索引语义与扁平 (S, ps) 完全一致(CuTe 按 colex
    # 拆扁平坐标),但字面 tuple 不相等。
    assert cute.size(scales, mode=[0]) == d // 16, scales.shape
    assert cute.size(scales, mode=[1]) == tokens, scales.shape

    d_blocks = cutlass.const_expr(d // 16)
    total_blocks = cutlass.const_expr(d_blocks * tokens)
    for block in cutlass.range(thread_idx, total_blocks, num_threads):
        d_block = block % d_blocks
        token = block // d_blocks
        src_block = cute.local_tile(src, (16, 1), (d_block, token))
        dst_block = cute.local_tile(dst, (16, 1), (d_block, token))
        dst_block.store(
            dequantize_nvfp4x16(
                src_block,
                scales[d_block, token],
                cutlass.Float8E4M3FN,
            )
        )


__all__ = [
    "dequantize_nvfp4_tile_d",
    "dequantize_nvfp4x8",
    "dequantize_nvfp4x8_ldsm",
    "dequantize_nvfp4x16",
    "mul_nvfp4x4_ldsm_e4m3_scales",
    "mul_nvfp4x8_e4m3_scale",
]
