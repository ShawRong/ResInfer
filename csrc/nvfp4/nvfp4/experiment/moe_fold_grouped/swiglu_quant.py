# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused SwiGLU + MExt-R1 quantiser for the batched expert chain.

EXPERIMENT. JIT-built into its own module, like `experiment/moe_fold.py`, so
a build failure here can never change which ops nvfp4._C exports.

This is the piece that made the chain real. Before it, the benchmark handed
GEMM2 a stand-in buffer of the right shape, so the reported chain time was
missing the quantise step entirely and GEMM2 was multiplying noise.

There is deliberately NO torch fallback. A quantiser that silently degrades
to something else produces numbers that look like a measurement and are not
one -- and unlike `fold_pairs`, there is no one-line torch expression that
reproduces the row-pair FP4 + swizzled-SF output, so any fallback would be a
second implementation to keep correct. Callers get an exception instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SOURCE = _HERE / "swiglu_quant.cu"

_module = None
_tried = False
_error: str | None = None


def _include_dir() -> Path:
    """Where fp4/nvfp4_utils.cuh lives.

    In the repo it is csrc/nvfp4/src, three levels up from this file. Remote
    harnesses that ship only the Python package must set NVFP4_CSRC_INCLUDE
    (the Modal probes add_local_dir it to /path/to/resinfer).
    """
    env = os.getenv("NVFP4_CSRC_INCLUDE")
    if env:
        return Path(env)
    return _HERE.parents[2] / "src"


def load(verbose: bool = False):
    """Build once; return the module, or None with `load_error()` set."""
    global _module, _tried, _error
    if _tried:
        return _module
    _tried = True
    try:
        from torch.utils.cpp_extension import load as _load

        inc = _include_dir()
        header = inc / "fp4" / "nvfp4_utils.cuh"
        if not header.exists():
            raise FileNotFoundError(
                f"{header} not found. Set NVFP4_CSRC_INCLUDE to the directory "
                "containing fp4/nvfp4_utils.cuh."
            )

        # torch's cpp_extension injects -D__CUDA_NO_HALF_CONVERSIONS__ (and
        # friends); the production build of this header does not. Without the
        # -U's, nvfp4_utils.cuh:538 `float(__hmax(...))` fails to compile with
        # "no suitable conversion function from __nv_bfloat16 to float". A
        # local nvcc check will NOT reproduce this -- the flags come from
        # cpp_extension, not from the source.
        flags = [
            "-O3", "--use_fast_math", f"-I{inc}",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ]
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            flags.append(
                f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"
            )
        _module = _load(
            name="nvfp4_experiment_swiglu_quant",
            sources=[str(_SOURCE)],
            extra_cuda_cflags=flags,
            extra_cflags=["-O3", f"-I{inc}"],
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
        import traceback

        _error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        _module = None
    return _module


def load_error() -> str | None:
    return _error


def available() -> bool:
    return load() is not None


def sf_bytes_per_expert(m_tok: int, cols: int) -> int:
    """Bytes the fold GEMM strides by per l. Padded to 128 rows / 64 cols."""
    return ((2 * m_tok + 127) // 128) * ((cols + 63) // 64) * 512


def make_buffers(experts: int, m_tok: int, cols: int, device) -> tuple:
    """Preallocate (out, out_sf) for `swiglu_quant`.

    The SF buffer is ZEROED, not empty. Each expert's block is padded up to
    128 rows and the fold GEMM's TMA reads the whole padded tile; FP8 E4M3
    encodes NaN at 0x7F/0xFF, so uninitialised padding can poison an output
    that is otherwise correct. A probe already lost a run to exactly that
    (all experts "unequal", maxdiff=nan, zero differing elements). Zero is a
    finite E4M3, so zeroing costs one memset and removes the failure mode.
    """
    out = torch.empty(experts * 2 * m_tok * (cols // 2), dtype=torch.uint8,
                      device=device)
    out_sf = torch.zeros(experts * sf_bytes_per_expert(m_tok, cols),
                         dtype=torch.uint8, device=device)
    return out, out_sf


LAYOUT_ROW_PAIR = 1   # base at 2t, residue at 2t+1        -- N-ext, store_mode="tma"
LAYOUT_TILE8 = 2      # [base0..7, res0..7] per 8 tokens   -- M-ext, store_mode="tma_mext"


def swiglu_quant(
    u: torch.Tensor,
    sf_scale: torch.Tensor,
    *,
    experts: int,
    fuse_swiglu: bool = True,
    layout_mode: int = LAYOUT_TILE8,
    out: torch.Tensor | None = None,
    out_sf: torch.Tensor | None = None,
    row_idx: torch.Tensor | None = None,
    m_tok: int | None = None,
    pre_scale: torch.Tensor | None = None,
    post_scale: torch.Tensor | None = None,
    linear_sf: bool = False,
):
    """SwiGLU + MExt-R1 quantise. Returns (fp4, sf).

    u:  (E*m_tok, 2I) bf16 [gate | up] when fuse_swiglu, else (E*m_tok, I)
    sf_scale: float32 [1] or [2] ([base_scale, residue_scale], dual-alpha)

    pre_scale / post_scale: optional float32 (experts,) per-SLOT factors.
        pre multiplies gate AND up before SwiGLU -- the per-expert GEMM1
        dequant ratio, which the nonlinearity forbids applying later.
        post multiplies h after SwiGLU, before quantisation -- the
        per-expert part of GEMM2's scale chain, so GEMM2's epilogue alpha
        can stay a global scalar. Indexed by the slot (blockIdx.z): with a
        gather table, pass vectors already gathered per slot, e.g.
        ratio[expert_ids]. In the unfused path the two collapse to one
        multiply. None keeps the exact pre-change bit pattern.

    row_idx: optional int32 (E*m_tok,) mapping slot -> source row of `u`. With
        it this call also does the PERMUTE -- `u` is the UNSORTED activation
        and each expert's slots gather their tokens straight out of it, so
        there is no reorder pass and no permuted copy in HBM. This is how the
        MoE input side avoids a separate `moe_permute`; the same table shape
        `build_routing` already produces. Requires m_tok, since `u`'s row
        count no longer implies it. Padding slots must point at a row the
        caller owns (conventionally a zero row appended to `u`) -- the kernel
        does not branch on them.

    Returns fp4 as (E*2*m_tok, I/2) uint8, experts l-outermost: expert e owns
    rows [e*2*m_tok, (e+1)*2*m_tok), so the GEMM takes a one-expert view and
    reaches the rest through its l-stride.

    layout_mode defaults to TILE8 because the chain uses store_mode="tma_mext"
    (M-ext). It is not a preference -- ROW_PAIR fed to an M-ext GEMM folds the
    wrong pair of rows and produces plausible garbage. See swiglu_quant.cu.
    """
    mod = load()
    if mod is None:
        raise RuntimeError(
            f"swiglu_quant kernel unavailable: {load_error()}"
        )
    if u.dim() != 2:
        raise ValueError(f"expected a 2-D activation, got {tuple(u.shape)}")

    if row_idx is None:
        if u.shape[0] % experts != 0:
            raise ValueError(
                f"{u.shape[0]} rows do not divide by experts={experts}"
            )
        m_tok = u.shape[0] // experts
    else:
        if m_tok is None:
            raise ValueError("row_idx needs an explicit m_tok")
        if row_idx.numel() != experts * m_tok:
            raise ValueError(
                f"row_idx has {row_idx.numel()} entries, expected "
                f"experts*m_tok = {experts * m_tok}"
            )

    cols = u.shape[1] // 2 if fuse_swiglu else u.shape[1]
    if out is None or out_sf is None:
        if linear_sf:
            # RESIDUE-GRAFT: plain row-major SF (2*m_tok*experts, cols/16),
            # exactly the SFA layout the cutedsl gather GEMM consumes.
            # EMPTY, not zeros: linear SF has no padding region -- every
            # byte is written (padding slots write sf=0 via the zero row),
            # and the zeros() memset was a measurable kernel in the timed
            # path (the zeroing convention exists for the SWIZZLED layout's
            # NaN-padding trap only).
            out = torch.empty(experts * 2 * m_tok * (cols // 2),
                              dtype=torch.uint8, device=u.device)
            out_sf = torch.empty(experts * 2 * m_tok * (cols // 16),
                                 dtype=torch.uint8, device=u.device)
        else:
            out, out_sf = make_buffers(experts, m_tok, cols, u.device)

    _empty_f32 = torch.empty(0, dtype=torch.float32, device=u.device)
    mod.swiglu_quant_mext_r1(
        out, out_sf, u.contiguous(), sf_scale, experts, fuse_swiglu,
        layout_mode,
        row_idx if row_idx is not None
        else torch.empty(0, dtype=torch.int32, device=u.device),
        m_tok,
        pre_scale if pre_scale is not None else _empty_f32,
        post_scale if post_scale is not None else _empty_f32,
        linear_sf)
    return out.view(experts * 2 * m_tok, cols // 2), out_sf


__all__ = [
    "LAYOUT_ROW_PAIR",
    "LAYOUT_TILE8",
    "available",
    "load",
    "load_error",
    "make_buffers",
    "sf_bytes_per_expert",
    "swiglu_quant",
]
