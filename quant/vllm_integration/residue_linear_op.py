"""Opaque `torch.library.custom_op` covering both residue NVFP4 dispatch paths.

Why this exists:
    Our residue NVFP4 linear method has two distinct kernel chains:
      • mext fold path: scaled_fp4_quant_mext_r1 + fold-tile8 GEMM
      • legacy K-ext path: scaled_fp4_quant_with_mask + cutlass_scaled_fp4_mm
    Which one a layer gets is NOT purely an M decision. A `mext_r1` layer
    (residue ratio 1.0, weight stored once) has no K-extended weight to fall
    back to, so it takes the fold at every M — see `is_mext_r1` in the
    dispatch rule below. Only a genuinely `extended_k` layer (ratio < 1.0)
    chooses between the two by M.
    Selecting between them at Python level inside `apply()` made the routing
    tensor-shape-dependent. torch.compile then specialised the trace at its
    dummy input shape (M ≈ max_num_batched_tokens), baked in the legacy branch,
    and *all* captured CUDA graphs replayed the slow path — verified by nsys
    (0 mext kernel symbols in the non-eager trace, only K-ext variants).

    This module wraps the entire residue NVFP4 GEMM pipeline (quant + GEMM)
    in a single opaque op (`torch.library.custom_op`). torch.compile treats
    the call as a black box: captures it as one node, bakes only per-layer
    constants (k_base, num_salient, use_fold, …) into that node's args, and
    replays it verbatim at every captured size. The internal Python (shape
    derivation, layout view, branch on use_fold, kernel calls) runs at
    execution time, not trace time, so layout choice is correct for the
    actual M of each capture.

    Same design pattern vLLM uses for its own `cutlass_scaled_fp4_mm` and
    `flashinfer_mm_fp4`: Python-side dispatch inside, opaque outside.

Shape derivation:
    All intermediate shapes are computed from `x.shape` (the model input,
    specialised per cudagraph_capture_size at trace time) and integer args
    (`num_salient`, `k_base`, etc., per-layer constants). We never read
    `x_fp4.shape[*]` from a kernel output — those are SymInts whose
    materialization would force a host sync at trace time (the exact bug
    we already fixed in `linear_method_mext_r1._apply_mext_r1`).

Backend scope:
    Cutlass NVFP4 backend only. The legacy V2 path also has flashinfer / cudnn
    / fbgemm fallbacks for non-cutlass kernels; those are not hot-path on our
    machines and remain on the original (non-custom-op) `super().apply()`
    branch in `linear_method_v2._apply_residue_v2`.

Production assumption:
    Always emits the clean-vec-store fold kernel for `use_fold=True`. The
    ext_b_prefix debug fallback that used to be named here is gone entirely —
    those C++ CUTLASS ops were deleted with the SM120 fold, and the CuTeDSL
    fold is the only fold implementation left.
"""

from __future__ import annotations

import json
import os

import torch

import nvfp4
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    slice_nvfp4_output,
)

# Row-pair layout used by the two-GEMM prefill fallback.
_LAYOUT_ROW_PAIR = 1


























_FOLD_MAX_M_DEFAULT_SM10X = 128
_FOLD_MAX_M_DEFAULT_OTHER = 64
_FOLD_MAX_M_ENV = os.environ.get("RESIDUE_DECODE_MEXT_MAX_M")

# Full-rank MExt can use the Fold families beyond the ordinary decode band.
# Keep the two limits independent: plain/partial-residue routing is unchanged.
_MEXT_FOLD_MAX_M_DEFAULT_SM10X = 512
_MEXT_FOLD_MAX_M_ENV = os.environ.get("RESIDUE_MEXT_FOLD_MAX_M")


def fold_max_m_for(arch: str | None) -> int:
    """The M at or below which a non-mext_r1 layer takes the fold.

    `arch` is `_sm10x_fold_arch()`'s answer: 'sm100'/'sm103', or None for
    SM120 and everything else.

    Single source of truth — warmup and dispatch must agree, or the server
    JITs inside cudagraph capture for the sizes warmup missed. The env
    override applies on EVERY arch; it is a performance knob only (since
    4083dee a mext_r1 layer folds at every M regardless of this value, so
    moving it can no longer change what math is computed).
    """
    if _FOLD_MAX_M_ENV is not None:
        return int(_FOLD_MAX_M_ENV)
    return (_FOLD_MAX_M_DEFAULT_SM10X if arch is not None
            else _FOLD_MAX_M_DEFAULT_OTHER)


def mext_fold_max_m_for(arch: str | None) -> int:
    """Largest M routed through Fold for a full-rank mext_r1 layer."""
    decode_max = fold_max_m_for(arch)
    if arch is None:
        return decode_max
    if _MEXT_FOLD_MAX_M_ENV is not None:
        return max(decode_max, int(_MEXT_FOLD_MAX_M_ENV))
    return max(decode_max, _MEXT_FOLD_MAX_M_DEFAULT_SM10X)


# Override knob for PD-disaggregated deployments: when set, every fold-eligible
# layer takes the fold branch regardless of M. Intended for the decode-only
# server in a prefill/decode split, where every forward is small-M anyway but
# we want a uniform captured kernel sequence (no M-threshold-dependent branch
# inside the captured graph set). At-call cost is 0 — only the dispatch
# expression changes; pack8 fold kernel works at arbitrary M via the K-split
# block/grid in invokeFP4QuantizationMExtR1.
_FORCE_FOLD = os.environ.get("RESIDUE_FORCE_DECODE_MEXT", "0") == "1"

# ── prefill-side residue kill switch (EXPERIMENT ONLY) ─────────────────────
# `RESIDUE_PREFILL_RESIDUE_OFF=1` makes every residue layer compute PLAIN
# NVFP4 above the M threshold while leaving decode untouched, i.e. the arm
# "no prefill residue + mext decode". It exists because the attribution
# control for the per-layer ratio search has never been run: every published
# comparison was b4.5 (4.5 effective bits) against b8.0 (8.0) or against
# plain (no residue anywhere), and neither isolates the prefill contribution.
# With this flag the same checkpoint serves both sides of the A/B, so nothing
# but the prefill path differs -- no re-export, no weight change.
#
# How each layer kind is neutered, and why that way:
#   K-extension  the extended activation's residue columns are ZEROED. fp4
#                code 0 is +0.0, so those columns contribute exactly nothing
#                and the output equals plain NVFP4 over the base columns.
#                Nothing about the weight layout, the scale layout or the
#                cutlass padding changes -- which is the point: slicing the
#                weight to k_base would mean a non-contiguous view and a
#                re-derived padding, i.e. two new ways to be silently wrong.
#   mext_r1      no extension exists to zero; the fold is simply not taken and
#                the layer falls through to the plain cutlass bypass, whose
#                weight/weight_scale already span the full (== base) K.
#
# The band is chosen by two optional bounds; the gate turns the switch on.
#
#   neither bound       MIN_M defaults to the fold band -> kills above it, so
#                       "prefill only" and decode keeps its residue everywhere
#   ..._MIN_M = t       kill where m >  t
#   ..._MAX_M = t       kill where m <= t
#   both                kill where either holds (residue survives only between
#                       them) -- an intersection nobody has needed yet, but the
#                       union is what the two rules mean read together
#
# _MAX_M completes the 2x2 the attribution study needs. Without it the fourth
# cell is unreachable:
#
#   both          no flags                          residue everywhere
#   decode-only   OFF=1                             kill above the fold band
#   plain         OFF=1  MIN_M=0                    kill everywhere
#   prefill-only  OFF=1  MAX_M=<fold band>          kill in the decode band
#
# and without the fourth cell you cannot tell whether both ~= prefill + decode
# (complementary) or both ~= max(prefill, decode) (redundant) -- i.e. whether
# two mechanisms are doing work or one.
#
# Naming wart, stated rather than hidden: the gate says PREFILL because when it
# was added only the "above" direction existed. With _MAX_M the same switch
# kills the DECODE band. Renaming it would break the arms already in flight and
# the results that cite them, so the name stays and this paragraph explains it.
_PREFILL_RESIDUE_OFF = (
    os.environ.get("RESIDUE_PREFILL_RESIDUE_OFF", "0") == "1")
_PREFILL_RESIDUE_OFF_MIN_M = os.environ.get("RESIDUE_PREFILL_RESIDUE_OFF_MIN_M")
_PREFILL_RESIDUE_OFF_MAX_M = os.environ.get("RESIDUE_PREFILL_RESIDUE_OFF_MAX_M")


# ── SM100/SM103 (B200/B300) CuTeDSL fold path ──────────────────────────────
# Blackwell datacenter GPUs use the CuTeDSL sm100/sm103 fork
# (nvfp4.cute_fold). (This comment used to contrast with 'the SM120 C++ fold
# GEMM' -- deleted 2a3e507; sm120 is CuTeDSL too now.) It consumes the row_pair mext_r1
# quant layout directly (no m_pad, no tile8) and, for pure mext_r1 layers,
# reads the full contiguous packed weight (k_base == full K, no prefix slice).
def _sm10x_fold_arch() -> str | None:
    """Return the SM10x kernel arch; a disabled Fold is an explicit opt-out."""
    if os.environ.get("RESIDUE_DISABLE_SM10X_FOLD", "0") == "1":
        return None
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        return None
    from nvfp4.cute_fold.tactics import kernel_arch_for_capability

    return kernel_arch_for_capability(int(major), int(minor))


_SHAPE_LOG_SEEN: set = set()


def _base_k_weight_view(weight: torch.Tensor, k_base: int) -> torch.Tensor:
    """Base-K prefix of a (possibly ext-K) packed FP4 weight, as a VIEW.

    For a pure `mext_r1` layer k_base already spans the whole tensor and this
    returns `weight` unchanged. For an `extended_k` layer the stored weight is
    [N, k_ext/2] and the fold only needs the first k_base/2 packed columns; the
    narrowed view keeps row stride k_ext/2, which both CuTeDSL folds now consume
    directly (see cute_fold/host.py and cute_fold_sm120/host.py). No copy, so
    the ext weight is never duplicated at decode.
    """
    k_base_packed = int(k_base) // 2
    if os.environ.get("NVFP4_FOLD_SHAPE_LOG", "0") == "1":
        _k = (int(weight.shape[0]), int(weight.shape[1]), int(k_base))
        if _k not in _SHAPE_LOG_SEEN:
            _SHAPE_LOG_SEEN.add(_k)
            print(f"[fold-shape] n_w={_k[0]} stored_packed={_k[1]} "
                  f"k_base={_k[2]} -> narrow={k_base_packed != weight.shape[1]}",
                  flush=True)
    if k_base_packed == weight.shape[1]:
        return weight
    assert k_base_packed < weight.shape[1], (
        f"k_base {k_base} exceeds stored weight width {weight.shape[1] * 2}")
    return weight[:, :k_base_packed]






















_RATIO1_PREFILL_PATH = os.environ.get(
    "RESIDUE_RATIO1_PREFILL_PATH",
    os.environ.get("RESIDUE_MEXT_PREFILL_PATH", "auto"),
)


def _run_ratio1_prefill_two_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """ratio-1 prefill: one backend GEMM at 2M rows + pair-sum epilogue.

    out[i] = W·base_i + W·res_i, exactly the fold's math. The mext_r1 quant's
    row_pair SF is bytewise the cutlass 128x4 swizzle over the 2M rows (the
    kernel places SFs with the same cvt_quant_to_fp4_get_sf_out_offset the
    stock quant uses), so the doubled activation feeds cutlass_scaled_fp4_mm
    directly and base/residue rows come back interleaved at even/odd indices.

    Every op here has a static compile space: no CuTeDSL, nothing to JIT at
    serving time, and cudagraph capture sees ordinary torch/cutlass calls.
    """
    x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(
        x, input_global_scale_inv, layout_mode=_LAYOUT_ROW_PAIR,
    )
    two_m, k = x_fp4.shape[0], x.shape[-1]
    num_m_tiles = (two_m + 127) // 128
    num_k_tiles = (k + 63) // 64
    x_sf_v = x_sf.view(torch.float8_e4m3fn).view(
        num_m_tiles * 128, num_k_tiles * 4)
    y = cutlass_scaled_fp4_mm(
        x_fp4, weight, x_sf_v, weight_scale_base, alpha, output_dtype)
    return y[0::2] + y[1::2]


def _linear_policy(k_base, num_salient, padding, output_size,
                   fold_eligible, use_swap, ratio1):
    from ..flashinfer_integration.nvfp4_linear_tuner import NVFP4Policy

    sm10x = _sm10x_fold_arch()
    arch = sm10x or ("sm120" if torch.cuda.get_device_capability()[0] == 12 else None)
    return NVFP4Policy(
        int(k_base), int(num_salient), int(padding), int(output_size),
        bool(fold_eligible), bool(use_swap), bool(ratio1), arch,
        fold_max_m_for(sm10x), mext_fold_max_m_for(sm10x), _FORCE_FOLD,
        _PREFILL_RESIDUE_OFF,
        int(_PREFILL_RESIDUE_OFF_MIN_M) if _PREFILL_RESIDUE_OFF_MIN_M else None,
        int(_PREFILL_RESIDUE_OFF_MAX_M) if _PREFILL_RESIDUE_OFF_MAX_M else None,
        _RATIO1_PREFILL_PATH,
        os.environ.get("RESIDUE_SM10X_FOLD_TUNER", "1") == "1",
        os.environ.get("RESIDUE_SM120_FOLD_TUNER", "1") == "1",
    )


def _nvfp4_linear_impl(inputs, policy):
    """Execute the static implementation of the shared numerical route."""
    from ..flashinfer_integration.nvfp4_linear_tuner import (
        _fold_inputs, _residue_off, _route,
    )

    x, weight, scale_inv, weight_scale, _base_sf, mask, alpha = inputs
    m, k = x.shape
    if m == 0:
        return x.new_empty((0, policy.output_size))
    route = _route(m, policy)
    if route == "invalid_mext":
        raise RuntimeError("Ratio-1 residue requires a prepared Fold path")
    if route in ("ratio1_prefill", "sm10x_fold", "sm120_fold"):
        args = _fold_inputs(inputs, policy)
        if route == "ratio1_prefill":
            if policy.prefill == "k_loop":
                from ..flashinfer_integration.ratio1_prefill_tuner import (
                    _kloop_valid_tactics, _run_kloop,
                )
                ready = _kloop_valid_tactics(args, False)
                if ready:
                    return _run_kloop(args, ready[0])
            return _run_ratio1_prefill_two_gemm(*args, x.dtype)
        if route == "sm10x_fold":
            from ..flashinfer_integration.sm10x_fold_tuner import _run_fallback
            return _run_fallback(args, x.dtype)[:m]
        from ..flashinfer_integration.sm120_fold_tuner import run_fold_table
        return run_fold_table(args, x.dtype)[:m]

    if route == "plain":
        x_fp4, x_sf = scaled_fp4_quant(
            x, scale_inv, is_sf_swizzled_layout=True, backend="cutlass")
    else:
        x_fp4, x_sf = nvfp4.scaled_fp4_quant_with_mask(
            x, scale_inv, mask, policy.num_salient,
            use_swapped_residue=policy.use_swap)
        if _residue_off(m, policy):
            x_fp4[:, k // 2:].zero_()
        x_sf = x_sf.view(torch.float8_e4m3fn).view(
            ((m + 127) // 128) * 128,
            ((k + policy.num_salient + 63) // 64) * 4)
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, policy.padding)
    out = cutlass_scaled_fp4_mm(
        x_fp4, weight, x_sf, weight_scale, alpha, x.dtype)
    return slice_nvfp4_output(out, policy.output_size)


@torch.library.custom_op(
    "residue::nvfp4_linear", mutates_args=(), device_types="cuda",
)
def nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_base: torch.Tensor,
    channel_mask: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    num_salient: int,
    weights_padding_cols: int,
    output_size: int,
    fold_eligible: bool,
    use_swap: bool,
    is_mext_r1: bool,
) -> torch.Tensor:
    """Dispatch all NVFP4 Linear sizes through one FlashInfer operation."""
    if x.shape[0] == 0:
        return x.new_empty((0, weight.shape[0]))
    from ..flashinfer_integration.nvfp4_linear_tuner import tuned_nvfp4_linear

    policy = _linear_policy(k_base, num_salient, weights_padding_cols,
                            output_size, fold_eligible, use_swap, is_mext_r1)
    return tuned_nvfp4_linear(
        [x, weight, input_global_scale_inv, weight_scale,
         weight_scale_base, channel_mask, alpha], policy,
    )


@nvfp4_linear.register_fake
def _nvfp4_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_base: torch.Tensor,
    channel_mask: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    num_salient: int,
    weights_padding_cols: int,
    output_size: int,
    fold_eligible: bool,
    use_swap: bool,
    is_mext_r1: bool,
) -> torch.Tensor:
    """Meta/abstract implementation for shape inference during torch.compile."""
    M = x.shape[0]
    N = int(output_size) if int(output_size) > 0 else weight.shape[0]
    return torch.empty(M, N, dtype=x.dtype, device=x.device)
