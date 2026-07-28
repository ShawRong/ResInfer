"""Opaque `torch.library.custom_op` covering both residue NVFP4 dispatch paths.

Why this exists:
    Our residue NVFP4 linear method has two distinct kernel chains for an
    `extended_k` layer:
      • mext fold path: scaled_fp4_quant_mext_r1 + fold-tile8 GEMM
      • legacy K-ext path: scaled_fp4_quant_with_mask + cutlass_scaled_fp4_mm
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
    ext_b_prefix + manual row-pair reduction debug fallback in
    `linear_method_mext_r1._apply_mext_r1` is not surfaced through this op.
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

try:
    from flashinfer.autotuner import (
        AutoTuner as _FlashInferAutoTuner,
        ConstraintSpec as _FlashInferConstraintSpec,
        DynamicTensorSpec as _FlashInferDynamicTensorSpec,
        TunableRunner as _FlashInferTunableRunner,
        TuningConfig as _FlashInferTuningConfig,
    )

    _FLASHINFER_AUTOTUNE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    _FlashInferAutoTuner = None
    _FlashInferConstraintSpec = None
    _FlashInferDynamicTensorSpec = None
    _FlashInferTunableRunner = object
    _FlashInferTuningConfig = None
    _FLASHINFER_AUTOTUNE_AVAILABLE = False

# mext_r1 quant layout selector. The C op defines:
#   0 = concat
#   1 = row_pair
#   2 = tile8  ← required by the clean-vec-store fold GEMM
_LAYOUT_ROW_PAIR = 1
_LAYOUT_TILE8 = 2
_MEXT_PATH_TILE8 = "tile8"
_MEXT_PATH_COL_SWAP = "col_swap"
_MEXT_PATH_AUTO = "auto"
_MEXT_COL_SWAP_TACTICS = frozenset(range(16))

# Pad input rows to a multiple of 8 so the tile8 fold kernel sees a clean
# 2x-doubled M dimension. The original M is preserved via output slicing.
_M_PAD_MULTIPLE = 8

# M threshold for choosing the fold path. Decided *inside* the op so each
# captured CUDA graph specialises to the real M of its capture size:
#   M ≤ _FOLD_MAX_M → fold (best for small-M / decode)
#   M >  _FOLD_MAX_M → legacy K-extension (better for large-M / prefill)
# The op is opaque to torch.compile, so the if-branch below runs at each
# real call (not at trace time). Different cudagraph_capture_sizes record
# different kernel sequences naturally — fixing the bug where a single
# trace at max-M used to lock every captured graph into one branch.
_FOLD_MAX_M = int(os.environ.get("RESIDUE_DECODE_MEXT_MAX_M", "64"))

# Override knob for PD-disaggregated deployments: when set, every fold-eligible
# layer takes the fold branch regardless of M. Intended for the decode-only
# server in a prefill/decode split, where every forward is small-M anyway but
# we want a uniform captured kernel sequence (no M-threshold-dependent branch
# inside the captured graph set). At-call cost is 0 — only the dispatch
# expression changes; pack8 fold kernel works at arbitrary M via the K-split
# block/grid in invokeFP4QuantizationMExtR1.
_FORCE_FOLD = os.environ.get("RESIDUE_FORCE_DECODE_MEXT", "0") == "1"

_MEXT_FLASHINFER_AUTOTUNE = "flashinfer_autotune"

# ── SM100/SM103 (B200/B300) CuTeDSL fold path ──────────────────────────────
# On Blackwell datacenter GPUs the SM120 C++ fold GEMM does not exist; use the
# CuTeDSL sm100/sm103 kernel (nvfp4.cute_fold). It consumes the row_pair
# mext_r1 quant layout directly and, for pure mext_r1 layers, reads the full
# contiguous packed weight (k_base == full K, no prefix slice).
# Cached: capability probe runs once.
_SM10X_FOLD_ARCH: str | None = None
_SM10X_FOLD_PROBED = False
_SM10X_FOLD_FIRED = [False]  # one-time "fold kernel actually ran" log guard


def _sm10x_fold_arch() -> str | None:
    """Return 'sm100'/'sm103' if this GPU should use the CuTeDSL fold, else
    None (SM120 / other → legacy path). Probed once."""
    global _SM10X_FOLD_ARCH, _SM10X_FOLD_PROBED
    if _SM10X_FOLD_PROBED:
        return _SM10X_FOLD_ARCH
    _SM10X_FOLD_PROBED = True
    if os.environ.get("RESIDUE_DISABLE_SM10X_FOLD", "0") == "1":
        _SM10X_FOLD_ARCH = None
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
        from nvfp4.cute_fold.tactics import kernel_arch_for_capability

        _SM10X_FOLD_ARCH = kernel_arch_for_capability(int(major), int(minor))
    except Exception:
        _SM10X_FOLD_ARCH = None
    return _SM10X_FOLD_ARCH


def _run_mext_fold_gemm_sm10x(
    x_fp4_rowpair: torch.Tensor,
    weight: torch.Tensor,
    x_sf: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """CuTeDSL sm100/sm103 fold GEMM for a pure mext_r1 layer.

    x_fp4_rowpair: [2M, K/2] uint8 (scaled_fp4_quant_mext_r1 row_pair output).
    weight:        [N, K/2] uint8 packed FP4 (full contiguous, k_base == K).
    x_sf:          raw swizzled 128x4 FP8 scale from the mext_r1 quant.
    weight_scale_base: swizzled base-K weight scale (== weight_scale for mext_r1).
    Returns [M, N] in output_dtype.
    """
    from nvfp4.cute_fold.tactics import run_fold_default

    if not _SM10X_FOLD_FIRED[0]:
        _SM10X_FOLD_FIRED[0] = True
        print(
            f"[residue] SM10x MExt fold kernel FIRED: act={tuple(x_fp4_rowpair.shape)} "
            f"weight={tuple(weight.shape)} arch={_sm10x_fold_arch()}",
            flush=True,
        )

    major, minor = torch.cuda.get_device_capability()
    return run_fold_default(
        int(major),
        int(minor),
        weight,
        x_fp4_rowpair,
        weight_scale_base,
        x_sf.view(torch.float8_e4m3fn),
        alpha,
        output_dtype,
    )


def _parse_int_tuple_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    parsed = tuple(sorted({int(part) for part in value.replace(",", " ").split()}))
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{name} must contain positive integer buckets, got {value!r}")
    return parsed


# Physical fold-GEMM M dimension, i.e. A rows after row-pair expansion.
_MEXT_AUTOTUNE_M2_BUCKETS = _parse_int_tuple_env(
    "RESIDUE_MEXT_AUTOTUNE_M2_BUCKETS",
    (16, 32, 64, 128, 256),
)
_MEXT_TACTIC_LOG_PATH = os.environ.get("RESIDUE_MEXT_TACTIC_LOG_PATH", "").strip()
_MEXT_TACTIC_LOG_LIMIT = int(os.environ.get("RESIDUE_MEXT_TACTIC_LOG_LIMIT", "256"))
_MEXT_TACTIC_LOG_COUNT = 0


def _parse_mext_path() -> str:
    value = os.environ.get("RESIDUE_MEXT_PATH", _MEXT_PATH_AUTO).strip().lower()
    aliases = {
        "": _MEXT_PATH_AUTO,
        "auto": _MEXT_PATH_AUTO,
        "default": _MEXT_PATH_AUTO,
        "tile8": _MEXT_PATH_TILE8,
        "rowwise": _MEXT_PATH_TILE8,
        "row_wise": _MEXT_PATH_TILE8,
        "colswap": _MEXT_PATH_COL_SWAP,
        "col_swap": _MEXT_PATH_COL_SWAP,
        "rowpair": _MEXT_PATH_COL_SWAP,
        "row_pair": _MEXT_PATH_COL_SWAP,
    }
    if value not in aliases:
        raise ValueError(
            "RESIDUE_MEXT_PATH must be auto, tile8, or col_swap, "
            f"got {value!r}"
        )
    return aliases[value]


_MEXT_PATH = _parse_mext_path()


def _resolve_auto_mext_path(m: int, n: int, k: int) -> str:
    # Current heuristic: use the new row-pair col-swap path for fold-eligible
    # small-M calls.  It binds quant layout and GEMM fold together, avoiding the
    # old mistake of treating col-swap as just another GEMM tactic.
    _ = (m, n, k)
    return _MEXT_PATH_COL_SWAP


def _effective_mext_path(m: int, n: int, k: int) -> str:
    if _MEXT_PATH == _MEXT_PATH_AUTO:
        return _resolve_auto_mext_path(m, n, k)
    return _MEXT_PATH


def _default_autotune_tactics_for_path(path: str) -> tuple[int, ...]:
    if path == _MEXT_PATH_COL_SWAP:
        return tuple(sorted(_MEXT_COL_SWAP_TACTICS))
    return tuple(range(16))


_MEXT_AUTOTUNE_TACTICS = _parse_int_tuple_env(
    "RESIDUE_MEXT_AUTOTUNE_TACTICS",
    _default_autotune_tactics_for_path(_effective_mext_path(1, 1, 1)),
)
if any(tactic < 0 or tactic > 15 for tactic in _MEXT_AUTOTUNE_TACTICS):
    raise ValueError(
        "RESIDUE_MEXT_AUTOTUNE_TACTICS are path-local tactic IDs and must "
        f"be in [0, 15], got {_MEXT_AUTOTUNE_TACTICS}"
    )
if _effective_mext_path(1, 1, 1) == _MEXT_PATH_COL_SWAP and any(
    tactic not in _MEXT_COL_SWAP_TACTICS for tactic in _MEXT_AUTOTUNE_TACTICS
):
    raise ValueError(
        "RESIDUE_MEXT_AUTOTUNE_TACTICS for col_swap path currently supports "
        f"only {sorted(_MEXT_COL_SWAP_TACTICS)}, got {_MEXT_AUTOTUNE_TACTICS}"
    )


def _encode_mext_kernel_tactic(path: str, tactic: int | None) -> int | None:
    if tactic is None:
        return None
    tactic = int(tactic)
    if path == _MEXT_PATH_TILE8:
        if tactic < 0 or tactic > 15:
            raise ValueError(f"tile8 MExt tactic must be in [0, 15], got {tactic}")
        return tactic
    if path == _MEXT_PATH_COL_SWAP:
        if tactic not in _MEXT_COL_SWAP_TACTICS:
            raise ValueError(
                "col_swap MExt path currently supports path-local tactics "
                f"{sorted(_MEXT_COL_SWAP_TACTICS)}, got {tactic}"
            )
        # Low-level C++ still receives a single integer for ABI stability.  The
        # public/autotune layer treats tactics as path-local IDs 0..15.
        return 16 + tactic
    raise ValueError(f"unsupported MExt path {path!r}")


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _map_m2_to_autotune_bucket(m2: int) -> int:
    for bucket in _MEXT_AUTOTUNE_M2_BUCKETS:
        if int(m2) <= bucket:
            return bucket
    return _MEXT_AUTOTUNE_M2_BUCKETS[-1]


def _log_mext_tactic_once(
    *,
    path: str,
    tactic: int,
    effective_tactic: int | None = None,
    encoded_tactic: int | None,
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    x_sf_block: torch.Tensor,
    k_base: int,
) -> None:
    if not _MEXT_TACTIC_LOG_PATH:
        return

    global _MEXT_TACTIC_LOG_COUNT
    if _MEXT_TACTIC_LOG_COUNT >= _MEXT_TACTIC_LOG_LIMIT:
        return
    _MEXT_TACTIC_LOG_COUNT += 1

    row = {
        "event": "mext_tactic_selected",
        "path": path,
        "tactic": int(tactic),
        "effective_tactic": (
            None if effective_tactic is None else int(effective_tactic)
        ),
        "encoded_tactic": None if encoded_tactic is None else int(encoded_tactic),
        "x_fp4_shape": list(x_fp4.shape),
        "weight_shape": list(weight.shape),
        "x_sf_block_shape": list(x_sf_block.shape),
        "m2": int(x_fp4.shape[0]),
        "n": int(weight.shape[0]),
        "k_base": int(k_base),
        "m2_bucket": _map_m2_to_autotune_bucket(int(x_fp4.shape[0])),
    }
    with open(_MEXT_TACTIC_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _run_mext_fold_gemm(
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    x_sf_block: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    output_dtype: torch.dtype,
    tactic: int | None,
) -> torch.Tensor:
    if tactic is None:
        return nvfp4.cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tile8_thread_vec_store_clean(
            x_fp4,
            weight,
            x_sf_block,
            weight_scale_base,
            alpha,
            int(k_base),
            output_dtype,
        )
    return nvfp4.cutlass_scaled_fp4_mm_ext_b_prefix_mext_fold_tactic(
        x_fp4,
        weight,
        x_sf_block,
        weight_scale_base,
        alpha,
        int(k_base),
        output_dtype,
        int(tactic),
    )


if _FLASHINFER_AUTOTUNE_AVAILABLE:

    class _MextFoldTunableRunner(_FlashInferTunableRunner):
        """FlashInfer autotuner adapter for the custom MExt fold GEMM."""

        def __init__(self, k_base: int, output_dtype: torch.dtype, path: str):
            self.k_base = int(k_base)
            self.output_dtype_name = str(output_dtype)
            self.output_dtype = output_dtype
            self.path = path

        def get_valid_tactics(self, inputs, profile):  # noqa: D401
            return list(_MEXT_AUTOTUNE_TACTICS)

        def get_cache_key_extras(self, inputs) -> tuple:
            _, weight, _, _, _ = inputs
            # FlashInfer's plain FP4 autotuner keys by full static GEMM shape.
            # MExt has dynamic M2, but N and the physical packed weight width
            # still affect tile choice substantially.  Without these fields,
            # QKV/O/Gate/Down projections with the same K can accidentally
            # share a cached tactic.
            return (
                self.path,
                self.k_base,
                self.output_dtype_name,
                int(weight.shape[0]),
                int(weight.shape[1]),
            )

        def forward(self, inputs, tactic: int = -1, do_preparation: bool = False, **kwargs):
            x_fp4, weight, x_sf_block, weight_scale_base, alpha = inputs
            if int(tactic) < 0:
                selected_tactic = (
                    _encode_mext_kernel_tactic(self.path, 6)
                    if self.path == _MEXT_PATH_COL_SWAP
                    else None
                )
            else:
                selected_tactic = _encode_mext_kernel_tactic(self.path, int(tactic))
            return _run_mext_fold_gemm(
                x_fp4,
                weight,
                x_sf_block,
                weight_scale_base,
                alpha,
                self.k_base,
                self.output_dtype,
                selected_tactic,
            )


    def _make_mext_autotune_config():
        return _FlashInferTuningConfig(
            dynamic_tensor_specs=(
                _FlashInferDynamicTensorSpec(
                    input_idx=(0,),
                    dim_idx=(0,),
                    gen_tuning_buckets=_MEXT_AUTOTUNE_M2_BUCKETS,
                    map_to_tuning_buckets=_map_m2_to_autotune_bucket,
                ),
            ),
            constraint_specs=(
                _FlashInferConstraintSpec(
                    input_idx=2,
                    dim_idx=0,
                    infer_shape=lambda shapes: _round_up(shapes[0][0], 128),
                ),
            ),
            use_cuda_graph=False,
        )


    _MEXT_AUTOTUNE_CONFIG = _make_mext_autotune_config()
    _MEXT_AUTOTUNE_RUNNERS: dict[tuple[int, str, str], _MextFoldTunableRunner] = {}
else:
    _MEXT_AUTOTUNE_CONFIG = None
    _MEXT_AUTOTUNE_RUNNERS = {}


def _run_mext_fold_gemm_flashinfer_autotuned(
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    x_sf_block: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    output_dtype: torch.dtype,
    path: str,
) -> torch.Tensor:
    if not _FLASHINFER_AUTOTUNE_AVAILABLE or _MEXT_AUTOTUNE_CONFIG is None:
        fallback_tactic = (
            _encode_mext_kernel_tactic(path, 6)
            if path == _MEXT_PATH_COL_SWAP
            else None
        )
        return _run_mext_fold_gemm(
            x_fp4,
            weight,
            x_sf_block,
            weight_scale_base,
            alpha,
            int(k_base),
            output_dtype,
            fallback_tactic,
        )

    runner_key = (int(k_base), str(output_dtype), path)
    runner = _MEXT_AUTOTUNE_RUNNERS.get(runner_key)
    if runner is None:
        runner = _MextFoldTunableRunner(int(k_base), output_dtype, path)
        _MEXT_AUTOTUNE_RUNNERS[runner_key] = runner

    inputs = [x_fp4, weight, x_sf_block, weight_scale_base, alpha]
    selected_runner, tactic = _FlashInferAutoTuner.get().choose_one(
        "mext_fold_fp4_gemm",
        [runner],
        _MEXT_AUTOTUNE_CONFIG,
        inputs,
    )
    effective_tactic = int(tactic)
    if effective_tactic < 0:
        # FlashInfer returns -1 when it asks the single runner to use its
        # default. For the row-pair col-swap path our runner default is the
        # previously best fixed tactic 6 (encoded as 22 for the C++ ABI).
        effective_tactic = 6 if path == _MEXT_PATH_COL_SWAP else -1
    encoded_tactic = (
        None
        if effective_tactic < 0
        else _encode_mext_kernel_tactic(path, effective_tactic)
    )
    _log_mext_tactic_once(
        path=path,
        tactic=int(tactic),
        effective_tactic=None if effective_tactic < 0 else effective_tactic,
        encoded_tactic=encoded_tactic,
        x_fp4=x_fp4,
        weight=weight,
        x_sf_block=x_sf_block,
        k_base=int(k_base),
    )
    return selected_runner(inputs, tactic=tactic)


@torch.library.custom_op(
    "residue::nvfp4_linear",
    mutates_args=(),
    device_types="cuda",
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
) -> torch.Tensor:
    """Single opaque op covering both residue NVFP4 paths.

    Dispatch rule (computed inside the opaque body, so each cudagraph capture
    sees the real M of its capture size):

        use_fold = fold_eligible AND (_FORCE_FOLD OR x.shape[0] <= _FOLD_MAX_M)

      • use_fold → scaled_fp4_quant_mext_r1 + fold-tile8 GEMM (k_base prefix)
      • else    → scaled_fp4_quant_with_mask + cutlass_scaled_fp4_mm (full K_ext)

    The two paths are numerically equivalent; only kernel layout differs.

    `_FORCE_FOLD` (env: `RESIDUE_FORCE_DECODE_MEXT=1`) skips the M threshold,
    so every fold-eligible layer takes fold regardless of M — used by the
    decode-only server in PD-disaggregated deployments to lock the captured
    kernel sequence to a single path.

    Args:
        x: (M, K) bfloat16 activation tensor.
        weight: (N, K_extended_packed) FP4-packed weight.
        input_global_scale_inv: scalar reciprocal of the input global scale.
        weight_scale: full-K block scale for `weight` (used by legacy path).
        weight_scale_base: base-K block scale (used by fold path; covers only
            the first k_base columns of `weight`). Caller passes the same as
            `weight_scale` if the layer has no separate base-scale.
        channel_mask: salient-channel bitmask (used by legacy path quant).
            Caller passes a 1-byte placeholder if the layer has no mask.
        alpha: per-tensor output scale.
        k_base: effective K dim for the fold path GEMM.
        num_salient: salient-channel count for the legacy path quant.
        weights_padding_cols: cutlass weight col-padding (legacy path needs to
            pad activation to match).
        output_size: output feature dim, used by legacy path to slice off any
            column padding the cutlass kernel produced.
        fold_eligible: per-layer constant — True iff the layer has the mext
            fold layout prepared (`_decode_mext_layout_enabled`, plus a valid
            base-K scale and k_base). The op decides at runtime whether to
            actually take the fold branch based on actual M.
        use_swap: swap-residue layout flag (legacy path only).

    Returns:
        (M, N) bfloat16 output. Caller adds bias and applies any final
        view/reshape outside this op.
    """
    original_m = x.shape[0]
    K_base = x.shape[-1]
    output_n = weight.shape[0]

    if original_m == 0:
        return x.new_empty((0, output_n))

    output_dtype = x.dtype

    # Runtime M-aware dispatch — happens *every* op invocation because the
    # function body is opaque to torch.compile (custom_op). Each captured
    # cudagraph at a specific M records the kernels for that branch only.
    # `_FORCE_FOLD` (env: RESIDUE_FORCE_DECODE_MEXT=1) bypasses the M threshold
    # — set this on decode-only servers in a PD-disaggregated deployment.
    # The fold-vs-plain crossover sits past m_tok=128 on SM10x; SM120 keeps its
    # env default.
    fold_max_m = 128 if _sm10x_fold_arch() is not None else _FOLD_MAX_M
    use_fold = fold_eligible and (_FORCE_FOLD or original_m <= fold_max_m)

    if use_fold:
        # ── mext-r1 quant + fold GEMM ─────────────────────────────────────
        # The quant layout must be chosen before GEMM autotuning because
        # row-pair col-swap tactics consume a different physical M layout than
        # the original tile8 row-wise tactics.  Default autotune currently
        # searches the path-local row-pair col-swap tactic space 0..15, so it quantizes
        # row-pair once and then lets FlashInfer's autotuner choose DP/StreamK.
        # ── SM100/SM103 CuTeDSL fold (B200/B300) ──────────────────────────
        # Row_pair quant, raw SF, full contiguous weight.
        if _sm10x_fold_arch() is not None:
            x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(
                x,
                input_global_scale_inv,
                None,
                use_swapped_residue=False,
                layout_mode=_LAYOUT_ROW_PAIR,
            )
            out = _run_mext_fold_gemm_sm10x(
                x_fp4,
                weight,
                x_sf,
                weight_scale_base,
                alpha,
                output_dtype,
            )
            return out[:original_m]

        path = _effective_mext_path(int(original_m), int(output_n), int(k_base))
        use_row_pair_layout = path == _MEXT_PATH_COL_SWAP

        m_padded = original_m if use_row_pair_layout else (
            (original_m + _M_PAD_MULTIPLE - 1) // _M_PAD_MULTIPLE
        ) * _M_PAD_MULTIPLE
        m2 = 2 * m_padded
        num_m_tiles = (m2 + 127) // 128
        num_k_tiles = (K_base + 63) // 64

        quant_kwargs = {
            "use_swapped_residue": False,
            "layout_mode": _LAYOUT_ROW_PAIR if use_row_pair_layout else _LAYOUT_TILE8,
        }
        if not use_row_pair_layout:
            quant_kwargs["m_pad_multiple"] = _M_PAD_MULTIPLE
        x_fp4, x_sf_block = nvfp4.scaled_fp4_quant_mext_r1(
            x,
            input_global_scale_inv,
            None,
            **quant_kwargs,
        )
        x_sf_block = x_sf_block.view(torch.float8_e4m3fn).view(
            num_m_tiles * 128,
            num_k_tiles * 4,
        )
        out = _run_mext_fold_gemm_flashinfer_autotuned(
            x_fp4,
            weight,
            x_sf_block,
            weight_scale_base,
            alpha,
            int(k_base),
            output_dtype,
            path,
        )
        return out[:original_m]

    # ── plain NVFP4 bypass (num_salient == 0) ──────────────────────────────
    # Plain layer entered the op but fold path not chosen (either
    # _DECODE_MEXT_ENABLED=0 → fold_eligible=False, or M > _FOLD_MAX_M).
    # The legacy K-ext path below would call scaled_fp4_quant_with_mask which
    # requires a real channel_mask tensor (plain layers pass a 1-byte dummy
    # via _apply_residue_via_op) — that combination crashes in the C++
    # binding. Use the vLLM stock cutlass kernel chain instead — bit-equivalent
    # to CutlassNvFp4LinearKernel.apply_weights.
    if int(num_salient) == 0:
        x_fp4, x_sf = scaled_fp4_quant(
            x,
            input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="cutlass",
        )
        x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, int(weights_padding_cols))
        out = cutlass_scaled_fp4_mm(
            x_fp4,
            weight,
            x_sf,
            weight_scale,
            alpha,
            output_dtype,
        )
        return slice_nvfp4_output(out, int(output_size))

    # ── legacy K-extension residue path (cutlass backend) ──────────────────
    # n_ext = K_base + num_salient (per residue contract); M stays as M.
    # All shape math derived from `x.shape` and the int `num_salient`, never
    # from x_fp4.shape.
    n_ext = K_base + int(num_salient)
    num_m_tiles = (original_m + 127) // 128
    num_k_tiles = (n_ext + 63) // 64

    x_fp4, x_sf_block = nvfp4.scaled_fp4_quant_with_mask(
        x,
        input_global_scale_inv,
        channel_mask,
        int(num_salient),
        use_swapped_residue=use_swap,
    )
    x_sf_block = x_sf_block.view(torch.float8_e4m3fn).view(
        num_m_tiles * 128,
        num_k_tiles * 4,
    )
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, int(weights_padding_cols))
    out = cutlass_scaled_fp4_mm(
        x_fp4,
        weight,
        x_sf_block,
        weight_scale,
        alpha,
        output_dtype,
    )
    return slice_nvfp4_output(out, int(output_size))


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
) -> torch.Tensor:
    """Meta/abstract implementation for shape inference during torch.compile."""
    M = x.shape[0]
    N = int(output_size) if int(output_size) > 0 else weight.shape[0]
    return torch.empty(M, N, dtype=x.dtype, device=x.device)
