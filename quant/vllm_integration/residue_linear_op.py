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

# mext_r1 quant layout selector. The C op defines:
#   0 = concat
#   1 = row_pair
#   2 = tile8  ← required by the clean-vec-store fold GEMM
_LAYOUT_ROW_PAIR = 1
_LAYOUT_TILE8 = 2

# Pad input rows to a multiple of 8 so the tile8 fold kernel sees a clean
# 2x-doubled M dimension. The original M is preserved via output slicing.
_M_PAD_MULTIPLE = 8

# ── how a layer represents its residue ─────────────────────────────────────
# Derived, never passed: the op already receives `num_salient` and
# `is_mext_r1` separately, so the information is unambiguous AT THE BOUNDARY.
# What was ambiguous was the CODE, which branched on `num_salient == 0` — a
# value that means two opposite things:
#
#   "plain layer, no residue at all"                        -> RESIDUE_NONE
#   "mext_r1 layer, the M-extension layout IS the residue,
#    so there are no separate salient channels"             -> RESIDUE_MEXT_R1
#
# That collision produced one silent-wrong-numerics bug (4083dee: above the
# fold's M threshold a mext_r1 layer took the plain bypass and returned
# bit-identical-to-plain output), and it has since misled two readers who
# stopped at the `== 0` test. Branch on the name, not the number.
RESIDUE_NONE = "none"
RESIDUE_K_EXT = "k_ext"
RESIDUE_MEXT_R1 = "mext_r1"


def residue_kind_of(num_salient: int, is_mext_r1: bool) -> str:
    """Which residue representation this layer uses.

    `is_mext_r1` wins: a mext_r1 layer legitimately reports num_salient == 0
    because every channel is salient and the residue lives in the doubled
    token rows, not in a channel subset.
    """
    if is_mext_r1:
        return RESIDUE_MEXT_R1
    return RESIDUE_NONE if int(num_salient) == 0 else RESIDUE_K_EXT


























_FOLD_MAX_M_DEFAULT_SM10X = 128
_FOLD_MAX_M_DEFAULT_OTHER = 64
_FOLD_MAX_M_ENV = os.environ.get("RESIDUE_DECODE_MEXT_MAX_M")


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


# Back-compat alias for the non-sm10x default. Prefer fold_max_m_for().
_FOLD_MAX_M = fold_max_m_for(None)

# Override knob for PD-disaggregated deployments: when set, every fold-eligible
# layer takes the fold branch regardless of M. Intended for the decode-only
# server in a prefill/decode split, where every forward is small-M anyway but
# we want a uniform captured kernel sequence (no M-threshold-dependent branch
# inside the captured graph set). At-call cost is 0 — only the dispatch
# expression changes; pack8 fold kernel works at arbitrary M via the K-split
# block/grid in invokeFP4QuantizationMExtR1.
_FORCE_FOLD = os.environ.get("RESIDUE_FORCE_DECODE_MEXT", "0") == "1"

# ── SM100/SM103 (B200/B300) CuTeDSL fold path ──────────────────────────────
# On Blackwell datacenter GPUs the SM120 C++ fold GEMM does not exist; use the
# CuTeDSL sm100/sm103 fork (nvfp4.cute_fold). It consumes the row_pair mext_r1
# quant layout directly (no m_pad, no tile8) and, for pure mext_r1 layers,
# reads the full contiguous packed weight (k_base == full K, no prefix slice).
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








_SM120_FOLD_OK: bool | None = None


def _sm120_fold_available() -> bool:
    """True iff this GPU is GeForce Blackwell and the CuTeDSL fold imports.

    This used to be `_sm120_fold_available()`, gated on
    RESIDUE_SM120_CUTEDSL_FOLD. The flag existed to fall back to the C++ CUTLASS
    fold; that kernel is gone, so there is nothing to fall back TO and a flag
    that only turns the fold off would just break the layer. What remains is a
    capability probe. Probed once; any failure -> legacy K-extension path.
    """
    global _SM120_FOLD_OK
    if _SM120_FOLD_OK is not None:
        return _SM120_FOLD_OK
    try:
        major, minor = torch.cuda.get_device_capability()
        from nvfp4.cute_fold_sm120 import supports_capability

        _SM120_FOLD_OK = bool(supports_capability(int(major), int(minor)))
    except Exception:
        _SM120_FOLD_OK = False
    return _SM120_FOLD_OK


# (n_w, k_base, k_ext_in_fp4_elements, is_mext_r1)
_SM120_FOLD_SHAPES: set[tuple[int, int, int, bool]] = set()


# Decode sizes the fold can serve. vLLM captures a cudagraph at each of these
# (plus intermediates) and the fold also runs eagerly in between, so all of them
# must be compiled before capture starts.
# vLLM's default cudagraph_capture_sizes for this config, in full. It runs to
# 512, not 128: a mext_r1 layer folds at every one of them, so stopping the list
# early would leave the prefill-sized captures to JIT.
_FOLD_WARMUP_M = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
                  112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
                  208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336,
                  352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512)


def register_sm120_fold_shape(n_w: int, k_base: int, k_ext: int,
                             is_mext_r1: bool = False) -> None:
    """Record a fold-eligible layer's shape AND pre-compile its kernels.

    Compiling here -- at the end of the layer's process_weights_after_loading --
    rather than from a vLLM warmup hook is deliberate. Two hook attempts failed
    silently: `kernel_warmup.flashinfer_autotune` is only wrapped for
    resolved_version == "mext_r1" (so every extended_k checkpoint skipped it),
    and `GPUModelRunner.capture_model` installs fine but is never reached --
    this vLLM has a second capture trigger in `profile_cudagraph_memory`. Both
    times the only symptom was cudagraph capture taking 97 s instead of 5 s,
    because a missed warmup does not fail, it just moves ~1.3 s of JIT into
    every capture step.

    PWAL always runs, always before capture, and the layer knows its own shape
    there. `compile_fold_gemm_sm120` is lru_cached, so the 100+ layers sharing
    four distinct shapes pay for four.
    """
    key = (int(n_w), int(k_base), int(k_ext), bool(is_mext_r1))
    if key in _SM120_FOLD_SHAPES:
        return
    _SM120_FOLD_SHAPES.add(key)
    # sm100: the kloop kernels are symbolic-shape, so ONE compile per tile
    # config covers every layer -- trigger it from the first mext_r1
    # registration and the whitelist is complete for the whole model.
    if key[3] and _sm10x_fold_arch() is not None and _mext_prefill_two_gemm():
        try:
            from .mext_prefill_tuner import (_PRECOMPILED,
                                             precompile_kloop_sm100)

            if not any(k0 == "sm100" for k0, *_ in _PRECOMPILED):
                n = precompile_kloop_sm100()
                if n:
                    print(f"[residue] sm100 kloop pre-compiled x {n} "
                          f"(tile, cluster) configs", flush=True)
        except Exception as e:                            # noqa: BLE001
            print(f"[residue] sm100 kloop precompile skipped: "
                  f"{type(e).__name__}: {str(e)[:70]}", flush=True)
    if not _sm120_fold_available():
        return
    try:
        from nvfp4.cute_fold_sm120 import warmup_fold_default

        major, minor = torch.cuda.get_device_capability()
        # Decode band only. The fold-at-prefill route was retired;
        # pre-compiling the full 51-size list for mext_r1 was not just waste:
        # 4 shapes x 51 sizes of graphed CuTeDSL kernels is what pushed a
        # multi-arm sweep into a cudaGraph-replay segfault.
        ms = [m for m in _FOLD_WARMUP_M
              if m <= fold_max_m_for(None)]          # sm120-only path
        warmup_fold_default(
            major, minor,
            [(m, key[0], key[1], torch.bfloat16, key[2]) for m in ms])
        # key[2] is the row stride in FP4 elements: == k_base for a contiguous
        # weight (warmup then compiles the a_ld=None variant), > k_base for an
        # ext-K prefix view. The kernel takes the token count at RUNTIME,
        # so the ms list only selects tiles -- the compile count
        # is the number of distinct tiles, not len(ms).
        from nvfp4.cute_fold_sm120.tactics import select_tile

        tiles = sorted({select_tile(m, key[0], key[1]) for m in ms})
        print(f"[residue] sm120 fold pre-compiled n_w={key[0]} k={key[1]} "
              f"k_ext={key[2]} mext_r1={key[3]} tiles={tiles} (dynamic m; "
              f"{len(_SM120_FOLD_SHAPES)} shapes so far)", flush=True)
        if key[3] and not mext_prefill_folds:
            # mext_r1 prefill routes to the tuned two_gemm/k_loop dispatch;
            # pre-compile the kloop bucket grid HERE so get_valid_tactics can
            # offer it at serving without ever compiling there.
            try:
                from .mext_prefill_tuner import precompile_kloop_sm120

                n_kloop = precompile_kloop_sm120(key[0], key[1])
                if n_kloop:
                    print(f"[residue] kloop pre-compiled n_w={key[0]} "
                          f"k={key[1]} x {n_kloop} tile kernels (dynamic m)",
                          flush=True)
            except Exception as e:                        # noqa: BLE001
                print(f"[residue] kloop precompile skipped for {key}: "
                      f"{type(e).__name__}: {str(e)[:70]}", flush=True)
    except Exception as e:                                # noqa: BLE001
        print(f"[residue] sm120 fold pre-compile failed for {key}: "
              f"{type(e).__name__}: {str(e)[:70]} -- kernels will JIT during "
              "cudagraph capture", flush=True)


def warmup_sm120_cutedsl_fold(m_tokens, out_dtype=torch.bfloat16) -> int:
    """Pre-compile every (registered shape x m_tok) the server may hit.

    Call at engine init, OUTSIDE cudagraph capture. Returns the number of
    (shape, m) pairs compiled; 0 when the path is off or nothing registered.
    Best-effort: a compile failure must not take the engine down, since the
    layer can still fall back to the legacy path.
    """
    # Say WHY nothing happened. A silent `return 0` here is how a broken
    # warmup hides: the engine still starts, the fold still works, and the only
    # symptom is cudagraph capture quietly taking 97s instead of 5s.
    if not _sm120_fold_available():
        print("[residue] sm120 fold warmup: CuTeDSL path disabled, nothing to "
              "pre-compile", flush=True)
        return 0
    if not _SM120_FOLD_SHAPES:
        print("[residue] sm120 fold warmup: NO shapes registered -- kernels "
              "will JIT during cudagraph capture", flush=True)
        return 0
    try:
        from nvfp4.cute_fold_sm120 import warmup_fold_default
    except Exception as e:                                # noqa: BLE001
        print(f"[residue] sm120 fold warmup unavailable: {e}", flush=True)
        return 0
    major, minor = torch.cuda.get_device_capability()
    m_tokens = list(m_tokens)          # may be a generator; used twice below
    shapes = [(int(m), n_w, k_base, out_dtype, k_ext)
              for (n_w, k_base, k_ext, _is_mext_r1) in sorted(_SM120_FOLD_SHAPES)
              for m in m_tokens]
    try:
        warmup_fold_default(major, minor, shapes)
    except Exception as e:                                # noqa: BLE001
        print(f"[residue] sm120 fold warmup failed ({type(e).__name__}: "
              f"{str(e)[:80]}); kernels will JIT on first use", flush=True)
        return 0
    print(f"[residue] sm120 CuTeDSL fold warmup: {len(shapes)} kernels "
          f"({len(_SM120_FOLD_SHAPES)} shapes x {len(m_tokens)} sizes)",
          flush=True)
    return len(shapes)


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





















_MEXT_PREFILL_PATH = os.environ.get("RESIDUE_MEXT_PREFILL_PATH", "auto")


def _mext_prefill_two_gemm() -> bool:
    """True iff mext_r1 layers above the fold band LEAVE the direct-fold path.

    (Historical name; the destination is the tuned dispatch.) Since the sm100
    kloop graduated its B200 gate (7/9 vs fold,
    numerics equal to every digit), auto enters the tuned dispatch on BOTH
    arches -- on sm10x the tuner's runners[0]/fallback IS the fold, so any
    cache miss serves exactly the old behavior.
    """
    # "fold" was retired as a prefill route (env contract now
    # rejects it): the tuned dispatch beat or tied it everywhere that
    # matters, and removing it deleted the per-m-JIT hazard class outright.
    # The decode band still folds -- that is a different dispatch term.
    return True


def _run_mext_r1_two_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """mext_r1 prefill: one backend GEMM at 2M rows + pair-sum epilogue.

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


def _run_mext_fold_gemm_sm10x(
    x_fp4_rowpair: torch.Tensor,
    weight: torch.Tensor,
    x_sf: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    family: tuple[str, str] | None = None,
) -> torch.Tensor:
    """CuTeDSL sm100/sm103 fold GEMM for a pure mext_r1 layer.

    `family` declares which layout the activation is in -- None means the
    row_pair default. It cannot be inferred from the tensor (both layouts are
    [2M, K/2]), so a tile8 caller must say so.

    x_fp4_rowpair: [2M, K/2] uint8 (scaled_fp4_quant_mext_r1 row_pair output).
    weight:        [N, K/2] uint8 packed FP4 (full contiguous, k_base == K).
    x_sf:          raw swizzled 128x4 FP8 scale from the mext_r1 quant.
    weight_scale_base: swizzled base-K weight scale (== weight_scale for mext_r1).
    Returns [M, N] in output_dtype.
    """
    from nvfp4.fold import run_fold

    if not _SM10X_FOLD_FIRED[0]:
        _SM10X_FOLD_FIRED[0] = True
        print(
            f"[residue] SM10x MExt fold kernel FIRED: act={tuple(x_fp4_rowpair.shape)} "
            f"weight={tuple(weight.shape)} arch={_sm10x_fold_arch()}",
            flush=True,
        )

    major, minor = torch.cuda.get_device_capability()
    return run_fold(
        int(major),
        int(minor),
        weight,
        x_fp4_rowpair,
        weight_scale_base,
        x_sf.view(torch.float8_e4m3fn),
        alpha,
        output_dtype,
        family=family,
    )


def _tune_mext_families_sm10x(
    x: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Warmup-only: measure N-ext vs M-ext for this layer's shape and m-bucket.

    Cross-family selection cannot live inside run_fold, because by then the
    activation has already been quantised into ONE layout and the families do
    not accept the same one. Here both are producible from the same `x`, so
    this is where the comparison belongs.

    Returns the winner's output so the warmup forward still produces a real
    result. Never call inside a cudagraph capture -- it measures (sync + JIT).

    """
    from nvfp4.cute_fold import tactics as T
    from nvfp4.fold import candidate_families

    x_rp, sf_rp = nvfp4.scaled_fp4_quant_mext_r1(
        x, input_global_scale_inv, layout_mode=_LAYOUT_ROW_PAIR,
    )
    x_t8, sf_t8 = nvfp4.scaled_fp4_quant_mext_r1(
        x, input_global_scale_inv, layout_mode=_LAYOUT_TILE8,
        m_pad_multiple=_M_PAD_MULTIPLE,
    )
    raw = {"row_pair": (x_rp, sf_rp), "tile8": (x_t8, sf_t8)}
    # The tuner calls the kernel directly, so it needs the fp8 SF view that
    # _run_mext_fold_gemm_sm10x would otherwise apply for us.
    acts = {k: (a, sf.view(torch.float8_e4m3fn)) for k, (a, sf) in raw.items()}

    major, minor = torch.cuda.get_device_capability()
    arch = T.kernel_arch_for_capability(int(major), int(minor))
    family, _tactic = T.tune_families(
        acts, weight, weight_scale_base, alpha, output_dtype,
        kernel_arch=arch, families=candidate_families(),
    )

    act_fp4, act_sf = raw[T.FAMILY_ACT_LAYOUT[family]]
    return _run_mext_fold_gemm_sm10x(
        act_fp4, weight, act_sf, weight_scale_base, alpha, output_dtype,
        family=family,
    )


# CuTeDSL SM120 fold (nvfp4.cute_fold_sm120) -- OPT-IN alternative to the C++
# SM120 fold op. Same row_pair mext_r1 quant layout as the SM10x fork; the
# kernel is validated bit-exact against the C++ op (52 fuzz cases + a
# four-corner loop vs an independent plain-NVFP4 twin). Enable with
# RESIDUE_SM120_CUTEDSL_FOLD=1; default OFF keeps the C++ path byte-identical.
_SM120_DSL_ENABLED: bool | None = None
_SM120_DSL_FIRED = [False]


def _run_mext_fold_gemm_sm120_cutedsl(
    x: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    original_m: int,
    output_n: int,
    k_base: int,
) -> torch.Tensor:
    'SM120 CuTeDSL fold: row_pair mext_r1 quant + compact fold GEMM.'
    from nvfp4.cute_fold_sm120 import act_pad_multiple
    from nvfp4.fold import run_fold

    pad_mult = act_pad_multiple(int(original_m), int(output_n), int(k_base))
    m_padded = ((int(original_m) + pad_mult - 1) // pad_mult) * pad_mult
    x_in = x
    if m_padded != x.shape[0]:
        x_in = torch.zeros(m_padded, x.shape[1], dtype=x.dtype, device=x.device)
        x_in[: x.shape[0]] = x

    x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(
        x_in,
        input_global_scale_inv,
        layout_mode=_LAYOUT_ROW_PAIR,
    )
    if not _SM120_DSL_FIRED[0]:
        _SM120_DSL_FIRED[0] = True
        print(
            f"[residue] SM120 CuTeDSL MExt fold FIRED: act={tuple(x_fp4.shape)} "
            f"weight={tuple(weight.shape)} m={original_m} pad->{m_padded}",
            flush=True,
        )
    major, minor = torch.cuda.get_device_capability()
    return run_fold(
        int(major),
        int(minor),
        weight,
        x_fp4,
        weight_scale_base,
        x_sf.view(torch.float8_e4m3fn),
        alpha,
        output_dtype,
        m_tok_real=int(original_m),
    )


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
    is_mext_r1: bool,
) -> torch.Tensor:
    """Single opaque op covering both residue NVFP4 paths.

    Dispatch rule (computed inside the opaque body, so each cudagraph capture
    sees the real M of its capture size):

        use_fold = fold_eligible AND (is_mext_r1 OR _FORCE_FOLD
                                      OR x.shape[0] <= fold_max_m_for(arch))

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

    
    
    
    
    
    
    
    
    fold_max_m = fold_max_m_for(_sm10x_fold_arch())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    use_fold = fold_eligible and (
        is_mext_r1 or _FORCE_FOLD or original_m <= fold_max_m)

    if use_fold:
        
        
        
        
        if (is_mext_r1 and original_m > fold_max_m
                and _mext_prefill_two_gemm()):
            _w_view = _base_k_weight_view(weight, k_base)
            if _MEXT_PREFILL_PATH in ("auto", "k_loop"):
                # FlashInfer-tuned two_gemm-vs-k_loop selection, per (shape,
                # m-bucket). runners[0]=two_gemm is FlashInfer's designated
                # fallback on any cache miss, and KloopRunner only reports
                # precompiled tiles outside the tuning window -- so this
                # branch can never JIT while serving. Any tuner-layer error
                # degrades to plain two_gemm rather than breaking the op.
                try:
                    from .mext_prefill_tuner import tuned_mext_prefill

                    return tuned_mext_prefill(
                        x, _w_view, input_global_scale_inv,
                        weight_scale_base, alpha, output_dtype,
                        force_kloop=(_MEXT_PREFILL_PATH == "k_loop"),
                    )
                except Exception:                         # noqa: BLE001
                    pass
            return _run_mext_r1_two_gemm(
                x, _w_view, input_global_scale_inv,
                weight_scale_base, alpha, output_dtype,
            )

        # ── mext-r1 quant + fold GEMM ─────────────────────────────────────
        # The quant layout must be chosen before GEMM autotuning because
        # row-pair col-swap tactics consume a different physical M layout than
        # the original tile8 row-wise tactics.  Default autotune currently
        # searches the path-local row-pair col-swap tactic space 0..15, so it quantizes
        # row-pair once and then lets FlashInfer's autotuner choose DP/StreamK.
        # ── SM100/SM103 CuTeDSL fold (B200/B300) ──────────────────────────
        # Blackwell datacenter GPUs use the CuTeDSL sm100/sm103 fork instead
        # of the SM120 C++ fold GEMM (which doesn't exist there). Row_pair
        # quant, raw SF, full contiguous weight — validated end-to-end
        # (5/5). No m_pad / tile8 needed.
        if _sm10x_fold_arch() is not None:
            
            
            
            
            
            from nvfp4.fold import (
                ENABLE_MEXT_FAMILY,
                is_tuning_mode,
                select_fold_plan,
            )

            major, minor = torch.cuda.get_device_capability()

            if ENABLE_MEXT_FAMILY and is_tuning_mode():
                # Warmup window: this is the ONLY place both layouts of the
                # same activation exist, so cross-family measurement has to
                # happen here rather than inside run_fold.
                return _tune_mext_families_sm10x(
                    x,
                    input_global_scale_inv,
                    weight,
                    weight_scale_base,
                    alpha,
                    output_dtype,
                )[:original_m]

            # family AND layout together: a layout does not determine a family, so
            # deriving one from the other can dispatch the wrong tuned tactic.
            family, layout = select_fold_plan(
                int(major), int(minor), int(output_n),
                int(weight.shape[1]) * 2, int(original_m))

            if layout == "tile8":
                # tile8 pads M to a multiple of 8 inside the quant op (same
                # contract the SM120 C++ fold uses); out[:original_m] trims.
                x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(
                    x,
                    input_global_scale_inv,
                    layout_mode=_LAYOUT_TILE8,
                    m_pad_multiple=_M_PAD_MULTIPLE,
                )
            else:
                x_fp4, x_sf = nvfp4.scaled_fp4_quant_mext_r1(
                    x,
                    input_global_scale_inv,
                    layout_mode=_LAYOUT_ROW_PAIR,
                )
            out = _run_mext_fold_gemm_sm10x(
                x_fp4,
                _base_k_weight_view(weight, k_base),
                x_sf,
                weight_scale_base,
                alpha,
                output_dtype,
                family=family,
            )
            return out[:original_m]

        
        
        
        
        
        
        
        
        return _run_mext_fold_gemm_sm120_cutedsl(
            x,
            input_global_scale_inv,
            _base_k_weight_view(weight, k_base),
            weight_scale_base,
            alpha,
            output_dtype,
            int(original_m),
            int(output_n),
            int(k_base),
        )

    # ── no residue at all: stock vLLM NVFP4 ────────────────────────────────
    # Reached when a layer that carries no residue entered the op anyway and
    # the fold was not chosen (RESIDUE_ENABLE_DECODE_MEXT=0 -> fold_eligible
    # False, or M > fold_max_m_for(arch)).
    #
    # The legacy K-ext path below would call scaled_fp4_quant_with_mask, which
    # needs a real channel_mask; a plain layer passes a 1-byte dummy via
    # _apply_residue_via_op and that combination crashes in the C++ binding.
    # The stock cutlass chain is bit-equivalent to
    # CutlassNvFp4LinearKernel.apply_weights.
    kind = residue_kind_of(num_salient, is_mext_r1)

    if kind == RESIDUE_MEXT_R1:
        # Unreachable today: use_fold is forced true for mext_r1, so we
        # returned above. Assert it rather than trust that, because the
        # protection is indirect -- fold_eligible can still be False on an
        # unsupported arch or with RESIDUE_DISABLE_SM10X_FOLD=1, and then this
        # would fall through to the plain path below and compute NO residue.
        # That is not a degraded answer, it is the wrong one, and there is no
        # correct fallback: a mext_r1 layer has no K-extended weight to use.
        # Failing here is how that stays a crash instead of a quiet -Npp.
        raise RuntimeError(
            "mext_r1 layer reached the non-fold path (fold_eligible="
            f"{bool(fold_eligible)}, M={int(x.shape[0])}). The residue cannot "
            "be applied without the fold kernel and there is no K-extended "
            "weight to fall back to. Check the fold arch probe / "
            "RESIDUE_DISABLE_SM10X_FOLD rather than letting this serve plain "
            "NVFP4."
        )

    if kind == RESIDUE_NONE:
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
    is_mext_r1: bool,
) -> torch.Tensor:
    """Meta/abstract implementation for shape inference during torch.compile."""
    M = x.shape[0]
    N = int(output_size) if int(output_size) > 0 else weight.shape[0]
    return torch.empty(M, N, dtype=x.dtype, device=x.device)
