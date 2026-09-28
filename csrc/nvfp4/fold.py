# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'One entry for the MExt fold GEMM, whichever Blackwell this is.'

from __future__ import annotations

import contextlib
import os
import sys
import warnings

import torch

_DEFAULT_PATH_WARNED = False












_SM10X_HOST = "nvfp4.cute_fold.host"
_SM120_HOST = "nvfp4.cute_fold_sm120.host"


def compile_stats() -> dict[str, dict | None]:
    """Fold kernels JIT-compiled by this process so far, per arch.

    Side-effect free: never imports a module that is not already loaded.
    `None` for an arch means its host module has not been imported at all,
    which is the normal state for the arch this GPU is not.
    """
    out: dict[str, dict | None] = {"sm10x": None, "sm120": None}

    for arch, module_name in (("sm10x", _SM10X_HOST), ("sm120", _SM120_HOST)):
        mod = sys.modules.get(module_name)
        cache = getattr(mod, "_KERNEL_CACHE", None)
        if cache is not None:
            out[arch] = {"compiled": len(cache), "keys": list(cache)}

    return out


def warmup(major: int, minor: int, shapes=None) -> int:
    """Pre-compile whatever this GPU's fold path can need. Returns the count.

    One contract, two implementations, because the arches JIT at different
    granularity and pretending otherwise would break one of them:

      sm10x   the compile key has no shape and no m_tok, so the space is a
              small fixed set of tactics. `shapes` is IGNORED.
      sm120   JITs per (n_w, k, m_tok, tile, a_ld), so `shapes` is REQUIRED --
              without it there is nothing to pre-compile and this returns 0.

    Call at engine init, before cudagraph capture. Wrap capture in
    observe_compiles() to find out whether this actually covered it; the two
    exist as a pair, and the detector is what turns a missed spec from "the
    server is mysteriously slow" into a line naming the key.
    """
    if major == 12:
        if not shapes:
            return 0
        from .cute_fold_sm120 import warmup_fold_default as _sm120_warmup

        _sm120_warmup(major, minor, shapes)
        return len(shapes)

    from .cute_fold import tactics as _t

    # The gate lives here, not in tactics: which families are reachable is a
    # dispatch policy question, and candidate_families() is already its answer.
    # Warming a family the dispatcher can never pick would just burn startup.
    fams = [f for f in candidate_families() if f != PRODUCTION_FAMILY]
    return _t.warmup_fold_default(major, minor, families=fams)


@contextlib.contextmanager
def observe_compiles(where: str, *, warn: bool = True):
    """Report any fold kernel JIT-compiled inside this block.

    Wrap the regions that must already be warm -- cudagraph capture above all,
    where a JIT is both slow and, for the sm10x path, a sync inside capture.
    A clean block prints nothing; the whole point is that the silent case
    stays silent and the broken case does not.
    """
    before = compile_stats()
    try:
        yield
    finally:
        after = compile_stats()
        for arch in ("sm10x", "sm120"):
            b, a = before.get(arch), after.get(arch)
            if not a:
                continue
            n = a["compiled"] - (b["compiled"] if b else 0)
            if n <= 0:
                continue
            detail = ""
            if a["keys"] is not None:
                fresh = a["keys"][len(b["keys"]) if b else 0:]
                detail = "".join(f"\n[residue]     {k}" for k in fresh[:8])
                if len(fresh) > 8:
                    detail += f"\n[residue]     ... and {len(fresh) - 8} more"
            if warn:
                print(
                    f"[residue] WARMUP UNDER-COVERAGE: {n} {arch} fold "
                    f"kernel(s) JIT-compiled during {where}. Each costs ~1.3 s "
                    f"and blocks the caller. The pre-compile missed:{detail}",
                    flush=True,
                )

# Mirrors FlashInfer's AutoTuner.is_tuning_mode: measuring is legal only in a
# warmup window, never in the serving path, because run_fold_autotuned times
# candidates (sync + JIT) and would deadlock or JIT inside a cudagraph capture.
# Outside the window run_fold only READS the cache.
_TUNING = False


@contextlib.contextmanager
def tuning_mode():
    """Let run_fold() measure and cache. Warmup only, outside capture."""
    global _TUNING
    prev, _TUNING = _TUNING, True
    try:
        yield
    finally:
        _TUNING = prev


def is_tuning_mode() -> bool:
    return _TUNING


# The layout production quantises to (scaled_fp4_quant_mext_r1 row_pair) and
# therefore the tactic family its shapes must be tuned under by default.
PRODUCTION_FAMILY = ("row_pair", "std")












ENABLE_MEXT_FAMILY = os.environ.get("RESIDUE_FOLD_MEXT", "0") == "1"












#
# A6 is legal only where 2*m_tok >= tile_M, i.e. m_tok >= 64; _rank_tactics
# returns nothing below that, so the tuner simply never picks it there.


def candidate_families() -> list[tuple[str, str]]:
    """Families the serving path may select among, given the env gate.

    Both M-ext families ride the same gate: they share the tile8 activation
    layout, and the thing still unproven is that layout's numerics in
    production, not the store design layered on top of it.
    """
    fams = [PRODUCTION_FAMILY]
    if ENABLE_MEXT_FAMILY:
        fams.append(("tile8", "std"))   # A4
        fams.append(("tile8", "pi"))    # A6
    return fams


def select_fold_plan(
    major: int, minor: int, n_w: int, k: int, m_tok: int
) -> tuple[tuple[str, str] | None, str]:
    """(family, activation layout) for this shape and m-bucket.

    Both in ONE call on purpose. The layout does not determine the family --
    "tile8" is shared by A4 ("tile8","std") and A6 ("tile8","pi"), which have
    different store modes and only one of which needs pi_via_layout. A caller
    that quantises by layout and then guesses the family will dispatch A6's
    tuned tactic as A4: wrong values, no crash. Returning the pair makes that
    disagreement unrepresentable.

    family is None when nothing was tuned for this bucket; the layout is then
    "row_pair" and run_fold falls through to its default path, exactly as an
    untuned build always behaved.
    """
    if major != 10 or not ENABLE_MEXT_FAMILY:
        return None, "row_pair"
    from .cute_fold import tactics as _t

    arch = _t.kernel_arch_for_capability(major, minor)
    if arch is None:
        return None, "row_pair"
    hit = _t.best_family_for(arch, n_w, k, m_tok)
    if hit is None:
        return None, "row_pair"
    return hit[0], _t.FAMILY_ACT_LAYOUT[hit[0]]


def act_layout_for(
    major: int, minor: int, n_w: int, k: int, m_tok: int
) -> str:
    """Just the activation layout. Prefer select_fold_plan() in callers that
    also have to dispatch -- see why there."""
    return select_fold_plan(major, minor, n_w, k, m_tok)[1]


def warn_default_path(arch: str, m_tok: int, detail: str = "") -> None:
    """Announce, once per process, that the untuned sm10x table is serving.

    sm10x ONLY. This warning means "a tuned path exists on this arch and you
    are not on it" -- true on sm10x, whose tuner primes at engine init. It
    used to also fire unconditionally on sm120, where the static table is
    itself an offline full-sweep product and a legitimate serving path; that
    was a false alarm, and sm120 now announces its table as information in
    cute_fold_sm120.tactics instead of calling here.

    Once, not per call: this sits in the decode path and the message is a
    build-configuration fact, not a per-request event.
    """
    global _DEFAULT_PATH_WARNED
    if _DEFAULT_PATH_WARNED:
        return
    _DEFAULT_PATH_WARNED = True
    warnings.warn(
        f"[residue] MExt fold is running on the STATIC tactic table (arch={arch}, m_tok={m_tok}{(', ' + detail if detail else '')}). Prime the tuner once at engine initialization outside CUDA Graph capture with cute_fold.run_fold_autotuned(), or set RESIDUE_FOLD_SILENCE_DEFAULT=1 if the fallback is intentional.",
        RuntimeWarning,
        stacklevel=3,
    )


def run_fold(
    major: int,
    minor: int,
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    m_tok_real: int | None = None,
    family: tuple[str, str] | None = None,
) -> torch.Tensor:
    """Fold GEMM on whichever fold kernel this GPU has.

    `m_tok_real` is the unpadded token count; only the sm120 path needs it
    (it pads the activation to the tile boundary and drops the extra rows in
    the epilogue), and it is ignored elsewhere.

    `family` declares which layout `act_fp4_rowpair` is actually in. Leave it
    None and the tuned choice for this bucket is used (row_pair if untuned) --
    but a caller that quantised to tile8 MUST pass the matching family, since
    the layouts are indistinguishable from the tensor's shape alone.
    """
    if major == 12:
        from .cute_fold_sm120 import run_fold_default as _sm120

        return _sm120(major, minor, weight_fp4, act_fp4_rowpair, weight_sf,
                      act_sf, alpha, out_dtype, m_tok_real=m_tok_real)

    from .cute_fold import tactics as _t

    arch = _t.kernel_arch_for_capability(major, minor)
    m_tok = act_fp4_rowpair.shape[0] // 2

    if _TUNING:
        
        
        
        
        return _t.run_fold_autotuned(
            family or PRODUCTION_FAMILY, weight_fp4, act_fp4_rowpair,
            weight_sf, act_sf, alpha, out_dtype, kernel_arch=arch)

    if family is None:
        
        
        hit = _t.best_family_for(
            arch, weight_fp4.shape[0], weight_fp4.shape[1] * 2, m_tok)
        family = PRODUCTION_FAMILY if hit is None else hit[0]

    tuned = _t.autotuned_tactic_for(
        family, arch, weight_fp4.shape[0], weight_fp4.shape[1] * 2, m_tok)
    if tuned is not None:
        mode, tiler, cluster, ab = _t._norm_tactic(tuned)
        from .cute_fold.host import mext_fold_gemm_sm103

        return mext_fold_gemm_sm103(
            weight_fp4, act_fp4_rowpair, weight_sf, act_sf, alpha, out_dtype,
            mma_tiler_mn=tiler, cluster_shape_mn=cluster, store_mode=mode,
            kernel_arch=arch, ab_stage_override=ab,
            pi_via_layout=_t._needs_pi_layout(family))

    # Nothing tuned this shape -- static table, and say so. The static table is
    # N-ext only, so this is correct only for the row_pair layout; a caller
    # that quantised to tile8 without a tuned entry is a wiring bug.
    if family != PRODUCTION_FAMILY:
        raise ValueError(
            f"family {family} was selected but has no tuned tactic for "
            f"n_w={weight_fp4.shape[0]} m_tok={m_tok}; the static fallback "
            f"table is N-ext/row_pair only. Tune this bucket at warmup "
            f"(tactics.tune_families) or quantise to row_pair.")
    return _t.run_fold_default(major, minor, weight_fp4, act_fp4_rowpair,
                               weight_sf, act_sf, alpha, out_dtype)
