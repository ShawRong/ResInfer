# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Tile/kernel selection for the SM100/SM103 MExt fold GEMM.'

from __future__ import annotations

import statistics

import torch

from ..diagnostics import silence_default_notice

# Escape hatch for harnesses that call the fallback on purpose.
_SILENCE_DEFAULT = silence_default_notice()

from .host import mext_fold_gemm_sm103
























_SENTINEL_M = 1 << 30
ARCH_TACTIC_BY_M: dict[str, list[tuple[int, str, tuple[int, int], tuple[int, int]]]] = {
    "sm100": [
        
        
        
        (64, "sm100", (128, 64), (1, 1)),            # decode: N=64 1SM floor
        (_SENTINEL_M, "sm100", (256, 128), (2, 1)),  # m>64: 2SM (safe blanket)
    ],
    "sm103": [
        
        
        
        
        
        
        (64, "sm100", (128, 64), (1, 1)),            # decode: sm100 kernel, N=64 1SM
        (256, "sm103", (256, 128), (2, 1)),          # transition band (mixed winners)
        (_SENTINEL_M, "sm103", (256, 256), (2, 1)),  # m>256: K=96 amortization wins
    ],
}


def kernel_arch_for_capability(major: int, minor: int) -> str | None:
    """Map a CUDA compute capability to the fold kernel arch, or None if the
    fold path is unsupported (caller falls back to unfused)."""
    if major != 10:
        return None
    return "sm103" if minor == 3 else "sm100"


def select_arch_tactic(
    arch: str, m_tok: int
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """(kernel_arch, mma_tiler, cluster) for this arch + token count.

    `arch` must come from kernel_arch_for_capability (KeyError otherwise). The
    last list entry is an unconditional fallback (matches any m_tok, incl. the
    _SENTINEL_M edge), so this never raises for a valid arch.
    """
    entries = ARCH_TACTIC_BY_M[arch]
    for m_max, kernel_arch, tiler, cluster in entries[:-1]:
        if m_tok <= m_max:
            return kernel_arch, tiler, cluster
    _m_max, kernel_arch, tiler, cluster = entries[-1]  # unconditional fallback
    return kernel_arch, tiler, cluster


def run_fold_default(
    major: int,
    minor: int,
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run the fold GEMM with the validated default tactic for this GPU/m_tok."""
    arch = kernel_arch_for_capability(major, minor)
    if arch is None:
        raise RuntimeError(f"no fold kernel for capability ({major},{minor})")
    m_tok = act_fp4_rowpair.shape[0] // 2
    kernel_arch, tiler, cluster = select_arch_tactic(arch, m_tok)
    if not _SILENCE_DEFAULT:
        from ..fold import warn_default_path

        warn_default_path(arch, m_tok, f"tile={tiler}")
    return mext_fold_gemm_sm103(
        weight_fp4, act_fp4_rowpair, weight_sf, act_sf, alpha, out_dtype,
        mma_tiler_mn=tiler, cluster_shape_mn=cluster,
        store_mode="tma", kernel_arch=kernel_arch,
    )


def warmup_specs(arch: str, families=(), strided=(False, True)) -> list[dict]:
    """Every _compile_fold_gemm kwargs this arch's serving path can reach.

    Shape-independent BY CONSTRUCTION, which is why this needs no shapes while
    the sm120 warmup does: this kernel's compile cache key holds only
    (tiler, cluster, dtype, mode, arch, ab_stage, dual_alpha, pi_via_layout,
    split_k, splitk_ablate, dsmem_split, strided_weight) -- no n_w, no k, no
    m_tok. So the whole compile space is a small fixed set.

    `strided` is the part that used to be missed. `mext_fold_gemm_sm103`
    derives strided_weight from the tensor at call time
    (`weight.stride(0) != k_packed`), so an extended_k layer -- whose weight is
    a [:, :k_base] view of the ext-K weight -- reaches a DIFFERENT compiled
    kernel than a contiguous mext_r1 layer. Warming only the contiguous one
    left every ext-K layer to JIT on first use. This is the same class of miss
    that cost sm120 46 s of capture instead of 5 s via its a_ld variant.
    Compiling both is ~1.3 s of insurance against a whole category.

    `families` beyond the production one are only reachable when the caller
    enabled them (RESIDUE_FOLD_MEXT=1) AND a tuned entry exists, so the gate
    decision belongs to the caller, not here.

    Pure and GPU-free: enumerating is separated from compiling so the coverage
    rules can be tested anywhere.
    """
    specs: list[dict] = []

    # The static per-m table: what an untuned build actually serves, N-ext only.
    for _m_max, kernel_arch, tiler, cluster in ARCH_TACTIC_BY_M[arch]:
        specs.append(dict(mma_tiler_mn=tiler, cluster_shape_mn=cluster,
                          mode="fold_tma", kernel_arch=kernel_arch,
                          ab_stage_override=None, pi_via_layout=False))

    # FAMILY_TACTICS entries carry a STORE_MODE, which is not the same
    # vocabulary as _compile_fold_gemm's `mode`. Translating with the call
    # path's own table rather than a second copy: getting this wrong silently
    # enumerates modes that do not exist, so every legality check below would
    # pass and the compile would trip an assert instead.
    from .host import STORE_MODE_TO_MODE

    for fam in families:
        pi = _needs_pi_layout(fam)
        for tactic in FAMILY_TACTICS.get(fam, ()):
            store_mode, tiler, cluster, ab = _norm_tactic(tactic)
            mode = STORE_MODE_TO_MODE[store_mode]
            # _compile_fold_gemm asserts these; skip rather than trip them.
            # M-ext is implemented on the sm100 kernel only, and the AB-depth
            # knob is an sm100 ctor parameter.
            if mode in ("fold_tma_mext", "fold_mext_direct") and arch != "sm100":
                continue
            if ab is not None and arch != "sm100":
                continue
            specs.append(dict(mma_tiler_mn=tiler, cluster_shape_mn=cluster,
                              mode=mode, kernel_arch=arch,
                              ab_stage_override=ab, pi_via_layout=pi))

    out = []
    for spec in specs:
        for sw in strided:
            out.append({**spec, "strided_weight": sw})
    return out


def warmup_fold_default(major: int, minor: int, *, families=(),
                        strided=(False, True), out_dtype=None) -> int:
    """Pre-compile the fold kernels this GPU can reach. Returns the count.

    Call eagerly before capture to prepare the requested specializations.

    Best-effort per spec: one illegal combination must not cost the rest of
    the coverage. Failures are printed, never swallowed -- a warmup that
    silently covers nothing is exactly the state this replaced.
    """
    from .host import _compile_fold_gemm

    arch = kernel_arch_for_capability(major, minor)
    if arch is None:
        return 0
    dtype = out_dtype if out_dtype is not None else torch.bfloat16

    compiled = failed = 0
    for spec in warmup_specs(arch, families=families, strided=strided):
        try:
            _compile_fold_gemm(spec["mma_tiler_mn"], spec["cluster_shape_mn"],
                               dtype, mode=spec["mode"],
                               kernel_arch=spec["kernel_arch"],
                               ab_stage_override=spec["ab_stage_override"],
                               pi_via_layout=spec["pi_via_layout"],
                               strided_weight=spec["strided_weight"])
            compiled += 1
        except Exception as e:                            # noqa: BLE001
            failed += 1
            print(f"[residue] fold warmup: {spec['mode']} "
                  f"{spec['mma_tiler_mn']} strided={spec['strided_weight']} "
                  f"failed: {type(e).__name__}: {str(e)[:70]}", flush=True)
    if failed:
        print(f"[residue] fold warmup: {compiled} compiled, {failed} FAILED "
              f"-- those shapes will JIT on first use", flush=True)
    return compiled





















FAMILY_TACTICS: dict[tuple[str, str], list[tuple]] = {
    ("row_pair", "std"): [
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ("tma", (128, 8), (1, 1), 5),
        ("tma", (128, 16), (1, 1), 6),
        ("tma", (128, 16), (1, 1), 5),
        ("tma", (128, 24), (1, 1)),
        ("tma", (128, 32), (1, 1), 5),
        ("tma", (128, 32), (1, 1)),
        ("tma", (128, 64), (1, 1)),
        ("tma", (128, 128), (1, 1)),
        ("tma", (256, 64), (2, 1)),
        ("tma", (256, 128), (2, 1)),
        ("tma", (256, 256), (2, 1)),
    ],
    ("tile8", "std"): [
        ("tma_mext", (128, 32), (1, 1)),
        ("tma_mext", (128, 64), (1, 1)),
        ("tma_mext", (128, 128), (1, 1)),
        ("tma_mext", (256, 128), (2, 1)),
    ],
    ("tile8", "pi"): [
        ("direct_mext", (128, 64), (1, 1)),
        ("direct_mext", (128, 128), (1, 1)),
        ("direct_mext", (256, 128), (2, 1)),
    ],
}

_AUTOTUNE_CACHE: dict[tuple, tuple] = {}


def _time_us(fn, warmup: int = 3, reps: int = 16, iters: int = 12) -> float:
    'Median us per call, resolved well enough to rank decode-sized kernels.'
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    capturing = False
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:
        pass

    if capturing:
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            e0.record()
            for _ in range(reps):
                fn()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3 / reps)
        return statistics.median(ts)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / reps)
    del g
    return statistics.median(ts)


def _cache_key(family, kernel_arch, n_w, k_logical, m_tok):
    """One key convention: K is the LOGICAL (unpacked) contraction dim.

    weight_fp4.shape[1] is K/2 because FP4 is nibble-packed, so a caller
    reading the cache with the real K used to miss every time -- which is how
    autotuned_tactic_for() silently returned None for every shape.
    """
    return (family, kernel_arch, n_w, int(k_logical), _m_bucket(m_tok))


def _m_bucket(m_tok: int) -> int:
    b = 1
    while b < m_tok and b < 1024:
        b <<= 1
    return b


def _norm_tactic(t: tuple) -> tuple[str, tuple[int, int], tuple[int, int], int | None]:
    """Accept 3- or 4-element entries; the 4th is ab_stage (None = auto)."""
    if len(t) == 3:
        return t[0], t[1], t[2], None
    return t[0], t[1], t[2], t[3]








AUTOTUNE_TOP_N = 8


def _token_axis(family: tuple[str, str]) -> int:
    """Which entry of mma_tiler_mn tiles the TOKENS, for this family.

    The two geometries put the tokens on opposite axes and the tactic tuples
    look identical, so this cannot be inferred from the tile:

      ("row_pair", *)  N-ext: weight on M, activation on N -> tokens are
                       tiler[1], and the GEMM's N dim IS 2*m_tok.
      ("tile8",    *)  M-ext: host.py swaps the operands (act as mA, weight as
                       mB), so tokens are tiler[0] and tiler[1] tiles the
                       weight's n_w (4096-14336).
    """
    return 0 if family[0] == "tile8" else 1


def _rank_tactics(family: tuple[str, str], m_tok: int, kernel_arch: str):
    'Order candidates best-first, cheapest-to-verify-first.'
    prob = 2 * m_tok
    axis = _token_axis(family)
    scored = []
    for raw in FAMILY_TACTICS[family]:
        mode, tiler, cluster, ab = _norm_tactic(raw)
        if mode == "direct_mext" and 2 * m_tok < tiler[0]:
            continue  # A6 full-tile constraint (raw STG has no TMA clamping)
        tile = tiler[axis]
        # Narrow tiles must cover the token extent outright: splitting the
        # token axis re-streams the whole weight tile per extra tile, and the
        # weight is the decode bottleneck. Wide tiles may over-cover; they only
        # waste MMA, which this regime has spare. Same guard as FlashInfer's
        # `tile_n < 64 and prob_n > tile_n` -- but only meaningful on the token
        # axis. (Inert for M-ext today: its tile_M is always 128 or 256.)
        if tile < 64 and prob > tile:
            continue
        n_tiles = (prob + tile - 1) // tile
        waste = n_tiles * tile - prob
        # exact fit first, then least waste, then deepest pipeline (ab=None
        # means the kernel picks, which is the deepest that its solve allows)
        scored.append(((waste, n_tiles, 0 if ab is None else 1),
                       (mode, tiler, cluster, ab)))
    scored.sort(key=lambda kv: kv[0])
    return [t for _, t in scored]


def _needs_pi_layout(family) -> bool:
    'A6 must be launched with pi carried by the LAYOUT.'
    return family in FAMILY_NEEDS_PI_WEIGHT


def _measure_family(
    family, weight_fp4, act_fp4, weight_sf, act_sf, alpha, out_dtype,
    kernel_arch, warmup_iters, measure_iters,
):
    """Best (tactic, us) for one family on this shape, or (None, None).

    Returns rather than raises when nothing launched: with cross-family
    selection a family having no legal config is a NORMAL outcome (A6 has
    none below m_tok=64), not an error.
    """
    m_tok = act_fp4.shape[0] // 2
    best, best_us = None, None
    for mode, tiler, cluster, ab in _rank_tactics(
            family, m_tok, kernel_arch)[:AUTOTUNE_TOP_N]:
        try:
            fn = lambda mode=mode, tiler=tiler, cluster=cluster, ab=ab: (
                mext_fold_gemm_sm103(
                    weight_fp4, act_fp4, weight_sf, act_sf, alpha,
                    out_dtype, mma_tiler_mn=tiler, cluster_shape_mn=cluster,
                    store_mode=mode, kernel_arch=kernel_arch,
                    ab_stage_override=ab,
                    pi_via_layout=_needs_pi_layout(family)))
            us = _time_us(fn, warmup=warmup_iters, iters=measure_iters)
            if best_us is None or us < best_us:
                best, best_us = (mode, tiler, cluster, ab), us
        except Exception:
            continue
    return best, best_us


def run_fold_autotuned(
    family: tuple[str, str],
    weight_fp4: torch.Tensor,
    act_fp4: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    kernel_arch: str = "sm100",
    warmup_iters: int = 3,
    measure_iters: int = 10,
) -> torch.Tensor:
    'Implementation details.'
    key = _cache_key(family, kernel_arch, weight_fp4.shape[0],
                     weight_fp4.shape[1] * 2, act_fp4.shape[0] // 2)
    hit = _AUTOTUNE_CACHE.get(key)
    if hit is None:
        hit, _us = _measure_family(
            family, weight_fp4, act_fp4, weight_sf, act_sf, alpha, out_dtype,
            kernel_arch, warmup_iters, measure_iters)
        if hit is None:
            raise ValueError(
                f"no legal tactic for {key} -- the ('tile8','pi') family has "
                f"no config at 2*m_tok < 128 (A6 full-tile constraint); route "
                f"tiny-m to the ('row_pair','std') N-ext family instead")
        _AUTOTUNE_CACHE[key] = hit
    mode, tiler, cluster, ab = _norm_tactic(hit)
    return mext_fold_gemm_sm103(
        weight_fp4, act_fp4, weight_sf, act_sf, alpha, out_dtype,
        mma_tiler_mn=tiler, cluster_shape_mn=cluster,
        store_mode=mode, kernel_arch=kernel_arch, ab_stage_override=ab,
        pi_via_layout=_needs_pi_layout(family))






















FAMILY_ACT_LAYOUT: dict[tuple[str, str], str] = {
    ("row_pair", "std"): "row_pair",
    ("tile8", "std"): "tile8",
    ("tile8", "pi"): "tile8",
}

# Families needing the pi weight permutation. pi is a pure axis transpose that
# COMMUTES with quantise+swizzle (a6_weight_perm_indices has the proof), so
# this is a LAYOUT requirement, not a second weight buffer and not a load-time
# commitment -- one physical weight can back both N-ext and A6. The kernel
# carries the transpose in its TMA/SF layouts (pi_via_layout, gated bitwise on
# B200 by gate_a6_pi_via_layout_sm100.py), so nothing is materialised per call.
FAMILY_NEEDS_PI_WEIGHT: frozenset = frozenset({("tile8", "pi")})

_FAMILY_CACHE: dict[tuple, tuple] = {}


def _family_key(kernel_arch, n_w, k_logical, m_tok):
    return (kernel_arch, n_w, int(k_logical), _m_bucket(m_tok))


def tune_families(
    acts: dict,
    weight_fp4: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    kernel_arch: str = "sm100",
    families=None,
    warmup_iters: int = 3,
    measure_iters: int = 10,
) -> tuple:
    """Measure every family the caller has an activation layout for; cache the
    winner for this (arch, n_w, k, m-bucket).

    acts: {layout_name: (act_fp4, act_sf)} -- e.g. {"row_pair": (...)} or both
          layouts. A family whose layout is absent is simply not considered,
          so a caller that can only produce row_pair degrades to the old
          behaviour instead of failing.

    Returns the winning (family, tactic). Warmup only: this measures, so it
    must not run inside a cudagraph capture.
    """
    if families is None:
        families = [f for f in FAMILY_TACTICS
                    if FAMILY_ACT_LAYOUT[f] in acts]
    any_act = next(iter(acts.values()))[0]
    key = _family_key(kernel_arch, weight_fp4.shape[0],
                      weight_fp4.shape[1] * 2, any_act.shape[0] // 2)

    best = None  # (us, family, tactic)
    for fam in families:
        layout = FAMILY_ACT_LAYOUT[fam]
        if layout not in acts:
            continue
        act_fp4, act_sf = acts[layout]
        tac, us = _measure_family(
            fam, weight_fp4, act_fp4, weight_sf, act_sf, alpha, out_dtype,
            kernel_arch, warmup_iters, measure_iters)
        if tac is None:
            continue  # no legal config for this family at this m -- normal
        _AUTOTUNE_CACHE[_cache_key(
            fam, kernel_arch, weight_fp4.shape[0], weight_fp4.shape[1] * 2,
            act_fp4.shape[0] // 2)] = tac
        if best is None or us < best[0]:
            best = (us, fam, tac)

    if best is None:
        raise ValueError(f"no family produced a legal tactic for {key}")
    _FAMILY_CACHE[key] = (best[1], best[2])
    return best[1], best[2]


def best_family_for(kernel_arch, n_w, k, m_tok):
    'Implementation details.'
    return _FAMILY_CACHE.get(_family_key(kernel_arch, n_w, k, m_tok))


def autotuned_tactic_for(family, kernel_arch, n_w, k, m_tok):
    """Inspect the cached choice (None if that bucket was never tuned).

    `k` is the LOGICAL contraction dim, not weight_fp4.shape[1] (= K/2)."""
    return _AUTOTUNE_CACHE.get(_cache_key(family, kernel_arch, n_w, k, m_tok))


def dump_state() -> dict:
    """JSON-shaped snapshot of everything this tuner decided.

    The FlashInfer layer's e2e tuning result is already a file (the autotune
    cache JSON the server writes); this gives the CuTeDSL fold layer the same
    property, so an offline kernel-level tune and a server's e2e tune can be
    diffed as files (scripts/audit_autotune_coverage.py). Keys/values are
    repr()-ed tuples -- ast.literal_eval round-trips them, the same convention
    FlashInfer uses for its cache keys.
    """
    return {
        "tactics": {repr(k): repr(v)
                    for k, v in sorted(_AUTOTUNE_CACHE.items())},
        "families": {repr(k): repr(v)
                     for k, v in sorted(_FAMILY_CACHE.items())},
    }
