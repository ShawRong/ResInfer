# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Tactic registry for the SM103 MExt fold GEMM.'

from __future__ import annotations

import torch

from .host import mext_fold_gemm_sm103








TACTICS: list[tuple[str, tuple[int, int], tuple[int, int]]] = [
    ("tma", (128, 128), (1, 1)),     
    
    ("tma", (256, 128), (2, 1)),     # 1: default -- beats floor for m_tok<=128
    ("tma", (256, 256), (2, 1)),     # 2: flattens the wave step, m_tok>128
]

def get_valid_tactics(m_tok: int) -> list[int]:
    """Tactics applicable at this m_tok (all are correct; this only prunes
    configurations that are strictly dominated to keep tuning cheap)."""
    if m_tok <= 128:
        return [1]
    # Above the single-wave boundary: (256,256) wins for large n_w,
    # (256,128) for small shards -- let the tuner pick.
    return [1, 2]


def default_tactic(m_tok: int) -> int:
    return 1 if m_tok <= 128 else 2


def run_fold_tactic(
    tactic: int,
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run the fold GEMM with an explicit tactic (or -1 for the heuristic).

    Returns D[m_tok, n_w] contiguous. See host.mext_fold_gemm_sm103 for the
    tensor contracts (row_pair activation layout, k_base-prefix weight,
    swizzled 128x4 scale factors).
    """
    if tactic < 0:
        tactic = default_tactic(act_fp4_rowpair.shape[0] // 2)
    mode, tiler, cluster = TACTICS[tactic]
    return mext_fold_gemm_sm103(
        weight_fp4,
        act_fp4_rowpair,
        weight_sf,
        act_sf,
        alpha,
        out_dtype,
        mma_tiler_mn=tiler,
        cluster_shape_mn=cluster,
        store_mode=mode,
    )


def warmup_tactics(device: torch.device | None = None) -> None:
    """Pre-compile all tactic variants (JIT costs seconds each; call once at
    engine init, outside any cudagraph capture)."""
    from .host import _compile_fold_gemm

    for mode, tiler, cluster in TACTICS:
        _compile_fold_gemm(tiler, cluster, torch.bfloat16, mode="fold_tma")

























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

# Back-compat: the decode/small-m default per arch (first list entry).
ARCH_DEFAULT: dict[str, tuple[str, tuple[int, int], tuple[int, int]]] = {
    arch: (entries[0][1], entries[0][2], entries[0][3])
    for arch, entries in ARCH_TACTIC_BY_M.items()
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
    return mext_fold_gemm_sm103(
        weight_fp4, act_fp4_rowpair, weight_sf, act_sf, alpha, out_dtype,
        mma_tiler_mn=tiler, cluster_shape_mn=cluster,
        store_mode="tma", kernel_arch=kernel_arch,
    )


def warmup_fold_default(major: int, minor: int) -> None:
    """Pre-compile ALL per-m fold kernels for this GPU (call at engine init,
    outside cudagraph capture; runtime selection then never JITs)."""
    from .host import _compile_fold_gemm

    arch = kernel_arch_for_capability(major, minor)
    if arch is None:
        return
    for _m_max, kernel_arch, tiler, cluster in ARCH_TACTIC_BY_M[arch]:
        _compile_fold_gemm(tiler, cluster, torch.bfloat16,
                           mode="fold_tma", kernel_arch=kernel_arch)











FAMILY_TACTICS: dict[tuple[str, str], list[tuple[str, tuple[int, int], tuple[int, int]]]] = {
    ("row_pair", "std"): [
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ("tma", (128, 24), (1, 1)),
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

_AUTOTUNE_CACHE: dict[tuple, tuple[str, tuple[int, int], tuple[int, int]]] = {}


def _m_bucket(m_tok: int) -> int:
    b = 1
    while b < m_tok and b < 1024:
        b <<= 1
    return b


def _legal(family: tuple[str, str], m_tok: int, kernel_arch: str):
    for mode, tiler, cluster in FAMILY_TACTICS[family]:
        if mode == "direct_mext" and 2 * m_tok < tiler[0]:
            continue  # A6 full-tile constraint (raw STG has no TMA clamping)
        yield mode, tiler, cluster


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
    m_tok = act_fp4.shape[0] // 2
    n_w = weight_fp4.shape[0]
    key = (family, kernel_arch, n_w, weight_fp4.shape[1], _m_bucket(m_tok))
    hit = _AUTOTUNE_CACHE.get(key)
    if hit is None:
        import time
        best, best_us = None, None
        for mode, tiler, cluster in _legal(family, m_tok, kernel_arch):
            try:
                fn = lambda: mext_fold_gemm_sm103(
                    weight_fp4, act_fp4, weight_sf, act_sf, alpha, out_dtype,
                    mma_tiler_mn=tiler, cluster_shape_mn=cluster,
                    store_mode=mode, kernel_arch=kernel_arch)
                for _ in range(warmup_iters):
                    fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(measure_iters):
                    fn()
                torch.cuda.synchronize()
                us = (time.perf_counter() - t0) / measure_iters * 1e6
                if best_us is None or us < best_us:
                    best, best_us = (mode, tiler, cluster), us
            except Exception:
                continue
        if best is None:
            raise ValueError(
                f"no legal tactic for {key} -- the ('tile8','pi') family has "
                f"no config at 2*m_tok < 128 (A6 full-tile constraint); route "
                f"tiny-m to the ('row_pair','std') N-ext family instead")
        _AUTOTUNE_CACHE[key] = best
        hit = best
    mode, tiler, cluster = hit
    return mext_fold_gemm_sm103(
        weight_fp4, act_fp4, weight_sf, act_sf, alpha, out_dtype,
        mma_tiler_mn=tiler, cluster_shape_mn=cluster,
        store_mode=mode, kernel_arch=kernel_arch)


def autotuned_tactic_for(family, kernel_arch, n_w, k, m_tok):
    """Inspect the cached choice (None if that bucket was never tuned)."""
    return _AUTOTUNE_CACHE.get((family, kernel_arch, n_w, k, _m_bucket(m_tok)))
