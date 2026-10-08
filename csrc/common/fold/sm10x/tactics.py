# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""SM100 Fold candidate tables, ranking and static fallback.

FlashInfer owns measurement and tactic caching. These tables are also shared
by MXFP4; preserve candidate order because cached tactic IDs refer to it.
"""
from __future__ import annotations

import torch

from ..diagnostics import silence_default_notice, warn_default_path
from .host import mext_fold_gemm

_SILENCE_DEFAULT = silence_default_notice()


_SENTINEL_M = 1 << 30
ARCH_TACTIC_BY_M: dict[str, list[tuple[int, str, tuple[int, int], tuple[int, int]]]] = {
    "sm100": [
        (64, "sm100", (128, 64), (1, 1)),            # decode: N=64 1SM floor
        (_SENTINEL_M, "sm100", (256, 128), (2, 1)),  # m>64: 2SM (safe blanket)
    ],
}


def kernel_arch_for_capability(major: int, minor: int) -> str | None:
    """Map a CUDA compute capability to the fold kernel arch, or None if the
    fold path is unsupported (caller falls back to unfused)."""
    return "sm100" if (major, minor) == (10, 0) else None


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
        warn_default_path(arch, m_tok, f"tile={tiler}")
    return mext_fold_gemm(
        weight_fp4, act_fp4_rowpair, weight_sf, act_sf, alpha, out_dtype,
        mma_tiler_mn=tiler, cluster_shape_mn=cluster,
        store_mode="tma", kernel_arch=kernel_arch,
    )


# (store_mode, MMA tile, cluster[, AB stages]). Explicit stages keep narrow
# row-pair tiles within SM100 shared memory; A4/A6 use tile8 activations.
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

FAMILY_ACT_LAYOUT: dict[tuple[str, str], str] = {
    ("row_pair", "std"): "row_pair",
    ("tile8", "std"): "tile8",
    ("tile8", "pi"): "tile8",
}

# A6 carries the weight transpose in its layout, without a second buffer.
FAMILY_NEEDS_PI_WEIGHT: frozenset = frozenset({("tile8", "pi")})


def _norm_tactic(t: tuple) -> tuple[str, tuple[int, int], tuple[int, int], int | None]:
    """Accept 3- or 4-element entries; the 4th is ab_stage (None = auto)."""
    if len(t) == 3:
        return t[0], t[1], t[2], None
    return t[0], t[1], t[2], t[3]


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
