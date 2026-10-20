# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""SM100 Fold candidates and default selection for NVFP4 and MXFP4."""
from __future__ import annotations

from functools import lru_cache

import torch

from .host import mext_fold_gemm


_SENTINEL_M = 1 << 30
ARCH_TACTIC_BY_M: dict[str, list[tuple[int, str, tuple[int, int], tuple[int, int]]]] = {
    "sm100": [
        (64, "sm100", (128, 64), (1, 1)),
        (_SENTINEL_M, "sm100", (256, 128), (2, 1)),
    ],
}


def kernel_arch_for_capability(major: int, minor: int) -> str | None:
    """Return the supported Fold architecture for a CUDA capability."""
    return "sm100" if (major, minor) == (10, 0) else None


def select_arch_tactic(
    arch: str, m_tok: int
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Return the fallback (kernel_arch, MMA tile, cluster) for a supported arch and M."""
    entries = ARCH_TACTIC_BY_M[arch]
    for m_max, kernel_arch, tiler, cluster in entries[:-1]:
        if m_tok <= m_max:
            return kernel_arch, tiler, cluster
    _m_max, kernel_arch, tiler, cluster = entries[-1]
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
    return mext_fold_gemm(
        weight_fp4, act_fp4_rowpair, weight_sf, act_sf, alpha, out_dtype,
        mma_tiler_mn=tiler, cluster_shape_mn=cluster,
        store_mode="tma", kernel_arch=kernel_arch,
    )


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


FAMILY_NEEDS_PI_WEIGHT: frozenset = frozenset({("tile8", "pi")})


def _norm_tactic(t: tuple) -> tuple[str, tuple[int, int], tuple[int, int], int | None]:
    """Normalize a tactic entry."""
    if len(t) == 3:
        return t[0], t[1], t[2], None
    return t[0], t[1], t[2], t[3]


def _token_axis(family: tuple[str, str]) -> int:
    """Return the token axis for a Fold family."""
    return 0 if family[0] == "tile8" else 1


@lru_cache(maxsize=None)
def expand_tactics(families: tuple) -> list[tuple]:
    """Stable candidate IDs shared by NVFP4 and MXFP4 adapters."""
    return [(family, *_norm_tactic(raw)) for family in families
            for raw in FAMILY_TACTICS[family]]


def legal_tactics(family: tuple[str, str], m_tok: int):
    """Return legal NVFP4 tactics for the token count."""
    tokens = 2 * m_tok
    axis = _token_axis(family)
    result = []
    for raw in FAMILY_TACTICS[family]:
        tactic = _norm_tactic(raw)
        mode, tiler, _cluster, _ab = tactic
        if mode == "direct_mext" and tokens < tiler[0]:
            continue
        if tiler[axis] < 64 and tokens > tiler[axis]:
            continue
        result.append(tactic)
    return result


def _needs_pi_layout(family) -> bool:
    """Check whether a Fold family uses the pi layout."""
    return family in FAMILY_NEEDS_PI_WEIGHT
