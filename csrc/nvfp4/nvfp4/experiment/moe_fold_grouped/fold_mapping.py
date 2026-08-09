# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Build the gather index that makes a grouped MoE GEMM foldable.'

from __future__ import annotations

PAIR_ADJACENT = "pair_adjacent"
TILE8 = "tile8"

#: The M-row distance the SM100 `.16x256b` TMEM-load atom co-locates in one
#: thread. Position p and p+8 of a tile land in the same accumulator fragment.
TILE8_STRIDE = 8

INVALID = -1


def _align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def base_row(token: int) -> int:
    """Row of `token`'s base half in a row_pair-quantised activation."""
    return 2 * token


def residue_row(token: int) -> int:
    """Row of `token`'s residue half in a row_pair-quantised activation."""
    return 2 * token + 1


def arrange_expert_block(tokens: list[int], arrangement: str) -> list[int]:
    """Order one expert's 2*len(tokens) rows.

    `tokens` is that expert's token list, in whatever order the router
    produced. The result is the source-row sequence for this expert's block,
    unpadded.
    """
    if arrangement == PAIR_ADJACENT:
        out: list[int] = []
        for t in tokens:
            out.append(base_row(t))
            out.append(residue_row(t))
        return out

    if arrangement != TILE8:
        raise ValueError(
            f"arrangement must be {PAIR_ADJACENT!r} or {TILE8!r}, got {arrangement!r}"
        )

    # 8 bases, then the SAME 8 tokens' residues, so that position p and p+8
    # are one token's two halves.
    out = []
    for start in range(0, len(tokens), TILE8_STRIDE):
        chunk = tokens[start : start + TILE8_STRIDE]
        out.extend(base_row(t) for t in chunk)
        if len(chunk) < TILE8_STRIDE:
            # A short tail would misalign the +8 relationship for the rows that
            # follow it, so pad the base half out to 8 before the residues.
            out.extend([INVALID] * (TILE8_STRIDE - len(chunk)))
        out.extend(residue_row(t) for t in chunk)
        if len(chunk) < TILE8_STRIDE:
            out.extend([INVALID] * (TILE8_STRIDE - len(chunk)))
    return out


def build_fold_token_id_mapping(
    expert_tokens: list[list[int]],
    mma_tiler_m: int,
    arrangement: str = TILE8,
) -> tuple[list[int], list[int], list[int]]:
    """Build the full mapping for a folded grouped MoE GEMM.

    Args:
        expert_tokens: per-expert token lists, `expert_tokens[e]` = the tokens
            routed to expert e. Indices are into the ORIGINAL M tokens, not
            into the doubled activation -- the doubling happens here.
        mma_tiler_m: tile alignment the kernel pads each expert block to.
        arrangement: TILE8 for an in-epilogue fold, PAIR_ADJACENT for the
            post-GEMM reduction.

    Returns:
        (token_id_mapping, aligned_group_m, mn_limits)
        - token_id_mapping: source row per GEMM row, INVALID where padded
        - aligned_group_m: padded row count per expert
        - mn_limits: per-expert end offset of the VALID rows, in the same
          cumulative form the kernel's `tile_idx_to_mn_limit` uses
    """
    if mma_tiler_m <= 0:
        raise ValueError(f"mma_tiler_m must be positive, got {mma_tiler_m}")

    mapping: list[int] = []
    aligned_group_m: list[int] = []
    mn_limits: list[int] = []
    cursor = 0

    for tokens in expert_tokens:
        block = arrange_expert_block(list(tokens), arrangement)
        aligned = _align_up(len(block), mma_tiler_m)
        mapping.extend(block)
        mapping.extend([INVALID] * (aligned - len(block)))
        aligned_group_m.append(aligned)
        mn_limits.append(cursor + len(block))
        cursor += aligned

    return mapping, aligned_group_m, mn_limits


def fold_output_index(expert_tokens: list[list[int]], mma_tiler_m: int,
                      arrangement: str = TILE8) -> list[int]:
    """Where each *folded* output row should go, as a token index.

    An in-epilogue fold writes one row per token instead of two. This says
    which token each surviving row belongs to, so the caller can scatter the
    result back. INVALID marks a padded row that produces nothing.

    Under TILE8 the surviving row of a pair is the base position; under
    PAIR_ADJACENT it is the even position.
    """
    out: list[int] = []
    for tokens in expert_tokens:
        block = arrange_expert_block(list(tokens), arrangement)
        aligned = _align_up(len(block), mma_tiler_m)
        for pos, src in enumerate(block):
            keep = (pos % (2 * TILE8_STRIDE)) < TILE8_STRIDE \
                if arrangement == TILE8 else (pos % 2 == 0)
            out.append(src // 2 if (keep and src != INVALID) else INVALID)
        out.extend([INVALID] * (aligned - len(block)))
    return out
