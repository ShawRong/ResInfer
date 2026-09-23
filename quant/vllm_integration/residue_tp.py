# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Tensor-parallel sharding for packed FP4 residue K-extensions.'
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

# Salient channels are selected top-k within blocks of this many channels.
SALIENT_BLOCK = 8
# One FP4 scale factor covers this many channels.
SF_VEC_SIZE = 16
# Swizzled block-scale tiles span 4 SF blocks along K.
SF_SWIZZLE_BLOCKS = 4
# One RESIDUE scale factor is a warp_group_max over 16/residue_per_8 threads,
# each covering 8 channels -- i.e. 128/residue_per_8 base channels, worst case
# 128 at the production ratio 0.125. A shard boundary inside such a group would
# give that rank a different residue amax than the single-GPU run: the residue
# values themselves change, silently, with no shape error anywhere. See
# cvt_residue_sf_get_offset / warp_group_max in nvfp4_utils.cuh.
RESIDUE_SF_GROUP_CHANNELS = 128


class ResidueTPError(ValueError):
    """A shape the residue TP sharding rule cannot honour."""


@dataclass(frozen=True)
class ResidueShardPlan:
    """How to cut one extended tensor for one rank.

    Attributes are in *logical channels*; helpers convert to packed-FP4 and
    scale-factor coordinates.
    """

    k_base: int
    num_salient: int
    tp_size: int
    tp_rank: int
    scale_vec_size: int = SF_VEC_SIZE
    scale_swizzle_blocks: int = SF_SWIZZLE_BLOCKS
    residue_sf_group_channels: int = RESIDUE_SF_GROUP_CHANNELS
    k_alignment: int = 1

    @property
    def k_ext(self) -> int:
        return self.k_base + self.num_salient

    @property
    def base_shard(self) -> int:
        return self.k_base // self.tp_size

    @property
    def salient_shard(self) -> int:
        return self.num_salient // self.tp_size

    @property
    def k_ext_shard(self) -> int:
        """Per-rank extended width. Equals k_ext // tp_size."""
        return self.base_shard + self.salient_shard

    @property
    def k_padded(self) -> int:
        return ((self.k_ext + self.k_alignment - 1) // self.k_alignment) * self.k_alignment

    @property
    def k_shard_padded(self) -> int:
        return ((self.k_ext_shard + self.k_alignment - 1) // self.k_alignment) * self.k_alignment

    def validate(self) -> None:
        """Reject shapes the two-range rule cannot serve, loudly.

        An unbalanced salient split would need ragged per-rank shapes. A
        boundary off the scale grid would split a quantization scale group.
        """
        tp = self.tp_size
        if (self.scale_vec_size not in (16, 32)
                or self.scale_swizzle_blocks <= 0
                or self.residue_sf_group_channels <= 0
                or self.k_alignment <= 0):
            raise ResidueTPError("invalid FP4 scale or K alignment")
        if tp <= 0 or self.tp_rank not in range(tp):
            raise ResidueTPError(f"bad tp_rank={self.tp_rank} for tp_size={tp}")
        if self.k_base % tp:
            raise ResidueTPError(
                f"K_base={self.k_base} not divisible by tp_size={tp}"
            )
        if self.num_salient % tp:
            raise ResidueTPError(
                f"num_salient={self.num_salient} not divisible by tp_size={tp};"
                " the salient split would be ragged"
            )
        # Balanced split needs the boundary to land on a salient block.
        if self.base_shard % SALIENT_BLOCK:
            raise ResidueTPError(
                f"K_base/tp={self.base_shard} is not a multiple of "
                f"{SALIENT_BLOCK}: ranks would get different salient counts"
            )
        # ...and must not cut a residue scale-factor group in half.
        if self.base_shard % self.residue_sf_group_channels:
            raise ResidueTPError(
                f"K_base/tp={self.base_shard} is not a multiple of "
                f"{self.residue_sf_group_channels}: the boundary would split a "
                "residue scale-factor group, so this rank would compute a "
                "different residue amax than the unsharded model"
            )
        # Both ranges are sliced in scale-factor space too.
        grid = self.scale_vec_size * self.scale_swizzle_blocks
        for name, width in (("K_base/tp", self.base_shard),
                            ("S/tp", self.salient_shard)):
            if width % grid:
                raise ResidueTPError(
                    f"{name}={width} is not a multiple of {grid}; the "
                    "block-scale group cannot be split there"
                )
        assert self.k_ext_shard * tp == self.k_ext, "K_ext must split evenly"
        if self.k_padded % self.scale_vec_size or self.k_shard_padded % self.scale_vec_size:
            raise ResidueTPError("padded K must align with scale groups")

    # -- range helpers ----------------------------------------------------

    def ranges(self, scale: int = 1) -> tuple[tuple[int, int], tuple[int, int]]:
        """(base, salient) ranges as (offset, length), divided by `scale`.

        `scale` = 1 for logical channels, 2 for packed FP4 (two nibbles per
        byte), or `scale_vec_size` for block scales.
        """
        if scale <= 0:
            raise ResidueTPError("scale must be positive")
        for name, v in (("k_base", self.k_base),
                        ("base_shard", self.base_shard),
                        ("salient_shard", self.salient_shard)):
            if v % scale:
                raise ResidueTPError(
                    f"{name}={v} not divisible by scale={scale}"
                )
        base = (self.tp_rank * self.base_shard // scale, self.base_shard // scale)
        sal = ((self.k_base + self.tp_rank * self.salient_shard) // scale,
               self.salient_shard // scale)
        return base, sal

    def gather(self, full: torch.Tensor, scale: int = 1,
               dim: int = -1) -> torch.Tensor:
        """Assemble this rank's shard from the two ranges."""
        (b_off, b_len), (s_off, s_len) = self.ranges(scale)
        want = full.shape[dim]
        if self.k_padded % scale or self.k_shard_padded % scale:
            raise ResidueTPError("padded K is not divisible by scale")
        need = self.k_padded // scale
        if want != need:
            raise ResidueTPError(
                f"expected extended dim {need} along dim {dim}, got {want}"
            )
        result = torch.cat(
            [full.narrow(dim, b_off, b_len), full.narrow(dim, s_off, s_len)],
            dim=dim,
        )
        pad = (self.k_shard_padded - self.k_ext_shard) // scale
        if pad:
            # FP4 zero has packed code 0; E8M0 code 0 is the producer's
            # zero-block scale. Input checkpoint padding is discarded.
            shape = list(result.shape)
            shape[dim] = pad
            result = torch.cat((result, result.new_zeros(shape)), dim=dim)
        return result

    # -- runtime state ----------------------------------------------------

    def local_channel_mask(self, full_mask: torch.Tensor) -> torch.Tensor:
        """This rank's slice of the per-8-channel salient bitmask."""
        if full_mask.numel() != self.k_base // SALIENT_BLOCK:
            raise ResidueTPError(
                f"channel_mask has {full_mask.numel()} bytes, expected "
                f"{self.k_base // SALIENT_BLOCK} for K_base={self.k_base}"
            )
        per = self.base_shard // SALIENT_BLOCK
        return full_mask.narrow(0, self.tp_rank * per, per).contiguous()

    def local_salient_indices(self, full_indices: torch.Tensor) -> torch.Tensor:
        """This rank's salient indices, rebased to rank-local channels.

        Relies on the indices being globally sorted, so this rank's are the
        r-th contiguous run -- verified rather than assumed.
        """
        lo = self.tp_rank * self.base_shard
        hi = lo + self.base_shard
        mine = full_indices[(full_indices >= lo) & (full_indices < hi)]
        if mine.numel() != self.salient_shard:
            raise ResidueTPError(
                f"rank {self.tp_rank} owns {mine.numel()} salient channels, "
                f"expected {self.salient_shard}: the export's per-block "
                "uniformity no longer holds, so TP sharding is unsafe"
            )
        run = full_indices.narrow(0, self.tp_rank * self.salient_shard,
                                  self.salient_shard)
        if not torch.equal(mine, run):
            raise ResidueTPError(
                "salient indices are not globally sorted; the contiguous-run "
                "assumption behind TP sharding is broken"
            )
        return mine - lo


def interleave_extended_for_tp(full: torch.Tensor, plan: ResidueShardPlan,
                               scale: int = 1, dim: int = -1) -> torch.Tensor:
    """Reorder ``[base | salient]`` into ``[base_0 | sal_0 | base_1 | ...]``.

    This avoids replacing vLLM's weight loader. After the permutation, rank
    r's *contiguous* physical range

        [ r*K_shard_padded , (r+1)*K_shard_padded )

    is exactly the two-range gather it needs. So the loader keeps doing its
    ordinary contiguous narrow and still lands on the right columns -- we
    only have to hand it a permuted checkpoint tensor.

    Applied to the full tensor, identically on every rank, so it stays a
    pure function of the checkpoint (no rank in the result).
    """
    tp = plan.tp_size
    pieces = []
    for r in range(tp):
        r_plan = replace(plan, tp_rank=r)
        pieces.append(r_plan.gather(full, scale=scale, dim=dim))
    return torch.cat(pieces, dim=dim).contiguous()


def plan_from_partition(
    *,
    extended_dim: int,
    input_size_per_partition: int,
    input_size: int,
    num_salient: int,
    tp_rank: int = 0,
    scale_vec_size: int = SF_VEC_SIZE,
    scale_swizzle_blocks: int = SF_SWIZZLE_BLOCKS,
    residue_sf_group_channels: int = RESIDUE_SF_GROUP_CHANNELS,
    k_alignment: int = 1,
) -> ResidueShardPlan | None:
    """Build a shard plan from what vLLM hands `create_weights`.

    Returns None for a column-parallel layer: its input is not sharded, so
    the whole extended K lives on every rank and there is nothing to cut.

    The row-parallel signal is `input_size_per_partition < input_size` --
    tp_size follows from their ratio, so this needs no process-group lookup
    and stays testable off-device.
    """
    if input_size_per_partition >= input_size:
        return None
    if input_size % input_size_per_partition:
        raise ResidueTPError(
            f"input_size={input_size} is not a whole multiple of "
            f"input_size_per_partition={input_size_per_partition}"
        )
    tp_size = input_size // input_size_per_partition
    k_base = extended_dim - num_salient
    if k_base != input_size:
        raise ResidueTPError(
            f"extended_dim={extended_dim} minus num_salient={num_salient} "
            f"is {k_base}, which does not match input_size={input_size}"
        )
    plan = ResidueShardPlan(
        k_base=k_base,
        num_salient=num_salient,
        tp_size=tp_size,
        tp_rank=tp_rank,
        scale_vec_size=scale_vec_size,
        scale_swizzle_blocks=scale_swizzle_blocks,
        residue_sf_group_channels=residue_sf_group_channels,
        k_alignment=k_alignment,
    )
    plan.validate()
    return plan


def current_tp_rank() -> int:
    """vLLM's TP rank, or 0 outside a distributed run."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except Exception:
        return 0
