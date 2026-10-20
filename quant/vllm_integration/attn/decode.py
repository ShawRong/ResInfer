# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResAttn decode launches and per-layer reduction workspace ownership."""
from __future__ import annotations

from typing import Any, cast

import torch

from .runtime import _load_decode_kernel, _validate_residue_geometry


class ResAttnDecodeMixin:
    """Requires the head geometry, scale and workspace cache of the adapter."""

    # Initialized by ResAttnFlashInferImpl after the stock FlashInfer base.
    num_heads: int
    num_kv_heads: int
    head_size: int
    scale: float
    _decode_reduction_workspaces: dict[tuple[str, int, int, int, str], Any]

    def _get_decode_reduction_workspace(
        self,
        host,
        splits: int,
        batch: int,
        device: torch.device,
        reduction: str = "kernel",
    ):
        # The CuTe kernel pads each GQA group to its supported N tile. Mirror
        # that effective head count here; using the model's unpadded HQ would
        # make the persistent workspace fail host validation for GQA < 16.
        group_size = host._residue_padded_group_size(self.num_heads, self.num_kv_heads)
        effective_heads = self.num_kv_heads * group_size
        if reduction not in ("kernel", "gmem_fused"):
            raise ValueError(f"unsupported persistent reduction: {reduction}")
        # (reduction, splits, batch) plus the impl-fixed head/dim above are the
        # complete shape determinants, so a key hit is a shape match; the host
        # entry validates the tensors it receives anyway.
        key = (reduction, splits, batch, effective_heads, str(device))
        workspace = self._decode_reduction_workspaces.get(key)
        if workspace is None:
            allocator = (
                host.allocate_qmul4_gmem_reduction_workspace
                if reduction == "gmem_fused"
                else host.allocate_qmul4_reduction_workspace
            )
            workspace = allocator(
                splits, batch, effective_heads, self.head_size, device
            )
            self._decode_reduction_workspaces[key] = workspace
        return workspace

    def _decode_forward_nvfp4(
        self,
        q_all: torch.Tensor,
        pool,
        md,
        output: torch.Tensor,
        output_scale: torch.Tensor,
        output_block_scale: torch.Tensor,
    ) -> None:
        """Decode directly into vLLM's packed activation buffers."""
        n = md.num_decode_tokens
        decode = md.decode
        q_dec = q_all if q_all.shape[0] == n else q_all[:n]
        host = _load_decode_kernel()
        _validate_residue_geometry(self.num_heads, self.num_kv_heads)
        plan = host.plan_qmul4_decode_nvfp4(
            n, self.num_heads, self.num_kv_heads,
            decode.max_seq_len, q_dec.device,
        )
        splits, reduction = plan
        reduction_workspace = None
        if reduction in ("kernel", "gmem_fused"):
            reduction_workspace = self._get_decode_reduction_workspace(
                host,
                splits,
                n,
                q_dec.device,
                reduction=reduction,
            )
        scale_rows = ((n + 127) // 128) * 128
        host.run_paged_fp4_decode_qmul4_tmem_nvfp4(
            q_dec,
            pool,
            pool.v_data,
            pool.v_sf,
            decode.block_tables,
            decode.seq_lens,
            self.scale * pool.k_global_scale,
            output_scale,
            out=output[:n].view(n, self.num_heads * self.head_size // 2),
            output_block_scale=output_block_scale[:scale_rows],
            max_seq_len=decode.max_seq_len,
            k_sf_mode="linear",
            value_scale=pool.v_global_scale,
            reduction_workspace=reduction_workspace,
            plan=plan,
        )

    def _decode_forward(
        self,
        q_all,
        pool,
        md,
        out3,
    ) -> None:
        """Decode into the stock BF16 output after stock has updated KV."""
        n = md.num_decode_tokens
        decode = md.decode
        q_dec = q_all if q_all.shape[0] == n else q_all[:n]

        host = _load_decode_kernel()
        # vLLM allocates contiguous output; the backend requires BF16.
        # The host validates this contract before launching.
        dec_out = out3[:n]
        reduction_workspace = None
        _validate_residue_geometry(
            self.num_heads, self.num_kv_heads
        )
        # Resolve through the same host policy that launches the kernel;
        # this prevents backend allocation thresholds from drifting away
        # from the measured dispatch rule.
        decode_plan = None
        # With unknown length, leave the original length fallback/selection
        # to the host rather than forwarding a plan chosen for length zero.
        if decode.max_seq_len:
            decode_plan = host.plan_qmul4_decode(
                n,
                self.num_kv_heads,
                decode.max_seq_len,
                q_dec.device,
                head_dim=self.head_size,
                grouped_head_tile=host._residue_padded_group_size(
                    self.num_heads, self.num_kv_heads),
            )
            splits, reduction = decode_plan
            if reduction == "kernel":
                # This branch requests the three-tensor kernel workspace.
                reduction_workspace = cast(
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    self._get_decode_reduction_workspace(
                        host,
                        splits,
                        n,
                        q_dec.device,
                    ),
                )
        host.run_paged_fp4_decode_qmul4_tmem(
            q_dec,
            pool,
            pool.v_data,
            pool.v_sf,
            decode.block_tables,
            decode.seq_lens,
            self.scale * pool.k_global_scale,
            out=dec_out,
            max_seq_len=decode.max_seq_len,
            k_sf_mode="linear",
            value_scale=pool.v_global_scale,
            reduction_workspace=reduction_workspace,
            plan=decode_plan,
        )
