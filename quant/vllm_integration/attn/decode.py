# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""ResAttn decode launches and per-layer reduction workspace ownership."""
from __future__ import annotations

import torch

from .runtime import _load_decode_kernel, _qk_residue_for_geometry


class ResAttnDecodeMixin:
    """Requires the head geometry, scale and workspace cache of the adapter."""

    def _try_tuned_decode(self, q, pool, md, out, *, output_scale=None,
                          output_block_scale=None):
        from quant.flashinfer_integration.resattn_tuner import (
            tuned_resattn_decode, tuner_enabled, last_decode_plan, last_tactic_source,
        )

        if not tuner_enabled() or not md.max_seq_len:
            return False
        host = _load_decode_kernel()
        residue = _qk_residue_for_geometry(self.num_heads, self.num_kv_heads)
        def workspace(plan):
            return self._get_decode_reduction_workspace(
                host, plan[0], q.shape[0], q.device, reduction=plan[1],
                residue_fold=residue, require_prepared=True)
        tuned_resattn_decode(
            q, pool, md.block_tables, md.seq_lens[:md.num_decodes],
            self.scale * pool.k_global_scale, out=out,
            max_seq_len=md.max_seq_len, k_sf_mode="linear",
            value_scale=pool.v_global_scale, residue_fold=residue,
            output_scale=output_scale, output_block_scale=output_block_scale,
            workspace_getter=workspace)
        self._decode_tuning_status = (
            last_decode_plan(), last_tactic_source(), q.shape[0], md.max_seq_len,
        )
        return True

    def _get_decode_reduction_workspace(
        self,
        host,
        splits: int,
        batch: int,
        device: torch.device,
        reduction: str = "kernel",
        residue_fold: bool = False,
        require_prepared: bool = False,
    ):
        # The CuTe kernel pads each GQA group to its supported N tile. Mirror
        # that effective head count here; using the model's unpadded HQ would
        # make the persistent workspace fail host validation for GQA < 16.
        group_size = (
            host._residue_padded_group_size(
                self.num_heads, self.num_kv_heads
            )
            if residue_fold
            else host._padded_group_size(
                self.num_heads, self.num_kv_heads
            )
        )
        effective_heads = self.num_kv_heads * group_size
        if reduction not in ("kernel", "gmem_fused"):
            raise ValueError(f"unsupported persistent reduction: {reduction}")
        # (reduction, splits, batch) plus the impl-fixed head/dim above are the
        # complete shape determinants, so a key hit is a shape match; the host
        # entry validates the tensors it receives anyway.
        key = (reduction, splits, batch, effective_heads, str(device))
        workspace = self._decode_reduction_workspaces.get(key)
        if workspace is None:
            if require_prepared and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare ResAttn reduction workspace before graph capture")
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
        host,
        plan: tuple[int, str],
    ) -> None:
        """Decode directly into vLLM's packed activation buffers."""
        n = md.num_decode_tokens
        q_dec = q_all if q_all.shape[0] == n else q_all[:n]
        if self._try_tuned_decode(
            q_dec, pool, md,
            output[:n].view(n, self.num_heads * self.head_size // 2),
            output_scale=output_scale,
            output_block_scale=output_block_scale[:((n + 127) // 128) * 128],
        ):
            return
        splits, reduction = plan
        qk_residue = _qk_residue_for_geometry(
            self.num_heads, self.num_kv_heads
        )
        reduction_workspace = None
        if reduction in ("kernel", "gmem_fused"):
            reduction_workspace = self._get_decode_reduction_workspace(
                host,
                splits,
                n,
                q_dec.device,
                reduction=reduction,
                residue_fold=qk_residue,
            )
        scale_rows = ((n + 127) // 128) * 128
        host.run_paged_fp4_decode_qmul4_tmem_nvfp4(
            q_dec,
            pool,
            pool.v_data,
            pool.v_sf,
            md.block_tables,
            md.seq_lens[: md.num_decodes],
            self.scale * pool.k_global_scale,
            output_scale,
            out=output[:n].view(n, self.num_heads * self.head_size // 2),
            output_block_scale=output_block_scale[:scale_rows],
            max_seq_len=md.max_seq_len,
            k_sf_mode="linear",
            value_scale=pool.v_global_scale,
            reduction_workspace=reduction_workspace,
            plan=plan,
            residue_fold=qk_residue,
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
        q_dec = q_all if q_all.shape[0] == n else q_all[:n]

        host = _load_decode_kernel()
        if host is None:
            raise RuntimeError(
                "[residue] ResAttn decode kernel not available: "
                "`nvfp4.cute_attn.host` could not be imported. Check that "
                "the kernel wheel from csrc is installed and nvidia-cutlass-dsl is "
                "installed (Blackwell GPU required)."
            )

        # The kernel is compiled with out_dtype=BFloat16; write straight into
        # the output slice when it already matches, else stage and cast.
        dec_out = out3[:n]
        kern_out = (
            dec_out
            if dec_out.dtype is torch.bfloat16 and dec_out.is_contiguous()
            else torch.empty_like(dec_out, dtype=torch.bfloat16)
        )
        if self._try_tuned_decode(q_dec, pool, md, kern_out):
            if kern_out is not dec_out:
                dec_out.copy_(kern_out)
            return
        reduction_workspace = None
        qk_residue = _qk_residue_for_geometry(
            self.num_heads, self.num_kv_heads
        )
        
        
        
        decode_plan = None
        # With unknown length, leave the original length fallback/selection
        # to the host rather than forwarding a plan chosen for length zero.
        if md.max_seq_len and hasattr(host, "plan_qmul4_decode"):
            decode_plan = host.plan_qmul4_decode(
                n,
                self.num_kv_heads,
                md.max_seq_len,
                q_dec.device,
            )
            splits, reduction = decode_plan
            if reduction == "kernel":
                reduction_workspace = self._get_decode_reduction_workspace(
                    host,
                    splits,
                    n,
                    q_dec.device,
                    residue_fold=qk_residue,
                )
        host.run_paged_fp4_decode_qmul4_tmem(
            q_dec,
            pool,
            pool.v_data,
            pool.v_sf,
            md.block_tables,
            md.seq_lens[: md.num_decodes],
            self.scale * pool.k_global_scale,
            out=kern_out,
            max_seq_len=md.max_seq_len,
            k_sf_mode="linear",
            value_scale=pool.v_global_scale,
            reduction_workspace=reduction_workspace,
            direct_output=True,
            plan=decode_plan,
            residue_fold=qk_residue,
        )
        if kern_out is not dec_out:
            dec_out.copy_(kern_out)
