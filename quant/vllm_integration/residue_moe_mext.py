# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'RETRACTED: folding at the expert OUTPUT deletes the residue.'

from __future__ import annotations

import os

import torch

import nvfp4

_LAYOUT_ROW_PAIR = 1


_RETRACTED = (
    'residue_moe_mext is disabled: folding after SwiGLU computes f(base)+f(res), not f(base+res). Fold the GEMM1 accumulators before applying SwiGLU.'
)


def moe_mext_enabled() -> bool:
    """Always False. See the module docstring -- the premise is invalid."""
    return False


def _sf_2d(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """View the flat mext_r1 scale buffer as the swizzled 128x4 FP8 tile grid.

    Same convention vLLM's `scaled_fp4_quant` returns, so the experts cannot
    tell our scale tensor from theirs. `rows` is the DOUBLED row count.
    """
    m_tiles = (rows + 127) // 128
    k_tiles = (cols + 63) // 64
    return flat.view(torch.float8_e4m3fn).view(m_tiles * 128, k_tiles * 4)


def double_rows_for_pairs(t: torch.Tensor) -> torch.Tensor:
    """Duplicate each row in place: [a, b] -> [a, a, b, b].

    This MUST match the `row_pair` quant layout, which emits
    [base_0, res_0, base_1, res_1, ...]. Row 2i and 2i+1 are the two halves of
    token i, so both must carry token i's routing. `repeat_interleave` is the
    interleaved duplication; `repeat`/`cat` is the concatenated one and would
    send every residue row to the wrong expert.
    """
    return t.repeat_interleave(2, dim=0)


def fold_row_pairs(t: torch.Tensor) -> torch.Tensor:
    """Sum adjacent rows along dim 0: (2M, ...) -> (M, ...).

    The inverse of `double_rows_for_pairs`, and the whole "fold" -- done here
    on the GEMM's output instead of inside its epilogue.

    Uses the experiment kernel (`nvfp4.experiment.moe_fold`) when it is
    available, which is ~2x the slice-add at decode sizes because the pair
    walk can issue 128-bit loads where two strided views cannot. Both forms
    are bitwise identical (fp32 accumulate, one rounding on store), so the
    choice is never a numerics question. Falls back silently to the slice-add
    if the kernel will not build -- that is a speed difference, not a
    correctness one.
    """
    if t.shape[0] % 2 != 0:
        raise ValueError(
            f"expected an even leading dim to fold row pairs, got {tuple(t.shape)}"
        )
    if t.is_cuda:
        try:
            from nvfp4.experiment.moe_fold import fold_pairs

            return fold_pairs(t)
        except Exception:  # noqa: BLE001 - experiment must never break serving
            pass
    return t[0::2] + t[1::2]


class ResidueMExtPrepareAndFinalize:
    """Wraps a vLLM `FusedMoEPrepareAndFinalizeModular`.

    prepare : quantize into (base, residue) row pairs -> 2M rows, and hand the
              doubled routing downstream so each residue row follows its base
              row to the same expert.
    finalize: sum the pairs back to M rows, then delegate to the inner
              implementation's weighting/reduction unchanged.

    Everything else is delegated, so this stays valid as vLLM adds methods to
    the interface (it is duck-typed via __getattr__ rather than subclassed for
    exactly that reason -- the ABC has grown several times).
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        # Only reached for attributes this class does not define.
        return getattr(self._inner, name)

    # -- prepare -----------------------------------------------------------

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config,
        defer_input_quant: bool = False,
    ):
        if defer_input_quant:
            # The backend wants to quantize inside its own fused kernel, so
            # there is no seam here to put the residue in. Fall through rather
            # than silently producing a non-residue result that looks like one.
            raise NotImplementedError(
                "RESIDUE_MOE_MEXT does not support defer_input_quant backends "
                "(AITER / FlashInfer fused quant+experts). Pick a backend that "
                "quantizes in prepare, or turn the flag off."
            )
        if not getattr(quant_config, "use_nvfp4_w4a4", False):
            raise NotImplementedError(
                "RESIDUE_MOE_MEXT requires an NVFP4 w4a4 MoE quant config; "
                f"got {quant_config!r}"
            )

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            # Applied BEFORE the residue split, so both halves carry it and
            # base+residue still sums to the weighted activation.
            a1 = a1 * topk_weights.to(a1.dtype)

        m, k = a1.shape
        a1q, a1q_scale_flat = nvfp4.scaled_fp4_quant_mext_r1(
            a1,
            quant_config.a1_gscale,
            layout_mode=_LAYOUT_ROW_PAIR,
        )
        a1q_scale = _sf_2d(a1q_scale_flat, 2 * m, k)

        # Hand the doubled routing downstream. The modular kernel prefers a
        # non-None dispatched topk over the caller's, so returning these is
        # what makes the residue rows follow their base rows.
        return (
            a1q,
            a1q_scale,
            None,
            double_rows_for_pairs(topk_ids),
            double_rows_for_pairs(topk_weights),
        )

    # -- finalize ----------------------------------------------------------

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl,
    ) -> None:
        m = output.shape[0]
        got = fused_expert_output.shape[0]
        if got != 2 * m:
            raise RuntimeError(
                "MExt MoE fold expected the expert output to carry 2M rows "
                f"(M={m} from `output`), got leading dim {got}; full shape "
                f"{tuple(fused_expert_output.shape)}. The row-pair contract "
                "only holds when the token axis is dim 0 -- a backend that "
                "flattens (M, topk) into dim 0 needs a different fold index."
            )

        folded = fold_row_pairs(fused_expert_output)
        # topk came back doubled from prepare; undo it the same way. Slicing
        # [0::2] is exact because repeat_interleave duplicated each row.
        self._inner.finalize(
            output,
            folded,
            topk_weights[0::2] if topk_weights.shape[0] == 2 * m else topk_weights,
            topk_ids[0::2] if topk_ids.shape[0] == 2 * m else topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )


def wrap_moe_kernel(moe_kernel) -> bool:
    """Wrap a built modular MoE kernel's prepare_finalize in place.

    Returns True if it wrapped, False if it declined (already wrapped, or the
    kernel has no `prepare_finalize` -- e.g. a monolithic backend).
    """
    pf = getattr(moe_kernel, "prepare_finalize", None)
    if pf is None:
        return False
    if isinstance(pf, ResidueMExtPrepareAndFinalize):
        return False
    moe_kernel.prepare_finalize = ResidueMExtPrepareAndFinalize(pf)
    return True


# ── seam 1: the FusedMoE method class ─────────────────────────────────────
#
# `ModelOptQuantConfigBase` declares LinearMethodCls / FusedMoEMethodCls /
# KVCacheMethodCls symmetrically, and `get_quant_method` dispatches on the
# attribute. Swapping FusedMoEMethodCls is the exact mirror of what
# `patch.py` already does for linear layers.


def make_residue_moe_mext_cls(base_cls: type) -> type:
    """Subclass a FusedMoE method so its modular kernel gets wrapped.

    `process_weights_after_loading` is where vLLM builds `self.moe_kernel`
    (modelopt.py:1607), so that is the earliest point at which there is a
    `prepare_finalize` object to wrap.
    """

    class ResidueMExtFusedMoE(base_cls):  # type: ignore[valid-type,misc]
        residue_mext_wrapped = False

        def process_weights_after_loading(self, layer) -> None:
            super().process_weights_after_loading(layer)
            if not moe_mext_enabled():
                return
            kernel = getattr(self, "moe_kernel", None)
            if kernel is None:
                return
            self.residue_mext_wrapped = wrap_moe_kernel(kernel)

    ResidueMExtFusedMoE.__name__ = f"ResidueMExt{base_cls.__name__}"
    ResidueMExtFusedMoE.__qualname__ = ResidueMExtFusedMoE.__name__
    return ResidueMExtFusedMoE


def patch_vllm_moe_mext_r1() -> bool:
    """Install the residue MExt MoE method. Returns True if it patched.

    Independent of the linear-side patch: a model can have residue linear
    layers and plain experts, which is exactly the state that made every MoE
    residue number so far close to a plain-NVFP4 measurement.
    """
    if os.environ.get("RESIDUE_MOE_MEXT", "0") == "1":
        # Refuse loudly rather than silently doing nothing: a null result here
        # would look like a finding about residue, not about placement.
        raise RuntimeError(_RETRACTED)
    return False
    try:  # noqa: B012 - unreachable, kept for the record
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
            ModelOptNvFp4FusedMoE,
        )
    except ImportError:
        return False

    if getattr(ModelOptNvFp4Config, "_residue_mext_moe_patched", False):
        return False

    base = getattr(
        ModelOptNvFp4Config, "_original_FusedMoEMethodCls", None
    ) or ModelOptNvFp4Config.FusedMoEMethodCls
    if base is None:
        base = ModelOptNvFp4FusedMoE
    ModelOptNvFp4Config._original_FusedMoEMethodCls = base
    ModelOptNvFp4Config.FusedMoEMethodCls = make_residue_moe_mext_cls(base)
    ModelOptNvFp4Config._residue_mext_moe_patched = True
    return True


def is_patched_moe_mext_r1() -> bool:
    """Read the class attribute, not a module global.

    vLLM spawns workers; a module-level flag resets in the child while the
    class attribute reflects the real state. Same reason `patch.is_patched`
    checks the class.
    """
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        return bool(
            getattr(ModelOptNvFp4Config, "_residue_mext_moe_patched", False)
        )
    except ImportError:
        return False
