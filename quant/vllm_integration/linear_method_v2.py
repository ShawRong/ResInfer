"""Residue NVFP4 linear method patch v2 for newer vLLM kernel APIs.

This version keeps the existing residue-specific metadata + masked quantization
logic, but delegates the standard NVFP4 path and weight post-processing to
vLLM's newer kernel registry.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm._custom_ops import cutlass_scaled_fp4_mm
from vllm.model_executor.kernels.linear import (
    CutlassNvFp4LinearKernel,
    FbgemmNvFp4LinearKernel,
    FlashInferCudnnNvFp4LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferTrtllmNvFp4LinearKernel,
    MarlinNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.emulation import (
    EmulationNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4LinearMethod,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    slice_nvfp4_output,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

try:
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        swizzle_blockscale,
    )
except ImportError:
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

from .layout_utils import prepare_decode_mext_weight_layout_from_raw_scale
from .linear_debug import maybe_tap_tensor
from .linear_utils import _InputQuantizerState, get_extended_dim, get_layer_metadata
from .residue_tp import (
    SF_VEC_SIZE as SF_VEC_SIZE_FOR_TP,
    current_tp_rank,
    plan_from_partition,
)
# Side-effect import: registers `torch.ops.residue.nvfp4_linear`, the opaque
# custom_op that wraps the residue NVFP4 GEMM pipeline (both mext-fold and
# legacy K-ext kernel chains) so torch.compile sees one black-box op call per
# residue forward — preserves correct per-layer routing across all captured
# CUDA graph sizes.
from . import residue_linear_op  # noqa: F401

try:
    from vllm.model_executor.kernels.linear import FlashInferB12xNvFp4LinearKernel
except ImportError:  # Older vLLM builds do not expose this optional backend.
    FlashInferB12xNvFp4LinearKernel = None  # type: ignore[assignment]

try:
    from vllm.model_executor.kernels.linear import FlashInferCuteDslNvFp4LinearKernel
except ImportError:  # Added after the pinned development wheel.
    FlashInferCuteDslNvFp4LinearKernel = None  # type: ignore[assignment]


# Process-startup cache: `RESIDUE_SWAP=1` toggles the swap-residue input
# layout in the legacy K-ext quant kernel. Read once at module import so
# the apply hot path does not call `os.environ.get` per forward (the V1
# inline path used to read it per-call; harmless for eager, but a Python
# side-effect that torch.compile would graph-break on under non-eager).
_USE_SWAP = os.environ.get("RESIDUE_SWAP", "0") == "1"

# Process-startup cache: `RESIDUE_ENABLE_DECODE_MEXT=1` enables fold-layout
# prep for legacy `extended_k` residue layers — V2.PWAL snapshots the raw
# FP8 weight scale before super-swizzle, then builds the base-K prefix
# tensors via `prepare_decode_mext_weight_layout_from_raw_scale`. Without
# this flag, extended_k layers never set `_decode_mext_layout_enabled` and
# the op dispatcher always takes the legacy K-extension branch. Pure
# `mext_r1` layers are always fold-eligible regardless of this flag — they
# don't have a legacy fallback layout to choose from.
_DECODE_MEXT_ENABLED = os.environ.get("RESIDUE_ENABLE_DECODE_MEXT", "0") == "1"


def _is_kernel_instance(kernel: object, kernel_cls: type | None) -> bool:
    return kernel_cls is not None and isinstance(kernel, kernel_cls)


_FOLD_REGISTER_WARNED = set()


def _warn_once_fold_register(tag, exc):
    if tag in _FOLD_REGISTER_WARNED:
        return
    _FOLD_REGISTER_WARNED.add(tag)
    print(f"[residue] sm120 fold shape registration failed ({tag}): "
          f"{type(exc).__name__}: {exc}", flush=True)


def _register_fold_shape_for(layer, tag: str) -> None:
    """Pre-compile the SM120 CuTeDSL fold for this layer's shape.

    There are THREE places that mark a layer fold-eligible -- ext-K prefix
    (layout_utils), plain NVFP4 with activation-side mext, and mext_r1 -- and
    every one of them must register or the warmup silently under-covers.
    Missing just the plain-NVFP4 branch left 4 of 9 live shapes uncompiled,
    which showed up only as cudagraph capture taking 46 s instead of 5 s.

    The key is derived from the SAME view the op will pass the kernel, so the
    warmup and the runtime cannot disagree about the row stride.
    """
    try:
        from .residue_linear_op import (_base_k_weight_view,
                                        register_sm120_fold_shape)

        k_base = int(getattr(layer, "_mext_decode_k_base", 0))
        if k_base <= 0:
            return
        view = _base_k_weight_view(layer.weight, k_base)
        register_sm120_fold_shape(
            int(view.shape[0]), k_base, int(view.stride(0)) * 2,
            is_mext_r1=(tag == "mext_r1"))
    except Exception as e:                                # noqa: BLE001
        _warn_once_fold_register(tag, e)


class ResidueNvFp4LinearMethodV2(ModelOptNvFp4LinearMethod):
    """Patch v2 of the residue linear method for the newer vLLM NVFP4 stack."""

    # Runtime mode tags stored in residue_kernel_metadata.json per layer.
    # Drives PWAL branching and `_get_extended_dim` behaviour.
    #   standard   — vanilla NVFP4, no residue extension
    #   extended_k — legacy K-extension residue (weight = K_base + num_salient)
    #   mext_r1    — pure M-extension residue (weight stays at K_base, activation
    #                doubled along M; PWAL bypasses the ModelOpt parent path)
    _RUNTIME_MODE_STANDARD = "standard"
    _RUNTIME_MODE_MEXT_R1 = "mext_r1"
    _RUNTIME_MODE_EXTENDED_K = "extended_k"

    def __init__(self, quant_config, layer_prefix: str = ""):
        super().__init__(quant_config)
        self.layer_prefix = layer_prefix
        self.group_size = getattr(quant_config, "group_size", 16)
        self._nvfp4 = None

    def _get_runtime_mode(self, layer_name: str) -> str:
        """Return the runtime mode for a layer.

        Precedence:
          1. Explicit residue metadata (exported residue checkpoints).
          2. Global mext_r1 patch (RESIDUE_PATCH_MODE=mext_r1) applied to a
             plain NVFP4 checkpoint: force mext_r1 fold on attention
             projection layers only (by name); everything else — MoE experts,
             MLP gate_up/down — stays STANDARD and delegates to native vLLM.
             This is how decode-MExt runs on a stock NVFP4 model with no
             residue export (the residue is computed at runtime).
        """
        meta = get_layer_metadata(self.quant_config, layer_name)
        if meta:
            return meta.get("runtime_mode", self._RUNTIME_MODE_EXTENDED_K)
        if self._forced_mext_r1_for_layer(layer_name):
            return self._RUNTIME_MODE_MEXT_R1
        return self._RUNTIME_MODE_STANDARD

    # Linear-projection layer-name fragments the global mext_r1 patch scopes the
    # fold to. DEFAULT: ALL linear projections (attention q/k/v/o + MLP
    # gate/up/down). MoE expert / router GEMMs are never folded.
    #
    # Override via RESIDUE_MEXT_PROJ_NAMES (comma-separated fragments). To fold
    # attention only, set it to "q_proj,k_proj,v_proj,qkv_proj,o_proj". The
    # legacy name RESIDUE_MEXT_ATTN_PROJ_NAMES is still honored as an alias.
    _DEFAULT_FOLD_PROJ_NAMES = (
        "q_proj,k_proj,v_proj,qkv_proj,o_proj,"
        "gate_proj,up_proj,gate_up_proj,down_proj"
    )
    _FOLD_PROJ_NAME_FRAGMENTS = tuple(
        s.strip() for s in os.environ.get(
            "RESIDUE_MEXT_PROJ_NAMES",
            os.environ.get(  # backward-compat alias
                "RESIDUE_MEXT_ATTN_PROJ_NAMES", _DEFAULT_FOLD_PROJ_NAMES),
        ).split(",") if s.strip()
    )

    def _forced_mext_r1_for_layer(self, layer_name: str) -> bool:
        """True iff the global mext_r1 patch should force this (metadata-less)
        layer to the fold path -- i.e. the layer is one of the configured linear
        projections (RESIDUE_MEXT_PROJ_NAMES; all linear projections by
        default)."""
        from .patch import is_patched_mext_r1

        if not is_patched_mext_r1():
            return False
        return any(frag in layer_name for frag in self._FOLD_PROJ_NAME_FRAGMENTS)

    def _uses_cutlass_nvfp4_kernel(self) -> bool:
        # Both vLLM's native CUTLASS backend and FlashInfer's CUTLASS runner
        # use the same NVFP4 weight/scale contract that our opaque residue op
        # consumes. Treat both as custom-op capable; otherwise selecting the
        # FlashInfer-CUTLASS backend silently disables decode-MExt layout prep.
        return isinstance(
            self.kernel,
            (CutlassNvFp4LinearKernel, FlashInferCutlassNvFp4LinearKernel),
        )

    def _has_decode_mext_layout(self, layer: torch.nn.Module) -> bool:
        return (
            getattr(layer, "_decode_mext_layout_enabled", False)
            and hasattr(layer, "weight_scale_base")
            and hasattr(layer, "_mext_decode_k_base")
        )

    def _needs_residue_or_mext_path(self, layer: torch.nn.Module) -> bool:
        """Whether this layer needs the custom residue/MExt forward.

        Standard NVFP4 layers should remain fully delegated to vLLM's selected
        backend. Only true residue layers, or cutlass layers with an explicitly
        prepared decode-MExt fold layout, need our custom path.
        """
        if getattr(layer, "_is_residue_layer", False):
            return True
        return (
            self._uses_cutlass_nvfp4_kernel()
            and self._has_decode_mext_layout(layer)
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create native vLLM NVFP4 weights, plus residue metadata state.

        V2 intentionally follows vLLM's ModelOptNvFp4LinearMethod weight
        contract. AWQ pre_quant_scale must be folded during export and is not
        registered or applied at runtime.
        """
        layer_name = getattr(layer, "prefix", "") or self.layer_prefix
        extended_dim = self._get_extended_dim(layer_name)
        if os.environ.get("RESIDUE_DEBUG_CREATE_WEIGHTS", "0") == "1":
            print(
                "[residue:create_weights] "
                f"layer_name={layer_name!r} "
                f"input_size_per_partition={input_size_per_partition} "
                f"extended_dim={extended_dim} "
                f"decode_mext_enabled={_DECODE_MEXT_ENABLED}",
                flush=True,
            )
        # Row-parallel layers (o_proj / down_proj) arrive with the input
        # already sharded, so the extended K has to be sharded too. The plan
        # is built from these arguments alone -- `input_size_per_partition <
        # input_size` IS the row-parallel signal -- so no process-group
        # lookup is needed and the rule stays unit-testable. Column-parallel
        # layers get None back: their input is whole, so the old
        # override-with-the-full-extended-K is exactly right.
        shard_plan = None
        if extended_dim is not None:
            meta = get_layer_metadata(self.quant_config, layer_name) or {}
            num_salient_meta = int(meta.get("num_salient", 0))
            shard_plan = plan_from_partition(
                extended_dim=extended_dim,
                input_size_per_partition=input_size_per_partition,
                input_size=input_size,
                num_salient=num_salient_meta,
                tp_rank=current_tp_rank(),
            )

        if extended_dim is not None:
            original_input = input_size_per_partition
            if shard_plan is not None:
                # Per-rank extended width == K_ext / tp. `input_size` stays
                # logical-and-whole: vLLM only uses it to derive shard sizes,
                # and our loader does the extended gather itself.
                input_size_per_partition = shard_plan.k_ext_shard
                input_size = extended_dim
            else:
                input_size_per_partition = extended_dim
                input_size = extended_dim
            print(
                f"[residue] {layer_name}: overriding input_size "
                f"{original_input} -> {input_size_per_partition}"
                + (
                    f" (row-parallel tp={shard_plan.tp_size} "
                    f"rank={shard_plan.tp_rank}: "
                    f"{shard_plan.base_shard} base + "
                    f"{shard_plan.salient_shard} salient)"
                    if shard_plan is not None
                    else ""
                )
            )

        ModelOptNvFp4LinearMethod.create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        # Row-parallel: permute the checkpoint's extended columns so that the
        # stock loader's contiguous narrow lands on this rank's two ranges.
        # Cheaper and far less invasive than replacing the loader, and it
        # keeps every vLLM parameter subclass working untouched.
        if shard_plan is not None:
            layer._residue_shard_plan = shard_plan
            self._wrap_extended_weight_loaders(layer, shard_plan)

        # Receive checkpoint keys such as
        # <layer>.input_quantizer._salient_indices without changing the native
        # vLLM NVFP4 weight contract.
        layer.register_module("input_quantizer", _InputQuantizerState())
        layer._is_residue_layer = False
        layer.num_salient = 0

    @staticmethod
    def _wrap_extended_weight_loaders(layer, shard_plan) -> None:
        """Interleave extended columns before vLLM's loader slices them.

        `weight` is packed FP4 (two nibbles per byte, so K_ext/2 columns) and
        `weight_scale` holds one scale per 16 channels -- hence the differing
        `scale` factors. Anything whose last dim is not an extended width is
        left alone: it is not a K-extended tensor.
        """
        from .residue_tp import interleave_extended_for_tp

        layer_tag = getattr(layer, 'prefix', '?')
        targets = (("weight", 2), ("weight_scale", SF_VEC_SIZE_FOR_TP))
        for name, scale in targets:
            param = getattr(layer, name, None)
            if param is None:
                continue
            orig = getattr(param, "weight_loader", None)
            if orig is None:
                continue

            def make(orig_loader, col_scale):
                def loader(param_, loaded_weight, *args, **kwargs):
                    expected = shard_plan.k_ext // col_scale
                    fired = loaded_weight.shape[-1] == expected
                    if fired:
                        loaded_weight = interleave_extended_for_tp(
                            loaded_weight, shard_plan, scale=col_scale
                        )
                    if os.environ.get("RESIDUE_DEBUG_TP", "0") == "1":
                        print(f"[residue:tp-loader] {layer_tag} "
                              f"loaded={tuple(loaded_weight.shape)} "
                              f"expected_last={expected} interleaved={fired} "
                              f"param={tuple(param_.data.shape)}", flush=True)
                    return orig_loader(param_, loaded_weight, *args, **kwargs)

                return loader

            param.weight_loader = make(orig, scale)

    def _get_extended_dim(self, layer_name: str) -> int | None:
        """Look up the extended K for a residue layer.

        `mext_r1` layers keep weights at original K (residue lives in the M
        dimension at runtime), so they must NOT receive an extended-K override
        from `create_weights`. All other modes use the metadata-driven extension.
        """
        if self._get_runtime_mode(layer_name) == self._RUNTIME_MODE_MEXT_R1:
            return None
        return get_extended_dim(self.quant_config, layer_name)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Set up residue layer state, dispatching on runtime_mode.

        Three branches:
          • mext_r1   — pure M-extension layer. Bypass the ModelOpt parent PWAL
                        entirely and build everything inline (`_pwal_mext_r1`).
                        Weight stays at original K; fold layout is the only
                        layout (no legacy fallback to swizzle for).
          • extended_k / standard — call the inherited ModelOpt PWAL after our
                        residue metadata setup. For cutlass + decode_mext_enabled
                        residue layers, snapshot the raw FP8 weight scale BEFORE
                        super-swizzle, then build the base-K prefix layout via
                        `prepare_decode_mext_weight_layout_from_raw_scale`.
        """
        layer_name = getattr(layer, "prefix", "") or self.layer_prefix
        runtime_mode = self._get_runtime_mode(layer_name)

        if runtime_mode == self._RUNTIME_MODE_MEXT_R1:
            self._pwal_mext_r1(layer, layer_name)
            return

        # extended_k / standard path
        #
        # The raw weight_scale snapshot has to happen here, at the very top of
        # PWAL, because the inherited ModelOpt PWAL (called at the end) swizzles
        # `layer.weight_scale` in place. The decode-mext layout prep needs the
        # un-swizzled FP8 scale to slice off the base-K prefix in raw layout.
        raw_weight_scale_ext = None
        if _DECODE_MEXT_ENABLED and self._uses_cutlass_nvfp4_kernel():
            raw_weight_scale_ext = layer.weight_scale.detach()

        salient_indices = None
        if hasattr(layer, "input_quantizer"):
            indices = layer.input_quantizer._salient_indices
            if indices is not None and indices.numel() > 0:
                salient_indices = indices.to(layer.weight.device)

        if salient_indices is None:
            meta = get_layer_metadata(self.quant_config, layer_name)
            if meta and meta.get("salient_indices"):
                salient_indices = torch.tensor(
                    meta["salient_indices"],
                    dtype=torch.int64,
                    device=layer.weight.device,
                )
                if hasattr(layer, "input_quantizer"):
                    layer.input_quantizer._salient_indices = salient_indices

        # Row-parallel: everything below derives from the weight this rank
        # actually holds (K_ext/tp wide), so the salient indices have to be
        # this rank's too -- otherwise `original_in_features` would subtract
        # the GLOBAL salient count from a per-rank extended width and the
        # channel mask would be built over the wrong K. Rebasing here keeps
        # both `num_salient` and `channel_mask` rank-local with no further
        # changes downstream.
        shard_plan = getattr(layer, "_residue_shard_plan", None)
        if shard_plan is not None and salient_indices is not None:
            salient_indices = shard_plan.local_salient_indices(salient_indices)

        if salient_indices is not None and salient_indices.numel() > 0:
            layer._is_residue_layer = True
            layer.num_salient = salient_indices.numel()
            layer._salient_indices = salient_indices
            if self._nvfp4 is None:
                import nvfp4  # noqa: F401

                self._nvfp4 = nvfp4

        packed_in = layer.weight.shape[1]
        extended_in = packed_in * 2
        layer.extended_in_features = extended_in

        if layer._is_residue_layer:
            layer.original_in_features = extended_in - layer.num_salient
            if self._nvfp4 is None:
                import nvfp4  # noqa: F401

                self._nvfp4 = nvfp4
            layer.channel_mask = self._nvfp4.indices_to_channel_masks(
                layer._salient_indices,
                layer.original_in_features,
                layer_name=layer_name,
            ).to(layer.weight.device)
        else:
            layer.original_in_features = extended_in
            layer.channel_mask = None

        ModelOptNvFp4LinearMethod.process_weights_after_loading(self, layer)

        layer._runtime_mode = runtime_mode
        layer._decode_mext_layout_enabled = False
        if _DECODE_MEXT_ENABLED and self._uses_cutlass_nvfp4_kernel():
            if (
                getattr(layer, "_is_residue_layer", False)
                and raw_weight_scale_ext is not None
            ):
                # Residue layer: K-ext weight; weight_scale_base = first k_base
                # groups of the extended scale (swizzled by prep helper).
                prepare_decode_mext_weight_layout_from_raw_scale(
                    layer,
                    raw_weight_scale_ext,
                )
            else:
                # Plain NVFP4 layer: no K-extension. Mext fold = activation-side
                # 2-stage FP4 quant (precision boost at small M); weight is
                # unchanged. weight_scale_base = full (already-swizzled)
                # weight_scale; k_base = full K. Op decides fold-vs-plain-bypass
                # at runtime based on M, so non-decode workloads pay nothing.
                layer._decode_mext_layout_enabled = True
                layer.weight_scale_base = layer.weight_scale
                layer._mext_decode_k_base = int(layer.original_in_features)
                _register_fold_shape_for(layer, "plain_nvfp4")
        if os.environ.get("RESIDUE_DEBUG_CREATE_WEIGHTS", "0") == "1":
            print(
                "[residue:post_process_weights] "
                f"layer_name={layer_name!r} "
                f"is_residue={getattr(layer, '_is_residue_layer', False)} "
                f"uses_cutlass={self._uses_cutlass_nvfp4_kernel()} "
                f"decode_mext_enabled={_DECODE_MEXT_ENABLED} "
                f"decode_layout={getattr(layer, '_decode_mext_layout_enabled', False)} "
                f"has_weight_scale_base={hasattr(layer, 'weight_scale_base')} "
                f"k_base={getattr(layer, '_mext_decode_k_base', None)}",
                flush=True,
            )

    def _pwal_mext_r1(self, layer: torch.nn.Module, layer_name: str) -> None:
        """PWAL for pure mext_r1 layers — bypasses the ModelOpt parent.

        Unlike `extended_k`, mext_r1 weights stay at original K and have no
        residue extension to swizzle out. We compute global scales, swizzle
        the block scale, and mark the layer fold-eligible with the full K as
        `_mext_decode_k_base` and `weight_scale_base = weight_scale` (i.e. the
        whole scale doubles as its own prefix scale).
        """
        if not self._uses_cutlass_nvfp4_kernel():
            raise RuntimeError(
                "M-extension R1 currently supports only the cutlass backend. "
                f"Got kernel={type(self.kernel).__name__!r} for layer {layer_name}."
            )

        if self._nvfp4 is None:
            import nvfp4  # noqa: F401

            self._nvfp4 = nvfp4

        layer._runtime_mode = self._RUNTIME_MODE_MEXT_R1
        # Mark fold-eligible for V2's op dispatcher. mext_r1 layers have no
        # separate salient channels — the M-extension layout IS the residue.
        layer._is_residue_layer = True
        layer._decode_mext_layout_enabled = True
        layer._salient_indices = None
        layer.num_salient = 0
        layer.channel_mask = None

        packed_in = layer.weight.shape[1]
        original_in = packed_in * 2
        layer.extended_in_features = original_in
        layer.original_in_features = original_in
        # Full K is the only K — fold reads the entire weight, no prefix slice.
        layer._mext_decode_k_base = int(original_in)
        # Register AFTER _mext_decode_k_base is set: the registrar reads it and
        # silently returns on 0. Registering ten lines earlier meant NO mext_r1
        # layer was ever recorded, which starved both the sm120 pre-compile and
        # the sm10x tuning-bucket ladder -- found live by the sm100 e2e warmup
        # audit (registry empty, tune_sizes=[128], prefill on the static table).
        _register_fold_shape_for(layer, "mext_r1")

        input_global_scale = layer.input_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
        del layer.input_scale

        weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
        layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
        del layer.weight_scale_2

        layer.alpha = Parameter(
            layer.input_global_scale * layer.weight_global_scale,
            requires_grad=False,
        )
        layer.input_global_scale_inv = Parameter(
            (1.0 / layer.input_global_scale).to(torch.float32),
            requires_grad=False,
        )

        assert layer.weight_scale.dtype == torch.float8_e4m3fn, (
            "Weight blockscale must be represented as FP8-E4M3"
        )
        layer.weight_scale = Parameter(
            swizzle_blockscale(layer.weight_scale.data),
            requires_grad=False,
        )
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        # Full scale doubles as "base scale" for the op dispatcher — fold reads
        # the entire weight, so no separate prefix scale is needed.
        layer.weight_scale_base = layer.weight_scale


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = maybe_tap_tensor(x)
        # Keep ordinary NVFP4 layers on vLLM's selected backend. We only
        # intercept layers that actually need residue quantization or an
        # explicitly prepared decode-MExt fold layout.
        if self._needs_residue_or_mext_path(layer):
            out = self._apply_residue_v2(layer, x, bias)
        else:
            out = self.kernel.apply_weights(layer=layer, x=x, bias=bias)
        return maybe_tap_tensor(out)

    def _apply_residue_v2(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dispatch residue forward through the right code path.

        Cutlass backend (the production hot path on sm_120) routes through the
        opaque `residue::nvfp4_linear` custom op so torch.compile sees a
        single black-box call — preserves correct mext-fold vs legacy-K-ext
        dispatch (per-layer attribute) across all captured CUDA graph sizes.

        Other NVFP4 backends (cudnn / fbgemm / trtllm / non-CUTLASS
        FlashInfer variants) keep the inline residue implementation. Standard
        non-residue layers do not enter this method and are delegated directly
        to vLLM's selected backend.
        """
        if self._uses_cutlass_nvfp4_kernel():
            return self._apply_residue_via_op(layer, x, bias)
        return self._apply_residue_inline(layer, x, bias)

    def _apply_residue_via_op(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Route residue forward through the opaque custom op.

        Per-layer attributes decide which path the op dispatches to:
          • runtime_mode == "mext_r1"          → fold with full K
          • _decode_mext_layout_enabled = True → fold with prefix K
          • otherwise                          → legacy K-extension path
        All three select the same op symbol; only the arguments differ.

        torch.compile captures one node per invocation with the per-layer arg
        values baked in. cudagraph replay re-executes that node verbatim, so
        kernel choice is consistent across all batch sizes — fixing the trace
        baking bug where a single trace at max-M used to lock every captured
        graph to the legacy K-ext branch.

        V2 stays agnostic to *how* the mext layout was produced — both real
        M-extension layouts (pure mext_r1 layers) and K-extension + prepare-
        time mext attrs (extended_k layers) end up with the same set of
        attributes: `_decode_mext_layout_enabled = True`, `weight_scale_base`,
        and `_mext_decode_k_base`. The MExt subclass's PWAL is responsible
        for normalising both into that shape so we have a single dispatch
        condition here.
        """
        # `fold_eligible` is a per-layer constant (does the layer have a
        # base-K scale and a valid prefix k_base?). The op's body uses this
        # *together with* the runtime M of each invocation to decide whether
        # to actually take the fold branch — so a residue layer is "fold-able
        # for small M, legacy for large M" automatically, with each captured
        # cudagraph specialising to the right path for its capture size.
        fold_eligible = (
            getattr(layer, "_decode_mext_layout_enabled", False)
            and hasattr(layer, "weight_scale_base")
            and hasattr(layer, "_mext_decode_k_base")
        )

        if fold_eligible:
            weight_scale_for_fold = layer.weight_scale_base
            k_base = int(layer._mext_decode_k_base)
        else:
            weight_scale_for_fold = layer.weight_scale  # placeholder, op ignores
            k_base = 0

        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        output_n = layer.weight.shape[0]

        if x_2d.shape[0] == 0:
            empty = x_2d.new_empty((0, output_n))
            if bias is not None:
                empty = empty + bias
            return empty.view(*output_shape)

        # Placeholder for legacy-mode-only param when fold mode is active.
        # In fold mode the op ignores channel_mask, but the op signature still
        # requires a tensor — provide a 1-byte zero buffer if the layer has
        # none (mext-only layers don't have a channel_mask).
        channel_mask = getattr(layer, "channel_mask", None)
        if channel_mask is None:
            channel_mask = x_2d.new_zeros(1, dtype=torch.uint8)

        output = torch.ops.residue.nvfp4_linear(
            x_2d,
            layer.weight,
            layer.input_global_scale_inv,
            layer.weight_scale,
            weight_scale_for_fold,
            channel_mask,
            layer.alpha,
            int(k_base),
            int(getattr(layer, "num_salient", 0)),
            int(getattr(layer, "weights_padding_cols", 0)),
            int(output_n),
            bool(fold_eligible),
            _USE_SWAP,
            # mext_r1 layers were charged for a residue, so they must apply it
            # at EVERY M -- see the dispatch comment in residue_linear_op. A
            # plain-NVFP4 layer is also fold_eligible (the small-M fold is a
            # free precision bonus there) but never paid, so it keeps the
            # threshold. The two are otherwise indistinguishable in this
            # signature: both carry num_salient=0 and k_base == full K.
            bool(getattr(layer, "_runtime_mode", None)
                 == self._RUNTIME_MODE_MEXT_R1),
        )

        if bias is not None:
            output = output + bias
        return output.view(*output_shape)

    def _apply_residue_inline(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Original non-op implementation, kept for non-cutlass backends.

        torch.compile traces into this function (it is not wrapped as a custom
        op), so any `x_fp4.shape[*]` access materialises a SymInt at trace
        time — fine for eager but a known graph-break / host-sync trigger
        under non-eager. The cutlass backend has been migrated to
        `_apply_residue_via_op` to avoid that. Backends below are not on the
        hot path and remain on this inline path for now.
        """
        assert self._nvfp4 is not None, (
            "nvfp4 extension must be loaded in process_weights_after_loading()"
        )

        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]
        x_2d = x.reshape(-1, x.shape[-1])

        use_swap = _USE_SWAP
        x_fp4, x_sf_block = self._nvfp4.scaled_fp4_quant_with_mask(
            x_2d,
            layer.input_global_scale_inv,
            layer.channel_mask,
            layer.num_salient,
            use_swapped_residue=use_swap,
        )

        n_ext = x_fp4.shape[1] * 2
        num_m_tiles = (x_fp4.shape[0] + 127) // 128
        num_k_tiles = (n_ext + 63) // 64
        x_sf_block = x_sf_block.view(torch.float8_e4m3fn).view(
            num_m_tiles * 128,
            num_k_tiles * 4,
        )

        kernel = self.kernel
        if isinstance(kernel, CutlassNvFp4LinearKernel):
            out = self._apply_cutlass_residue(
                layer, x_fp4, x_sf_block, output_dtype
            )
        elif isinstance(kernel, FlashInferCutlassNvFp4LinearKernel):
            out = self._apply_flashinfer_residue(
                layer, x_fp4, x_sf_block, output_dtype, backend="cutlass"
            )
        elif isinstance(kernel, FlashInferCudnnNvFp4LinearKernel):
            out = self._apply_flashinfer_residue(
                layer, x_fp4, x_sf_block, output_dtype, backend="cudnn"
            )
        elif _is_kernel_instance(kernel, FlashInferCuteDslNvFp4LinearKernel):
            out = self._apply_flashinfer_residue(
                layer, x_fp4, x_sf_block, output_dtype, backend="cute-dsl"
            )
        elif _is_kernel_instance(kernel, FlashInferB12xNvFp4LinearKernel):
            out = self._apply_flashinfer_residue(
                layer, x_fp4, x_sf_block, output_dtype, backend="b12x"
            )
        elif isinstance(kernel, FlashInferTrtllmNvFp4LinearKernel):
            out = self._apply_flashinfer_residue(
                layer, x_fp4, x_sf_block, output_dtype, backend="trtllm"
            )
        elif isinstance(kernel, FbgemmNvFp4LinearKernel):
            out = self._apply_fbgemm_residue(
                layer, x_fp4, x_sf_block, output_dtype
            )
        elif isinstance(kernel, MarlinNvFp4LinearKernel):
            raise RuntimeError(
                "Residue NVFP4 patch v2 does not support the Marlin backend."
            )
        elif isinstance(kernel, EmulationNvFp4LinearKernel):
            raise RuntimeError(
                "Residue NVFP4 patch v2 does not support the emulation backend."
            )
        else:
            raise RuntimeError(
                f"Unsupported NVFP4 kernel for residue patch v2: {type(kernel).__name__}"
            )

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)

    @staticmethod
    def _apply_cutlass_residue(
        layer: torch.nn.Module,
        x_fp4: torch.Tensor,
        x_sf_block: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        x_fp4 = pad_nvfp4_activation_for_cutlass(
            x_fp4, getattr(layer, "weights_padding_cols", 0)
        )
        out = cutlass_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_sf_block,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
        )
        return slice_nvfp4_output(out, layer.output_size_per_partition)

    @staticmethod
    def _apply_flashinfer_residue(
        layer: torch.nn.Module,
        x_fp4: torch.Tensor,
        x_sf_block: torch.Tensor,
        output_dtype: torch.dtype,
        *,
        backend: str,
    ) -> torch.Tensor:
        if backend in {"cutlass", "cudnn", "cute-dsl", "b12x"}:
            x_fp4 = pad_nvfp4_activation_for_cutlass(
                x_fp4, getattr(layer, "weights_padding_cols", 0)
            )

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_sf_block,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
            backend=backend,
        )
        return slice_nvfp4_output(out, layer.output_size_per_partition)

    @staticmethod
    def _apply_fbgemm_residue(
        layer: torch.nn.Module,
        x_fp4: torch.Tensor,
        x_sf_block: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        import fbgemm_gpu  # noqa: F401

        out = torch.ops.fbgemm.f4f4bf16(
            x_fp4,
            layer.weight,
            x_sf_block.view(-1).view(torch.uint8),
            layer.weight_scale,
            layer.alpha,
            use_mx=False,
        ).to(output_dtype)
        return slice_nvfp4_output(out, layer.output_size_per_partition)
