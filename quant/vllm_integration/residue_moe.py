# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
'Residue coverage for MoE expert layers.'
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field


@dataclass
class MoECoverageReport:
    """What the residue patch would have to cover on this model."""

    layers: dict[str, dict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, name: str, **facts) -> None:
        with self._lock:
            self.layers.setdefault(name, {}).update(facts)

    def summary(self) -> dict:
        n_experts = sum(v.get("num_experts", 0) for v in self.layers.values())
        params = sum(v.get("expert_params", 0) for v in self.layers.values())
        applies = sum(v.get("applies", 0) for v in self.layers.values())
        return {
            "moe_layers": len(self.layers),
            "total_experts": n_experts,
            "expert_params": params,
            "apply_calls": applies,
        }

    def render(self) -> str:
        s = self.summary()
        head = (
            f"[residue:moe] {s['moe_layers']} MoE layers, "
            f"{s['total_experts']} experts total, "
            f"{s['expert_params'] / 1e9:.2f} B expert params, "
            f"{s['apply_calls']} apply() calls -- "
            "NONE of this is under residue today"
        )
        rows = []
        for name, v in sorted(self.layers.items())[:3]:
            rows.append(
                f"    {name}: E={v.get('num_experts')} "
                f"w13={v.get('w13_shape')} w2={v.get('w2_shape')}"
            )
        return "\n".join([head, *rows])


REPORT = MoECoverageReport()


def observer_enabled() -> bool:
    return os.environ.get("RESIDUE_MOE_OBSERVE", "0") == "1"


def moe_method_enabled() -> bool:
    return os.environ.get("RESIDUE_ENABLE_MOE", "0") == "1"


def resize_w13_for_extended_k(layer, num_experts: int, w13_rows: int,
                              k_ext: int, weight_loader) -> None:
    """Re-register w13 weight+scale at the K-extended width.

    The stock create_weights sizes every param from the model's hidden
    size; an extended_k checkpoint carries w13 tensors at K_ext = K +
    num_salient, and the loader hard-asserts param.shape == loaded.shape.
    Only w13 changes: w2 uses hidden as its OUTPUT dim and stays base-K.
    Mirrors the dense method's input_size override, at the FusedMoE seam.
    """
    import torch
    from vllm.model_executor.parameter import ModelWeightParameter

    w13 = layer.w13_weight
    scale = layer.w13_weight_scale
    group = w13.shape[-1] * 2 // scale.shape[-1]  # fp4 packing x2 / group
    for name, cols, dtype in (
            ("w13_weight", k_ext // 2, w13.dtype),
            ("w13_weight_scale", k_ext // group, scale.dtype)):
        param = ModelWeightParameter(
            data=torch.empty(num_experts, w13_rows, cols, dtype=dtype,
                             device=getattr(layer, name).data.device),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter(name, param)


def make_residue_moe_method_cls(base_cls: type) -> type:
    """Subclass vLLM's NVFP4 MoE method; experts go to ResidueCuteDSLExperts.

    Built at patch time from whatever `FusedMoEMethodCls` currently is, so no
    vLLM version is pinned. Three touch points, everything else inherited:

      * __init__ re-runs the support check with OUR experts class and, when
        it passes, sets `self.experts_cls` -- the base's modular-kernel
        machinery (make_nvfp4_moe_kernel, is_monolithic, apply) picks it up
        from there with no further changes.
      * process_weights_after_loading skips the stock backend repack: the
        residue experts consume the RAW modelopt layout (linear SF blocks,
        scales as loaded) and run their own algebra + swizzle when the
        method calls `fused_experts.process_weights_after_loading(layer)`,
        exactly as the stock path does for its own experts.
      * unsupported configs print the reason and fall back to the stock
        method wholesale.
    """
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    # The class name MUST contain the substring "ModelOpt": vLLM's
    # RoutedExperts.weight_loader dispatches its ModelOpt-specific loading
    # path on `"ModelOpt" in type(quant_method).__name__` (routed_experts.py,
    # marked TODO upstream). A name without it falls through to the generic
    # scale loader, which demands a `quant_method` attr ModelOpt params never
    # set -- and the load dies with "quant method must be one of [...]".
    # Found the hard way on B200: baseline loaded, this subclass (then named
    # ResidueNvFp4FusedMoEMethod) did not.
    class ModelOptNvFp4FusedMoEResidue(base_cls):  # type: ignore[misc,valid-type]
        __doc__ = (
            "vLLM's NVFP4 FusedMoE method with the expert compute routed "
            "through ResidueCuteDSLExperts (MExt-R1 on the CuTeDSL graft)."
        )

        def __init__(self, quant_config, moe_config) -> None:
            super().__init__(quant_config, moe_config)
            self._residue_impl = None
            # Per-layer policy (metadata "moe" section). The method instance
            # is per-layer (ModelOptQuantConfigBase.get_quant_method builds a
            # fresh one per RoutedExperts), but the layer NAME only appears at
            # process_weights_after_loading -- so __init__ records the stock
            # state needed to honor an "off" verdict there, and pwal decides.
            from .residue_moe_policy import load_moe_policy_section
            self._moe_policy_section = load_moe_policy_section()
            self._stock_experts_cls = getattr(self, "experts_cls", None)
            self._stock_nvfp4_backend = getattr(self, "nvfp4_backend", None)
            if getattr(self, "use_a16", False):
                print("[residue:moe] W4A16 checkpoint -- stock experts",
                      flush=True)
                return
            
            
            
            
            
            
            
            
            impl = os.environ.get("RESIDUE_MOE_IMPL", "cutedsl")
            if impl != "cutedsl":
                raise ValueError(
                    f"[residue:moe] RESIDUE_MOE_IMPL={impl!r} is not "
                    "supported. "
                    "Unset it or set 'cutedsl'.")
            # The grafted flashinfer-CuTeDSL path. Weights must go
            # through the STOCK FLASHINFER_CUTEDSL prep (interleave +
            # MMA-layout SF + alpha vectors), so pwal is NOT overridden
            # here -- only the experts class and backend are swapped.
            from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (  # noqa: E501
                NvFp4MoeBackend,
            )

            from .residue_cutedsl_experts import ResidueCuteDSLExperts

            supported, reason = ResidueCuteDSLExperts.is_supported_config(
                ResidueCuteDSLExperts, self.moe, kNvfp4Static,
                kNvfp4Dynamic, mk.FusedMoEActivationFormat.Standard)
            if supported:
                self.experts_cls = ResidueCuteDSLExperts
                self.nvfp4_backend = NvFp4MoeBackend.FLASHINFER_CUTEDSL
                self._residue_impl = "cutedsl"
                print("[residue:moe] experts_cls -> "
                      "ResidueCuteDSLExperts (graft)", flush=True)
            else:
                print(f"[residue:moe] cutedsl graft unsupported: "
                      f"{reason}; stock experts kept", flush=True)

        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype,
                           **extra_weight_attrs) -> None:
            super().create_weights(layer, num_experts, hidden_size,
                                   intermediate_size_per_partition,
                                   params_dtype, **extra_weight_attrs)
            # extended_k layers carry PHYSICALLY K-extended w13 tensors;
            # resize the params now or weight loading dies on the shape
            # assert. Resolution must succeed here -- sizing cannot be
            # deferred -- so a policy that exists but cannot be resolved
            # for this layer is a loud failure by design.
            if self._residue_impl is None or self._moe_policy_section is None:
                return
            from .residue_moe_policy import (
                MOE_IMPL_EXTENDED_K,
                resolve_moe_layer_policy,
            )
            policy = resolve_moe_layer_policy(
                self._moe_policy_section, getattr(layer, "layer_name", ""))
            if policy.impl != MOE_IMPL_EXTENDED_K:
                return
            num_salient = len(policy.w13_salient_indices)
            expected = policy.ratio * hidden_size
            if abs(num_salient - expected) > 0.5:
                raise ValueError(
                    f"[residue:moe] {layer.layer_name}: extended_k salient "
                    f"count {num_salient} does not match ratio "
                    f"{policy.ratio} x K {hidden_size}.")
            w13_rows = layer.w13_weight.shape[1]
            resize_w13_for_extended_k(
                layer, num_experts, w13_rows, hidden_size + num_salient,
                extra_weight_attrs.get("weight_loader"))
            print(f"[residue:moe] {layer.layer_name}: extended_k r="
                  f"{policy.ratio:g} -- w13 sized at K_ext="
                  f"{hidden_size + num_salient}", flush=True)

        def _attach_extk_state(self, layer, layer_name, policy) -> None:
            """Stash the layer-shared salient mask on the experts instance.

            apply() cannot read metadata (it only sees tensors), so the
            mask travels via instance attributes, set right after the stock
            pwal built the kernel. Failing to locate the experts object is
            fatal: serving extended_k weights without the mask would
            quantise at base K and feed a K_ext GEMM garbage.
            """
            import torch

            from nvfp4.quant import indices_to_channel_masks

            experts = getattr(getattr(self, "moe_kernel", None),
                              "fused_experts", None)
            if experts is None:
                raise RuntimeError(
                    f"[residue:moe] {layer_name}: cannot locate "
                    "fused_experts on the built kernel to attach the "
                    "extended_k salient mask.")
            k_base = int(self.moe.hidden_dim)
            idx = torch.tensor(policy.w13_salient_indices,
                               dtype=torch.long,
                               device=layer.w13_weight.device)
            experts._extk_channel_masks = indices_to_channel_masks(
                idx, k_base)
            experts._extk_num_salient = int(idx.numel())
            print(f"[residue:moe] {layer_name}: experts_cls -> "
                  f"ExtendedKCuteDSLExperts (K_ext={k_base + idx.numel()}, "
                  f"r={policy.ratio:g})", flush=True)

        def process_weights_after_loading(self, layer) -> None:
            # Per-layer policy verdict. Only consulted when this instance
            # would serve residue at all -- a config-level fallback to stock
            # (unsupported arch/EP/...) already IS off for every layer.
            if self._residue_impl is not None:
                from .residue_moe_policy import (
                    MOE_IMPL_EXTENDED_K,
                    MOE_IMPL_OFF,
                    resolve_moe_layer_policy,
                )
                layer_name = getattr(layer, "layer_name", "")
                policy = resolve_moe_layer_policy(
                    self._moe_policy_section, layer_name)
                if policy.impl == MOE_IMPL_OFF:
                    print(f"[residue:moe] {layer_name}: policy=off -> "
                          "stock experts", flush=True)
                    # Undo __init__'s takeover for THIS layer's instance:
                    # the stock pwal below then builds the stock kernel from
                    # the restored experts_cls/backend.
                    self._residue_impl = None
                    self.experts_cls = self._stock_experts_cls
                    self.nvfp4_backend = self._stock_nvfp4_backend
                elif policy.impl == MOE_IMPL_EXTENDED_K:
                    # A K-extended checkpoint has NO stock fallback (its
                    # w13 shape is wrong for every stock path), so an
                    # unavailable extk runtime must refuse, not degrade.
                    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (  # noqa: E501
                        NvFp4MoeBackend,
                    )

                    from .residue_cutedsl_experts import (
                        ExtendedKCuteDSLExperts,
                    )

                    if not ExtendedKCuteDSLExperts._supports_current_device():
                        raise RuntimeError(
                            f"[residue:moe] {layer_name}: policy demands "
                            "extended_k but this runtime cannot serve it "
                            "(needs sm100 family + the W3 quantiser).")
                    self._residue_impl = "extended_k"
                    self.experts_cls = ExtendedKCuteDSLExperts
                    self.nvfp4_backend = NvFp4MoeBackend.FLASHINFER_CUTEDSL
                    ret = super().process_weights_after_loading(layer)
                    self._attach_extk_state(layer, layer_name, policy)
                    return ret
            # stock experts OR the cutedsl graft: both want the stock
            # weight prep (the graft's forced FLASHINFER_CUTEDSL backend
            # included). The raw-layout kernel construction that used to
            # follow served only the deleted fold-chain impl.
            return super().process_weights_after_loading(layer)

    return ModelOptNvFp4FusedMoEResidue


def patch_moe_method() -> bool:
    """Install the real residue MoE method on FusedMoEMethodCls.

    Same pattern as the observer; RESIDUE_ENABLE_MOE=1 gates it. When both
    envs are set the method wins -- the observer exists to answer a question
    the real method makes moot.
    """
    if not moe_method_enabled():
        return False
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )
    except ImportError:
        return False

    current = getattr(ModelOptNvFp4Config, "FusedMoEMethodCls", None)
    if current is None:
        print("[residue:moe] vLLM has no FusedMoEMethodCls hook; skipping",
              flush=True)
        return False
    if getattr(current, "__name__", "") == "ModelOptNvFp4FusedMoEResidue":
        return True

    if not hasattr(ModelOptNvFp4Config, "_original_FusedMoEMethodCls"):
        ModelOptNvFp4Config._original_FusedMoEMethodCls = current
    ModelOptNvFp4Config.FusedMoEMethodCls = make_residue_moe_method_cls(
        current)
    print(f"[residue:moe] FusedMoEMethodCls -> ModelOptNvFp4FusedMoEResidue "
          f"(was {current.__name__})", flush=True)
    return True


def make_observing_moe_cls(base_cls: type) -> type:
    """Subclass `base_cls`, adding only observation.

    Built at patch time from whatever vLLM's current NVFP4 MoE method is, so
    this does not pin a vLLM version or duplicate any of its logic.
    """

    class ResidueObservingNvFp4FusedMoE(base_cls):  # type: ignore[misc,valid-type]
        __doc__ = (
            "vLLM's NVFP4 FusedMoE method, instrumented. Identical numerics; "
            "records what residue is not covering."
        )

        def process_weights_after_loading(self, layer) -> None:
            super().process_weights_after_loading(layer)
            try:
                w13 = getattr(layer, "w13_weight", None)
                w2 = getattr(layer, "w2_weight", None)
                name = getattr(layer, "prefix", None) or f"moe@{id(layer):x}"
                # w13/w2 are FP4-packed: the last dim is half the logical K.
                params = 0
                for w in (w13, w2):
                    if w is not None:
                        params += w.numel() * 2
                REPORT.record(
                    name,
                    num_experts=int(getattr(layer, "local_num_experts", 0)
                                    or getattr(layer, "global_num_experts", 0)),
                    w13_shape=tuple(w13.shape) if w13 is not None else None,
                    w2_shape=tuple(w2.shape) if w2 is not None else None,
                    expert_params=params,
                )
            except Exception as e:  # observation must never break a model load
                REPORT.record(f"error@{id(layer):x}", error=repr(e)[:200])

        def apply(self, layer, *args, **kwargs):
            name = getattr(layer, "prefix", None) or f"moe@{id(layer):x}"
            REPORT.record(name, applies=REPORT.layers.get(name, {})
                          .get("applies", 0) + 1)
            return super().apply(layer, *args, **kwargs)

    return ResidueObservingNvFp4FusedMoE


def patch_moe_observer() -> bool:
    """Install the observer on `ModelOptNvFp4Config.FusedMoEMethodCls`.

    Mirrors patch.py's handling of LinearMethodCls, including keeping the
    original around so unpatching is possible.
    """
    if not observer_enabled():
        return False
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )
    except ImportError:
        return False

    current = getattr(ModelOptNvFp4Config, "FusedMoEMethodCls", None)
    if current is None:
        print("[residue:moe] vLLM has no FusedMoEMethodCls hook; skipping",
              flush=True)
        return False
    if getattr(current, "__name__", "") == "ResidueObservingNvFp4FusedMoE":
        return True

    if not hasattr(ModelOptNvFp4Config, "_original_FusedMoEMethodCls"):
        ModelOptNvFp4Config._original_FusedMoEMethodCls = current
    ModelOptNvFp4Config.FusedMoEMethodCls = make_observing_moe_cls(current)
    print(f"[residue:moe] observing FusedMoEMethodCls (was {current.__name__})",
          flush=True)
    return True


def unpatch_moe_observer() -> bool:
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )
    except ImportError:
        return False
    orig = getattr(ModelOptNvFp4Config, "_original_FusedMoEMethodCls", None)
    if orig is not None:
        ModelOptNvFp4Config.FusedMoEMethodCls = orig
        return True
    return False
