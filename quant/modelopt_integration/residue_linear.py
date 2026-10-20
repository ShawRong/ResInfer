# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
#
# This file is adapted from NVIDIA ModelOpt / CUTLASS codebase.
# Modifications include:
# 
#     Residue-aware QuantLinear module modeled after ModelOpt's SVDQuantLinear.
#     The weight/input quantizers should be ResidueTensorQuantizer so the qtensors
#     carry residue metadata internally.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from modelopt.torch.quantization.nn.modules.quant_linear import QuantLinearConvBase
from modelopt.torch.quantization.nn.modules.quant_module import QuantModuleRegistry
from .tensor_quantizer import ResidueTensorQuantizer
from modelopt.torch.quantization.qtensor import NVFP4QTensor, MXFP4QTensor
from .util import append_quant_residual_channels


class ResidueQuantLinear(QuantLinearConvBase):
    def _setup(self):
        """Overrides and bypass the _setup function. We don't need setup for the linear registered in a SVDLQuantLinear way"""
        if not isinstance(self.weight_quantizer, ResidueTensorQuantizer):
            self.weight_quantizer.__class__ = ResidueTensorQuantizer

        if not isinstance(self.input_quantizer, ResidueTensorQuantizer):
            self.input_quantizer.__class__ = ResidueTensorQuantizer

        if not hasattr(self.input_quantizer, "_enable_salient_indices"):
            self.input_quantizer._enable_salient_indices = True
        if not hasattr(self.weight_quantizer, "_enable_salient_indices"):
            self.weight_quantizer._enable_salient_indices = True

    def _apply_pre_quant_scale(self, input: torch.Tensor) -> torch.Tensor:
        pre_quant_scale = self.input_quantizer.pre_quant_scale
        if pre_quant_scale is None:
            return input
        print("[ResidueQuantLinear] unfused input_quantizer.pre_quant_scale is applied")
        return input * pre_quant_scale.to(device=input.device, dtype=input.dtype)

    def _input_qdq_without_pre_quant_scale(self, input: torch.Tensor) -> torch.Tensor:
        """Run input quantizer QDQ after caller has already applied pre_quant_scale."""
        with self.input_quantizer.disable_pre_quant_scale():
            with self.input_quantizer.allow_residue_linear_fake_quant():
                return self.input_quantizer(input)

    def _make_residue_input_qdq(
        self,
        input: torch.Tensor,
        salient_indices: torch.Tensor,
    ) -> torch.Tensor:
        input_for_quant = self._apply_pre_quant_scale(input)
        is_mx = self.input_quantizer.block_sizes.get("scale_bits") == (8, 0)
        if is_mx:
            input_for_quant = input_for_quant.float()
        salient_indices = salient_indices.to(device=input.device, dtype=torch.long)

        input_base = self._input_qdq_without_pre_quant_scale(input_for_quant)
        input_residue = input_for_quant.index_select(-1, salient_indices) - input_base.index_select(
            -1, salient_indices
        )
        input_residue_q = self._input_qdq_without_pre_quant_scale(input_residue)

        result = torch.cat([input_base, input_residue_q], dim=-1)
        if is_mx and result.shape[-1] < self.weight.shape[-1]:
            result = F.pad(result, (0, self.weight.shape[-1] - result.shape[-1]))
        return result.to(input.dtype) if is_mx else result

    def _make_residue_weight(
        self,
        input: torch.Tensor,
        input_ext: torch.Tensor,
        salient_indices: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight.data if isinstance(self.weight, torch.nn.Parameter) else self.weight
        if weight.shape[-1] == input_ext.shape[-1]:
            return weight
        if weight.shape[-1] != input.shape[-1]:
            raise RuntimeError(
                "Residue weight/input shape mismatch: "
                f"weight K={weight.shape[-1]}, input K={input.shape[-1]}, "
                f"extended input K={input_ext.shape[-1]}."
            )
        weight_indices = salient_indices.to(device=weight.device)
        return torch.cat([weight, weight.index_select(-1, weight_indices)], dim=-1)

    def _apply_weight_pre_quant_scale(
        self,
        weight: torch.Tensor,
        salient_indices: torch.Tensor,
    ) -> torch.Tensor:
        pre_quant_scale = self.weight_quantizer.pre_quant_scale
        if pre_quant_scale is None:
            return weight

        scale = pre_quant_scale.to(device=weight.device, dtype=weight.dtype)
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)
        if scale.shape[-1] == weight.shape[-1]:
            return weight * scale
        if scale.shape[-1] + salient_indices.numel() == weight.shape[-1]:
            scale_indices = salient_indices.to(device=scale.device, dtype=torch.long)
            scale = torch.cat([scale, scale.index_select(-1, scale_indices)], dim=-1)
            return weight * scale
        raise RuntimeError(
            "weight_quantizer.pre_quant_scale shape does not match residue weight: "
            f"scale K={scale.shape[-1]}, weight K={weight.shape[-1]}, "
            f"salient={salient_indices.numel()}."
        )

    def _forward_residue_qdq(self, input: torch.Tensor) -> torch.Tensor:
        salient_indices = self.input_quantizer.salient_indices
        assert salient_indices is not None and salient_indices.numel() > 0

        input_ext_q = self._make_residue_input_qdq(input, salient_indices)
        weight = self._make_residue_weight(input, input_ext_q, salient_indices)
        weight = self._apply_weight_pre_quant_scale(weight, salient_indices)
        with self.weight_quantizer.disable_pre_quant_scale():
            with self.weight_quantizer.allow_residue_linear_fake_quant():
                if (self.weight_quantizer.block_sizes.get("scale_bits") == (8, 0)
                        and self.weight_quantizer.is_enabled
                        and self.weight_quantizer.fake_quant
                        and getattr(self.weight_quantizer, "_if_quant", True)):
                    packed, scale = MXFP4QTensor.quantize(weight.contiguous(), 32)
                    weight_q = packed.dequantize(scale=scale, block_sizes={-1: 32})
                else:
                    weight_q = self.weight_quantizer(weight)

        output = F.linear(input_ext_q, weight_q, self.bias)
        return self.output_quantizer(output) if hasattr(self, "output_quantizer") else output

    def forward(self, input, *args, **kwargs):
        input_has_salient_attr = hasattr(self.input_quantizer, "salient_indices")
        weight_has_salient_attr = hasattr(self.weight_quantizer, "salient_indices")
        if not input_has_salient_attr and not weight_has_salient_attr:
            # ModelOpt auto-search can activate a plain NVFP4 candidate on the
            # same module. In that case quantizers are regular TensorQuantizer
            # instances and the layer should behave exactly like QuantLinear.
            return super().forward(input, *args, **kwargs)
        if input_has_salient_attr != weight_has_salient_attr:
            raise RuntimeError(
                "Mixed residue/plain quantizer state is invalid: "
                f"input_has_salient_attr={input_has_salient_attr}, "
                f"weight_has_salient_attr={weight_has_salient_attr}."
            )

        salient_indices = self.input_quantizer.salient_indices
        has_salient_indices = salient_indices is not None and salient_indices.numel() > 0

        if not has_salient_indices:
            weight = self.weight.data if isinstance(self.weight, torch.nn.Parameter) else self.weight
            if weight.shape[-1] > input.shape[-1]:
                raise RuntimeError(
                    "Extended residue weight requires input_quantizer.salient_indices "
                    "during forward, but none were loaded."
                )
            active_fake_quant = (
                (
                    self.input_quantizer.is_enabled
                    and getattr(self.input_quantizer, "_if_quant", True)
                    and self.input_quantizer.fake_quant
                )
                or (
                    self.weight_quantizer.is_enabled
                    and getattr(self.weight_quantizer, "_if_quant", True)
                    and self.weight_quantizer.fake_quant
                )
            )
            if (active_fake_quant and
                    (self.input_quantizer.block_sizes or {}).get("scale_bits") == (8, 0)):
                with self.input_quantizer.allow_residue_linear_fake_quant(), self.weight_quantizer.allow_residue_linear_fake_quant():
                    return super().forward(input, *args, **kwargs)
            if active_fake_quant:
                raise RuntimeError(
                    "Residue fake-quant forward requires salient_indices. "
                    "This fallback is only valid for calibration collect / normal forward."
                )
            # Keep the normal ModelOpt QuantLinear path for residue calibration's
            # statistics-collection forward. Before postprocess there is no
            # salience metadata or K extension to apply.
            return super().forward(input, *args, **kwargs)

        return self._forward_residue_qdq(input)

    def input_extension(
        self,
        input: torch.Tensor,
        salient_indices: torch.Tensor
    ) -> torch.Tensor:
        if(os.getenv("RESIDUE_DEBUG", "0") == "1"):
            layer_name = getattr(self, "_layer_name", "unknown")
            print(f"[{layer_name}] extending input")
            print(f"[{layer_name}] weight_quantizer (disable, if_quant, if_calib, fake_quant): {self.weight_quantizer._disabled}, {self.weight_quantizer._if_quant}, {self.weight_quantizer._if_calib}, {self.weight_quantizer._fake_quant}")
            print(f"[{layer_name}] input_quantizer (disable, if_quant, if_calib, fake_quant): {self.input_quantizer._disabled}, {self.input_quantizer._if_quant}, {self.input_quantizer._if_calib}, {self.input_quantizer._fake_quant}")
        if self.weight_quantizer._fake_quant:
            self.weight_quantizer._fake_quant = False
        if self.input_quantizer._fake_quant:
            self.input_quantizer._fake_quant = False
        if salient_indices is None:
            raise RuntimeError("Salient indices not loaded. Call load_salient_indices() before input_extension().")

        if self.input_quantizer.block_sizes.get("scale_bits") == (8, 0):
            return self.input_quantizer.residue_extension(input, salient_indices)

        # Dynamic quantizer - compute amax from input
        amax = input.abs().max()
        weight_scaling_factor_2 = amax.float() / (448.0 * 6.0)

        if os.getenv("RESIDUE_DEBUG", "0") == "1":
            print(f"[{layer_name}] Using append_quant_residual_channels for input extension")

        return append_quant_residual_channels(
            input,
            16,
            weight_scaling_factor=None,
            weight_scaling_factor_2=weight_scaling_factor_2,
            salient_indices=salient_indices,
            quantizer=NVFP4QTensor, #TODO fixed to be nvfp4 qtensor for now
        )


# Alias to mimic ModelOpt naming
ResidueLinear = ResidueQuantLinear

__all__ = [
    "ResidueQuantLinear",
    "ResidueLinear",
]


# TODO: The RealQuantLinear conversion is functionally not working for the ResidueQuantLinear due to dynamic registration
# We use manually set the forward after mtq.compress for now.

# Register for automatic conversion of nn.Linear → ResidueQuantLinear
QuantModuleRegistry.register({nn.Linear: ResidueQuantLinear})

# Register for RealQuantLinear conversion during compress()
# This ensures ResidueQuantLinear → RealQuantLinear fusion happens
try:
    from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
    from modelopt.torch.quantization.compress import RealQuantModuleRegistry

    RealQuantModuleRegistry.register({ResidueQuantLinear: "ResidueQuantLinear"})(
        RealQuantLinear
    )
    if os.getenv("RESIDUE_DEBUG", "0") == "1":
        print("[INIT] ResidueQuantLinear registered in RealQuantModuleRegistry")
except (ImportError, AssertionError) as e:
    if os.getenv("RESIDUE_DEBUG", "0") == "1":
        print(f"[INIT] ResidueQuantLinear registration skipped: {e}")
    pass
