# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
#
# This file is adapted from NVIDIA ModelOpt / CUTLASS codebase.
# Modifications include:
# - New class `ResidueTensorQuantizer`

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional
import torch

from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.quantization.qtensor import MXFP4QTensor, NVFP4QTensor
from .util import append_copy_channels, append_quant_residual_channels

# Equipped with extension method compared with TensorQuantizer
class ResidueTensorQuantizer(TensorQuantizer):
    def __init__(
        self,
        quant_attribute_cfg=None,
        if_quant: bool = True,
        if_calib: bool = False,
        amax=None,
    ) -> None:
        super().__init__(
            quant_attribute_cfg=quant_attribute_cfg,
            if_quant=if_quant,
            if_calib=if_calib,
            amax=amax,
        )
        self._enable_salient_indices = True

    # mimic pre quant scale in the modelopt, in (Tensorquantizer, svdlinear, awq)
    @property
    def salient_indices(self):
        """Return salient_indices used for residue."""
        if not hasattr(self, "_salient_indices") or not self._enable_salient_indices:
            return None
        return self._salient_indices

    @salient_indices.setter
    def salient_indices(self, value):
        assert value is not None, "salient_indices cannot be set to None."
        assert self._enable_salient_indices, (
            "salient_indices cannot be set when forward_with_pre_quant_scale is False."
        )
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if not hasattr(self, "_salient_indices"):
            self.register_buffer("_salient_indices", value.clone().detach())
        else:
            if self._salient_indices.shape != value.shape:
                raise RuntimeError("Changing shape when setting salient_indices is not allowed.")
            self._salient_indices.data.copy_(
                value.clone().detach().to(self._salient_indices.device)
            )

    def forward(self, inputs):
        """Forward pass through the tensor quantizer."""
        return super().forward(inputs)

    def _fake_quantize(self, inputs):
        """Reject standalone fake quantization for residue quantizers.

        Residue fake quant is defined at Linear level because activation
        residuals and copied weight channels must be handled together. Returning
        NaNs makes accidental direct quantizer fake-quant use fail loudly.
        """
        if getattr(self, "_allow_residue_linear_fake_quant", False):
            return super()._fake_quantize(inputs)
        return torch.full_like(inputs, float("nan"))

    @contextmanager
    def allow_residue_linear_fake_quant(self):
        """Temporarily allow ResidueQuantLinear to use TensorQuantizer.forward()."""
        was_allowed = getattr(self, "_allow_residue_linear_fake_quant", False)
        self._allow_residue_linear_fake_quant = True
        try:
            yield
        finally:
            self._allow_residue_linear_fake_quant = was_allowed

    def copy_extension(
        self,
        input: torch.Tensor,
        salient_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Append selected channels unchanged. Used for materialized weights."""
        if salient_indices is None:
            raise RuntimeError("Salient indices not loaded. Call load_salient_indices() before copy_extension().")
        return append_copy_channels(input, salient_indices)

    def residue_extension(
        self,
        input: torch.Tensor,
        salient_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Append selected quantization residual channels."""
        if salient_indices is None:
            raise RuntimeError("Salient indices not loaded. Call load_salient_indices() before residue_extension().")

        quantizer = None
        if self._block_sizes.get("scale_bits") == (8, 0) and self._block_sizes.get("type") == "dynamic":
            # MX quantization
            if self._num_bits == (2, 1):
                quantizer = MXFP4QTensor
            else:
                raise ValueError(
                    f"Real quantization for MX {self._num_bits} format is not supported."
                )
        elif self._block_sizes.get("scale_bits") == (4, 3):
            # NVFP4 default quantization
            quantizer = NVFP4QTensor

        # Get amax: use self.amax if available (static), otherwise compute from input (dynamic)
        if hasattr(self, "amax") and self.amax is not None:
            # Static quantizer - use pre-computed amax
            weight_scaling_factor_2 = self.amax.float() / (448.0 * 6.0)
        else:
            # Dynamic quantizer - compute amax from input
            amax = input.abs().max()
            weight_scaling_factor_2 = amax.float() / (448.0 * 6.0)

        return append_quant_residual_channels(
            input,
            self._block_sizes[-1],
            weight_scaling_factor=None,
            weight_scaling_factor_2=weight_scaling_factor_2,
            salient_indices=salient_indices,
            quantizer=quantizer,
        )

    def input_extension(
        self,
        input: torch.Tensor,
        mode: str,
        salient_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deprecated compatibility wrapper for old copy/residue mode call sites."""
        if mode == "copy":
            return self.copy_extension(input=input, salient_indices=salient_indices)
        if mode == "residue":
            return self.residue_extension(input=input, salient_indices=salient_indices)
        raise ValueError(f"Unsupported residue extension mode: {mode!r}")


__all__ = [
    "ResidueTensorQuantizer",
]
