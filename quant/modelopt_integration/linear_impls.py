# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
Linear implementations registration for runner system.

Registers concrete Linear implementations to the runner's LinearRegistry.

Architecture:
- ResidueLinear/ResidueSwapLinear: legacy RealQuantLinear wrapper path
- NVFP4Linear: Use RealQuantLinear + standard Nvfp4Linear (from ModelOpt)

Note: old backend_infer/backend GEMM functions are archived; official eval uses vLLM.
      RealQuantLinear is the actual module that uses these functions.
"""
import os
import torch
import torch.nn as nn

from quant.linear_registry import LinearRegistry

# Import ModelOpt's RealQuantLinear (post-compress module)
from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear


@LinearRegistry.register("ResidueLinear")
class ResidueLinear(nn.Module):
    """
    Residue quantization linear (no swap).

    Legacy wrapper around RealQuantLinear. Official eval uses vLLM.
    Weight matrix extended: W' = [W, W_salient]
    Input extended by backend GEMM function.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        # Use RealQuantLinear as the underlying implementation
        self.linear = None  # Will be set in from_linear

    @classmethod
    def from_linear(cls, linear: nn.Module, r: float = 0.125, fmt: str = "nvfp4", **kwargs):
        """
        Create ResidueLinear from existing linear.

        Args:
            linear: Source linear module (RealQuantLinear with extended weights)
            r: Residue ratio (metadata only, not used during inference)
            fmt: Quantization format
            **kwargs: Additional arguments

        Returns:
            ResidueLinear instance wrapping RealQuantLinear
        """
        # Ensure swap mode disabled
        old_swap = os.environ.get("RESIDUE_SWAP")
        os.environ["RESIDUE_SWAP"] = "0"

        try:
            # If already RealQuantLinear (from compress), wrap it
            if isinstance(linear, RealQuantLinear):
                instance = object.__new__(cls)
                super(ResidueLinear, instance).__init__()
                instance.in_features = linear.in_features
                instance.out_features = linear.out_features
                instance.has_bias = linear.bias is not None
                instance.linear = linear
                return instance

            # Otherwise, unsupported (need compressed model)
            raise TypeError(
                f"ResidueLinear can only wrap RealQuantLinear (compressed model), "
                f"got {type(linear).__name__}"
            )

        finally:
            if old_swap is None:
                os.environ.pop("RESIDUE_SWAP", None)
            else:
                os.environ["RESIDUE_SWAP"] = old_swap

    def forward(self, x):
        return self.linear(x)


@LinearRegistry.register("ResidueSwapLinear")
class ResidueSwapLinear(nn.Module):
    """
    Residue quantization with swap mode.

    Swap mode: Q(x_salient) → x_salient (FP16) for higher accuracy.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.linear = None

    @classmethod
    def from_linear(cls, linear: nn.Module, r: float = 0.125, fmt: str = "nvfp4", **kwargs):
        """Create ResidueSwapLinear from RealQuantLinear."""
        # Ensure swap mode enabled
        old_swap = os.environ.get("RESIDUE_SWAP")
        os.environ["RESIDUE_SWAP"] = "1"

        try:
            if isinstance(linear, RealQuantLinear):
                instance = object.__new__(cls)
                super(ResidueSwapLinear, instance).__init__()
                instance.in_features = linear.in_features
                instance.out_features = linear.out_features
                instance.has_bias = linear.bias is not None
                instance.linear = linear
                return instance

            raise TypeError(
                f"ResidueSwapLinear can only wrap RealQuantLinear, got {type(linear).__name__}"
            )

        finally:
            if old_swap is None:
                os.environ.pop("RESIDUE_SWAP", None)
            else:
                os.environ["RESIDUE_SWAP"] = old_swap

    def forward(self, x):
        return self.linear(x)


@LinearRegistry.register("NVFP4Linear")
class NVFP4Linear(nn.Module):
    """
    W4A4 quantized linear using ModelOpt's standard NVFP4 backend.

    This is standard W4A4 quantization without residue.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, fmt: str = "nvfp4", **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.fmt = fmt
        self.linear = None

    @classmethod
    def from_linear(cls, linear: nn.Module, fmt: str = "nvfp4", **kwargs):
        """Create NVFP4Linear from RealQuantLinear."""
        if isinstance(linear, RealQuantLinear):
            instance = object.__new__(cls)
            super(NVFP4Linear, instance).__init__()
            instance.in_features = linear.in_features
            instance.out_features = linear.out_features
            instance.has_bias = linear.bias is not None
            instance.fmt = fmt
            instance.linear = linear
            return instance

        raise TypeError(
            f"NVFP4Linear can only wrap RealQuantLinear, got {type(linear).__name__}"
        )

    def forward(self, x):
        return self.linear(x)


if os.getenv("RESIDUE_DEBUG", "0") == "1":
    print("[linear_impls] Registered to LinearRegistry:")
    print("  - ResidueLinear (swap=0, legacy wrapper)")
    print("  - ResidueSwapLinear (swap=1, legacy wrapper)")
    print("  - NVFP4Linear (W4A4, uses ModelOpt backend)")
