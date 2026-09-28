# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
Linear implementations registration for runner system.

Registers concrete Linear implementations to the runner's LinearRegistry.

Architecture:
- ResidueLinear/ResidueSwapLinear: legacy RealQuantLinear wrapper path
- NVFP4Linear: Use RealQuantLinear + standard Nvfp4Linear (from ModelOpt)

Each wrapper delegates forward to the supplied RealQuantLinear. It does not
install a GEMM backend. Existing registry keys and module paths are retained.
"""
import os
import torch.nn as nn

from quant.linear_registry import LinearRegistry

# Import ModelOpt's RealQuantLinear (post-compress module)
from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear


def _wrap_compressed(cls, linear, *, compressed_hint=False):
    """Share wrapping without changing class paths or the `linear.*` state keys."""
    if not isinstance(linear, RealQuantLinear):
        detail = " (compressed model)" if compressed_hint else ""
        raise TypeError(
            f"{cls.__name__} can only wrap RealQuantLinear{detail}, "
            f"got {type(linear).__name__}"
        )
    instance = object.__new__(cls)
    nn.Module.__init__(instance)
    instance.in_features = linear.in_features
    instance.out_features = linear.out_features
    instance.has_bias = linear.bias is not None
    instance.linear = linear
    return instance


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
        return _wrap_compressed(cls, linear, compressed_hint=True)

    def forward(self, x):
        return self.linear(x)


@LinearRegistry.register("ResidueSwapLinear")
class ResidueSwapLinear(nn.Module):
    """
    Residue quantization with swap mode.

    The wrapped RealQuantLinear owns forward behavior; the legacy class name
    alone does not establish which residue mode it executes.
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
        return _wrap_compressed(cls, linear)

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
        instance = _wrap_compressed(cls, linear)
        instance.fmt = fmt
        return instance

    def forward(self, x):
        return self.linear(x)


if os.getenv("RESIDUE_DEBUG", "0") == "1":
    print("[linear_impls] Registered to LinearRegistry:")
    print("  - ResidueLinear (swap=0, legacy wrapper)")
    print("  - ResidueSwapLinear (swap=1, legacy wrapper)")
    print("  - NVFP4Linear (W4A4, uses ModelOpt backend)")
