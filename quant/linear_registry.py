# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Registry of named linear implementations."""

from __future__ import annotations

from typing import Type

import torch.nn as nn


class LinearRegistry:
    """Global mapping from string identifiers to linear implementation classes."""

    _registry: dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        """Register a linear class by name."""

        def decorator(linear_cls: Type[nn.Module]):
            if not hasattr(linear_cls, "from_linear"):
                raise TypeError(
                    f"{linear_cls.__name__} must implement from_linear() class method\n"
                    "Signature: @classmethod from_linear(cls, linear: nn.Module, "
                    "**kwargs) -> nn.Module"
                )

            if name in cls._registry:
                print(f"Warning: Overwriting registered Linear type '{name}'")

            cls._registry[name] = linear_cls
            return linear_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """Return a registered linear class."""
        if name not in cls._registry:
            raise KeyError(
                f"Linear type {name!r} not registered.\n"
                f"Available types: {list(cls._registry.keys())}\n"
                "Did you forget to import and register the implementation?"
            )
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """List all registered linear type names."""
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return whether ``name`` is registered."""
        return name in cls._registry

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Intended for tests."""
        cls._registry.clear()


@LinearRegistry.register("TorchLinear")
class TorchLinearWrapper(nn.Linear):
    """Passthrough wrapper for standard ``torch.nn.Linear``."""

    @classmethod
    def from_linear(cls, linear: nn.Module, **kwargs) -> nn.Module:
        return linear


__all__ = ["LinearRegistry", "TorchLinearWrapper"]
