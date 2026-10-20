# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Apply ModelOpt compression and residue post-processing hooks."""

from typing import List, Optional, Tuple, Type

import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
from modelopt.torch.utils.network import bind_forward_method


# =============================================================================
# Default Post-Compress Hooks
# =============================================================================

# Base hooks (for all quant modes)
BASE_POST_COMPRESS_HOOKS = [
    "rebind_forward",      # Fix forward binding issue
]

# Mode-specific hooks
MODE_HOOKS = {
    "residue": ["precompute_masks"],  # Pre-compute channel masks
    "awq_lite_residue": ["precompute_masks"],  # Same as residue
    "w4a4": [],
    "w4a8": [],
}

# Legacy: keep for backward compatibility
DEFAULT_POST_COMPRESS_HOOKS = BASE_POST_COMPRESS_HOOKS


# =============================================================================
# Hook Registry
# =============================================================================

HOOK_REGISTRY = {}


def register_hook(name: str):
    """Decorator to register a post-compress hook."""
    def decorator(fn):
        HOOK_REGISTRY[name] = fn
        return fn
    return decorator


# =============================================================================
# Hook Implementations
# =============================================================================

def _check_forward_mismatch(module: nn.Module) -> bool:
    """
    Check if module's forward method differs from its class forward.

    Returns True if the instance has a bound method that doesn't match
    the class-level forward method (indicating stale binding).
    """
    if hasattr(module.forward, "__func__"):
        return module.forward.__func__ is not module.__class__.forward
    return False


def _rebind_forward_methods(
    model: nn.Module,
    target_types: Tuple[Type[nn.Module], ...] = (RealQuantLinear,),
    verbose: bool = False,
) -> int:
    """
    Rebind forward methods for target module types.

    Iterates all modules of the specified types and rebinds their forward
    method to the class-level forward if there's a mismatch.

    Args:
        model: The model to process.
        target_types: Tuple of module types to check.
        verbose: If True, print rebind details.

    Returns:
        Number of modules that were rebinded.
    """
    rebind_count = 0
    for name, module in model.named_modules():
        if isinstance(module, target_types):
            if _check_forward_mismatch(module):
                bind_forward_method(module, module.__class__.forward)
                rebind_count += 1
                if verbose:
                    print(f"  [compress] Rebinded forward for: {name}")
    return rebind_count


@register_hook("rebind_forward")
def _hook_rebind_forward(model: nn.Module, verbose: bool = False):
    """Hook: rebind forward methods after compress."""
    _rebind_forward_methods(model, verbose=verbose)


def _precompute_channel_masks(model: nn.Module, verbose: bool = False) -> int:
    """
    Pre-compute channel masks for residue quantization.

    This sets channel_mask and num_salient on input_quantizers, which are
    required by the residue GEMM backends.

    Args:
        model: The compressed model.
        verbose: If True, print diagnostic information.

    Returns:
        Number of modules with masks computed.
    """
    try:
        from quant.modelopt_integration import precompute_channel_masks
        count = precompute_channel_masks(model)
        if verbose and count > 0:
            print(f"  [compress] Pre-computed channel masks for {count} modules")
        return count
    except ImportError:
        # residue not installed, skip silently
        return 0


@register_hook("precompute_masks")
def _hook_precompute_masks(model: nn.Module, verbose: bool = False):
    """Hook: precompute channel masks for residue quantization."""
    _precompute_channel_masks(model, verbose=verbose)


def _run_hook(model: nn.Module, hook_name: str, verbose: bool) -> None:
    """
    Execute a named post-compress hook from registry.

    Args:
        model: The compressed model.
        hook_name: Name of the hook to run.
        verbose: If True, print diagnostic information.

    Raises:
        ValueError: If hook_name is not in HOOK_REGISTRY.
    """
    if hook_name not in HOOK_REGISTRY:
        raise ValueError(f"Unknown hook: {hook_name}. Available: {list(HOOK_REGISTRY.keys())}")
    HOOK_REGISTRY[hook_name](model, verbose=verbose)


# =============================================================================
# Main Compress Function
# =============================================================================

def compress(
    model: nn.Module,
    config=None,
    *,
    hooks: Optional[List[str]] = None,
    skip_hooks: Optional[List[str]] = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Compress model with configurable post-compress hooks.

    This is our project's wrapper around mtq.compress() that applies
    configurable post-compress hooks. Use this instead of mtq.compress()
    directly.

    Args:
        model: The quantized model to compress.
        config: Compression configuration (passed to mtq.compress).
        hooks: List of hooks to run after compress.
            None = use DEFAULT_POST_COMPRESS_HOOKS.
        skip_hooks: Hooks to skip (useful to disable specific defaults).
        verbose: If True, print diagnostic information.

    Returns:
        The compressed model with hooks applied.

    Hooks:
        - "rebind_forward": Fix forward method binding after compress.
          Required when calibration algorithms use instance-level forward binding.

        - "precompute_masks": Pre-compute channel masks for residue quantization.
          Sets channel_mask and num_salient on input_quantizers.

    Example:
        >>> from quant.modelopt_integration.compress import compress
        >>>
        >>> # Default: all hooks enabled
        >>> model = mtq.quantize(model, config, forward_loop=calib_fn)
        >>> compress(model)
        >>>
        >>> # Skip specific hooks (e.g., for export)
        >>> compress(model, skip_hooks=["precompute_masks"])
        >>>
        >>> # Custom hook list
        >>> compress(model, hooks=["rebind_forward"])

    Note:
        If a hook finds nothing to do, it's a no-op. This makes the wrapper
        safe to use even if ModelOpt fixes issues in a future version.

    """
    # Run compression
    if verbose:
        print("[compress] Running mtq.compress()...")

    compressed = mtq.compress(model, config)

    # Handle in-place modification (some versions return None)
    if compressed is None:
        compressed = model

    # Determine which hooks to run
    active_hooks = hooks if hooks is not None else DEFAULT_POST_COMPRESS_HOOKS
    if skip_hooks:
        active_hooks = [h for h in active_hooks if h not in skip_hooks]

    # Execute hooks
    if verbose and active_hooks:
        print(f"[compress] Running post-compress hooks: {active_hooks}")

    for hook_name in active_hooks:
        _run_hook(compressed, hook_name, verbose)

    if verbose:
        print("[compress] Done")

    return compressed


# =============================================================================
# Compressor Factory
# =============================================================================

def compressor(quant_mode: str = ""):
    """
    Create a compress function for the specified quant mode.

    Usage:
        compressor("residue")(model)  # With residue hooks
        compressor("w4a4")(model)     # Without residue hooks
        compressor()(model)           # Base hooks only

    Args:
        quant_mode: Quantization mode ("residue", "w4a4", "w4a8", etc.)

    Returns:
        A compress function with mode-specific hooks.
    """
    # Build hooks: base + mode-specific
    hooks = BASE_POST_COMPRESS_HOOKS + MODE_HOOKS.get(quant_mode, [])

    def _compress(model: nn.Module, config=None, verbose: bool = False) -> nn.Module:
        return compress(model, config, hooks=hooks, verbose=verbose)

    return _compress
