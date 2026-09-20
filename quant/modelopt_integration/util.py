# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0

import torch
from modelopt.torch.quantization.qtensor import BaseQuantizedTensor, MXFP4QTensor

def _select_salient(
    tensor: torch.Tensor,
    salient_indices: torch.Tensor,
) -> torch.Tensor:
    # Ensure salient_indices is on the same device as tensor (defensive)
    salient_indices = salient_indices.to(tensor.device)
    selected = tensor.index_select(dim=-1, index=salient_indices)
    return selected

def append_copy_channels(tensor: torch.Tensor, salient_indices: torch.Tensor) -> torch.Tensor:
    """Append selected channels unchanged along the last dimension."""
    selected = _select_salient(tensor, salient_indices)
    in_features = tensor.shape[-1]
    num_selected = selected.shape[-1]
    extended_shape = (*tensor.shape[:-1], in_features + num_selected)
    extended = torch.empty(
        extended_shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    extended[..., :in_features].copy_(tensor)
    extended[..., in_features:].copy_(selected)
    del selected
    return extended


def append_quant_residual_channels(
    tensor: torch.Tensor,
    block_size: int,
    weight_scaling_factor: torch.Tensor,
    weight_scaling_factor_2: torch.Tensor,
    salient_indices: torch.Tensor,
    quantizer: BaseQuantizedTensor,
) -> torch.Tensor:
    """Append selected quantization residual channels along the last dimension."""
    import os
    debug = os.getenv("RESIDUE_DEBUG", "0") == "1"

    if debug:
        print("[append_quant_residual_channels] mode=residue")
        print(f"[append_quant_residual_channels] Input shape: {tensor.shape}")
        print(f"[append_quant_residual_channels] Salient indices: {salient_indices.numel()} channels")
        print(f"[append_quant_residual_channels] Output last dim will be: {tensor.shape[-1] + salient_indices.numel()}")

    # Quantize and dequantize to get base on last dimension
    if quantizer is MXFP4QTensor:
        if block_size != 32:
            raise ValueError("MXFP4 residue requires block size 32")
        # Activation QDQ uses RNE; MXFP4QTensor is the weight exporter and
        # uses ties toward zero in ModelOpt 0.45. Do not use it for A residue.
        from modelopt.torch.quantization.tensor_quant import dynamic_block_quant
        base_dense = dynamic_block_quant(tensor.float().contiguous(), block_size,
                                         None, None, (2, 1), (8, 0))
        base_qtensor = base_scale = base_double_scale = None
    else:
        base_qtensor, base_scale, base_double_scale = quantizer.quantize(
            tensor, block_size, weight_scaling_factor, weight_scaling_factor_2,
            keep_high_precision=False, try_tensorrt=True,
        )
        base_dense = base_qtensor.dequantize(
            scale=base_scale, double_scale=base_double_scale,
            block_sizes={-1: block_size}, fast=True,
        )

    residue_source = tensor - base_dense
    residue = _select_salient(residue_source, salient_indices)
    del residue_source

    # Concatenate along last dimension (in_features dimension)
    # tensor: [..., in], residue: [..., residue_channels]
    # result: [out, in + residue_channels]
    tensor_extended = torch.cat([tensor, residue], dim=-1)

    if debug:
        print(f"[append_quant_residual_channels] Extended shape: {tensor_extended.shape}")
        print(f"[append_quant_residual_channels] Residue stats: min={residue.min():.6f}, max={residue.max():.6f}, mean={residue.mean():.6f}")

    # Free intermediate tensors
    del base_qtensor, base_scale, base_double_scale, base_dense, residue

    return tensor_extended


# Deprecated compatibility wrapper for older call sites.
def weight_extension(weight: torch.Tensor, block_size: int, weight_scaling_factor: torch.Tensor, weight_scaling_factor_2: torch.Tensor, salient_indices: torch.Tensor, mode: str, quantizer: BaseQuantizedTensor) -> torch.Tensor:
    if mode == "copy":
        return append_copy_channels(weight, salient_indices)
    if mode == "residue":
        return append_quant_residual_channels(
            weight,
            block_size,
            weight_scaling_factor,
            weight_scaling_factor_2,
            salient_indices,
            quantizer,
        )
    raise ValueError(f"Unsupported residue extension mode: {mode!r}")

# For output extension. use when custmoized kernel is not supported, or debug
def swap_extension(weight: torch.Tensor, block_size: int, weight_scaling_factor: torch.Tensor, weight_scaling_factor_2: torch.Tensor, salient_indices: torch.Tensor, mode: str, quantizer: BaseQuantizedTensor) -> torch.Tensor:
    import os
    debug = os.getenv("RESIDUE_DEBUG", "0") == "1"

    if debug:
        print(f"[swap_extension] mode={mode}")
        print(f"[swap_extension] Input shape: {weight.shape}")
        print(f"[swap_extension] Salient indices: {salient_indices.numel()} channels")

    if mode == "copy":
        # Select salient channels
        salient_channels = _select_salient(weight, salient_indices)  # Selects on dim=-1

        # create extended weight: [base, salient_channels]
        out_features, in_features = weight.shape
        num_residue = salient_channels.shape[1]

        weight_extended = torch.empty(
            out_features,
            in_features + num_residue,
            dtype=weight.dtype,
            device=weight.device
        )

        weight_extended[:, :in_features].copy_(weight)
        weight_extended[:, in_features:].copy_(salient_channels)

        del salient_channels

        return weight_extended

    # select original salient channels FIRST, before any modification
    # These will be appended to the extended tensor
    salient_channels = _select_salient(weight, salient_indices)

    # quantize salient channels separately to compute their residue
    # This matches kernel lines 419-436
    salient_qtensor, salient_scale, salient_double_scale = quantizer.quantize(
        salient_channels,
        block_size,
        weight_scaling_factor,
        weight_scaling_factor_2,
        keep_high_precision=False,
        try_tensorrt=True,
    )

    salient_dense = salient_qtensor.dequantize(
        scale=salient_scale,
        double_scale=salient_double_scale,
        block_sizes={-1: block_size},
        fast=True,
    )

    # Compute residue for salient channels
    # residue = original_salient - quantized_salient
    if mode == "residue":
        residue = salient_channels - salient_dense
    else:
        residue = salient_channels.clone()

    # Free intermediate tensors for salient quantization
    del salient_qtensor, salient_scale, salient_double_scale, salient_dense

    weight_swapped = weight.clone()

    weight_swapped.index_copy_(dim=-1, index=salient_indices, source=residue)

    weight_extended = torch.cat([weight_swapped, salient_channels], dim=-1)

    if debug:
        print(f"[swap_extension] Extended shape: {weight_extended.shape}")
        print(f"[swap_extension] Swapped weight stats: min={weight_swapped.min():.6f}, max={weight_swapped.max():.6f}")

    # Free intermediate tensors
    del residue, salient_channels, weight_swapped

    return weight_extended


def weight_extension_copy(weight: torch.Tensor, salient_indices: torch.Tensor) -> torch.Tensor:
    return append_copy_channels(weight, salient_indices)


def precompute_channel_masks(model: torch.nn.Module) -> int:
    """
    Pre-compute channel masks from salient_indices for all quantized modules.

    This should be called after mtq.compress() to convert salient_indices to
    channel_masks once, avoiding repeated conversion during inference.

    Adds the following attributes to each input_quantizer:
        - channel_mask: Pre-computed uint8 bitmask tensor
        - num_salient: Number of salient channels

    Args:
        model: The quantized model after compress

    Returns:
        Number of modules that had masks pre-computed
    """
    import nvfp4

    count = 0
    for name, module in model.named_modules():
        # Check if module has input_quantizer with salient_indices
        if hasattr(module, 'input_quantizer'):
            input_quantizer = module.input_quantizer
            if hasattr(input_quantizer, 'salient_indices') and input_quantizer.salient_indices is not None:
                salient_indices = input_quantizer.salient_indices

                # Get the logical input dimension K used to create the mask.
                # Older materialized residue flows stored weight as [N, K + S],
                # but current calibration is salience-only and leaves weight as
                # [N, K]. Prefer explicit metadata when available; otherwise use
                # the module's declared in_features instead of blindly subtracting
                # S from weight.shape[1].
                if hasattr(module, 'weight'):
                    weight = module.weight
                    if hasattr(weight, 'get_qtensor'):
                        # QTensorWrapper - get actual shape
                        qtensor = weight.get_qtensor()
                        in_features = qtensor._quantized_data.shape[1] * 2  # FP4 packed
                    else:
                        in_features = weight.shape[1]

                    num_salient = len(salient_indices)
                    original_in_features = getattr(module, "_original_in_features", None)
                    if original_in_features is not None:
                        num_channels = int(original_in_features)
                    else:
                        declared_in_features = getattr(module, "in_features", None)
                        if declared_in_features is not None:
                            num_channels = int(declared_in_features)
                        elif int(salient_indices.max().item()) < int(in_features):
                            num_channels = int(in_features)
                        else:
                            # Last-resort compatibility with old materialized
                            # modules that did not preserve _original_in_features.
                            num_channels = int(in_features) - int(num_salient)
                else:
                    # Fallback: infer from salient_indices max value
                    num_channels = (salient_indices.max().item() + 8) // 8 * 8

                # Pre-compute the mask
                channel_mask = nvfp4.indices_to_channel_masks(salient_indices, num_channels)

                # Store on the quantizer
                input_quantizer.channel_mask = channel_mask
                input_quantizer.num_salient = len(salient_indices)

                count += 1

    return count
