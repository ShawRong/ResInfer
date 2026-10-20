# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-FileCopyrightText: Copyright (c) 2023 MIT HAN Lab (get_act_scale)
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# This file is adapted from NVIDIA ModelOpt / CUTLASS codebase.
# Modifications include:
# 
# - New residue max calibration algorithm helper functions

import torch
import torch.nn.functional as F
import torch.nn as nn

from modelopt.torch.quantization import utils as quant_utils
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg_context
from modelopt.torch.quantization.model_calib import max_calibrate
from modelopt.torch.utils.network import bind_forward_method, unpatch_forward_method

__all__ = [
    "ResidueHelper",
    "get_salient_indices",
    "get_nvfp4_error_weighted_score",
    "customized_forward",
    "apply_salient_indices",
]


# The single source of truth for which salience metrics exist. It used to be
# an inline set inside the constructor, which meant any caller wanting to
# validate one before loading a 7.6GB model had to write out a second copy --
# and a second copy of a fact is a copy that drifts. Exported so the export
# preflight can reject a typo in seconds instead of after the model load.
SALIENCE_METRICS = frozenset({"activation_max", "nvfp4_error_weighted"})


class ResidueHelper:
    cache_mode: bool = False
    
    def __init__(
        self, 
        module, 
        name,
        block_size: int = 8,
        residue_ratio: float = 0.25,
        salience_metric: str = "activation_max",
    ):
        self.name = name
        self.module = module
        # Residue-specific parameters
        self.block_size = block_size
        self.residue_ratio = residue_ratio
        self.salience_metric = salience_metric

        if salience_metric not in SALIENCE_METRICS:
            raise ValueError(
                f"Unsupported residue salience metric: {salience_metric!r}; "
                f"known: {sorted(SALIENCE_METRICS)}")

        self.act_scale = 0.0
        self.num_tokens = 0
        self.num_cache_steps = 0
        self.is_enabled = True
        self.is_input_quantized = module.input_quantizer.is_enabled # for smooth, but not enabled here

    def setup(self):
        module = self.module
        bind_forward_method(module, customized_forward, "_forward_no_residuemax")
        if module.input_quantizer.is_enabled:
            module.input_quantizer.disable()
            if module.input_quantizer.axis not in [None, -1]:
                self.is_enabled = False
                return
            # Preserve the AWQ final scalar amax so this branch isolates the effect of
            # raw-input act_scale for salient channel ranking only.
            self._saved_awq_amax = (
                module.input_quantizer.amax.clone()
                if module.input_quantizer.amax is not None
                else None
            )
            # Reset so load_calib_amax() can allocate a per-channel buffer for act_scale
            # collection (axis=-1). Without reset, copying per-channel into a scalar buffer fails.
            module.input_quantizer.reset_amax()
            module.input_quantizer.axis = -1

    def cleanup(self):
        module = self.module
        if hasattr(module, "_if_calib"):
            delattr(module, "_if_calib")
        unpatch_forward_method(module, "_forward_no_residuemax")
    
    
def get_salient_indices(act_scale_mean: torch.Tensor, block_size, ratio: float) -> torch.Tensor:
    org_shape = act_scale_mean.shape 
    slice_after_padding = None
    if block_size:
        if org_shape[-1] % block_size != 0:
            slice_after_padding = slice(org_shape[-1])
            act_scale_mean = F.pad(act_scale_mean, (0, block_size - org_shape[-1] % block_size), "constant", 0)
            org_shape = act_scale_mean.shape
        act_scale_mean = act_scale_mean.contiguous().view(-1, block_size)
    channels_per_block = int(ratio * block_size)
    _, local_indices = torch.topk(act_scale_mean, k=channels_per_block, dim=1, largest=True)

    # convert local indices to global indices
    # local_indices has shape [num_blocks, channels_per_block]
    # add global shift: block_idx * block_size to each block's local indices
    block_indices = torch.arange(act_scale_mean.shape[0], device=act_scale_mean.device).unsqueeze(1)
    global_indices = local_indices + block_indices * block_size
    salient_indices = global_indices.flatten().sort().values

    if slice_after_padding is not None:
        act_scale_mean = act_scale_mean[..., slice_after_padding]
    
    return salient_indices


def get_nvfp4_error_weighted_score(
    accumulated_error_sq: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Return ||X-Q(X)||_2 * ||W[:, j]||_2 for every input channel."""
    weight_col_norm = weight.detach().float().square().sum(dim=0).sqrt()
    return accumulated_error_sq.float().clamp_min(0).sqrt() * weight_col_norm

def customized_forward(self, input, *args, **kwargs):
    # collect actual output without quantization
    self.weight_quantizer.disable()
    if hasattr(self.input_quantizer, "_salient_indices"):
        delattr(self.input_quantizer, "_salient_indices")
    if hasattr(self.weight_quantizer, "_salient_indices"):
        delattr(self.weight_quantizer, "_salient_indices")
    out_actual = self._forward_no_residuemax(input, *args, **kwargs)
    self.weight_quantizer.enable()

    if input.numel() == 0 or not self.residue_max.is_enabled:
        # Some layers may see an empty batch
        return out_actual

    if ResidueHelper.cache_mode:
        # get local tensor from dtensor
        input = input.to_local() if hasattr(input, "to_local") else input

        # Salient channel ranking should reflect the raw activation distribution rather than
        # the AWQ-smoothed path.
        if self.residue_max.salience_metric == "nvfp4_error_weighted":
            # Use ModelOpt's exact dynamic NVFP4 fake quantizer.  Calling the
            # protected primitive is intentional: stats collection puts the
            # public quantizer in calibration/pass-through mode.
            with self.input_quantizer.allow_residue_linear_fake_quant():
                quantized_input = self.input_quantizer._fake_quantize(input)
            flat_error = (input.float() - quantized_input.float()).reshape(-1, input.shape[-1])
            self.residue_max.act_scale += flat_error.square().sum(dim=0)
        else:
            self.residue_max.act_scale += get_act_scale(input)
        self.residue_max.num_cache_steps += 1
        self.residue_max.num_tokens += input.numel() / input.shape[-1]
        if self.residue_max.is_input_quantized and self.residue_max._saved_awq_amax is None:
            with set_quantizer_by_cfg_context(self.input_quantizer, {"*": {"enable": True}}):
                max_calibrate(
                    self.input_quantizer,
                    lambda quantizer: quantizer(input),
                    distributed_sync=False,
                )
    return out_actual


# Average absolute value per channel; adapted from MIT HAN Lab llm-awq (MIT).
def get_act_scale(x):
    return x.abs().contiguous().view(-1, x.shape[-1]).mean(0).to(torch.float32)

# for extension, and copied logic from apply_pre_quant_scale_and_smooth in model_calib.py
def apply_salient_indices(
    linear: nn.Module, salient_indices: torch.Tensor | None = None
):

    assert salient_indices is not None, "salient indices should be provided or already set"
    linear.input_quantizer._enable_salient_indices = True
    linear.input_quantizer.salient_indices = salient_indices
