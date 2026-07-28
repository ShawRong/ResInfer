# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
#
# This file is adapted from NVIDIA ModelOpt / CUTLASS codebase.
# Modifications include:
# 
# - New residue max calibration algorithm

import os
import json

import warnings
from pathlib import Path
from typing import Literal
import torch.distributed as dist
from modelopt.torch.utils.distributed import DistributedProcessGroup
import torch
import torch.nn as nn
from modelopt.torch.opt.config import ModeloptField
from modelopt.torch.quantization.config import QuantizeAlgorithmConfig, QuantizeConfig
from modelopt.torch.quantization.mode import (
    BaseCalibrateModeDescriptor,
    CalibrateModeRegistry,
)
from modelopt.torch.quantization.model_calib import ForwardLoop
from modelopt.torch.quantization.model_calib import enable_stats_collection
from modelopt.torch.quantization.utils import is_quantized_linear
from modelopt.torch.quantization.conversion import _replace_quant_module
from modelopt.torch.opt.conversion import ModeloptStateManager
from modelopt.torch.quantization.model_calib import max_calibrate 
from modelopt.torch.utils import print_rank_0
from modelopt.torch.quantization.utils import enable_weight_access_and_writeback

from .algo_helper import ResidueHelper, get_salient_indices, apply_salient_indices
from .residue_linear import ResidueQuantLinear
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.quantization.conversion import restore_quantizer_state
from modelopt.torch.opt.mode import RestoreEntrypoint, MetadataDict

ResidueQuantModuleRegistry = _DMRegistryCls("ResidueQuant")


# Right now we keep on using this replacement function while the residue_max_calibrate is called.
# I think we should keep this SVDQuant registration style
def create_and_replace_residue_linear_on_the_fly(model):
    needs_replacement = False
    for _, module in model.named_modules():
        if not is_quantized_linear(module) or isinstance(module, ResidueQuantLinear):
            continue
        needs_replacement = True
        if type(module) not in ResidueQuantModuleRegistry:
            ResidueQuantModuleRegistry.register({type(module): module.__class__.__name__})(
                ResidueQuantLinear
            )
    if not needs_replacement:
        return model
    if os.getenv("RESIDUE_DEBUG", "0") == "1":
        print_rank_0("Replacing instances of QuantLinear with ResidueQuantLinear.")
    # ModelOpt layerwise calibration calls residue_max_calibrate() on a single
    # decoder layer, not the root model that owns modelopt state.  The current
    # replacement helper only recurses through children and does not use
    # state_version, so do not require ModeloptStateManager(model) here.
    _replace_quant_module(model, version=None, registry=ResidueQuantModuleRegistry)
    return model

def restore_residue_model(model: nn.Module, config: QuantizeConfig, metadata: MetadataDict):
    """Restore the residue quantization states from the given state dict.

    NOTE: This only restores architecture and creates empty buffers.
    Weight extension happens AFTER load_state_dict() via finalize_restore_residue_model().
    """
    print("[restore_residue_model] Step 1: Replacing architecture...")
    create_and_replace_residue_linear_on_the_fly(model)

    print("[restore_residue_model] Step 2: Restoring quantizer state (creates empty buffers)...")
    restore_quantizer_state(model, config, metadata)

    print("[restore_residue_model] ✓ Architecture restored. Call finalize_restore_residue_model() after mto.restore() to extend weights.")
    return model


def finalize_restore_residue_model(model: nn.Module) -> nn.Module:
    """Finalize residue model restoration by extending weights.

    This MUST be called AFTER mto.restore() completes, because it requires
    salient_indices to be loaded from the checkpoint's state_dict.

    Args:
        model: Model returned from mto.restore()

    Returns:
        Model with extended weights

    Example:
        >>> model = mto.restore(base_model, checkpoint_path)
        >>> model = finalize_restore_residue_model(model)
    """
    print("[finalize_restore_residue_model] Extending weights using loaded salient_indices...")
    extended_count = 0

    for name, module in model.named_modules():
        if is_quantized_linear(module) and hasattr(module, 'input_quantizer'):
            salient_indices = getattr(module.input_quantizer, '_salient_indices', None)
            if salient_indices is not None and salient_indices.numel() > 0:
                # Only extend once (idempotency check)
                if not hasattr(module, '_original_in_features'):
                    # Device alignment - ensure indices on same device as weight
                    weight_device = module.weight.device
                    if salient_indices.device != weight_device:
                        module.input_quantizer._salient_indices = salient_indices.to(weight_device)
                        salient_indices = module.input_quantizer._salient_indices

                    # Bounds validation
                    in_features = module.weight.shape[1]
                    if salient_indices.numel() > 0:
                        max_idx = salient_indices.max().item()
                        min_idx = salient_indices.min().item()
                        if max_idx >= in_features or min_idx < 0:
                            print(
                                f"  WARNING: Skipping {name} - salient_indices out of bounds "
                                f"(min={min_idx}, max={max_idx}, in_features={in_features})"
                            )
                            continue

                    if extended_count < 3:  # Only print first 3 for brevity
                        print(f"  Extending {name}: {module.weight.shape} + {salient_indices.numel()} salient")
                    trigger_weight_extension(model, module, salient_indices)
                    extended_count += 1

    print(f"[finalize_restore_residue_model] ✓ Extended {extended_count} layers")
    return model


def residue_restore(model, checkpoint_path, **load_kwargs):
    """Custom restore function for residue quantization.

    This properly handles weight extension by:
    1. Loading checkpoint
    2. Restoring architecture (creates empty buffers)
    3. Extracting salient_indices from checkpoint's model_state_dict
    4. Extending weights BEFORE loading state_dict
    5. Loading full state_dict

    Args:
        model: Base model or ModelLike object
        checkpoint_path: Path to checkpoint saved with mto.save()
        **load_kwargs: Additional kwargs for torch.load()

    Returns:
        Restored model with extended weights

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
        >>> model = residue_restore(base_model, "checkpoint.pt")
    """
    import os
    from pathlib import Path
    from modelopt.torch.opt.conversion import (
        restore_from_modelopt_state,
        ModelLikeModule,
        unwrap_model,
    )

    print("[residue_restore] Loading checkpoint...")

    # Initialize ModelLikeModule if needed
    model = model if isinstance(model, nn.Module) else ModelLikeModule(model)

    # Check if the model is in a wrapper
    model = unwrap_model(model, raise_error=True)

    # Load checkpoint
    load_kwargs.setdefault("map_location", "cpu")
    load_kwargs.setdefault("weights_only", False)
    objs = torch.load(checkpoint_path, **load_kwargs)

    print("[residue_restore] Restoring architecture...")
    # Restore model architecture (creates empty buffers)
    model_restored = restore_from_modelopt_state(model, objs["modelopt_state"])

    print("[residue_restore] Extending weights using salient_indices from checkpoint...")
    # Extract salient_indices from checkpoint and extend weights
    state_dict = objs["model_state_dict"]
    extended_count = 0

    for name, module in model_restored.named_modules():
        if is_quantized_linear(module) and hasattr(module, 'input_quantizer'):
            # Get salient_indices from checkpoint's state_dict
            salient_key = f"{name}.input_quantizer._salient_indices"
            if salient_key in state_dict:
                salient_indices = state_dict[salient_key]

                if salient_indices.numel() > 0 and not hasattr(module, '_original_in_features'):
                    # Device alignment
                    weight_device = module.weight.device
                    if salient_indices.device != weight_device:
                        salient_indices = salient_indices.to(weight_device)

                    # Extend weights
                    if extended_count < 3:  # Only print first 3
                        print(f"  Extending {name}: {module.weight.shape} + {salient_indices.numel()} salient")
                    trigger_weight_extension(model_restored, module, salient_indices)
                    extended_count += 1

    print(f"[residue_restore] ✓ Extended {extended_count} layers")

    print("[residue_restore] Loading state_dict...")
    # Now load the full state_dict (weights will match extended dimensions)
    model_restored.load_state_dict(state_dict)

    print("[residue_restore] ✓ Restore complete")
    return model_restored


@torch.no_grad()
def inject_residue_salient_map(
    model: nn.Module,
    salient_indices_map: dict[str, torch.Tensor],
    block_size: int = 8,
    residue_ratio: float | None = None,
    debug: bool = False,
    replace_modules: bool = True,
    preserve_existing_amax: bool = True,
):
    """Inject externally computed salient indices and reuse residue postprocess.

    ``block_size`` and ``residue_ratio`` are accepted for compatibility with
    pipeline configs, but external salient indices fully determine the extension
    pattern.

    External injection is intended to preserve the base calibration state
    (MSE/AWQ/GPTQ). It should materialize residue weights and metadata without
    recalibrating weight/input scales.
    """
    if not salient_indices_map:
        raise ValueError("salient_indices_map must be non-empty")

    if replace_modules:
        create_and_replace_residue_linear_on_the_fly(model)

    return _inject_residue_salient_map_direct(
        model,
        salient_indices_map=salient_indices_map,
        debug=debug,
        preserve_existing_amax=preserve_existing_amax,
    )


def _clone_quantizer_amax(quantizer):
    amax = getattr(quantizer, "amax", None)
    if isinstance(amax, torch.Tensor):
        return amax.detach().clone()
    return amax


def _restore_quantizer_amax(quantizer, amax):
    quantizer.amax = amax


def _inject_residue_salient_map_direct(
    model: nn.Module,
    salient_indices_map: dict[str, torch.Tensor],
    debug: bool = False,
    preserve_existing_amax: bool = True,
):
    """Materialize externally provided residue salient indices without calibration."""
    modules = dict(model.named_modules())
    processed_count = 0
    newly_extended_count = 0
    already_extended_count = 0
    verbose = debug or os.getenv("RESIDUE_DEBUG", "0") == "1"

    for name, indices in salient_indices_map.items():
        if name not in modules:
            raise KeyError(f"Layer from residue metadata not found in model: {name}")

        module = modules[name]
        if not is_quantized_linear(module):
            raise TypeError(
                f"Layer '{name}' is not a quantized linear module and cannot be residue-injected"
            )
        if not getattr(module.weight_quantizer, "is_enabled", False):
            if verbose:
                print_rank_0(f"[residue inject] Skipping disabled weight quantizer: {name}")
            continue

        salient_indices = torch.as_tensor(indices, dtype=torch.int64)
        salient_indices = salient_indices.to(module.weight.device, dtype=torch.int64).contiguous()
        existing_input_salient = getattr(module.input_quantizer, "_salient_indices", None)
        existing_input_salient_cpu = (
            existing_input_salient.detach().cpu().clone()
            if isinstance(existing_input_salient, torch.Tensor)
            else None
        )

        if hasattr(module.weight_quantizer, "_salient_indices"):
            delattr(module.weight_quantizer, "_salient_indices")

        processed_count += 1
        if hasattr(module, "_original_in_features"):
            if existing_input_salient_cpu is None:
                raise RuntimeError(
                    f"Layer '{name}' is already extended but input_quantizer._salient_indices was not loaded."
                )
            if not torch.equal(existing_input_salient_cpu, salient_indices.detach().cpu()):
                raise ValueError(
                    f"Layer '{name}' is already extended with different salient_indices."
                )
            expected_in_features = int(module._original_in_features) + int(salient_indices.numel())
            current_in_features = int(getattr(module, "in_features", module.weight.shape[1]))
            if current_in_features != expected_in_features:
                raise RuntimeError(
                    f"Layer '{name}' has inconsistent extended dimension: "
                    f"expected {expected_in_features}, got {current_in_features}."
                )
            already_extended_count += 1
            continue

        apply_salient_indices(module, salient_indices)
        saved_weight_amax = (
            _clone_quantizer_amax(module.weight_quantizer)
            if preserve_existing_amax
            else None
        )
        trigger_weight_extension(model, module, salient_indices)
        if preserve_existing_amax:
            _restore_quantizer_amax(module.weight_quantizer, saved_weight_amax)
        newly_extended_count += 1

    return {
        "processed_layers": processed_count,
        "newly_extended_layers": newly_extended_count,
        "already_extended_layers": already_extended_count,
    }


@torch.no_grad()
def residue_max_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop,
    block_size: int = 8,
    residue_ratio: float = 0.25,
    debug: bool = False,
    external_salient_metadata_path: str | None = None,
    **kwargs,
):
    # DEBUG: Print received parameters to verify they're passed correctly
    if os.getenv("RESIDUE_DEBUG", "0") == "1":
        print_rank_0(
            f"[residue_max_calibrate] RECEIVED PARAMS: block_size={block_size}, "
            f"residue_ratio={residue_ratio}, "
            f"external_salient_metadata_path={external_salient_metadata_path}"
        )
        print_rank_0(f"[residue_max_calibrate] kwargs={list(kwargs.keys())}")

    if forward_loop is None:
        warnings.warn("forward_loop must be provided for residue_max; skipping")
        return

    # Phase 1: Module replacement (SVDQuant style class generation)
    create_and_replace_residue_linear_on_the_fly(model)

    # Phase 2-5: Core calibration logic. This intentionally records salience
    # only; K+S weight extension is an explicit export/injection step.
    return _residue_calibrate_core(
        model,
        forward_loop,
        block_size,
        residue_ratio,
        debug,
        external_salient_metadata_path=external_salient_metadata_path,
        **kwargs,
    )


def _restore_input_quantizer_state(module):
    """Restore input_quantizer state after residue act_scale collection.

    Reads `module.residue_max.is_input_quantized` and the optional
    `_saved_awq_amax` snapshot (taken during ResidueHelper.setup) and rebinds
    the input quantizer to its pre-residue-stats configuration.
    """
    if module.residue_max.is_input_quantized:
        saved_amax = getattr(module.residue_max, "_saved_awq_amax", None)
        calibrated_amax = module.input_quantizer.amax
        module.input_quantizer.reset_amax()
        module.input_quantizer.axis = None
        if saved_amax is not None:
            module.input_quantizer.amax = saved_amax
        elif calibrated_amax is not None:
            act_amax = calibrated_amax
            module.input_quantizer._amax_for_smoothing = act_amax
            module.input_quantizer.amax = act_amax.amax()
        module.input_quantizer.enable()


def _cleanup_residue_postprocess(module, *, debug: bool):
    module.residue_max.cleanup()
    if not debug:
        delattr(module, "residue_max")


def _compute_residue_salient_indices(
    module,
    name: str,
    current_layer_idx: int,
    total_modules: int,
    verbose: bool,
):
    salient_indices = get_salient_indices(
        module.residue_max.act_scale,
        module.residue_max.block_size,
        module.residue_max.residue_ratio,
    )

    if verbose and current_layer_idx == 0:
        num_channels = module.residue_max.act_scale.shape[0]
        print_rank_0(
            f"[postprocess] Layer 0: ratio={module.residue_max.residue_ratio}, "
            f"salient_indices={len(salient_indices)}/{num_channels} "
            f"(expected: {int(module.residue_max.residue_ratio * num_channels)})"
        )

    return salient_indices


def _load_external_residue_salient_map(
    metadata_path: str | os.PathLike[str],
    *,
    block_size: int,
    residue_ratio: float,
) -> dict[str, torch.Tensor]:
    """Load per-layer salient indices for salience-only auto-search scoring.

    This is intentionally not the materialized injection path. It lets residue
    calibration keep its usual quantizer calibration/forward-loop behavior while
    replacing only the channel-selection rule.
    """
    path = Path(metadata_path)
    data = json.loads(path.read_text())
    layers = data.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"External residue metadata has no non-empty layers dict: {path}")

    global_meta = data.get("global", {})
    declared_block_size = global_meta.get("block_size")
    if declared_block_size is not None and int(declared_block_size) != int(block_size):
        raise ValueError(
            f"External residue metadata block_size mismatch for {path}: "
            f"metadata={declared_block_size}, requested={block_size}"
        )
    declared_ratio = global_meta.get("residue_ratio")
    if declared_ratio is not None and abs(float(declared_ratio) - float(residue_ratio)) > 1e-6:
        raise ValueError(
            f"External residue metadata ratio mismatch for {path}: "
            f"metadata={declared_ratio}, requested={residue_ratio}"
        )

    salient_map: dict[str, torch.Tensor] = {}
    for layer_name, layer_meta in layers.items():
        raw = layer_meta.get("salient_indices") if isinstance(layer_meta, dict) else None
        if raw is None:
            continue
        salient_map[layer_name] = torch.tensor(raw, dtype=torch.long)
    if not salient_map:
        raise ValueError(f"External residue metadata has no salient_indices: {path}")
    return salient_map


def _external_salient_indices_for_module(
    external_salient_map: dict[str, torch.Tensor],
    module: nn.Module,
    name: str,
) -> torch.Tensor:
    salient = external_salient_map.get(name)
    if salient is None:
        raise KeyError(f"External residue metadata is missing enabled layer: {name}")
    if salient.numel() == 0:
        return salient.to(device=module.weight.device, dtype=torch.long)
    num_channels = int(getattr(module, "in_features", module.weight.shape[1]))
    if int(salient.max().item()) >= num_channels or int(salient.min().item()) < 0:
        raise ValueError(
            f"External residue salient index out of range for {name}: "
            f"min={int(salient.min().item())}, max={int(salient.max().item())}, "
            f"in_features={num_channels}"
        )
    return salient.to(device=module.weight.device, dtype=torch.long)


@torch.no_grad()
def _residue_calibrate_core(
    model: nn.Module,
    forward_loop: ForwardLoop,
    block_size: int = 8,
    residue_ratio: float = 0.25,
    debug: bool = False,
    external_salient_metadata_path: str | None = None,
    **kwargs,
):
    """Residue calibration core logic (assumes module replacement already done).

    Extracted from residue_max_calibrate to keep module replacement separate
    from salience collection.

    This function is salience-only: it records calibrated salient indices on
    quantizers and never extends weights. Export/injection code is responsible
    for materializing K+S weights from the emitted metadata.
    """
    unsupported_external_keys = {
        "external_salient_map",
        "skip_residue_stats_collection",
        "preserve_existing_amax",
    } & set(kwargs)
    if unsupported_external_keys:
        raise ValueError(
            "External residue salient injection must use inject_residue_salient_map(), "
            f"not _residue_calibrate_core(); got {sorted(unsupported_external_keys)}"
        )
    verbose = debug or os.getenv("RESIDUE_DEBUG", "0") == "1"
    external_salient_map = (
        _load_external_residue_salient_map(
            external_salient_metadata_path,
            block_size=block_size,
            residue_ratio=residue_ratio,
        )
        if external_salient_metadata_path
        else None
    )

    for name, module in model.named_modules():
        if is_quantized_linear(module) and module.weight_quantizer.is_enabled:
            with enable_weight_access_and_writeback(module, model):
                module.residue_max = ResidueHelper(module, name, block_size, residue_ratio)
            module.residue_max.setup()
            module._layer_name = name

    ResidueHelper.cache_mode = True
    print_rank_0("residue_max: caching activation statistics...")

    enable_stats_collection(model)
    forward_loop(model)

    max_calibrate(model, lambda model: None)

    def sync_act_scale_across_dp(module, data_parallel_group):
        """sync activation scale across data parallel (dp)."""
        if data_parallel_group.is_initialized():
            dist.all_reduce(
                module.residue_max.act_scale, op=dist.ReduceOp.AVG, group=data_parallel_group.group
            )

    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and hasattr(module, "residue_max")
            and module.residue_max.num_cache_steps > 0
        ):
            module._if_calib = True
            module.residue_max.act_scale = module.residue_max.act_scale / module.residue_max.num_cache_steps

            has_nan_local = torch.any(torch.isnan(module.residue_max.act_scale))
            has_nan = DistributedProcessGroup.get_dist_syncd_obj(
                has_nan_local, module.parallel_state.data_parallel_group, lambda objs: any(objs)
            )

            if has_nan:
                module.residue_max.is_enabled = False
            else:
                sync_act_scale_across_dp(
                    module,
                    module.parallel_state.data_parallel_group,
                )

    ResidueHelper.cache_mode = False
    print_rank_0("residue_max: finished activation statistics.")

    total_modules = sum(1 for _ in model.modules() if is_quantized_linear(_) and hasattr(_, "residue_max"))
    processed_count = [0]
    salience_only_count = [0]
    skipped_count = [0]

    def postprocess(module, name):
        current_layer_idx = processed_count[0]
        is_enabled = module.residue_max.is_enabled
        if not is_enabled or (
            external_salient_map is None and module.residue_max.num_cache_steps <= 0
        ):
            salient_indices = torch.empty(0, device=module.weight.device, dtype=torch.long)
            is_enabled = False
        elif external_salient_map is not None:
            salient_indices = _external_salient_indices_for_module(
                external_salient_map,
                module,
                name,
            )
        else:
            salient_indices = _compute_residue_salient_indices(
                module,
                name,
                current_layer_idx,
                total_modules,
                verbose,
            )

        processed_count[0] += 1

        if is_enabled and salient_indices.numel() > 0:
            if hasattr(module.weight_quantizer, "_salient_indices"):
                delattr(module.weight_quantizer, "_salient_indices")
            if hasattr(module.input_quantizer, "_salient_indices"):
                delattr(module.input_quantizer, "_salient_indices")
            apply_salient_indices(module, salient_indices)
            module._residue_ratio = getattr(module.residue_max, "residue_ratio", None)
            module._residue_materialized = False
            salience_only_count[0] += 1
        else:
            module._residue_materialized = False
            skipped_count[0] += 1

        _restore_input_quantizer_state(module)

    for name, module in model.named_modules():
        if hasattr(module, "residue_max"):
            postprocess(module, name)
            _cleanup_residue_postprocess(module, debug=debug)

    return {
        "processed_layers": processed_count[0],
        "salience_only_layers": salience_only_count[0],
        "skipped_layers": skipped_count[0],
        "newly_extended_layers": 0,
        "already_extended_layers": 0,
    }

class ResidueMaxCalibConfig(QuantizeAlgorithmConfig):
    """The config for residue max calibration algorithm.

    Residue Max calibration adds block amax tracking for residue quantization.
    """

    method: Literal["residue_max"] = ModeloptField("residue_max")

    distributed_sync: bool | None = ModeloptField(
        default=True,
        title="Whether to sync the amax across the distributed processes.",
        description="If True, the amax will be synced across the distributed processes.",
    )

    block_size: int | None = ModeloptField(
        default=8,
        ge=1,
        title="block size for max calibration.",
        description="The block size for per-block max calibration.",
    )

    residue_ratio: float | None = ModeloptField(
        default=0.25,
        ge=0.0,
        title="Residue ratio for residue quantization.",
        description="The residue ratio for residue quantization.",
    )

    track_amax: bool | None = ModeloptField(
        default=True,
        title="Whether to track amax during calibration.",
        description="If True, the calibrator will track amax during calibration.",
    )

    external_salient_metadata_path: str | None = ModeloptField(
        default=None,
        title="External residue salient metadata for salience-only scoring.",
        description=(
            "Optional residue metadata path used by auto-search candidates. It "
            "replaces the act_scale top-k selection with precomputed per-layer "
            "salient_indices, without materializing weight extension."
        ),
    )


@CalibrateModeRegistry.register_mode
class ResidueMaxCalibrateModeDescriptor(BaseCalibrateModeDescriptor):
    """Mode for residue max calibration algorithm."""

    @property
    def config_class(self) -> type[QuantizeAlgorithmConfig]:
        """Specifies the config class for the mode."""
        return ResidueMaxCalibConfig

    _calib_func = residue_max_calibrate

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_residue_model
    # @property
    # def convert(self) -> ConvertEntrypoint:
    #   pass


def trigger_weight_extension(model: nn.Module, module: nn.Module, salient_indices: torch.Tensor = None):
    if is_quantized_linear(module) and module.weight_quantizer.is_enabled:
        with enable_weight_access_and_writeback(module, model):
            wq = module.weight_quantizer

            # Save metadata ONLY (not references to weight data)
            requires_grad = module.weight.requires_grad
            weight_device = module.weight.device

            # CRITICAL: Make a copy of weight data, then FREE the original immediately
            # This ensures we don't have both old and new weights in memory during extension
            weight_data_copy = module.weight.data.clone()

            # FREE the original weight IMMEDIATELY before extending
            with torch.no_grad():
                module.weight.data = torch.empty(0, device=weight_device)

            # Delete the parameter attribute
            del module.weight

            # Force garbage collection to free memory NOW
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # Materialized residue weights append a copy of the selected
            # channels. Runtime activation extension supplies the residual.
            extended_weight = wq.copy_extension(
                input=weight_data_copy,
                salient_indices=salient_indices
            )

            # Free the temporary copy
            del weight_data_copy
            gc.collect()
            torch.cuda.empty_cache()

            # Assign NEW extended weight as new parameter
            module.weight = nn.Parameter(extended_weight, requires_grad=requires_grad)

            # Store original in_features before updating (for collect_extended_linear_dims)
            if hasattr(module, 'in_features'):
                module._original_in_features = module.in_features
                module.in_features = extended_weight.shape[1]  # Update to extended size
