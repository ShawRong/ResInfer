"""Startup and loader diagnostics; no tensor reads or execution decisions."""
from __future__ import annotations

import logging
import os


def _logger():
    # Lazy import keeps configuration/metadata importable without the Runtime.
    # Registration also needs to report errors when vLLM itself is unavailable.
    try:
        from vllm.logger import init_logger
    except ImportError:
        return logging.getLogger("vllm.residue")
    return init_logger("vllm.residue")


def log_selected_kernel_once(method):
    logger = _logger()
    compatible = isinstance(method.kernel, method._CUTLASS_LAYOUT_KERNELS)
    logger.info_once(
        "NVFP4 kernel: %s (residue-compatible layout: %s)",
        type(method.kernel).__name__, compatible,
    )
    if not compatible:
        logger.warning_once(
            "NVFP4 kernel %s cannot provide the residue CUTLASS layout; "
            "residue features may be disabled or rejected. Select cutlass, "
            "flashinfer_cutlass or flashinfer_cutedsl.",
            type(method.kernel).__name__,
        )


def log_weight_creation(method, layer_name, input_size, extended_dim):
    if os.environ.get("RESIDUE_DEBUG_CREATE_WEIGHTS") != "1":
        return
    _logger().info("create_weights %s: input=%s extended=%s decode_mext=%s",
                   layer_name, input_size, extended_dim, method._settings.decode_mext)


def log_extended_input(layer_name, original_input, input_size, plan):
    if os.environ.get("RESIDUE_DEBUG_CREATE_WEIGHTS") != "1":
        return
    detail = (f"row-parallel tp={plan.tp_size} rank={plan.tp_rank}: "
              f"{plan.base_shard} base + {plan.salient_shard} salient"
              if plan is not None else "unsharded input")
    _logger().info("%s: overriding input_size %s -> %s (%s)",
                   layer_name, original_input, input_size, detail)


def log_weight_layout(method, layer, layer_name):
    if os.environ.get("RESIDUE_DEBUG_CREATE_WEIGHTS") != "1":
        return
    _logger().info(
        "post_process_weights %s: residue=%s cutlass=%s decode_mext=%s "
        "decode_layout=%s base_scale=%s k_base=%s",
        layer_name, getattr(layer, "_is_residue_layer", False),
        method._uses_cutlass_compatible_layout(), method._settings.decode_mext,
        getattr(layer, "_decode_mext_layout_enabled", False),
        hasattr(layer, "weight_scale_base"), getattr(layer, "_mext_decode_k_base", None),
    )


def log_tp_load(layer, loaded_weight, param, expected, interleaved):
    if os.environ.get("RESIDUE_DEBUG_TP") != "1":
        return
    _logger().info("tp-loader %s: loaded=%s expected_last=%s interleaved=%s param=%s",
                   getattr(layer, "prefix", "?"), tuple(loaded_weight.shape),
                   expected, interleaved, tuple(param.shape))


def log_settings(cfg):
    _logger().info("mode=%s decode_mext=%s max_m=%s ratio1_prefill=%s",
                   cfg.mode, cfg.decode_mext, cfg.decode_max_m, cfg.ratio1_prefill)


def log_patch(method_name, mode):
    _logger().info("ModelOpt NVFP4 -> %s (%s)", method_name, mode)


def log_registration_failure(reason, detail=None):
    _logger().warning("ResAttn registration refused: %s%s. Backend left unchanged.",
                      reason, "" if detail is None else f" ({detail})")


def log_registration_success(class_path):
    _logger().info("ResAttn registered on CUSTOM (%s); select --attention-backend "
                   "CUSTOM --kv-cache-dtype nvfp4 --block-size 128", class_path)


def log_decode_import_failure(exc):
    _logger().error("ResAttn decode kernel nvfp4.cute_attn.host failed to import "
                    "(%s: %s); backend will refuse the first decode forward",
                    type(exc).__name__, exc)
