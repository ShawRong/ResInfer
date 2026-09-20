# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
#
# Light-weight registry for custom residue quantizers
import copy
import modelopt.torch.quantization as mtq


_QUANT_SCOPE_PREFIX_PATTERNS = {
    "ffn_only": ["*mlp*"],
    # Intentionally exclude self_attn.o_proj. This scope is for isolating
    # q/k/v projection quantization impact from output projection and FFN.
    "qkv_only": [
        "*self_attn.q_proj*",
        "*self_attn.k_proj*",
        "*self_attn.v_proj*",
    ],
}


def apply_quant_scope(custom_cfg, quant_scope: str | None = None):
    """Restrict which TensorQuantizers ModelOpt enables.

    ``ffn_only`` means only modules whose quantizer name contains ``mlp`` keep
    weight/input quantization enabled.

    ``qkv_only`` means only ``self_attn.q_proj/k_proj/v_proj`` quantizers keep
    weight/input quantization enabled. It intentionally excludes ``o_proj``.

    Both scopes directly control ModelOpt quantizer enablement instead of using
    calibration-time environment-variable filters.
    """
    quant_scope = quant_scope or "all"
    if quant_scope == "all":
        return custom_cfg
    if quant_scope not in _QUANT_SCOPE_PREFIX_PATTERNS:
        raise ValueError(f"Unsupported quant_scope: {quant_scope}")
    scope_patterns = _QUANT_SCOPE_PREFIX_PATTERNS[quant_scope]

    quant_cfg = custom_cfg["quant_cfg"]
    if isinstance(quant_cfg, list):
        weight_cfg = None
        input_cfg = None
        kept_entries = []
        for entry in quant_cfg:
            qname = entry.get("quantizer_name")
            if qname == "*weight_quantizer":
                weight_cfg = copy.deepcopy(entry)
                continue
            if qname == "*input_quantizer":
                input_cfg = copy.deepcopy(entry)
                continue
            kept_entries.append(entry)

        if weight_cfg is None or input_cfg is None:
            raise ValueError(
                f"quant_scope={quant_scope} requires global weight/input quantizer entries"
            )

        scope_entries = []
        for pattern in scope_patterns:
            scoped_weight_cfg = copy.deepcopy(weight_cfg)
            scoped_input_cfg = copy.deepcopy(input_cfg)
            scoped_weight_cfg["quantizer_name"] = f"{pattern}weight_quantizer"
            scoped_input_cfg["quantizer_name"] = f"{pattern}input_quantizer"
            scope_entries.extend([scoped_weight_cfg, scoped_input_cfg])

        custom_cfg["quant_cfg"] = kept_entries + scope_entries
        return custom_cfg

    weight_cfg = copy.deepcopy(quant_cfg.pop("*weight_quantizer", None))
    input_cfg = copy.deepcopy(quant_cfg.pop("*input_quantizer", None))
    if weight_cfg is None or input_cfg is None:
        raise ValueError(
            f"quant_scope={quant_scope} requires global weight/input quantizer entries"
        )

    for pattern in scope_patterns:
        quant_cfg[f"{pattern}weight_quantizer"] = copy.deepcopy(weight_cfg)
        quant_cfg[f"{pattern}input_quantizer"] = copy.deepcopy(input_cfg)
    return custom_cfg


def _disable_lm_head(custom_cfg):
    """Disable lm_head quantizer across ModelOpt config schema variants."""
    quant_cfg = custom_cfg["quant_cfg"]
    if isinstance(quant_cfg, list):
        quant_cfg.append({"quantizer_name": "*lm_head*", "enable": False})
    else:
        quant_cfg["*lm_head*"] = {"enable": False}


def build_residue_cfg(
    fmt: str,
    ratio: float,
    calib_block_size: int = 8,
    quant_scope: str | None = None,
    use_sequential: bool = False,
    external_salient_metadata_path: str | None = None,
    salience_metric: str = "activation_max",
):
    """Build quant_cfg + algorithm dict for ModelOpt conversion.

    Note: Mode (copy vs residue) is handled by ResidueQuantLinear._setup()
    which swaps TensorQuantizer to CopyTensorQuantizer (weight) or
    ActivationResidueTensorQuantizer (input). Config only specifies format.

    When ``use_sequential=True``, the config uses ModelOpt's official
    ``layerwise`` calibration path. Residue calibration remains salience-only;
    materialization is handled by the external metadata injection/export stage.
    """
    CUSTOM_CFG = None
    if fmt == "nvfp4":
        CUSTOM_CFG = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG )
    elif fmt == "mxfp4":
        CUSTOM_CFG = copy.deepcopy(mtq.MXFP4_DEFAULT_CFG)
    else:
        raise ValueError(f"Unsupported residue format: {fmt}")
    _disable_lm_head(CUSTOM_CFG)
    apply_quant_scope(CUSTOM_CFG, quant_scope)

    algo_cfg = {
        "method": "residue_max",
        "residue_ratio": ratio,
        "block_size": calib_block_size,
        "track_amax": True,
        "distributed_sync": False,
        "salience_metric": salience_metric,
    }
    if use_sequential:
        algo_cfg["layerwise"] = True
    if external_salient_metadata_path:
        algo_cfg["external_salient_metadata_path"] = external_salient_metadata_path
    CUSTOM_CFG["algorithm"] = algo_cfg

    return CUSTOM_CFG


def build_awq_lite_cfg(
    fmt: str = "nvfp4",
    alpha_step: float = 0.1,
    quant_scope: str | None = None,
):
    """Build config for standalone AWQ-lite W4A4 (no residue).

    Uses ModelOpt's built-in awq_lite calibration which computes per-channel
    pre_quant_scale to smooth activations before W4A4 quantization.

    Only NVFP4 is supported. ModelOpt does not provide MXFP4_AWQ config.
    """
    assert fmt == "nvfp4", f"AWQ-lite only supports nvfp4, got {fmt}"
    CUSTOM_CFG = copy.deepcopy(mtq.NVFP4_AWQ_LITE_CFG)
    _disable_lm_head(CUSTOM_CFG)
    apply_quant_scope(CUSTOM_CFG, quant_scope)

    CUSTOM_CFG["algorithm"] = {
        "method": "awq_lite",
        "alpha_step": alpha_step,
    }

    return CUSTOM_CFG


def build_awq_full_cfg(
    fmt: str = "nvfp4",
    alpha_step: float = 0.1,
    max_co_batch_size: int = 1024,
    max_tokens_per_batch: int = 64,
    min_clip_ratio: float = 0.5,
    shrink_step: float = 0.05,
    quant_scope: str | None = None,
):
    """Build config for ModelOpt AWQ-full W4A4.

    AWQ-full runs AWQ-lite smoothing followed by AWQ-clip.  The calibration
    output may contain both pre-quant scales and clipped weight amax values, so
    external residue injection should fuse pre-quant scales before extending
    weights.
    """
    assert fmt == "nvfp4", f"AWQ-full only supports nvfp4, got {fmt}"
    CUSTOM_CFG = copy.deepcopy(mtq.NVFP4_AWQ_FULL_CFG)
    _disable_lm_head(CUSTOM_CFG)
    apply_quant_scope(CUSTOM_CFG, quant_scope)

    CUSTOM_CFG["algorithm"] = {
        "method": "awq_full",
        "alpha_step": alpha_step,
        "max_co_batch_size": max_co_batch_size,
        "max_tokens_per_batch": max_tokens_per_batch,
        "min_clip_ratio": min_clip_ratio,
        "shrink_step": shrink_step,
    }

    return CUSTOM_CFG


def build_awq_clip_cfg(
    fmt: str = "nvfp4",
    max_co_batch_size: int = 1024,
    max_tokens_per_batch: int = 64,
    min_clip_ratio: float = 0.5,
    shrink_step: float = 0.05,
    quant_scope: str | None = None,
):
    """Build config for ModelOpt AWQ-clip W4A4."""
    assert fmt == "nvfp4", f"AWQ-clip only supports nvfp4, got {fmt}"
    CUSTOM_CFG = copy.deepcopy(mtq.NVFP4_AWQ_CLIP_CFG)
    _disable_lm_head(CUSTOM_CFG)
    apply_quant_scope(CUSTOM_CFG, quant_scope)

    CUSTOM_CFG["algorithm"] = {
        "method": "awq_clip",
        "max_co_batch_size": max_co_batch_size,
        "max_tokens_per_batch": max_tokens_per_batch,
        "min_clip_ratio": min_clip_ratio,
        "shrink_step": shrink_step,
    }

    return CUSTOM_CFG


def build_gptq_cfg(
    fmt: str = "nvfp4",
    perc_damp: float = 0.01,
    block_size: int = 128,
    quant_scope: str | None = None,
    layerwise: bool = False,
):
    """Build config for standalone GPTQ W4A4/NVFP4.

    Uses the official GPTQ calibration from modelopt main (PR #853),
    which correctly populates input_quantizer.amax via its internal
    ``max_calibrate(model, forward_loop=...)`` call. Non-sequential mode
    (``use_sequential=False``, the default) runs on the full model at once.
    """
    assert fmt == "nvfp4", f"GPTQ only supports nvfp4, got {fmt}"
    CUSTOM_CFG = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    _disable_lm_head(CUSTOM_CFG)
    apply_quant_scope(CUSTOM_CFG, quant_scope)

    CUSTOM_CFG["algorithm"] = {
        "method": "gptq",
        "perc_damp": perc_damp,
        "block_size": block_size,
    }
    if layerwise:
        # Sequential per-decoder-layer calibration: each layer's Hessian is
        # built from the PREVIOUS layers' already-quantized activations instead
        # of the original ones. modelopt runs a full model forward per layer, so
        # this costs roughly num_layers x the non-layerwise calibration.
        #
        # Whether it helps HERE is not obvious. The residue is injected AFTER
        # this stage, so layerwise propagates uncompensated quantization error
        # -- an input distribution worse than what the served model sees on the
        # K-ext layers (which compensate at every M). It is arguably a BETTER
        # match for mext_r1 layers, which carry num_salient=0 and take the
        # plain-NVFP4 bypass above the fold's M threshold, i.e. run
        # uncompensated exactly as layerwise assumes. Measure, do not assume.
        CUSTOM_CFG["algorithm"]["layerwise"] = True

    return CUSTOM_CFG
