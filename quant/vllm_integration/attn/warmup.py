# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Declare static decode compiler variants to vLLM's CuTeDSL warmup."""
from __future__ import annotations

import functools
import torch
from vllm.config import get_current_vllm_config_or_none

from .runtime import _load_decode_kernel, _validate_residue_geometry


class _DecodeWarmupProvider:
    """Compile variants only; no model forwards, timing or KV-page samples."""

    def __init__(self):
        self._geometries: set[tuple[int, int, int, bool]] = set()

    def add_geometry(self, num_qo_heads, num_kv_heads, head_dim, fuse_attn_quant):
        self._geometries.add((num_qo_heads, num_kv_heads, head_dim, fuse_attn_quant))

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        host = _load_decode_kernel()
        if host is None or not torch.cuda.is_available():
            return []
        units, seen = [], set()
        for hq, hkv, dim, packed in sorted(self._geometries):
            _validate_residue_geometry(hq, hkv)
            for output_nvfp4 in ((False, True) if packed else (False,)):
                for spec in host.decode_compile_variants(
                    hq, hkv, dim, residue_fold=True, output_nvfp4=output_nvfp4,
                ):
                    key = ('resattn_decode', spec['args'], tuple(sorted(spec['kwargs'].items())))
                    if key in seen:
                        continue
                    seen.add(key)
                    units.append(CuTeDSLCompileUnit(
                        name=f"resattn_decode[{spec['label']}]", key=key,
                        compile=functools.partial(host.compile_decode_variant, spec),
                    ))
        return units


# vLLM stores providers in a WeakSet; retain this provider for the worker lifetime.
_DECODE_WARMUP_PROVIDER: _DecodeWarmupProvider | None = None


def _register_decode_warmup(num_qo_heads, num_kv_heads, head_dim) -> None:
    global _DECODE_WARMUP_PROVIDER
    from vllm.model_executor.warmup.cutedsl_warmup import register_cutedsl_warmup_provider

    cfg = get_current_vllm_config_or_none()
    packed = bool(cfg is not None and cfg.compilation_config is not None
                  and cfg.compilation_config.pass_config.fuse_attn_quant)
    if _DECODE_WARMUP_PROVIDER is None:
        _DECODE_WARMUP_PROVIDER = _DecodeWarmupProvider()
    _DECODE_WARMUP_PROVIDER.add_geometry(num_qo_heads, num_kv_heads, head_dim, packed)
    register_cutedsl_warmup_provider(_DECODE_WARMUP_PROVIDER)
