# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Declare ResAttn specializations to vLLM's CuTeDSL warmup executor."""
from __future__ import annotations

import functools
import torch
from vllm.config import get_current_vllm_config_or_none

from .runtime import _load_decode_kernel, _qk_residue_for_geometry


class _DecodeWarmupProvider:
    """CuTeDSL warmup provider for the decode kernel (vLLM first-class hook).

    vLLM's `kernel_warmup` drains `register_cutedsl_warmup_provider` units
    under `enable_cutedsl_warmup` -- the mechanism built for exactly this
    case (JIT'd CuTeDSL kernels), replacing the old `kernel_warmup`
    monkeypatch, which was silently dead under real serving:
    `gpu_worker.py:53` from-imports `kernel_warmup` by name, so rebinding the
    module attribute after that import changed nothing and the only symptom
    was a slow first token.

    One unit per `_compiled` key reachable for the registered geometry, from
    `host.plan_decode_compile_variants` -- which shares its construction with
    the production entry, so warmup compiles what live requests hit. vLLM
    holds providers in a WeakSet; the module-level `_DECODE_WARMUP_PROVIDER`
    strong ref keeps this one alive.
    """

    def __init__(self):
        self._geometries: dict[
            tuple[int, int, int], tuple[int, int, bool]
        ] = {}

    def add_geometry(self, num_qo_heads: int, num_kv_heads: int,
                     head_dim: int, max_model_len: int,
                     max_num_seqs: int,
                     fuse_attn_quant: bool) -> None:
        self._geometries[(num_qo_heads, num_kv_heads, head_dim)] = (
            max_model_len, max_num_seqs, fuse_attn_quant
        )

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import (
            CuTeDSLCompileUnit,
        )

        host = _load_decode_kernel()
        if host is None or not torch.cuda.is_available():
            return []
        device = torch.device("cuda", torch.cuda.current_device())
        units = []
        for (hq, hkv, d), (
            mml,
            mns,
            fuse_attn_quant,
        ) in self._geometries.items():
            qk_residue = _qk_residue_for_geometry(hq, hkv)
            plans = host.plan_decode_compile_variants(
                hq,
                hkv,
                d,
                mml,
                mns,
                device,
                residue_fold=qk_residue,
            )
            if fuse_attn_quant:
                plans += host.plan_decode_nvfp4_compile_variants(
                    hq,
                    hkv,
                    d,
                    mml,
                    mns,
                    device,
                    residue_fold=qk_residue,
                )
            for plan in plans:
                units.append(CuTeDSLCompileUnit(
                    name=f"resattn_decode[{plan['label']}]",
                    key=("resattn_decode", plan["args"],
                         tuple(sorted(plan["kwargs"].items()))),
                    compile=functools.partial(
                        host.compile_decode_variant, plan
                    ),
                ))
        return units


_DECODE_WARMUP_PROVIDER: _DecodeWarmupProvider | None = None


def _register_decode_warmup(num_qo_heads: int, num_kv_heads: int,
                            head_dim: int) -> None:
    """Register (or extend) the decode-kernel warmup provider. Best effort."""
    global _DECODE_WARMUP_PROVIDER
    try:
        from vllm.model_executor.warmup.cutedsl_warmup import (
            register_cutedsl_warmup_provider,
        )
    except ImportError:  # pragma: no cover - vLLM layout drift
        return
    cfg = get_current_vllm_config_or_none()
    max_model_len = (
        cfg.model_config.max_model_len
        if cfg is not None and cfg.model_config is not None else 65536
    )
    max_num_seqs = (
        cfg.scheduler_config.max_num_seqs
        if cfg is not None and cfg.scheduler_config is not None else 64
    )
    fuse_attn_quant = bool(
        cfg is not None
        and cfg.compilation_config is not None
        and cfg.compilation_config.pass_config.fuse_attn_quant
    )
    if _DECODE_WARMUP_PROVIDER is None:
        _DECODE_WARMUP_PROVIDER = _DecodeWarmupProvider()
    _DECODE_WARMUP_PROVIDER.add_geometry(
        num_qo_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
        max_num_seqs,
        fuse_attn_quant,
    )
    register_cutedsl_warmup_provider(_DECODE_WARMUP_PROVIDER)
