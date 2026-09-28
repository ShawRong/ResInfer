"""Prepare Linear kernels once layouts are ready, before any graph capture.

The kernel compiler caches own deduplication (including device and dtype).
Required static kernels fail at loading; optional tuning candidates may fail.
"""
from __future__ import annotations

import torch

from ..flashinfer_integration.fi_tuner_base import precompile
from .env_contract import settings


def precompile_linear(layer, *, ratio1: bool) -> None:
    cfg = settings()
    k = int(layer._mext_decode_k_base)
    weight = layer.weight[:, :k // 2]
    n = int(weight.shape[0])
    stride = int(weight.stride(0)) * 2
    dtype = getattr(layer, "_residue_output_dtype", torch.bfloat16)
    with torch.cuda.device(weight.device):
        major, minor = torch.cuda.get_device_capability()
        if major == 10:
            if cfg.disable_sm10x:
                if ratio1:
                    raise RuntimeError("Ratio-1 residue requires Fold on this device")
                return
            from nvfp4.cute_fold.host import _compile_fold_gemm
            from nvfp4.cute_fold.tactics import kernel_arch_for_capability, warmup_specs
            from ..flashinfer_integration.sm10x_fold_tuner import _families

            arch = kernel_arch_for_capability(major, minor)
            def specs(families=()):
                return [((s["mma_tiler_mn"], s["cluster_shape_mn"], dtype),
                         {key: value for key, value in s.items()
                          if key not in ("mma_tiler_mn", "cluster_shape_mn")})
                        for s in warmup_specs(arch, families=families,
                                              strided=(stride != k,))]
            for args, kwargs in specs():
                _compile_fold_gemm(*args, **kwargs)
            if cfg.sm10x_tuner:
                precompile("sm10x Fold candidates", _compile_fold_gemm, specs(_families()))
        elif major == 12:
            from nvfp4.cute_fold_sm120.host import compile_fold_gemm_sm120
            from nvfp4.cute_fold_sm120.tactics import _TILE_TABLE, select_tile
            from ..flashinfer_integration.sm120_fold_tuner import precompile_fold_tiles

            # M is dynamic in compiled kernels. Table boundaries enumerate
            # every default tile; there is no separate capture-size ladder.
            bounds = {bound for rows in _TILE_TABLE.values() for bound, _ in rows}
            tiles = {select_tile(m, n, k) for m in bounds}
            a_ld = None if stride == k else stride
            for tile in tiles:
                compile_fold_gemm_sm120(n, k, tile, dtype, a_ld=a_ld)
            if cfg.sm120_tuner:
                precompile_fold_tiles(n, k, a_ld=a_ld, out_dtype=dtype)
        elif ratio1:
            raise RuntimeError(f"Ratio-1 residue has no Fold kernel for SM{major}{minor}")

        if ratio1 and major in (10, 12):
            if cfg.ratio1_prefill == "two_gemm":
                return
            from ..flashinfer_integration.ratio1_prefill_tuner import (
                precompile_kloop_sm100, precompile_kloop_sm120,
            )
            if major == 10:
                precompile_kloop_sm100(dtype)
            else:
                precompile_kloop_sm120(n, k, dtype)
