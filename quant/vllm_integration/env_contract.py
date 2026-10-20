# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Parse and validate NVFP4 runtime settings."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from . import diagnostics


@dataclass(frozen=True)
class NVFP4Settings:
    mode: str
    projections: tuple[str, ...]
    decode_mext: bool
    swap: bool
    force_fold: bool
    disable_sm10x: bool
    decode_max_m: int | None
    ratio1_max_m: int | None
    ratio1_prefill: str
    sm10x_tuner: bool
    sm120_tuner: bool


def parse_settings(env) -> NVFP4Settings:
    """Validate NVFP4 environment variables without reading global state."""
    def flag(name, default='0'):
        value = env.get(name, default)
        if value not in ('0', '1'):
            raise ValueError(f'{name} must be 0 or 1, got {value!r}')
        return value == '1'

    def threshold(name):
        raw = env.get(name)
        if raw is None:
            return None
        value = int(raw)
        if value < 0:
            raise ValueError(f'{name} must be nonnegative')
        return value

    mode = env.get('RESIDUE_PATCH_MODE', 'v2').strip().lower().replace('-', '_') or 'v2'
    if mode not in ('v2', 'mext_r1'):
        raise ValueError(f'Unknown RESIDUE_PATCH_MODE={mode!r}; expected v2 or mext_r1')
    ratio1_prefill = env.get('RESIDUE_RATIO1_PREFILL_PATH', 'auto')
    if ratio1_prefill not in ('auto', 'two_gemm', 'k_loop'):
        raise ValueError(f'Unknown ratio1 prefill path {ratio1_prefill!r}')

    projection_names = env.get(
        'RESIDUE_MEXT_PROJ_NAMES',
        'q_proj,k_proj,v_proj,qkv_proj,o_proj,gate_proj,up_proj,gate_up_proj,down_proj',
    )
    return NVFP4Settings(
        mode=mode,
        projections=tuple(part.strip() for part in projection_names.split(',') if part.strip()),
        decode_mext=flag('RESIDUE_ENABLE_DECODE_MEXT'),
        swap=False,  # Only the standard layout is exposed.
        force_fold=flag('RESIDUE_FORCE_DECODE_MEXT'),
        disable_sm10x=flag('RESIDUE_DISABLE_SM10X_FOLD'),
        decode_max_m=threshold('RESIDUE_DECODE_MEXT_MAX_M'),
        ratio1_max_m=threshold('RESIDUE_MEXT_FOLD_MAX_M'),
        ratio1_prefill=ratio1_prefill,
        sm10x_tuner=flag('RESIDUE_SM10X_FOLD_TUNER', '1'),
        sm120_tuner=flag('RESIDUE_SM120_FOLD_TUNER', '1'),
    )


@lru_cache(maxsize=1)
def settings() -> NVFP4Settings:
    """Return the process-wide startup snapshot; later env edits do not apply."""
    return parse_settings(os.environ)


def enforce(model_dir=None) -> NVFP4Settings:
    """Reject invalid startup paths and production use of dev-only switches."""
    cfg = settings()
    if model_dir and not Path(model_dir).is_dir():
        raise ValueError(f'RESIDUE_VLLM_MODEL_DIR is not a directory: {model_dir!r}')
    migrated = (
        'RESIDUE_SWAP',
        'RESIDUE_SHARE_AUTOTUNE_CACHE', 'RESIDUE_PREFILL_RESIDUE_OFF',
        'RESIDUE_PREFILL_RESIDUE_OFF_MIN_M', 'RESIDUE_PREFILL_RESIDUE_OFF_MAX_M',
        'RESIDUE_TENSOR_DUMP', 'RESIDUE_TENSOR_DUMP_MAX_M',
        'RESIDUE_STACK_DUMP_SIGNAL', 'RESIDUE_OBSERVE_COMPILES',
    )
    active = [name for name in migrated if os.environ.get(name) not in (None, '', '0')]
    if active or os.environ.get('RESINFER_DEV_EXPERIMENTS') == '1':
        raise ValueError('Experimental launcher switches are not supported by this distribution')
    diagnostics.log_settings(cfg)
    return cfg
