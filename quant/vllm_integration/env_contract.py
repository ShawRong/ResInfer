"""Parse NVFP4 runtime settings once, before constructing the model.

Kernel-specific and attention settings remain owned by those integrations.
Development experiments are accepted only through the repository launcher.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


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
    new = env.get('RESIDUE_RATIO1_PREFILL_PATH')
    old = env.get('RESIDUE_MEXT_PREFILL_PATH')
    if new is not None and old is not None and new != old:
        raise ValueError('RESIDUE_RATIO1_PREFILL_PATH conflicts with RESIDUE_MEXT_PREFILL_PATH')
    prefill = new if new is not None else old if old is not None else 'auto'
    if prefill not in ('auto', 'two_gemm', 'k_loop'):
        raise ValueError(f'Unknown ratio1 prefill path {prefill!r}')
    projections = env.get('RESIDUE_MEXT_PROJ_NAMES', env.get(
        'RESIDUE_MEXT_ATTN_PROJ_NAMES',
        'q_proj,k_proj,v_proj,qkv_proj,o_proj,gate_proj,up_proj,gate_up_proj,down_proj'))
    return NVFP4Settings(
        mode, tuple(part.strip() for part in projections.split(',') if part.strip()),
        flag('RESIDUE_ENABLE_DECODE_MEXT'), flag('RESIDUE_SWAP'),
        flag('RESIDUE_FORCE_DECODE_MEXT'), flag('RESIDUE_DISABLE_SM10X_FOLD'),
        threshold('RESIDUE_DECODE_MEXT_MAX_M'), threshold('RESIDUE_MEXT_FOLD_MAX_M'),
        prefill, flag('RESIDUE_SM10X_FOLD_TUNER', '1'), flag('RESIDUE_SM120_FOLD_TUNER', '1'),
    )


@lru_cache(maxsize=1)
def settings() -> NVFP4Settings:
    return parse_settings(os.environ)


def enforce(model_dir=None) -> NVFP4Settings:
    cfg = settings()
    if model_dir and not Path(model_dir).is_dir():
        raise ValueError(f'RESIDUE_VLLM_MODEL_DIR is not a directory: {model_dir!r}')
    migrated = (
        'RESIDUE_SHARE_AUTOTUNE_CACHE', 'RESIDUE_PREFILL_RESIDUE_OFF',
        'RESIDUE_PREFILL_RESIDUE_OFF_MIN_M', 'RESIDUE_PREFILL_RESIDUE_OFF_MAX_M',
        'RESIDUE_TENSOR_DUMP', 'RESIDUE_TENSOR_DUMP_MAX_M',
        'RESIDUE_STACK_DUMP_SIGNAL', 'RESIDUE_OBSERVE_COMPILES',
    )
    active = [name for name in migrated if os.environ.get(name) not in (None, '', '0')]
    if active:
        raise ValueError(f'Unsupported development switches: {", ".join(active)}')
    print(f'[residue:env] mode={cfg.mode} decode_mext={cfg.decode_mext} '
          f'max_m={cfg.decode_max_m} ratio1_prefill={cfg.ratio1_prefill}', flush=True)
    return cfg
