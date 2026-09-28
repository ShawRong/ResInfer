"""Install the Residue LinearMethod in vLLM 0.26's ModelOpt loader.

Call before model construction. Installation is idempotent; dependency and
loader failures propagate so an explicitly enabled residue model cannot
silently become a plain NVFP4 model.
"""
from __future__ import annotations

import os

from . import diagnostics

_installed_mode: str | None = None


def patch_vllm_modelopt(model_dir: str | None = None, version: str = 'auto') -> bool:
    """Return True after installation, False when already installed."""
    global _installed_mode
    if model_dir is not None:
        os.environ['RESIDUE_VLLM_MODEL_DIR'] = str(model_dir)
    requested = version if version != 'auto' else os.environ.get('RESIDUE_PATCH_MODE', 'v2')
    requested = requested.strip().lower().replace('-', '_') or 'v2'
    if requested not in ('v2', 'mext_r1'):
        raise ValueError(f'Unknown residue patch mode: {requested!r}')
    if _installed_mode is not None:
        if requested != _installed_mode:
            raise ValueError('Residue patch mode cannot change after installation')
        return False
    os.environ['RESIDUE_PATCH_MODE'] = requested
    from .env_contract import enforce, settings

    settings.cache_clear()
    cfg = enforce(os.environ.get('RESIDUE_VLLM_MODEL_DIR'))
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    from vllm.model_executor.layers import linear
    from .linear_method_v2 import ResidueNvFp4LinearMethodV2 as method_cls

    # Merged projections need vLLM's scalar-to-shard weight loader.
    supported = linear.WEIGHT_LOADER_V2_SUPPORTED
    if method_cls.__name__ not in supported:
        supported.append(method_cls.__name__)
    original_init = ModelOptNvFp4Config.__init__
    original_get = ModelOptNvFp4Config.get_quant_method

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # The parent sets this on each instance, overriding its class attribute.
        if getattr(self, 'quant_method', None) == 'NVFP4':
            self.LinearMethodCls = method_cls

    def get_quant_method(self, layer, prefix):
        method = original_get(self, layer, prefix)
        if isinstance(method, method_cls):
            method.layer_prefix = prefix
        return method

    ModelOptNvFp4Config.LinearMethodCls = method_cls
    ModelOptNvFp4Config.__init__ = initialize
    ModelOptNvFp4Config.get_quant_method = get_quant_method
    _installed_mode = cfg.mode
    diagnostics.log_patch(method_cls.__name__, cfg.mode)
    return True


def patch_vllm_modelopt_v2(model_dir: str | None = None) -> bool:
    return patch_vllm_modelopt(model_dir, version='v2')


def patch_vllm_modelopt_mext_r1(model_dir: str | None = None) -> bool:
    return patch_vllm_modelopt(model_dir, version='mext_r1')


def is_patched() -> bool:
    """Compatibility query for external callers; installation itself raises on failure."""
    try:
        from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
        from .linear_method_v2 import ResidueNvFp4LinearMethodV2
    except ImportError:
        return False
    return ModelOptNvFp4Config.LinearMethodCls is ResidueNvFp4LinearMethodV2


def is_patched_mext_r1() -> bool:
    return _installed_mode == 'mext_r1' and is_patched()
