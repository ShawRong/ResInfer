"""Quantization integration package.

The public integration modules are intentionally not imported eagerly:

- ``quant.modelopt_integration`` for calibration/export-time ModelOpt hooks.
- ``quant.vllm_integration`` for serving-time vLLM patches.

Keeping this package import-light avoids importing optional ModelOpt/vLLM
dependencies unless the caller explicitly requests an integration.
"""

__all__ = []
