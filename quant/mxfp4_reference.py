"""Historical CPU MXFP4 mathematical oracle; not the runtime quantizer.

See docs/mxfp4-development.md for the format and conversion contract.
NumPy float64 intermediates preserve subnormal scales without FTZ; decoded
values and GEMM results are float32. This is not a performance backend.
The historical OCP branch is retained for old experiment reproduction only.
Current runtime tests use FlashInfer itself as the authoritative producer;
this oracle's exact division/subnormal behavior does not model that kernel.
"""

from __future__ import annotations

import numpy as np

BLOCK_SIZE = 32
_MAGNITUDES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
_MIDPOINTS = (_MAGNITUDES[:-1] + _MAGNITUDES[1:]) / 2


def quantize(x: np.ndarray, *, recipe: str = "ocp",
             min_scale_exponent: int = -127) -> tuple[np.ndarray, np.ndarray]:
    """Quantize finite float16/32 input [M,K], K a positive multiple of 32.

    Returns uint8 payload [M,K/2] and E8M0 scales [M,K/32]. The first
    element occupies the low nibble. Scale exponent is
    clip(floor(log2(amax)) - 2, -127, 127), with code 0 for zero blocks.
    Elements use round-to-nearest, ties-to-even and saturate to +/-6.
    Signed zeros are retained. No tensor-level scale or implicit padding.
    recipe="ceil" uses ceil(log2(amax/6)); min_scale_exponent=-126
    selects the retired local GPU kernel's minimum-normal-scale policy.
    """
    if recipe not in ("ocp", "ceil") or min_scale_exponent not in (-127, -126):
        raise ValueError("recipe must be ocp/ceil; minimum exponent must be -127/-126")
    x = np.asarray(x)
    if x.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError("MXFP4 reference input must be float16 or float32")
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % BLOCK_SIZE:
        raise ValueError("expected [M,K] with positive K divisible by 32")
    if not np.isfinite(x).all():
        raise ValueError("MXFP4 reference quantization requires finite inputs")

    blocks = x.astype(np.float64).reshape(x.shape[0], x.shape[1] // BLOCK_SIZE, BLOCK_SIZE)
    amax = np.abs(blocks).max(axis=-1)
    # frexp avoids log2 rounding across exact power-of-two boundaries.
    exponent = np.frexp(amax)[1] - 3
    if recipe == "ceil":
        fraction, power = np.frexp(amax / 6.0)
        exponent = power - (fraction == 0.5)
    exponent = np.where(amax == 0, min_scale_exponent,
                        np.clip(exponent, min_scale_exponent, 127))
    scales = (exponent + 127).astype(np.uint8)
    magnitude = np.abs(blocks) / np.ldexp(1.0, exponent)[..., None]
    codes = np.searchsorted(_MIDPOINTS, magnitude, side="left").astype(np.uint8)
    # At odd lower codes, a midpoint rounds up to the even upper code.
    for lower in (1, 3, 5):
        codes += (magnitude == _MIDPOINTS[lower]).astype(np.uint8)
    codes |= np.signbit(blocks).astype(np.uint8) << 3
    codes = codes.reshape(x.shape)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), scales


def dequantize(payload: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode linear MXFP4 to float32; scale 255 makes the whole block NaN.

    Out-of-float32-range decoded values become signed infinity. E8M0 code
    zero decodes as 2**-127. GPU swizzled scale buffers are not accepted.
    """
    payload, scales = np.asarray(payload), np.asarray(scales)
    if payload.dtype != np.uint8 or scales.dtype != np.uint8:
        raise TypeError("payload and scales must both be uint8")
    if payload.ndim != 2 or payload.shape[1] == 0 or payload.shape[1] % 16:
        raise ValueError("expected payload [M,K/2] with positive K divisible by 32")
    if scales.shape != (payload.shape[0], payload.shape[1] // 16):
        raise ValueError("expected linear scales [M,K/32]")
    codes = np.stack((payload & 15, payload >> 4), axis=-1).reshape(
        payload.shape[0], payload.shape[1] * 2
    )
    values = np.copysign(_MAGNITUDES[codes & 7], np.where(codes & 8, -1.0, 1.0))
    scale_values = np.ldexp(1.0, scales.astype(np.int16) - 127)
    scale_values = np.where(scales == 255, np.nan, scale_values)
    with np.errstate(over="ignore"):
        return (values * np.repeat(scale_values, BLOCK_SIZE, axis=1)).astype(np.float32)


def gemm(
    a: np.ndarray, a_scales: np.ndarray, b: np.ndarray, b_scales: np.ndarray
) -> np.ndarray:
    """Reference D[M,N] = decode(A[M,K]) @ decode(B[N,K]).T in float32."""
    a_dense, b_dense = dequantize(a, a_scales), dequantize(b, b_scales)
    if a_dense.shape[1] != b_dense.shape[1]:
        raise ValueError("A and B must have the same K")
    return a_dense @ b_dense.T
