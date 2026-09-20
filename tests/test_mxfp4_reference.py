import numpy as np
import pytest

from quant.mxfp4_reference import dequantize, gemm, quantize


def test_all_codes_and_nibble_order():
    values = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    x = np.array([values * 2], dtype=np.float32)
    payload, scales = quantize(x)
    np.testing.assert_array_equal(payload, [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2])
    np.testing.assert_array_equal(scales, [[127]])
    decoded = dequantize(payload, scales)
    np.testing.assert_array_equal(decoded, x)
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(x))


def test_midpoints_neighbors_and_saturation():
    midpoints = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)
    for points, expected in [
        (np.nextafter(midpoints, np.float32(0)), [0, 1, 2, 3, 4, 5, 6]),
        (midpoints, [0, 2, 2, 4, 4, 6, 6]),
        (np.nextafter(midpoints, np.float32(8)), [1, 2, 3, 4, 5, 6, 7]),
    ]:
        x = np.zeros((1, 32), dtype=np.float32)
        x[0, :7], x[0, 8:15], x[0, -2:] = points, -points, [7, -7]
        payload, scales = quantize(x)
        codes = np.stack((payload & 15, payload >> 4), axis=-1).reshape(1, 32)
        np.testing.assert_array_equal(scales, [[127]])
        np.testing.assert_array_equal(codes[0, :7], expected)
        np.testing.assert_array_equal(codes[0, 8:15], np.array(expected) + 8)
        np.testing.assert_array_equal(codes[0, -2:], [7, 15])


def test_scale_boundaries_zero_subnormal_and_max_float():
    maxima = np.array([0, 2**-149, 2**-127, 1, 6, 7, np.nextafter(np.float32(8), np.float32(0)), 8, np.finfo(np.float32).max], dtype=np.float32)
    x = np.repeat(maxima[:, None], 32, axis=1)
    payload, scales = quantize(x)
    np.testing.assert_array_equal(scales[:, 0], [0, 0, 0, 125, 127, 127, 127, 128, 252])
    decoded = dequantize(payload, scales)
    np.testing.assert_array_equal(decoded[:8, 0], [0, 0, 2**-127, 1, 6, 6, 6, 8])
    assert np.isfinite(decoded).all()


def test_scale_zero_nan_and_overflow_decoding():
    payload = np.full((3, 16), 0x22, dtype=np.uint8)  # +/-1 code = 2
    scales = np.array([[0], [255], [254]], dtype=np.uint8)
    decoded = dequantize(payload, scales)
    assert (decoded[0] == np.float32(2**-127)).all()
    assert np.isnan(decoded[1]).all()
    assert (decoded[2] == np.float32(2**127)).all()
    assert np.isposinf(dequantize(np.full((1, 16), 0x77, np.uint8), scales[2:])).all()


def test_two_block_gemm_with_independent_expected_values():
    # Non-square M/N and distinct scales along K catch transpose/block errors.
    a = np.repeat(np.array([[1, 2], [-4, 0.5]], np.float32), 32, axis=1)
    b = np.repeat(np.array([[2, -1], [0.5, 4], [-2, -2]], np.float32), 32, axis=1)
    aq, sa = quantize(a)
    bq, sb = quantize(b)
    actual = gemm(aq, sa, bq, sb)
    np.testing.assert_array_equal(actual, [[0, 272, -192], [-272, 0, 224]])
    assert actual.dtype == np.float32


def test_noncontiguous_float16_and_empty_rows():
    x = np.tile(np.array([1, -2], np.float16), (3, 64))[:, ::2]
    np.testing.assert_array_equal(dequantize(*quantize(x)), x)
    assert dequantize(*quantize(np.empty((0, 32), np.float32))).shape == (0, 32)


@pytest.mark.parametrize("bad", [np.zeros((32,), np.float32), np.zeros((1, 31), np.float32), np.zeros((1, 0), np.float32), np.full((1, 32), np.inf, np.float32), np.full((1, 32), np.nan, np.float32)])
def test_invalid_input(bad):
    with pytest.raises(ValueError):
        quantize(bad)


def test_invalid_types_and_layouts():
    with pytest.raises(TypeError):
        quantize(np.ones((1, 32), np.float64))
    with pytest.raises(TypeError):
        dequantize(np.zeros((1, 16), np.int8), np.zeros((1, 1), np.uint8))
    with pytest.raises(ValueError):
        dequantize(np.zeros((1, 16), np.uint8), np.zeros(512, np.uint8))
    with pytest.raises(ValueError):
        gemm(*quantize(np.ones((1, 32), np.float32)), *quantize(np.ones((1, 64), np.float32)))


def test_explicit_scale_recipes_and_gpu_minimum():
    x = np.repeat(np.array([[0, 7, 8, 12]], np.float32), 32, axis=1)
    _, floor_sf = quantize(x)
    _, ceil_sf = quantize(x, recipe="ceil")
    np.testing.assert_array_equal(floor_sf, [[0,127,128,128]])
    np.testing.assert_array_equal(ceil_sf, [[0,128,128,128]])
    _, gpu_sf = quantize(x, min_scale_exponent=-126)
    np.testing.assert_array_equal(gpu_sf, [[1,127,128,128]])
    with pytest.raises(ValueError):
        quantize(x, recipe="nvfp4")
