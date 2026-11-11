import math

import pytest

from lob.export_fixed import quantize_multiplier


# The integer multiplier and shift must reproduce the real scale factor closely.
@pytest.mark.parametrize("real", [1e-6, 0.000123, 0.25, 0.5, 0.999999, 1.0, 3.7, 1234.5])
def test_quantize_multiplier_approximates_the_scale(real):
    multiplier, shift = quantize_multiplier(real)

    # The multiplier is a normalised 31-bit mantissa and fits a signed int32.
    assert 1 << 30 <= multiplier < 1 << 31
    assert math.isclose(multiplier / 2**shift, real, rel_tol=1e-9)


# Powers of two have an exact representation.
def test_quantize_multiplier_is_exact_for_powers_of_two():
    multiplier, shift = quantize_multiplier(0.5)
    assert multiplier / 2**shift == 0.5


# Scales must be positive; anything else is a bug upstream.
@pytest.mark.parametrize("real", [0.0, -1.0])
def test_quantize_multiplier_rejects_non_positive_scales(real):
    with pytest.raises(ValueError):
        quantize_multiplier(real)
