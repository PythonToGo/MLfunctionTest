import pytest
from hw1.py import compute_linear_fit, compute_polynomial_fit

def test_linear_fit():
    x, y, a = compute_linear_fit()
    assert a >= 0  

def test_polynomial_fit():
    x, y, a = compute_polynomial_fit()
    assert len(a) == 5  # Since we expect 5 coefficients
    

