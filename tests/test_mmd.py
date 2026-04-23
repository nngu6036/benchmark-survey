import numpy as np
from empirical_comparison.metrics.descriptor.mmd import mmd_unbiased

def test_mmd_zero_for_identical_samples():
    xs = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
    ys = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
    assert abs(mmd_unbiased(xs, ys)) < 1e-6
