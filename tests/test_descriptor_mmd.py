import numpy as np

from empirical_comparison.metrics.descriptor.mmd import gaussian_emd_kernel_matrix, mmd_gaussian_emd


def test_gaussian_emd_kernel_handles_zero_sum_histograms():
    x = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y = np.asarray([[0.0, 0.0, 0.0]])

    kernel = gaussian_emd_kernel_matrix(x, y, sigma=1.0)

    assert kernel.shape == (2, 1)
    assert np.all(np.isfinite(kernel))
    assert kernel[0, 0] == 1.0


def test_mmd_gaussian_emd_handles_all_zero_generated_histograms():
    ref = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]])
    gen = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    value = mmd_gaussian_emd(ref, gen, sigma=1.0)

    assert np.isfinite(value)
    assert value >= 0.0
