from empirical_comparison.metrics.descriptor.mmd import mmd_unbiased

def feature_mmd(xs, ys, sigma: float = 1.0) -> float:
    return mmd_unbiased(xs, ys, sigma=sigma)
