from __future__ import annotations

import numpy as np


def make_binary_dataset(
    ref_features: np.ndarray,
    gen_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a binary classifier dataset.

    Label convention:
    - 0 = reference / real graph
    - 1 = generated graph
    """
    ref_features = np.asarray(ref_features, dtype=np.float64)
    gen_features = np.asarray(gen_features, dtype=np.float64)

    if ref_features.ndim != 2 or gen_features.ndim != 2:
        raise ValueError("Both feature arrays must be 2D.")

    if ref_features.shape[1] != gen_features.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: {ref_features.shape[1]} vs {gen_features.shape[1]}"
        )

    x = np.vstack([ref_features, gen_features])
    y = np.concatenate(
        [
            np.zeros(ref_features.shape[0], dtype=np.int64),
            np.ones(gen_features.shape[0], dtype=np.int64),
        ]
    )

    return x, y