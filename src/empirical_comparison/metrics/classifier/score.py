from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def classifier_scores(
    prob_generated: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Compute classifier-based separability scores.

    Label convention:
    - 0 = reference / real graph
    - 1 = generated graph

    Main interpretation:
    - auc near 0.5: generated and reference are hard to distinguish, good.
    - auc near 1.0: generated and reference are easy to distinguish, bad.
    - pgs_js_normalized near 0: low discrepancy, good.
    - pgs_js_normalized near 1: high discrepancy, bad.
    """
    p = np.asarray(prob_generated, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)

    p = np.clip(p, eps, 1.0 - eps)

    auc = roc_auc_score(y, p)
    pred = (p >= 0.5).astype(np.int64)
    acc = accuracy_score(y, pred)
    ce = log_loss(y, p, labels=[0, 1])

    # Mean classifier log-likelihood on a balanced test set:
    # real: log(1 - D(x)), generated: log(D(x))
    mean_ll = float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # GAN-style JS lower-bound relation under equal priors:
    # JS >= mean_ll + log(2).
    # Normalize by log(2), so the value is in [0, 1] approximately.
    js_lb = max(0.0, mean_ll + np.log(2.0))
    js_norm = min(1.0, js_lb / np.log(2.0))

    return {
        "classifier_auc": float(auc),
        "classifier_accuracy": float(acc),
        "classifier_log_loss": float(ce),
        "pgs_js_lower_bound": float(js_lb),
        "pgs_js_normalized": float(js_norm),
    }


def separation_score(prob_generated: np.ndarray, y_true: np.ndarray) -> float:
    """
    Backward-compatible score.

    Returns AUC by default:
    - 0.5 is ideal / indistinguishable.
    - 1.0 means highly separable.
    """
    return classifier_scores(prob_generated, y_true)["classifier_auc"]