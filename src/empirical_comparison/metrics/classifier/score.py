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
    - classifier_auc close to 0.5 is good.
    - classifier_auc close to 1.0 or 0.0 means separable.
    - classifier_separation = 2 * abs(AUC - 0.5), so:
        0.0 = indistinguishable, good
        1.0 = perfectly separable, bad
    """
    p = np.asarray(prob_generated, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)

    p = np.clip(p, eps, 1.0 - eps)

    auc = float(roc_auc_score(y, p))
    pred = (p >= 0.5).astype(np.int64)
    acc = float(accuracy_score(y, pred))
    ce = float(log_loss(y, p, labels=[0, 1]))

    # Direction-free separability.
    # Handles both AUC > 0.5 and AUC < 0.5.
    classifier_separation = float(2.0 * abs(auc - 0.5))

    # Optional calibrated JS-style score.
    # This can be useful, but should not be the main score if it collapses to zero.
    mean_ll = float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    js_lb = max(0.0, mean_ll + np.log(2.0))
    js_norm = min(1.0, js_lb / np.log(2.0))

    return {
        "classifier_auc": auc,
        "classifier_separation": classifier_separation,
        "classifier_accuracy": acc,
        "classifier_log_loss": ce,
        "pgs_js_lower_bound": float(js_lb),
        "pgs_js_normalized": float(js_norm),
    }


def separation_score(prob_generated: np.ndarray, y_true: np.ndarray) -> float:
    """
    Backward-compatible scalar score.

    Returns direction-free classifier separation:
    - 0.0 is ideal / indistinguishable.
    - 1.0 is perfectly separable.
    """
    return classifier_scores(prob_generated, y_true)["classifier_separation"]