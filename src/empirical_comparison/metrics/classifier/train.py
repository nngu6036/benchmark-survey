from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PolyGraphScoreClassifier:
    """
    A practical PolyGraphScore-style binary classifier.

    This is not the full original PGS implementation with TabPFN,
    but it follows the same evaluation idea:
    train a classifier to distinguish reference from generated graphs.
    """

    def __init__(
        self,
        max_iter: int = 2000,
        C: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=max_iter,
                        C=C,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolyGraphScoreClassifier":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        unique = np.unique(y)
        if unique.size != 2:
            raise ValueError(
                f"Classifier training requires both classes, got labels {unique.tolist()}."
            )

        self.model.fit(x, y)
        return self

    def predict_proba_generated(self, x: np.ndarray) -> np.ndarray:
        """
        Return P(y = generated | x).
        """
        x = np.asarray(x, dtype=np.float64)
        return self.model.predict_proba(x)[:, 1]