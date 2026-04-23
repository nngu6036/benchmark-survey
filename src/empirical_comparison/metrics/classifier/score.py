import numpy as np

def separation_score(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = (probs >= 0.5).astype(int); return float((preds == labels).mean())
