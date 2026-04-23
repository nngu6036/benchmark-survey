from pathlib import Path
import numpy as np

def save_feature_array(path, arr):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); np.save(path, arr)
