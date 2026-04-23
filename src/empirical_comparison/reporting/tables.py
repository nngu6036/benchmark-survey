import pandas as pd

def combine_metric_frames(frames):
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
