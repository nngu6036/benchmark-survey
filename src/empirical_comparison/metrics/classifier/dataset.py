import numpy as np

def make_binary_dataset(real_features, gen_features):
    x = np.vstack([real_features, gen_features]); y = np.concatenate([np.zeros(len(real_features)), np.ones(len(gen_features))]); return x, y
