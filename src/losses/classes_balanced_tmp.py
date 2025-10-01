import numpy as np
import torch

def class_balanced_weight(num_per_class, beta=0.9999, device="cpu"):
    eff_num = 1.0 - np.power(beta, num_per_class)
    weights = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    weights = weights / weights.sum() * len(num_per_class)
    return torch.tensor(weights, dtype=torch.float32, device=device)
