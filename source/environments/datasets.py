import math

import torch


def make_dataset(name: str, seed: int, n: int = 2000):
    """Returns Xtr, ytr, Xte, yte (CPU tensors) and n_classes."""
    g = torch.Generator().manual_seed(seed)
    if name == "spirals":  # two interleaved spirals, 1.5 turns, in [-1,1]^2
        half = n // 2
        t = torch.sqrt(torch.rand(half, generator=g)) * 3 * math.pi + 0.25 * math.pi
        arm = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1)
        X = torch.cat([arm, -arm]) / (3.25 * math.pi)
        X = X + 0.02 * torch.randn(X.shape, generator=g)
        y = torch.cat([torch.zeros(half), torch.ones(half)]).long()
        C = 2
    elif name == "digits":  # sklearn's bundled 8x8 digits (no download needed)
        from sklearn.datasets import load_digits
        d = load_digits()
        X = torch.tensor(d.data, dtype=torch.float32) / 16.0
        y = torch.tensor(d.target, dtype=torch.long)
        C = 10
    else:
        raise ValueError(name)
    perm = torch.randperm(len(y), generator=g)
    X, y = X[perm].float(), y[perm]
    k = int(0.8 * len(y))
    return X[:k], y[:k], X[k:], y[k:], C
