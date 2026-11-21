"""Generate artifacts in one view or in every view."""
import numpy as np

from .data import minmax


def _scale(X):
    return float(np.median(X.max(0)))


def add_every(views, frac, seed, groups=5, sigma=0.3):
    rng = np.random.RandomState(seed)
    n = views[0].shape[0]
    out, masks = [], []
    for X in views:
        q = max(1, int(round(frac * X.shape[1])))
        u = rng.randint(groups, size=n)
        mu = rng.rand(groups, q)
        N = mu[u] + sigma * rng.randn(n, q) * mu.std()
        out.append(np.hstack([X, _scale(X) * minmax(N)]))
        masks.append(np.r_[np.zeros(X.shape[1], bool), np.ones(q, bool)])
    return out, np.concatenate(masks)


def add_one(views, frac, seed, groups=5, sigma=0.3):
    rng = np.random.RandomState(seed)
    n = views[0].shape[0]
    D = sum(X.shape[1] for X in views)
    target = int(np.argmax([X.shape[1] for X in views]))
    q = max(1, int(round(frac * D)))
    u = rng.randint(groups, size=n)
    mu = rng.rand(groups, q)
    N = minmax(mu[u] + sigma * rng.randn(n, q) * mu.std())
    out, masks = [], []
    for v, X in enumerate(views):
        if v == target:
            out.append(np.hstack([X, _scale(X) * N]))
            masks.append(np.r_[np.zeros(X.shape[1], bool), np.ones(q, bool)])
        else:
            out.append(X)
            masks.append(np.zeros(X.shape[1], bool))
    return out, np.concatenate(masks)


def corrupt(views, mode, frac, seed, groups=5, sigma=0.3):
    if mode == "one":
        return add_one(views, frac, seed, groups, sigma)
    if mode == "every":
        return add_every(views, frac, seed, groups, sigma)
    raise ValueError(mode)
