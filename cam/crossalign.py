"""Cross-view kernel-target alignment and feature ranking."""
import numpy as np
from sklearn.cluster import KMeans

from .graphs import knn_affinity, norm_adj, top_eigvecs


def standardize(X):
    mu, sd = X.mean(0), X.std(0)
    Z = np.zeros_like(X)
    ok = sd > 1e-12
    Z[:, ok] = (X[:, ok] - mu[ok]) / sd[ok]
    return Z


def centred_basis(F, keep):
    Fc = F - F.mean(0, keepdims=True)
    U, _, _ = np.linalg.svd(Fc, full_matrices=False)
    return U[:, :keep]


def spectral_partition(A, c, seed=0):
    F = top_eigvecs(A, c, seed=seed)[0]
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    lab = KMeans(c, n_init=10, random_state=seed).fit_predict(Fn)
    return lab, F


def indicator_basis(lab, c):
    n = lab.size
    Y = np.zeros((n, c)); Y[np.arange(n), lab] = 1
    return Y / np.sqrt(np.maximum(Y.sum(0, keepdims=True), 1))


def _mix(A, w, idx):
    Am, ws = None, 0.0
    for u in idx:
        Am = A[u] * w[u] if Am is None else Am + A[u] * w[u]
        ws += w[u]
    return Am / ws


def greedy_alignment(Xc, rel, K):
    """Greedy maximisation of rho(S) = sum_{f in S} rel_f / ||X_S^T X_S||_F (the constant sqrt(c-1) is dropped).

    ||X_S^T X_S||_F^2 = sum_{f,g in S} (x_f^T x_g)^2 is updated exactly: adding feature j adds
    2 * q_j + ||x_j||^4 with q_j = sum_{g in S} (x_g^T x_j)^2, and q is refreshed with one product X^T x_j.
    Zero-norm columns are placed after all non-constant columns.
    """
    D = Xc.shape[1]
    diag = (Xc * Xc).sum(0) ** 2
    q = np.zeros(D)
    num, den2 = 0.0, 0.0
    active = diag > 0
    available = active.copy()
    order, trace = [], []
    for _ in range(min(K, int(active.sum()))):
        score = (num + rel) / np.sqrt(den2 + 2 * q + diag + 1e-300)
        score[~available] = -np.inf
        j = int(np.argmax(score))
        order.append(j)
        available[j] = False
        num += rel[j]; den2 += 2 * q[j] + diag[j]
        trace.append(num / np.sqrt(den2))
        q += (Xc.T @ Xc[:, j]) ** 2
    rest = np.flatnonzero(available)
    order.extend(rest[np.argsort(-rel[rest])])
    order.extend(np.flatnonzero(~active))
    return np.asarray(order, dtype=int), np.asarray(trace)


def crossalign(views, c, k=10, target="cross", discrete=True, select="align", space="raw", K=None, seed=0,
               return_info=False):
    """Rank all features of `views` (list of n x d_v arrays) for c clusters.

    target  : "cross" (leave-one-view-out, default) | "self" (the view itself)
    discrete: True -> target = spectral-clustering partition; False -> continuous spectral embedding
    select  : "align" (greedy subset alignment, default) | "ratio" | "relevance" (per-feature rankings)
    space   : "raw" (clustering geometry, default) | "std" (standardised features)
    K       : number of greedy steps (default ceil(D/2) + 1, enough for every budget up to 50%)
    """
    V = len(views)
    A = [norm_adj(knn_affinity(X, k)) for X in views]
    w = np.ones(V) / V
    if target == "cross":
        if V < 2:
            raise ValueError("Cross-view targeting requires at least two views.")
        mixes = [_mix(A, w, [u for u in range(V) if u != v]) for v in range(V)]
    elif target == "self":
        mixes = [A[v] for v in range(V)]
    else:
        raise ValueError(f"Unknown target: {target}")
    T, cache = [], {}
    for v in range(V):
        key = id(mixes[v])
        if key not in cache:
            lab, F = spectral_partition(mixes[v], c, seed)
            cache[key] = centred_basis(indicator_basis(lab, c) if discrete else F, c - 1)
        T.append(cache[key])
    Xs = [(X - X.mean(0)) if space == "raw" else standardize(X) for X in views]
    for X, Xc in zip(views, Xs):
        Xc[:, np.ptp(X, axis=0) == 0] = 0
    rel = np.concatenate([((T[v].T @ Xs[v]) ** 2).sum(0) for v in range(V)])
    Xall = np.hstack(Xs)
    D = Xall.shape[1]
    K = int(np.ceil(0.5 * D)) + 1 if K is None else K
    if select in {"ratio", "relevance"}:
        energy = (Xall ** 2).sum(0)
        active = energy > 0
        merit = rel / np.maximum(energy, 1e-300) if select == "ratio" else rel
        ranking = np.argsort(-merit)
        ranking = np.concatenate((ranking[active[ranking]], np.flatnonzero(~active)))
        trace = None
    else:
        ranking, trace = greedy_alignment(Xall, rel, K)
    if return_info:
        return ranking, dict(rel=rel, trace=trace, targets=T)
    return ranking
