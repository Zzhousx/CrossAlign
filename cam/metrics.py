"""Clustering accuracy (Hungarian matching), NMI, and the k-means evaluation protocol for a feature ranking."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

import config


def acc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    k = max(y.max(), p.max()) + 1
    w = np.zeros((k, k), dtype=np.int64)
    np.add.at(w, (p, y), 1)
    r, c = linear_sum_assignment(-w)
    return w[r, c].sum() / y.size


def nmi(y, p):
    return normalized_mutual_info_score(y, p, average_method="arithmetic")


def kmeans_scores(Xs, y, reps, seed0):
    """k-means (k-means++ init, one start) with seeds seed0 .. seed0+reps-1; returns ACC and NMI arrays."""
    c = int(y.max() + 1)
    A, N = [], []
    for s in range(reps):
        p = KMeans(n_clusters=c, n_init=1, init="k-means++", random_state=seed0 + s, max_iter=300).fit_predict(Xs)
        A.append(acc(y, p)); N.append(nmi(y, p))
    return np.array(A), np.array(N)


def rownorm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def feature_count(width, ratio):
    """Convert a feature-budget ratio to a nonempty prefix length."""
    return max(1, int(np.floor(ratio * width)))


def evaluate_ranking(XX, y, ranking, reps, seed0, cosine=False, ratios=config.RATIOS):
    """k-means on the top floor(r*D) features for every budget r.
    cosine=True renormalises the rows of the selected block (text data)."""
    D = XX.shape[1]
    out = {}
    for r in ratios:
        k = feature_count(D, r)
        Xs = XX[:, ranking[:k]]
        if cosine:
            Xs = rownorm(Xs)
        A, N = kmeans_scores(Xs, y, reps=reps, seed0=seed0)
        out[r] = dict(acc=float(A.mean()), acc_std=float(A.std()), nmi=float(N.mean()), nmi_std=float(N.std()))
    return out


def mean_over_budgets(res):
    return (float(np.mean([v["acc"] for v in res.values()])), float(np.mean([v["nmi"] for v in res.values()])))
