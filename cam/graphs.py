"""Graph utilities: self-tuning kNN affinities, normalised adjacency, leading eigenvectors."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from sklearn.neighbors import NearestNeighbors


def knn_affinity(X, k=10):
    """Symmetric kNN affinity with the self-tuning heat kernel of Zelnik-Manor & Perona (2005):
    w_ij = exp(-d_ij^2 / (sigma_i sigma_j)), sigma_i = distance to the k-th neighbour (no bandwidth to set)."""
    n = X.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sig = np.maximum(dist[:, -1], 1e-12)
    w = np.exp(-(dist ** 2) / (sig[:, None] * sig[idx]))
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    return W.maximum(W.T).tocsr()


def norm_adj(W):
    d = np.asarray(W.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    Dm = sp.diags(dinv)
    return (Dm @ W @ Dm).tocsr()


def top_eigvecs(A, c, seed=0):
    """Top-c eigenvectors of a symmetric (sparse or dense) matrix."""
    n = A.shape[0]
    if n <= 600 or not sp.issparse(A):
        Ad = A.toarray() if sp.issparse(A) else A
        vals, vecs = np.linalg.eigh((Ad + Ad.T) / 2)
        o = np.argsort(vals)[::-1][:c]
        return vecs[:, o], vals[o]
    v0 = np.random.RandomState(seed).rand(n)
    try:
        vals, vecs = eigsh(A, k=c, which="LA", v0=v0, tol=1e-6, maxiter=5000)
    except ArpackNoConvergence:          # degenerate spectrum: exact dense solver
        vals, vecs = np.linalg.eigh(A.toarray())
        o = np.argsort(vals)[::-1][:c]
        return vecs[:, o], vals[o]
    o = np.argsort(vals)[::-1]
    return vecs[:, o], vals[o]
