"""Feature-selection baselines."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator, ArpackNoConvergence
from sklearn.cluster import KMeans
from sklearn.linear_model import Lars
from sklearn.neighbors import NearestNeighbors

from .graphs import top_eigvecs


# ============================================================================ shared helpers
def heat_knn(X, k=5):
    """kNN graph with heat kernel, t = mean squared kNN distance (Cai/He style)."""
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    t = np.mean(dist ** 2) + 1e-12
    w = np.exp(-(dist ** 2) / t)
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    return W.maximum(W.T).tocsr()


def proj_simplex_rows(V):
    """Euclidean projection of every row of V onto the probability simplex."""
    n, m = V.shape
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, m + 1)
    cond = U - css / ind > 0
    rho = m - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(n), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0)


def sqdist(A, B=None):
    B = A if B is None else B
    aa = (A * A).sum(1)[:, None]
    bb = (B * B).sum(1)[None, :]
    return np.maximum(aa + bb - 2 * A @ B.T, 0)


def spectral_init(XX, c, k=5, seed=0):
    W = heat_knn(XX, k)
    d = np.asarray(W.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(d + 1e-12))
    A = (Dm @ W @ Dm).tocsr()
    n = XX.shape[0]
    if n <= 600:
        vals, vecs = np.linalg.eigh(A.toarray())
        E = vecs[:, np.argsort(vals)[::-1][:c]]
    else:
        try:
            vals, E = eigsh(A, k=c, which="LA", v0=np.random.RandomState(seed).rand(n))
        except ArpackNoConvergence:      # degenerate spectrum (e.g. a disconnected graph): exact dense solver
            vals, vecs = np.linalg.eigh(A.toarray())
            E = vecs[:, np.argsort(vals)[::-1][:c]]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return KMeans(c, n_init=10, random_state=seed).fit_predict(E)


def indicator(lab, c):
    Y = np.zeros((lab.size, c))
    Y[np.arange(lab.size), lab] = 1
    return Y


def norm_lap_dense(XX, k=5):
    W = heat_knn(XX, k).toarray()
    dm = 1 / np.sqrt(W.sum(1) + 1e-12)
    return np.eye(W.shape[0]) - dm[:, None] * W * dm[None, :]


def solve_reg_ls(X, Dvec, beta, F):
    """W = (X X^T + beta diag(Dvec))^{-1} X F for X: d x n; also returns X^T (...)^{-1} X."""
    d, n = X.shape
    if d <= n:
        A = X @ X.T + beta * np.diag(Dvec)
        Ainv_X = np.linalg.solve(A, X)
        return Ainv_X @ F, X.T @ Ainv_X
    Dinv = 1.0 / Dvec
    K = X.T @ (Dinv[:, None] * X)
    B = np.linalg.solve(beta * np.eye(n) + K, np.eye(n))
    return (Dinv[:, None] * X) @ (B @ F), K @ B


# ============================================================================ LS, MCFS
def laplacian_score(XX, k=5):
    W = heat_knn(XX, k)
    d = np.asarray(W.sum(1)).ravel()
    mu = (d @ XX) / d.sum()
    F = XX - mu[None, :]
    num = np.einsum("ij,ij->j", F, F * d[:, None]) - np.einsum("ij,ij->j", F, W @ F)
    den = np.einsum("ij,ij->j", F, F * d[:, None])
    score = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.inf)
    return np.argsort(score)


def mcfs(XX, c, k=5, K=None):
    n, D = XX.shape
    W = heat_knn(XX, k)
    d = np.asarray(W.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(d + 1e-12))
    E, _ = top_eigvecs((Dm @ W @ Dm).tocsr(), c + 1)
    Y = Dm @ E[:, 1:c + 1]
    Y = Y / np.linalg.norm(Y, axis=0, keepdims=True)
    K = int(np.ceil(0.5 * D)) if K is None else K
    nz = int(min(K, n - 1))
    Xc = XX - XX.mean(0)
    S = np.zeros(D)
    for j in range(Y.shape[1]):
        m = Lars(n_nonzero_coefs=nz, fit_intercept=False, copy_X=True).fit(Xc, Y[:, j] - Y[:, j].mean())
        S = np.maximum(S, np.abs(m.coef_))
    corr = np.abs(Xc.T @ (Y - Y.mean(0))).max(1)          # tie-break features with zero LARS weight
    return np.argsort(-(S * 1e6 + corr / (corr.max() + 1e-12)))


# ============================================================================ NDFS
def ndfs(XX, c, alpha=1.0, beta=1.0, k=5, gamma=1e8, iters=30, seed=0):
    X = XX.T
    d, n = X.shape
    L = norm_lap_dense(XX, k)
    F = indicator(spectral_init(XX, c, k, seed), c)
    F = F / np.sqrt(F.sum(0, keepdims=True) + 1e-12) + 0.2
    Dvec = np.ones(d)
    for _ in range(iters):
        W, hat = solve_reg_ls(X, Dvec, beta, F)
        M = L + alpha * (np.eye(n) - hat)
        M = (M + M.T) / 2
        Mp, Mn = np.maximum(M, 0), np.maximum(-M, 0)
        F = F * (gamma * F + Mn @ F) / (Mp @ F + gamma * F @ (F.T @ F) + 1e-12)
        Dvec = 1.0 / (2 * np.sqrt((W ** 2).sum(1)) + 1e-8)
    W, _ = solve_reg_ls(X, Dvec, beta, F)
    return np.argsort(-(W ** 2).sum(1))


# ============================================================================ UDFS
def _udfs_local(XX, k, lam):
    n, d = XX.shape
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(XX).kneighbors(XX)
    H = np.eye(k + 1) - 1.0 / (k + 1)
    Mi = np.zeros((n, n))
    for i in range(n):
        Xi = XX[idx[i]]
        Bi = np.linalg.inv(H @ (Xi @ Xi.T) @ H + lam * np.eye(k + 1))
        Mi[np.ix_(idx[i], idx[i])] += H @ Bi @ H
    return Mi


def udfs(XX, c, gamma=0.1, lam=1e-3, k=5, iters=20, seed=0):
    n, d = XX.shape
    Mi = _udfs_local(XX, k, lam)
    X = XX.T
    Dvec = np.ones(d)
    if d <= 2500:
        M = X @ Mi @ X.T
        for _ in range(iters):
            A = M + gamma * np.diag(Dvec)
            vals, vecs = np.linalg.eigh((A + A.T) / 2)
            W = vecs[:, :c]
            Dvec = 1.0 / (2 * np.sqrt((W ** 2).sum(1)) + 1e-8)
    else:
        rng = np.random.RandomState(seed)
        W = None
        for _ in range(iters):
            ub = np.abs(Mi).sum(1).max() * (X ** 2).sum() + gamma * Dvec.max()

            def mm(Vm, Dvec=Dvec, ub=ub):     # smallest eigvecs of M + gamma D == largest of ub I - (...)
                Vm = Vm.reshape(d, -1)
                return ub * Vm - X @ (Mi @ (X.T @ Vm)) - gamma * Dvec[:, None] * Vm
            op = LinearOperator((d, d), matvec=lambda v: mm(v).ravel(), matmat=mm, dtype=float)
            v0 = W[:, 0] if W is not None else rng.rand(d)
            vals, W = eigsh(op, k=c, which="LA", v0=v0, tol=1e-4, maxiter=2000)
            Dvec = 1.0 / (2 * np.sqrt((W ** 2).sum(1)) + 1e-8)
    return np.argsort(-(W ** 2).sum(1))


# ============================================================================ JMVFG (port)
def _rownorm_heat_knn_V(Xv, V, k=5):
    """As in the JMVFG code: constructW(k=5, HeatKernel, t = median(EuDist)^2), row-normalised, times V."""
    n = Xv.shape[0]
    D2 = sqdist(Xv)
    t = np.median(np.sqrt(D2)) ** 2 + 1e-12
    idx = np.argsort(D2, axis=1)[:, :k + 1]
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k + 1)
    W[rows, idx.ravel()] = np.exp(-D2[rows, idx.ravel()] / (2 * t ** 2))
    np.fill_diagonal(W, 0)
    W = np.maximum(W, W.T)
    return V * W / (W.sum(1, keepdims=True) + 1e-12)


def _solve_delta(S, A):
    V = len(A)
    p = np.array([np.sum(A[v] * S) for v in range(V)])
    q = np.array([np.sum(A[v] * A[v]) for v in range(V)])
    g = p / q + (1 - np.sum(p / q)) / (q * np.sum(1 / q))
    if g.min() >= 0:
        return np.maximum(g, 0)
    sumq = np.sum(1 / q)
    der_t = (1 / sumq) / q
    miu, f, it = 0.0, 1.0, 0
    while abs(f) > 1e-10 and it < 1000:
        m0 = (miu / sumq) / q - g
        pos = m0 > 0
        der = der_t[pos].sum() - 1 if pos.any() else -1
        f = m0[pos].sum() - miu
        miu = miu - f / der
        it += 1
    return np.maximum(g - (miu / sumq) / q, 0)


def jmvfg(views, c, eta=1.0, gamma=1.0, beta=1e-3, k=5, maxiter=20, seed=0):
    V = len(views)
    n = views[0].shape[0]
    alpha = 1e4
    XX = np.hstack(views)
    A = [_rownorm_heat_knn_V(Xv, V, k) for Xv in views]
    delta = np.ones(V) / V
    S = sum(delta[v] * A[v] for v in range(V))
    SS = (S + S.T) / 2
    L = np.diag(SS.sum(1)) - SS
    lab = spectral_init(XX, c, 5, seed)
    H = np.random.RandomState(5489).rand(n, c)
    H[np.arange(n), lab] = 1
    H = H + 0.2
    Xs = [Xv.T for Xv in views]
    W, B, Dv = [], [], []
    for v in range(V):
        dv = Xs[v].shape[0]
        W.append(np.eye(dv, c)); B.append(W[v].T @ Xs[v] @ H); Dv.append(np.ones(dv))
    prevW = None
    for _ in range(maxiter):
        delta = _solve_delta(S, A)
        Q = np.eye(n) + gamma * L
        Ys = []
        for v in range(V):
            X = Xs[v]
            rhs = X @ H @ B[v].T
            if X.shape[0] <= n:
                W[v] = np.linalg.solve(X @ Q @ X.T + eta * np.diag(Dv[v]), rhs)
            else:                       # (X Q X^T + eta D)^{-1} X = (eta D)^{-1} X (I + Q K)^{-1}
                Dinv = 1.0 / (eta * Dv[v])
                K = X.T @ (Dinv[:, None] * X)
                W[v] = (Dinv[:, None] * X) @ np.linalg.solve(np.eye(n) + Q @ K, H @ B[v].T)
            Dv[v] = 0.5 / np.sqrt((W[v] ** 2).sum(1) + np.finfo(float).eps)
            Ys.append(W[v].T @ X)
        for v in range(V):
            Ub, _, Vt = np.linalg.svd(W[v].T @ Xs[v] @ H, full_matrices=False)
            B[v] = Ub @ Vt
        M = sum(Xs[v].T @ W[v] @ B[v] for v in range(V)) + alpha * np.maximum(H, 0)
        Ub, _, Vt = np.linalg.svd(M, full_matrices=False)
        H = Ub @ Vt
        G = sum(sqdist(Ys[v].T) for v in range(V))
        dA = sum(2 * delta[v] * A[v] for v in range(V))
        S = proj_simplex_rows((dA - gamma / (2 * beta) * G) / (2 * V))
        SS = (S + S.T) / 2
        L = np.diag(SS.sum(1)) - SS
        WW = np.vstack(W)
        if prevW is not None and np.linalg.norm(WW - prevW) / (np.linalg.norm(prevW) + 1e-12) < 1e-4:
            break
        prevW = WW.copy()
    return np.argsort(-(np.vstack(W) ** 2).sum(1))


# ============================================================================ CvLP-DCL
def _heat_knn_dense(Xv, k=5):
    n = Xv.shape[0]
    D2 = sqdist(Xv)
    idx = np.argsort(D2, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(n), k)
    s2 = np.mean(D2[rows, idx.ravel()]) + 1e-12
    W = np.zeros((n, n))
    W[rows, idx.ravel()] = np.exp(-D2[rows, idx.ravel()] / s2)
    W = np.maximum(W, W.T)
    return W / (W.sum(1, keepdims=True) + 1e-12)


def _simplex_qp(Q, f, iters=500):
    """min 0.5 g^T Q g + f^T g over the simplex (projected gradient)."""
    g = np.ones(len(f)) / len(f)
    Lc = np.linalg.eigvalsh((Q + Q.T) / 2).max() + 1e-12
    for _ in range(iters):
        g = proj_simplex_rows((g - (Q @ g + f) / Lc)[None, :])[0]
    return np.maximum(g, 1e-6) / np.maximum(g, 1e-6).sum()


def cvlp_dcl(views, c, lam=1.0, alpha=1.0, beta=1.0, k=5, rho=1e6, iters=30, seed=0):
    V = len(views)
    n = views[0].shape[0]
    XX = np.hstack(views)
    Sv = [_heat_knn_dense(Xv, k) for Xv in views]
    Mg = np.array([[np.sum(Sv[p] * Sv[q]) for q in range(V)] for p in range(V)])
    gam = np.ones(V) / V
    Sb = sum(gam[v] * Sv[v] for v in range(V))
    Yb = indicator(spectral_init(XX, c, 5, seed), c)
    Yb = Yb / np.sqrt(Yb.sum(0, keepdims=True) + 1e-12) + 0.2
    Xs = [Xv.T for Xv in views]
    Yv = [np.zeros((n, c)) for _ in range(V)]
    Gv = [np.ones(X.shape[0]) for X in Xs]
    W = [solve_reg_ls(Xs[v], Gv[v], 1.0, Yb)[0] for v in range(V)]
    for _ in range(iters):
        SS = (Sb + Sb.T) / 2
        Dg = SS.sum(1)
        P = sum(Xs[v].T @ W[v] for v in range(V))
        SY = sum(Yv)
        num = 2 * rho * Yb + lam * SS @ Yb + np.maximum(P, 0) + np.maximum(-SY, 0)
        den = V * Yb + lam * Dg[:, None] * Yb + 2 * rho * Yb @ (Yb.T @ Yb) + np.maximum(-P, 0) + np.maximum(SY, 0)
        Yb = Yb * num / (den + 1e-12)
        for v in range(V):
            R = Xs[v].T @ W[v] - Yb
            Yv[v] = np.sign(R) * np.maximum(np.abs(R) - gam[v] / 2, 0)
        for v in range(V):
            W[v], _ = solve_reg_ls(Xs[v], Gv[v], max(1 - gam[v], 1e-6), Yb + Yv[v])
            Gv[v] = 1.0 / (2 * np.sqrt((W[v] ** 2).sum(1)) + 1e-8)
        Sb = proj_simplex_rows(sum(gam[v] * Sv[v] for v in range(V)) - lam / (4 * alpha) * sqdist(Yb))
        f = np.array([np.abs(Yv[v]).sum() + np.sqrt((W[v] ** 2).sum(1)).sum() - 2 * alpha * np.sum(Sb * Sv[v])
                      for v in range(V)])
        gam = _simplex_qp((2 * alpha + beta) * Mg, f)
    return np.argsort(-(np.vstack(W) ** 2).sum(1))


# ============================================================================ dispatcher
def run_baseline(method, views, c, params):
    XX = np.hstack(views)
    if method == "LS":
        return laplacian_score(XX)
    if method == "MCFS":
        return mcfs(XX, c)
    if method == "UDFS":
        return udfs(XX, c, **params)
    if method == "NDFS":
        return ndfs(XX, c, **params)
    if method == "JMVFG":
        return jmvfg(views, c, **params)
    if method == "CvLP":
        return cvlp_dcl(views, c, **params)
    raise ValueError(method)
