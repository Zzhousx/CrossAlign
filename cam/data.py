"""Loaders for the six benchmarks."""
import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

import config

# name: (file in DATA_DIR, feature key, label key, samples stored in columns)
SOURCES = {
    "HW":       ("HW.mat", "X", "Y", False),
    "OutScene": ("OutScene.mat", "fea", "gt", False),
    "ORL":      ("ORL.mat", "X", "gt", True),
    "Yale":     ("Yale.mat", "X", "gt", True),
    "BBCSport": ("BBCSport.mat", "X", "gt", True),
    "MNIST4":   ("MNIST4.mat", "X", "Y", False),
}

VIEW_NAMES = {
    "HW": ["Pix", "Fou", "Fac", "Zer", "Kar", "Mor"],
    "OutScene": ["GIST", "CM", "HOG", "LBP"],
    "ORL": ["Intensity", "LBP", "Gabor"],
    "Yale": ["Intensity", "LBP", "Gabor"],
    "BBCSport": ["Segment 1", "Segment 2"],
    "MNIST4": ["ISO", "LDA", "NPE"],
}

DOMAIN = {"HW": "handwritten digits", "OutScene": "outdoor scenes",
          "ORL": "faces", "Yale": "faces", "BBCSport": "news text", "MNIST4": "digits"}


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(0), x.max(0)
    rng = hi - lo
    out = np.zeros_like(x)
    ok = rng > 1e-12
    out[:, ok] = (x[:, ok] - lo[ok]) / rng[ok]
    return out


def tfidf_l2(x):
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.preprocessing import normalize
    return normalize(TfidfTransformer().fit_transform(x).toarray())


def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


def load(name, normalise=True):
    fname, fk, lk, transpose = SOURCES[name]
    d = sio.loadmat(os.path.join(config.DATA_DIR, fname))
    views = []
    for cell in d[fk].ravel():
        a = _dense(cell).astype(np.float64)
        views.append(a.T if transpose else a)
    y = np.asarray(d[lk]).ravel()
    _, y = np.unique(y, return_inverse=True)
    if normalise:
        views = [tfidf_l2(v) for v in views] if name in config.TEXT_DATASETS else [minmax(v) for v in views]
    return views, y.astype(int)


def describe(name):
    X, y = load(name)
    return dict(name=name, domain=DOMAIN[name], n=int(X[0].shape[0]), V=len(X), dims=[int(v.shape[1]) for v in X],
                D=int(sum(v.shape[1] for v in X)), c=int(y.max() + 1), views=VIEW_NAMES[name])
