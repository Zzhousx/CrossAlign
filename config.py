"""Experiment configuration."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CAM_DATA", os.path.join(ROOT, "data", "raw"))
RESULTS = os.environ.get("CAM_RESULTS", os.path.join(ROOT, "results"))
RANKINGS = os.path.join(RESULTS, "rankings")
TABLES = os.environ.get("CAM_TABLES", os.path.join(ROOT, "tables"))
FIGURES = os.environ.get("CAM_FIGURES", os.path.join(ROOT, "figures"))
LOGS = os.path.join(ROOT, "logs")
EXPECTED = os.path.join(ROOT, "expected")
SMOKE = bool(os.environ.get("CAM_SMOKE"))   # tests/smoke.py only: tiny synthetic data, 2 k-means seeds

# ---------------------------------------------------------------- benchmark
DATASETS = ["HW", "OutScene", "ORL", "Yale", "BBCSport", "MNIST4"]
TEXT_DATASETS = {"BBCSport"}          # TF-IDF + L2 views, evaluated in cosine geometry

# ---------------------------------------------------------------- evaluation protocol
RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5)    # feature budgets as a fraction of all features D
TUNE = dict(reps=10, seed0=0)         # tuning seeds
REPORT = dict(reps=20, seed0=100)     # k-means seeds 100-119: every reported number
ROBUST = dict(reps=10, seed0=100)     # k-means seeds 100-109: robustness experiment
if SMOKE:
    TUNE, REPORT, ROBUST = dict(reps=2, seed0=0), dict(reps=2, seed0=100), dict(reps=2, seed0=100)

# ---------------------------------------------------------------- CrossAlign
CROSSALIGN_K = 10                     # default kNN graph size
SENSITIVITY_K = (5, 10, 15, 20, 30)

# ---------------------------------------------------------------- baselines
BASELINES = ["LS", "MCFS", "UDFS", "NDFS", "JMVFG", "CvLP"]
LARGE_N = 2000                        # above this sample size JMVFG and CvLP use the reduced grid
G4 = (1e-3, 1e-1, 1e1, 1e3)
G3 = (1e-2, 1, 1e2)


def grid(method, n):
    """Baseline hyperparameter grid."""
    g = _grid(method, n)
    return g[:2] if SMOKE else g


def _grid(method, n):
    if method in ("LS", "MCFS"):
        return [dict()]
    if method == "NDFS":
        return [dict(alpha=a, beta=b) for a in G4 for b in G4]
    if method == "UDFS":
        return [dict(gamma=g, lam=l) for g in (1e-3, 1e-1) for l in (1e-3, 1.0, 1e3)]
    if method == "JMVFG":
        betas = (1e-3, 1e-1, 1e1) if n <= LARGE_N else (1e-3,)
        return [dict(eta=e, gamma=g, beta=b) for e in G3 for g in G3 for b in betas]
    if method == "CvLP":
        if n <= LARGE_N:
            return [dict(lam=l, alpha=a, beta=b) for l in G3 for a in G3 for b in G3]
        return [dict(lam=l, alpha=1e2, beta=b) for l in G3 for b in (1, 1e2)] + [dict(lam=1, alpha=1, beta=1)]
    raise ValueError(method)


# ---------------------------------------------------------------- robustness to corrupted views
NUISANCE_FRACS = (0.25, 0.5, 1.0)     # nuisance features added, as a fraction of D (one) or d_v (every)
NUISANCE_MODES = ("one", "every")     # one corrupted view / every view corrupted
NUISANCE_SEED = 1
NUISANCE_GROUPS = 5
NUISANCE_SIGMA = 0.3
ROBUST_METHODS = ["CrossAlign", "Align-self", "LS", "MCFS", "NDFS", "JMVFG", "CvLP"]
if SMOKE:
    NUISANCE_FRACS, SENSITIVITY_K = (0.5,), (5, 10)

# ---------------------------------------------------------------- component analysis
ABLATION = {
    "CrossAlign": {},
    "within-view target": dict(target="self"),
    "continuous target (no discretisation)": dict(discrete=False),
    "per-feature ranking: relevance ratio": dict(select="ratio"),
    "per-feature ranking: relevance": dict(select="relevance"),
    "standardised geometry": dict(space="std"),
}

for _d in (RESULTS, RANKINGS, TABLES, FIGURES, LOGS):
    os.makedirs(_d, exist_ok=True)
