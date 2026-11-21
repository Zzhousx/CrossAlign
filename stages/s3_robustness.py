"""Stage 3: robustness to source-specific structured artifacts (cam/nuisance.py)."""
import numpy as np

from common import config, log, read, append, args, main_params, Timer
from cam.data import load
from cam.metrics import evaluate_ranking, feature_count, mean_over_budgets
from cam.crossalign import crossalign
from cam.baselines import run_baseline
from cam.nuisance import corrupt


def rank(method, X, c, ds):
    if method == "CrossAlign":
        return crossalign(X, c, k=config.CROSSALIGN_K)
    if method == "Align-self":
        return crossalign(X, c, k=config.CROSSALIGN_K, target="self")
    params = main_params(ds, method) if method in ("NDFS", "JMVFG", "CvLP", "UDFS") else {}
    return run_baseline(method, X, c, params)


if __name__ == "__main__":
    a = args()
    done = {(r["dataset"], r["mode"], r["frac"], r["method"]) for r in read("robustness.jsonl")}
    for ds in a.datasets:
        X0, y = load(ds)
        c = int(y.max() + 1)
        cosine = ds in config.TEXT_DATASETS
        for mode in config.NUISANCE_MODES:
            for frac in config.NUISANCE_FRACS:
                X, mask = corrupt(X0, mode, frac, config.NUISANCE_SEED, config.NUISANCE_GROUPS, config.NUISANCE_SIGMA)
                XX = np.hstack(X)
                D = XX.shape[1]
                for m in config.ROBUST_METHODS:
                    if (ds, mode, frac, m) in done:
                        continue
                    with Timer() as t:
                        rk = np.asarray(rank(m, X, c, ds))
                    res = evaluate_ranking(XX, y, rk, cosine=cosine, **config.ROBUST)
                    ma, mn = mean_over_budgets(res)
                    share = {str(r): float(mask[rk[:feature_count(D, r)]].mean()) for r in config.RATIOS}
                    append("robustness.jsonl", dict(dataset=ds, mode=mode, frac=frac, seed=config.NUISANCE_SEED,
                                                    method=m, acc=ma, nmi=mn, share=share, fit_seconds=t.seconds,
                                                    res={str(k): v for k, v in res.items()}))
                    log(f"{ds:9s} {mode:5s} frac={frac:.2f} {m:10s} ACC={ma:.4f} NMI={mn:.4f} "
                        f"nuisance@10%={share['0.1']:.2f} ({t.seconds:.1f}s)")
    log("stage 3 done")
