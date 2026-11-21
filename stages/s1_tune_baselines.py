"""Tune baseline hyperparameters."""
import numpy as np

from common import config, log, read, append, pkey, phash, args, Timer
from cam.data import load
from cam.metrics import evaluate_ranking, mean_over_budgets
from cam.baselines import run_baseline


def extra(ap):
    ap.add_argument("--methods", default=",".join(config.BASELINES))


if __name__ == "__main__":
    a = args(extra)
    methods = [m for m in a.methods.split(",") if m]
    done = {(r["dataset"], r["method"], pkey(r["params"])) for r in read("tuning.jsonl")}
    for ds in a.datasets:
        X, y = load(ds)
        c, n, XX = int(y.max() + 1), X[0].shape[0], np.hstack(X)
        cosine = ds in config.TEXT_DATASETS
        for m in methods:
            for p in config.grid(m, n):
                if (ds, m, pkey(p)) in done:
                    continue
                with Timer() as t:
                    rk = np.asarray(run_baseline(m, X, c, p))
                fname = f"{ds}_{m}_{phash(p)}.npy"
                np.save(f"{config.RANKINGS}/{fname}", rk.astype(np.int32))
                res = evaluate_ranking(XX, y, rk, cosine=cosine, **config.TUNE)
                ta, tn = mean_over_budgets(res)
                append("tuning.jsonl", dict(dataset=ds, method=m, params=p, fit_seconds=t.seconds,
                                            tune_acc=ta, tune_nmi=tn, ranking=fname))
                log(f"{ds:9s} {m:6s} {pkey(p):48s} tuneACC={ta:.4f} fit={t.seconds:.1f}s")
    log("stage 1 done")
