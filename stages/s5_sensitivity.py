"""Evaluate sensitivity to neighbourhood size."""
import numpy as np

from common import config, log, read, append, args, Timer
from cam.data import load
from cam.metrics import evaluate_ranking, mean_over_budgets
from cam.crossalign import crossalign

if __name__ == "__main__":
    a = args()
    done = {(r["dataset"], r["k"]) for r in read("sensitivity.jsonl")}
    for ds in a.datasets:
        X, y = load(ds)
        c, XX = int(y.max() + 1), np.hstack(X)
        cosine = ds in config.TEXT_DATASETS
        for k in config.SENSITIVITY_K:
            if (ds, k) in done:
                continue
            with Timer() as t:
                rk = crossalign(X, c, k=k)
            res = evaluate_ranking(XX, y, rk, cosine=cosine, **config.REPORT)
            ma, mn = mean_over_budgets(res)
            append("sensitivity.jsonl", dict(dataset=ds, k=k, acc=ma, nmi=mn, fit_seconds=t.seconds,
                                             res={str(q): v for q, v in res.items()}))
            log(f"{ds:9s} k={k:3d} ACC={ma:.4f} NMI={mn:.4f}")
    log("stage 5 done")
