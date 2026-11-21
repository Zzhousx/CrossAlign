"""Evaluate CrossAlign variants."""
import numpy as np

from common import config, log, read, append, args, Timer
from cam.data import load
from cam.metrics import evaluate_ranking, mean_over_budgets
from cam.crossalign import crossalign

if __name__ == "__main__":
    a = args()
    done = {(r["dataset"], r["variant"]) for r in read("ablation.jsonl")}
    for ds in a.datasets:
        X, y = load(ds)
        c, XX = int(y.max() + 1), np.hstack(X)
        cosine = ds in config.TEXT_DATASETS
        for name, kw in config.ABLATION.items():
            if (ds, name) in done:
                continue
            with Timer() as t:
                rk = crossalign(X, c, k=config.CROSSALIGN_K, **kw)
            res = evaluate_ranking(XX, y, rk, cosine=cosine, **config.REPORT)
            ma, mn = mean_over_budgets(res)
            append("ablation.jsonl", dict(dataset=ds, variant=name, kw=kw, acc=ma, nmi=mn, fit_seconds=t.seconds,
                                          res={str(k): v for k, v in res.items()}))
            log(f"{ds:9s} {name:42s} ACC={ma:.4f} NMI={mn:.4f}")
    log("stage 4 done")
