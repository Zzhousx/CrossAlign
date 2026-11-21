"""Evaluate selectors on clean benchmarks."""
import numpy as np

from common import config, log, read, append, args, selected, Timer
from cam.data import load
from cam.metrics import evaluate_ranking, mean_over_budgets, kmeans_scores, rownorm
from cam.crossalign import crossalign

METHODS = ["AllFea"] + config.BASELINES + ["CrossAlign"]

if __name__ == "__main__":
    a = args()
    done = {(r["dataset"], r["method"]) for r in read("main.jsonl")}
    for ds in a.datasets:
        X, y = load(ds)
        c, XX = int(y.max() + 1), np.hstack(X)
        cosine = ds in config.TEXT_DATASETS
        for m in METHODS:
            if (ds, m) in done:
                continue
            params, fit, grid_size, tuning_seconds = None, None, None, None
            if m == "AllFea":
                A, N = kmeans_scores(rownorm(XX) if cosine else XX, y, **config.REPORT)
                res = {r: dict(acc=float(A.mean()), acc_std=float(A.std()), nmi=float(N.mean()),
                               nmi_std=float(N.std())) for r in config.RATIOS}
            elif m == "CrossAlign":
                with Timer() as t:
                    rk = crossalign(X, c, k=config.CROSSALIGN_K)
                fit, grid_size, tuning_seconds = t.seconds, 1, t.seconds
                np.save(f"{config.RANKINGS}/{ds}_CrossAlign.npy", rk.astype(np.int32))
                res = evaluate_ranking(XX, y, rk, cosine=cosine, **config.REPORT)
            else:
                rec = selected(ds, m)
                tun = [r for r in read("tuning.jsonl") if r["dataset"] == ds and r["method"] == m]
                params, fit = rec["params"], rec["fit_seconds"]
                grid_size, tuning_seconds = len(tun), float(sum(r["fit_seconds"] for r in tun))
                rk = np.load(f"{config.RANKINGS}/{rec['ranking']}")
                res = evaluate_ranking(XX, y, rk, cosine=cosine, **config.REPORT)
            ma, mn = mean_over_budgets(res)
            append("main.jsonl", dict(dataset=ds, method=m, params=params, fit_seconds=fit, grid_size=grid_size,
                                      tuning_seconds=tuning_seconds, acc=ma, nmi=mn,
                                      res={str(k): v for k, v in res.items()}))
            log(f"{ds:9s} {m:10s} ACC={ma:.4f} NMI={mn:.4f} params={params}")
    log("stage 2 done")
