"""Measure sequential ranking runtimes."""
import os
import platform

import numpy as np

from common import config, log, read, append, args, main_params, Timer
from cam.data import load
from cam.crossalign import crossalign
from cam.baselines import run_baseline

METHODS = config.BASELINES + ["CrossAlign"]

if __name__ == "__main__":
    a = args()
    done = {(r["dataset"], r["method"]) for r in read("runtime.jsonl")}
    machine = dict(cpu=platform.processor() or platform.machine(), cores=os.cpu_count(),
                   python=platform.python_version())
    for ds in a.datasets:
        X, y = load(ds)
        c, n = int(y.max() + 1), X[0].shape[0]
        for m in METHODS:
            if (ds, m) in done:
                continue
            times = []
            for rep in range(3):
                with Timer() as t:
                    if m == "CrossAlign":
                        crossalign(X, c, k=config.CROSSALIGN_K)
                    else:
                        run_baseline(m, X, c, main_params(ds, m))
                times.append(t.seconds)
                if t.seconds > 60:
                    break
            grid_size = 1 if m == "CrossAlign" else len(config.grid(m, n))
            sec = float(np.median(times))
            append("runtime.jsonl", dict(dataset=ds, method=m, seconds=sec, repeats=len(times), grid_size=grid_size,
                                         search_seconds=sec * grid_size, machine=machine))
            log(f"{ds:9s} {m:10s} one run {sec:8.2f}s  x{grid_size:2d} settings = {sec * grid_size:9.1f}s")
    log("stage 6 done")
