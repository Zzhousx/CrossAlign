"""Shared helpers for benchmark stages."""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import config  # noqa: E402


def path(name):
    return os.path.join(config.RESULTS, name)


def read(name):
    p = path(name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def append(name, record):
    with open(path(name), "a") as f:
        f.write(json.dumps(record) + "\n")


def pkey(params):
    return json.dumps(params, sort_keys=True)


def phash(params):
    return hashlib.md5(pkey(params).encode()).hexdigest()[:8]


def selected(dataset, method):
    """Hyper-parameters chosen for a baseline: best mean ACC over the five budgets on the tuning seeds."""
    recs = [r for r in read("tuning.jsonl") if r["dataset"] == dataset and r["method"] == method]
    if not recs:
        raise RuntimeError(f"no tuning records for {dataset}/{method}: run stages/s1_tune_baselines.py first")
    return max(recs, key=lambda r: r["tune_acc"])


def main_params(dataset, method):
    for r in read("main.jsonl"):
        if r["dataset"] == dataset and r["method"] == method:
            return r["params"] or {}
    raise RuntimeError(f"no main record for {dataset}/{method}: run stages/s2_main_eval.py first")


def args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    if extra:
        extra(ap)
    a = ap.parse_args()
    a.datasets = [d for d in a.datasets.split(",") if d]
    return a


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def result_file(name):
    """Resolve an output file or its bundled reference."""
    from pathlib import Path
    current = Path(config.RESULTS) / name
    return current if current.is_file() else Path(config.EXPECTED) / name


def benchmark_rows(name):
    """Load the latest benchmark records, falling back to the reference CSV."""
    from csv import DictReader
    if name in {"main.csv", "reference_per_budget.csv"}:
        records = read("main.jsonl")
        if records:
            if name == "reference_per_budget.csv":
                return [dict(dataset=r["dataset"], method=r["method"], budget=budget, **scores)
                        for r in records for budget, scores in r["res"].items()]
            runtime = {(r["dataset"], r["method"]): r for r in read("runtime.jsonl")}
            return [dict(r, params=json.dumps(r["params"]),
                         one_run_seconds=runtime.get((r["dataset"], r["method"]), {}).get("seconds", r["fit_seconds"]),
                         search_seconds=runtime.get((r["dataset"], r["method"]), {}).get(
                             "search_seconds", (r["fit_seconds"] or 0) * (r["grid_size"] or 1)))
                    for r in records]
    if name == "ESM_1.csv":
        records = read("robustness.jsonl")
        if records:
            return [dict(r, **{f"artifact_share_at_{b}": v for b, v in r["share"].items()})
                    for r in records]
    if name == "sensitivity.csv":
        records = read("sensitivity.jsonl")
        if records:
            return records
    with result_file(name).open(newline="", encoding="utf-8") as source:
        return list(DictReader(source))
