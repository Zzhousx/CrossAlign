
"""Measure selected-feature redundancy."""
import csv
from pathlib import Path

import numpy as np

from common import config, log
from cam.crossalign import crossalign
from cam.data import load
from cam.metrics import evaluate_ranking, feature_count

def read_rows(file):
    if not file.exists():
        return []
    with file.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def add_record(path, row):
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def column_correlation(selected):
    """Mean off-diagonal squared Pearson correlation, excluding constant columns."""
    centred = selected - selected.mean(axis=0)
    norms = np.linalg.norm(centred, axis=0)
    keep = norms > 1e-12
    columns = int(keep.sum())
    if columns < 2:
        raise ValueError("At least two non-constant columns are required.")
    unit = centred[:, keep] / norms[keep]
    gram = unit @ unit.T if unit.shape[0] < columns else unit.T @ unit
    squared_offdiag = max(0., float(np.square(gram).sum() - columns))
    return squared_offdiag / (columns*(columns-1)), columns


def redundancy_experiment(file, datasets=config.DATASETS):
    existing = read_rows(file)
    keys = {(r["dataset"],float(r["budget"]),r["method"]) for r in existing}
    for dataset in datasets:
        methods = ("CrossAlign", "Relevance only")
        if all((dataset, rate, method) in keys for method in methods for rate in config.RATIOS):
            continue
        views, labels = load(dataset)
        data = np.hstack(views)
        c = int(labels.max() + 1)
        rankings = {name: crossalign(views, c, k=config.CROSSALIGN_K, select=select)
                    for name, select in (("CrossAlign", "align"), ("Relevance only", "relevance"))}
        cosine = dataset in config.TEXT_DATASETS
        for method, rank in rankings.items():
            rates = tuple(r for r in config.RATIOS if (dataset,r,method) not in keys)
            scores = evaluate_ranking(data, labels, rank, cosine=cosine, ratios=rates, **config.REPORT)
            for ratio in rates:
                count = feature_count(data.shape[1], ratio)
                correlation, valid = column_correlation(data[:, rank[:count]])
                add_record(file, dict(dataset=dataset, budget=ratio, method=method, features=count,
                                      nonconstant=valid, mean_squared_correlation=correlation, **scores[ratio]))
        log(f"{dataset} redundancy diagnostics complete")


if __name__ == "__main__":
    redundancy_experiment(Path(config.RESULTS) / "subset_redundancy.csv")
