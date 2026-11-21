"""Reproduce the BBCSport feature coverage and corruption analysis."""
from pathlib import Path
import csv
import json
from collections import Counter

import numpy as np
from sklearn.manifold import TSNE
from common import config, benchmark_rows
from cam.baselines import run_baseline
from cam.crossalign import crossalign
from cam.data import load
from cam.metrics import feature_count, rownorm, kmeans_scores
from cam.nuisance import corrupt

def main():
    OUTPUT = Path(config.RESULTS)
    views, labels = load("BBCSport")
    n = labels.size
    c = np.unique(labels).size
    widths = [v.shape[1] for v in views]
    clean = np.hstack(views)
    budget = feature_count(clean.shape[1], 0.1)
    rank = crossalign(views, c, K=budget)
    relevance = crossalign(views, c, select="relevance")
    delta = np.stack([clean[labels == k].mean(0) - clean[labels != k].mean(0) for k in range(c)])
    feature_topics = delta.argmax(0)
    sigma = clean.std(0)
    covered = (sigma > 1e-12) & (delta.max(0) > 0)
    segments = np.repeat(np.arange(len(widths)), widths)
    clean_summary = []
    for tag, ranked in (("CrossAlign", rank), ("Relevance only", relevance)):
        selected = ranked[:budget]
        counts = Counter((int(feature_topics[j]) + 1, int(segments[j]) + 1) for j in selected if covered[j])
        matrix = {str(k): [counts[k, 1], counts[k, 2]] for k in range(1,c+1)}
        good = selected[covered[selected]]
        cor = np.corrcoef(clean[:, good].T)
        squared_corr = (np.square(cor).sum() - good.size) / (good.size * (good.size - 1))
        clean_summary.append(dict(method=tag, assigned=int(good.size), width=budget, counts=matrix,
                                  mean_squared_correlation=float(squared_corr)))
        print(tag, matrix, squared_corr, flush=True)
    robustness = benchmark_rows("ESM_1.csv")
    external = {"LS","MCFS","NDFS","JMVFG","CvLP"}
    base = max((r for r in robustness if r["dataset"] == "BBCSport" and r["mode"] == "one" and
                float(r["frac"]) == 1 and r["method"] in external), key=lambda row: float(row["acc"]))["method"]
    params = json.loads(next(r["params"] for r in benchmark_rows("main.csv") if r["dataset"] == "BBCSport" and r["method"] == base))
    dirty, mask = corrupt(views, "one", 1., config.NUISANCE_SEED, config.NUISANCE_GROUPS, config.NUISANCE_SIGMA)
    merged = np.hstack(dirty)
    m = feature_count(merged.shape[1], 0.2)
    ours = crossalign(dirty,c,K=m)
    within = crossalign(dirty,c,target="self",K=m)
    print("CrossAlign", int(mask[ours[:20]].sum()), "Within-view", int(mask[within[:20]].sum()), flush=True)
    external_rank = run_baseline(base, dirty, c, params)
    top = {"CrossAlign": int(mask[ours[:20]].sum()), "Within-view": int(mask[within[:20]].sum()),
           base: int(mask[external_rank[:20]].sum())}
    print("top", top, flush=True)
    panels = (("All features","af", np.arange(mask.size)), ("Within-view","within", within[:m]),
              ("CrossAlign","ours",ours[:m]))
    scores = {}
    coords = []
    for title, _, selected in panels:
        matrix = rownorm(merged[:,selected])
        acc,nmi = kmeans_scores(matrix, labels, **config.ROBUST)
        scores[title] = {"acc": float(acc.mean()), "nmi": float(nmi.mean()), "selected_artifact_share": float(mask[selected].mean())}
        xy = TSNE(n_components=2, perplexity=25, init="pca", learning_rate="auto", max_iter=1000,
                  random_state=42).fit_transform(matrix)
        for i, (x,y) in enumerate(xy):
            coords.append(("BBCSport",title,i+1,int(labels[i])+1,float(x),float(y)))
        print(title,scores[title],flush=True)
    with (OUTPUT / "visual_embeddings.csv").open("w", newline="",encoding="utf-8") as stream:
        output = csv.writer(stream)
        output.writerow(["dataset","representation","sample","class_index","x","y"])
        output.writerows(coords)
    with (OUTPUT / "bbc_case_study.json").open("w", encoding="utf-8") as stream:
        json.dump({"clean": clean_summary, "top20_artifacts": top, "embeddings": scores}, stream, indent=2)
        stream.write("\n")



if __name__ == "__main__":
    main()
