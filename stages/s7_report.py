"""Summarise benchmark results in tables."""
import json
import os

import numpy as np
from scipy.stats import rankdata

from common import config, log, read

DS = config.DATASETS
BASE = config.BASELINES
METHODS = BASE + ["CrossAlign"]
LABEL = {"LS": "LS", "MCFS": "MCFS", "UDFS": "UDFS", "NDFS": "NDFS", "JMVFG": "JMVFG", "CvLP": "CvLP-DCL",
         "CrossAlign": "CrossAlign", "AllFea": "All features",
         "Align-self": "Within-view target"}
R = [str(r) for r in config.RATIOS]
def out(name):
    return os.path.join(config.TABLES, name)


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write(name, text):
    with open(out(name), "w") as f:
        f.write(text)


def latex_table(header, rows, caption, label, bold=None):
    cols = "l" + "c" * (len(header) - 1)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\small", f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        cells = []
        for j, c in enumerate(r):
            s = str(c)
            if bold and bold.get((i, j)) == 1:
                s = f"\\textbf{{{s}}}"
            elif bold and bold.get((i, j)) == 2:
                s = f"\\underline{{{s}}}"
            cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


# ============================================================================ datasets
def table_datasets(summary):
    p = os.path.join(config.RESULTS, "datasets.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    header = ["Dataset", "Domain", "n", "Views", "Dimensions", "D", "c"]
    body = [[d["name"], d["domain"], d["n"], d["V"], "/".join(map(str, d["dims"])), d["D"], d["c"]] for d in rows]
    write("datasets.md", md_table(header, body))
    write("datasets.tex", latex_table(header, body, "Benchmark datasets.", "tab:datasets"))
    summary["datasets"] = {d["name"]: dict(n=d["n"], V=d["V"], D=d["D"], c=d["c"]) for d in rows}


# ============================================================================ main results
def main_tables(summary):
    recs = {(r["dataset"], r["method"]): r for r in read("main.jsonl")}
    if not recs:
        return None
    # per-budget CSV
    with open(out("per_budget.csv"), "w") as f:
        f.write("dataset,method,budget,acc,acc_std,nmi,nmi_std\n")
        for (ds, m), r in sorted(recs.items()):
            for b in R:
                v = r["res"][b]
                f.write(f"{ds},{m},{b},{v['acc']:.4f},{v['acc_std']:.4f},{v['nmi']:.4f},{v['nmi_std']:.4f}\n")
    stats = {}
    for metric in ("acc", "nmi"):
        tab = np.array([[recs[(d, m)][metric] if (d, m) in recs else np.nan for m in METHODS] for d in DS])
        ranks = np.full_like(tab, np.nan)
        for i in range(len(DS)):
            ok = ~np.isnan(tab[i])
            ranks[i, ok] = rankdata(-tab[i, ok])
        avg_rank = {m: float(np.nanmean(ranks[:, j])) for j, m in enumerate(METHODS)}
        header = ["Dataset", "All features"] + [LABEL[m] for m in METHODS]
        body, bold = [], {}
        for i, d in enumerate(DS):
            allf = recs.get((d, "AllFea"), {}).get(metric)
            row = [d, fmt(allf)] + [fmt(x) for x in tab[i]]
            order = np.argsort(-np.nan_to_num(tab[i], nan=-1))
            bold[(i, 2 + order[0])] = 1
            bold[(i, 2 + order[1])] = 2
            body.append(row)
        body.append(["Average rank", ""] + [f"{avg_rank[m]:.2f}" for m in METHODS])
        name = "ACC" if metric == "acc" else "NMI"
        md_rows = [[(f"**{c}**" if bold.get((i, j)) == 1 else (f"_{c}_" if bold.get((i, j)) == 2 else c))
                    for j, c in enumerate(r)] for i, r in enumerate(body)]
        write(f"main_{metric}.md", md_table(header, md_rows))
        write(f"main_{metric}.tex", latex_table(
            header, body, f"Clustering {name} averaged over the five feature budgets (10--50\\%), k-means seeds "
            f"100--119. Best in bold, second underlined.",
            f"tab:main_{metric}", bold))
        # Compare scores at each dataset and budget.
        tests = {}
        for method in BASE:
            complete = all((d, method) in recs and (d, "CrossAlign") in recs for d in DS)
            if not complete:
                continue
            differences = np.array([
                [recs[d, "CrossAlign"]["res"][q][metric] - recs[d, method]["res"][q][metric] for q in R]
                for d in DS
            ])
            tests[method] = dict(cells=differences.size, wins=int((differences > 0).sum()),
                                 datasets=len(DS), mean_gain_points=float(100 * differences.mean()))
        best_ds = {d: max((recs[(d, m)][metric], m) for m in BASE if (d, m) in recs) for d in DS
                   if (d, "CrossAlign") in recs}
        margin = {d: float(100 * (recs[(d, "CrossAlign")][metric] - best_ds[d][0])) for d in best_ds}
        stats[metric] = dict(avg_rank=avg_rank, tests=tests, margin_vs_best_baseline=margin,
                             best_baseline={d: best_ds[d][1] for d in best_ds},
                             crossalign_best_on=int(sum(1 for d in margin if margin[d] > 0)),
                             datasets=len(margin))
        rows = [[LABEL[m], f"{t['wins']}/{t['cells']}", f"{t['mean_gain_points']:+.2f}"]
                for m, t in tests.items()]
        write(f"comparisons_{metric}.md", md_table(["CrossAlign vs", "Wins", "Mean gain (points)"], rows))
    summary["main"] = stats
    return recs


# ============================================================================ robustness
def robustness_tables():
    recs = read("robustness.jsonl")
    if not recs:
        return
    get = {(r["dataset"], r["mode"], r["frac"], r["method"]): r for r in recs}
    present = [d for d in DS if any(k[0] == d for k in get)]
    meths = config.ROBUST_METHODS
    rows_md, head = [], ["Dataset", "Scenario", "Nuisance"] + [LABEL[m] for m in meths]
    for d in present:
        for mode in config.NUISANCE_MODES:
            for fr in config.NUISANCE_FRACS:
                if (d, mode, fr, "CrossAlign") not in get:
                    continue
                accs = [get[(d, mode, fr, m)]["acc"] if (d, mode, fr, m) in get else np.nan for m in meths]
                best = int(np.nanargmax(accs))
                cells = [(f"**{fmt(x)}**" if j == best else fmt(x)) for j, x in enumerate(accs)]
                rows_md.append([d, "one view" if mode == "one" else "every view", f"{fr:g}"] + cells)
    write("robustness_acc.md", md_table(head, rows_md))
    rows_sh = []
    for d in present:
        for mode in config.NUISANCE_MODES:
            for fr in config.NUISANCE_FRACS:
                if (d, mode, fr, "CrossAlign") not in get:
                    continue
                rows_sh.append([d, "one view" if mode == "one" else "every view", f"{fr:g}"] +
                               [fmt(100 * get[(d, mode, fr, m)]["share"]["0.1"], 0) if (d, mode, fr, m) in get else "-"
                                for m in meths])
    write("robustness_share.md", md_table(head, rows_sh))


# ============================================================================ ablation
def ablation_tables(summary):
    recs = read("ablation.jsonl")
    if not recs:
        return
    get = {(r["dataset"], r["variant"]): r for r in recs}
    variants = list(config.ABLATION)
    present = [d for d in DS if (d, "CrossAlign") in get]
    head = ["Variant"] + present + ["Mean ACC", "Change", "Mean NMI", "Change"]
    full_a = np.mean([get[(d, "CrossAlign")]["acc"] for d in present])
    full_n = np.mean([get[(d, "CrossAlign")]["nmi"] for d in present])
    body, out_sum = [], {}
    for v in variants:
        if not all((d, v) in get for d in present):
            continue
        a = np.mean([get[(d, v)]["acc"] for d in present]); nm = np.mean([get[(d, v)]["nmi"] for d in present])
        body.append([v] + [fmt(get[(d, v)]["acc"]) for d in present] +
                    [fmt(a), "" if v == "CrossAlign" else f"{100 * (a - full_a):+.1f}", fmt(nm),
                     "" if v == "CrossAlign" else f"{100 * (nm - full_n):+.1f}"])
        out_sum[v] = dict(acc=float(a), nmi=float(nm), d_acc_points=float(100 * (a - full_a)),
                          d_nmi_points=float(100 * (nm - full_n)))
    write("ablation.md", md_table(head, body))
    write("ablation.tex", latex_table(head, body, "Component analysis: ACC per dataset and the mean change "
                                      "(points) when one design choice is replaced.", "tab:ablation"))
    summary["ablation"] = out_sum


# ============================================================================ sensitivity
def sensitivity_tables(summary):
    recs = read("sensitivity.jsonl")
    if not recs:
        return
    get = {(r["dataset"], r["k"]): r for r in recs}
    present = [d for d in DS if (d, config.CROSSALIGN_K) in get]
    ks = config.SENSITIVITY_K
    body, gap = [], {}
    for d in present:
        vals = [get[(d, k)]["acc"] if (d, k) in get else np.nan for k in ks]
        gap[d] = float(100 * (np.nanmax(vals) - get[(d, config.CROSSALIGN_K)]["acc"]))
        body.append([d] + [fmt(v) for v in vals] + [f"{gap[d]:.1f}"])
    write("sensitivity.md", md_table(["Dataset"] + [f"k={k}" for k in ks] + ["k=10 vs best k (points)"], body))
    summary["sensitivity_gap_default_vs_best_points"] = gap


# ============================================================================ runtime
def runtime_tables(summary):
    recs = read("runtime.jsonl")
    source = "stage 6 (controlled, sequential)"
    if not recs:   # fall back to the timings recorded while tuning (stage 1) and in stage 2
        source = "stages 1-2 (recorded during the runs)"
        recs = [dict(dataset=r["dataset"], method=r["method"], seconds=r["fit_seconds"],
                     search_seconds=r["tuning_seconds"], grid_size=r["grid_size"])
                for r in read("main.jsonl") if r["method"] in METHODS and r.get("fit_seconds") is not None]
    if not recs:
        return
    get = {(r["dataset"], r["method"]): r for r in recs}
    body, out_sum = [], {}
    for d in DS:
        row = [d]
        for m in METHODS:
            if (d, m) in get:
                r = get[(d, m)]
                row.append(f"{r['seconds']:.1f} / {r['search_seconds']:.0f}")
                out_sum.setdefault(d, {})[m] = dict(one_run=r["seconds"], with_search=r["search_seconds"],
                                                    settings=r["grid_size"])
            else:
                row.append("-")
        body.append(row)
    write("runtime.md", md_table(["Dataset"] + [LABEL[m] for m in METHODS], body) +
          "\nCell: seconds for one run / seconds including the label-tuned hyper-parameter search. "
          f"Timings from {source}.\n")
    write("runtime.tex", latex_table(["Dataset"] + [LABEL[m] for m in METHODS], body,
                                     "Seconds for one run / including hyper-parameter search.", "tab:runtime"))
    summary["runtime"] = out_sum
    summary["runtime_source"] = source


if __name__ == "__main__":
    summary = {}
    table_datasets(summary)
    main_tables(summary)
    ablation_tables(summary)
    robustness_tables()
    sensitivity_tables(summary)
    runtime_tables(summary)
    with open(out("summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"tables -> {config.TABLES}")
