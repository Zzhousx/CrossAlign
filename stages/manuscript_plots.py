
"""Regenerate the manuscript comparison, sensitivity, timing and embedding figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from common import config, benchmark_rows

OUT = Path(config.FIGURES)
DATASETS = ("HW", "OutScene", "ORL", "Yale", "BBCSport", "MNIST4")
BASELINES = ("LS", "MCFS", "UDFS", "NDFS", "JMVFG", "CvLP")
ROBUST = ("LS", "MCFS", "NDFS", "JMVFG", "CvLP")
BUDGETS = (.1, .2, .3, .4, .5)
CORRUPTION = (.25, .5, 1.)
GRAPH_SIZES = (5, 10, 15, 20, 30)
STYLES = {
    "CrossAlign": ("Ours", "#174B78", "o", "-", 1.55),
    "AllFea": ("AF", "#666972", "", ":", 1.02),
    "LS": ("LS", "#87949D", "v", "--", .83),
    "MCFS": ("MCFS", "#927D9D", "^", "-.", .83),
    "UDFS": ("UDFS", "#A07737", "D", "--", .83),
    "NDFS": ("NDFS", "#678B78", "s", "-", .83),
    "JMVFG": ("JMVFG", "#B36458", "P", "-.", .83),
    "CvLP": ("CvLP-DCL", "#58666F", "x", "--", 1.02),
}
REPRESENTATIONS = {"All features": "af", "Within-view": "within", "CrossAlign": "ours"}
CLASS_COLORS = ("#3368A2", "#DD782A", "#328453", "#9D439A", "#C8474B", "#658029", "#615B9C")
CLASS_SYMBOLS = ("o", "s", "^", "D", "v", "P", "X")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.4, "axes.titlesize": 8,
    "axes.labelsize": 7.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .53, "axes.edgecolor": "#7F8994",
    "axes.labelcolor": "#263442", "text.color": "#263442",
    "xtick.color": "#4B5661", "ytick.color": "#4B5661",
    "xtick.labelsize": 6.7, "ytick.labelsize": 6.7, "pdf.fonttype": 42,
})


def save(figure, name):
    figure.savefig(OUT / (name + ".pdf"), facecolor="white")
    plt.close(figure)

def draw_comparison(ax, series, dataset, ticks, methods, robust):
    for method in reversed(methods):
        label, color, marker, linestyle, width = STYLES[method]
        ax.plot(ticks, series[dataset, method], label=label, color=color, marker=marker,
                ls=linestyle, lw=width, ms=2.6 if method == "CrossAlign" else 2.1,
                markerfacecolor=color if method == "CrossAlign" else "white",
                markeredgewidth=.64)
    ax.set_xticks(ticks, ["0", "0.25", "0.5", "1"] if robust else ticks)
    ax.tick_params(length=2, width=.53, pad=2)
    ax.grid(axis="y", color="#E1E6EC", lw=.52)
    ax.set_axisbelow(True)
    ax.margins(x=.061, y=.14)
    if robust:
        ax.set(ylim=(0, 1), yticks=[0, .25, .5, .75, 1])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:g}")
    )


def comparison_legend(ax, methods, *, corner, size, columns):
    handles = {line.get_label(): line for line in ax.lines}
    labels = [STYLES[method][0] for method in methods]
    legend = ax.legend([handles[label] for label in labels], labels, loc=corner,
                       ncols=columns, fontsize=size, frameon=True, fancybox=False,
                       framealpha=.62, edgecolor="#E0E5EB", handlelength=1.58,
                       handletextpad=.40, columnspacing=.55, labelspacing=.24,
                       borderpad=.32, borderaxespad=.39)
    legend.get_frame().set_linewidth(.36)
    legend.get_texts()[0].set_weight("bold")


def clean_benchmark_panels(per_budget):
    ticks = [round(budget * 100) for budget in BUDGETS]
    methods = ("CrossAlign", "AllFea", *BASELINES)
    for dataset in DATASETS:
        fig, (acc, nmi) = plt.subplots(2, 1, sharex=True, figsize=(1.76, 1.76))
        fig.subplots_adjust(left=.223, right=.979, top=.982, bottom=.160, hspace=.21)
        for ax, metric in ((acc, "acc"), (nmi, "nmi")):
            series = {(dataset, method): [float(per_budget[dataset, method, budget][metric])
                                          for budget in BUDGETS] for method in methods}
            draw_comparison(ax, series, dataset, ticks, methods, robust=False)
            ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10]))
            ax.tick_params(labelsize=5.65, pad=1.15, length=1.6)
            ax.set_ylabel(metric.upper(), fontsize=6.2, labelpad=1.4)
        nmi.set_xlabel("Feature budget (%)", fontsize=6.1, labelpad=1.5)
        comparison_legend(nmi, methods, corner="lower left", size=4.60, columns=2)
        save(fig, f"fig_clean_{dataset.lower()}")


def artifact_panels(main_rows, corrupted):
    methods = ("CrossAlign", *ROBUST)
    ticks = [0, *CORRUPTION]
    for dataset in DATASETS:
        series = {(dataset, method): [float(main_rows[dataset, method]["acc"]),
                  *(corrupted[dataset, frac, method] for frac in CORRUPTION)]
                  for method in methods}
        fig, ax = plt.subplots(figsize=(1.76, 1.27))
        fig.subplots_adjust(left=.223, right=.970, top=.951, bottom=.233)
        draw_comparison(ax, series, dataset, ticks, methods, robust=True)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_xticks(ticks, [f"{frac:.0%}" for frac in ticks])
        ax.tick_params(labelsize=5.65, pad=1.15, length=1.6)
        ax.set_ylabel("ACC", fontsize=6.2, labelpad=1.4)
        ax.set_xlabel("Added features / original total", fontsize=6.1, labelpad=1.5)
        comparison_legend(ax, methods, corner="center right", size=4.60, columns=2)
        legend_height = .86 if dataset == "OutScene" else .44
        ax.get_legend().set_bbox_to_anchor((1, legend_height))
        save(fig, f"fig_robustness_{dataset.lower()}")


def sensitivity_panels():
    rows = benchmark_rows("sensitivity.csv")
    values = {(row["dataset"], int(row["k"])): row for row in rows}
    expected = {(dataset, k) for dataset in DATASETS for k in GRAPH_SIZES}
    if expected != values.keys() or len(rows) != len(expected):
        raise ValueError("Expected one sensitivity record per dataset and graph size")

    for metric in ("acc", "nmi"):
        fig, ax = plt.subplots(figsize=(2.45, 1.69))
        fig.subplots_adjust(left=.176, right=.978, top=.962, bottom=.380)
        ax.axvline(10, color="#7F8994", linewidth=.80, linestyle="--", zorder=1)
        for dataset, color, marker in zip(DATASETS, CLASS_COLORS, CLASS_SYMBOLS):
            scores = [float(values[dataset, k][metric]) for k in GRAPH_SIZES]
            ax.plot(GRAPH_SIZES, scores, label=dataset, color=color, marker=marker,
                    linewidth=.99, markersize=2.7, markeredgewidth=.56)
        ax.set(xlim=(3.9, 31.1), ylim=(.54, .92), xticks=GRAPH_SIZES,
               yticks=(.55, .65, .75, .85), xlabel="Graph neighbourhood size $k$",
               ylabel=metric.upper())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
        ax.tick_params(length=1.9, pad=2.1, labelsize=6.4)
        ax.xaxis.label.set_fontsize(7)
        ax.grid(axis="y", color="#E1E6EC", linewidth=.5)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(.41, -.398), ncols=3,
                  fontsize=5.4, frameon=False, handlelength=1.34, handletextpad=.37,
                  columnspacing=.67, labelspacing=.42, borderaxespad=0)
        save(fig, f"fig_sensitivity_{metric}")


def ranking_cost(rows):
    methods = ("NDFS", "JMVFG", "CvLP", "CrossAlign")
    styles = {
        "NDFS": ("NDFS", "#727C83", "s"),
        "JMVFG": ("JMVFG", "#546168", "P"),
        "CvLP": ("CvLP-DCL", "#364048", "D"),
        "CrossAlign": ("Ours", "#174B78", "o"),
    }
    fig, ax = plt.subplots(figsize=(5.15, 2.32))
    fig.subplots_adjust(left=.125, right=.985, top=.956, bottom=.188)
    for index, dataset in enumerate(DATASETS):
        baseline_costs = [float(rows[dataset, method]["search_seconds"]) for method in methods[:-1]]
        ours = float(rows[dataset, "CrossAlign"]["one_run_seconds"])
        ax.hlines(index, min(baseline_costs), max(baseline_costs), color="#858D93", linewidth=1)
        for method, seconds in zip(methods, (*baseline_costs, ours)):
            label, color, marker = styles[method]
            ax.plot(seconds, index, marker=marker, color=color, ls="none",
                    ms=4.7 if method == "CrossAlign" else 3.65,
                    label=label if index == 0 else None)
        ratio = min(baseline_costs) / ours
        ratio_text = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}"
        ax.annotate(ratio_text + "\u00d7", (max(baseline_costs), index), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=6.7)
    ax.set_xscale("log")
    ax.set(xlim=(.4, 8400), ylim=(len(DATASETS) - .58, -.75), yticks=range(len(DATASETS)),
           yticklabels=DATASETS, xlabel="Ranking time (seconds, log scale)")
    ax.set_xticks([1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(axis="x", color="#E1E6EC", lw=.52)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=3)
    legend = ax.legend(loc="upper right", ncols=1, borderaxespad=.46, borderpad=.47,
                       labelspacing=.36, fontsize=7.1, handlelength=1.2, handletextpad=.55,
                       frameon=True, framealpha=1, edgecolor="#D7DDE4", fancybox=False)
    legend.get_frame().set_linewidth(.56)
    save(fig, "fig_cost")


def embedding_panels():
    points = benchmark_rows("visual_embeddings.csv")
    for dataset in ("BBCSport",):
        for representation, suffix in REPRESENTATIONS.items():
            subset = [row for row in points
                      if (row["dataset"], row["representation"]) == (dataset, representation)]
            assert subset, (dataset, representation)
            fig = plt.figure(figsize=(1.68, 1.42))
            ax = fig.add_axes((.014, .018, .972, .964))
            classes = sorted({int(row["class_index"]) for row in subset})
            for group in classes:
                members = [row for row in subset if int(row["class_index"]) == group]
                ax.scatter([float(row["x"]) for row in members],
                           [float(row["y"]) for row in members], s=6.9, alpha=.78,
                           color=CLASS_COLORS[group-1], marker=CLASS_SYMBOLS[group-1],
                           edgecolors="white", linewidths=.14)
            ax.margins(.045)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            save(fig, f"fig_embedding_{dataset.lower()}_{suffix}")

def benchmark_panels():
    per_budget = {(r["dataset"], r["method"], float(r["budget"])): r
                  for r in benchmark_rows("reference_per_budget.csv")}
    main_rows = {(r["dataset"], r["method"]): r for r in benchmark_rows("main.csv")}
    corrupted = {(r["dataset"], float(r["frac"]), r["method"]): float(r["acc"])
                 for r in benchmark_rows("ESM_1.csv") if r["mode"] == "one"}
    clean_benchmark_panels(per_budget)
    artifact_panels(main_rows, corrupted)
    return main_rows


def main():
    main_rows = benchmark_panels()
    ranking_cost(main_rows)
    sensitivity_panels()
    embedding_panels()


if __name__ == "__main__":
    main()

