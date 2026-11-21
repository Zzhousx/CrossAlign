
"""Plot feature redundancy and clustering NMI."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from target_diagnostics import config, read_rows

from common import result_file
OUTPUT = Path(config.FIGURES)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.4,"axes.titlesize":8,
                     "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.55,
                     "pdf.fonttype":42,"axes.edgecolor":"#7f8994","text.color":"#243447"})
def summarise_redundancy(rows):
    methods=("CrossAlign","Relevance only")
    expected={(d,p,m) for d in config.DATASETS for p in config.RATIOS for m in methods}
    items={(r["dataset"],float(r["budget"]),r["method"]):r for r in rows}
    assert len(rows)==len(items) and set(items)==expected
    for (d,p,m),r in items.items():
        assert r["features"]==items[d,p,"CrossAlign"]["features"]
        assert int(r["nonconstant"])<=int(r["features"])
    cols=tuple(config.RATIOS)
    metrics=("mean_squared_correlation","nmi")
    matrices={}
    for metric in metrics:
        matrices[metric]=100*np.array([[float(items[d,p,"CrossAlign"][metric])
                               - float(items[d,p,"Relevance only"][metric])
                               for p in cols] for d in config.DATASETS])
        assert np.isfinite(matrices[metric]).all()
    panels = (
        (metrics[0], "correlation", r"$\Delta$ squared correlation", "BrBG_r"),
        (metrics[1], "nmi", r"$\Delta$ NMI", "PuOr"),
    )
    for metric, suffix, colorbar_label, cmap in panels:
        fig, ax = plt.subplots(figsize=(2.48, 2.38))
        fig.subplots_adjust(left=.223, right=.840, bottom=.166, top=.969)
        values = matrices[metric]
        limit = max(float(np.abs(values).max()), .1)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        im = ax.pcolormesh(values, cmap=cmap, norm=norm, shading="flat",
                           edgecolors=(1, 1, 1, .5), linewidth=.30)
        ax.set_ylim(len(config.DATASETS), 0)
        ax.set_xticks(np.arange(len(cols)) + .5, [str(round(p * 100)) for p in cols])
        ax.set_yticks(np.arange(len(config.DATASETS)) + .5, config.DATASETS)
        ax.tick_params(length=0, labelsize=6.8, pad=3.4)
        ax.set_xlabel("Feature budget (%)", labelpad=4)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                text = f"{value:+.1f}" if abs(value) >= .05 else "0.0"
                color = "white" if abs(value) > .65 * limit else "#183044"
                ax.text(j + .5, i + .5, text, ha="center", va="center", fontsize=6.1, color=color)
        colorbar = fig.colorbar(im, ax=ax, fraction=.077, pad=.05,
                                ticks=[-limit, 0, limit], format="%.1f")
        colorbar.ax.tick_params(labelsize=6.0, length=1.6, pad=1.2)
        colorbar.outline.set_linewidth(.45)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label(colorbar_label, fontsize=6.9, labelpad=2.0)
        fig.savefig(OUTPUT / f"fig_subset_redundancy_{suffix}.pdf")
        plt.close(fig)
    return {metric:{"mean_points":float(x.mean()),"by_dataset":dict(zip(config.DATASETS,x.mean(axis=1).tolist())),
                    "below_count":int((x<0).sum()),"above_count":int((x>0).sum()),
                    "by_budget":x.tolist()} for metric,x in matrices.items()}


if __name__ == "__main__":
    summary = summarise_redundancy(read_rows(result_file("subset_redundancy.csv")))
    (OUTPUT / "subset_redundancy.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
