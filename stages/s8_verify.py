"""Compare benchmark results with the reference results."""
import csv
import json
import os
import sys

from common import config, read

TOL = {"crossalign": 0.005, "baseline": 0.02, "robust_ca": 0.01, "robust_base": 0.03}
CA_FAMILY = {"CrossAlign", "Align-self"}
# UDFS eigenbases can vary across BLAS builds.
PLATFORM_DEPENDENT = {("ORL", "UDFS"), ("Yale", "UDFS"), ("BBCSport", "UDFS")}


def load_csv(name):
    p = os.path.join(config.EXPECTED, name)
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No reference results in {p}")
    return rows


def verdict(diff, tol):
    d = abs(diff)
    return "PASS" if d <= tol else ("WARN" if d <= 3 * tol else "FAIL")


def main():
    lines, counts = [], {"PASS": 0, "WARN": 0, "FAIL": 0, "MISSING": 0}

    def add(status, text):
        counts[status] += 1
        if status != "PASS":
            lines.append(f"- {status}: {text}")

    got = {(r["dataset"], r["method"]): r for r in read("main.jsonl")}
    for e in load_csv("main.csv"):
        k = (e["dataset"], e["method"])
        if k not in got:
            add("MISSING", f"main {k}"); continue
        tol = TOL["crossalign"] if e["method"] == "CrossAlign" else TOL["baseline"]
        for metric in ("acc", "nmi"):
            diff = got[k][metric] - float(e[metric])
            status = verdict(diff, tol)
            note = ""
            if k in PLATFORM_DEPENDENT and status == "FAIL":
                status, note = "WARN", " (UDFS with d > n: platform-dependent eigenbasis, expected to vary)"
            add(status, f"main {k} {metric}: got {got[k][metric]:.4f}, expected {float(e[metric]):.4f}{note}")
        if e.get("params") and e["params"] != "null":
            if json.dumps(got[k]["params"], sort_keys=True) != json.dumps(json.loads(e["params"]), sort_keys=True):
                add("WARN", f"selection {k}: got {got[k]['params']}, expected {e['params']}")

    got = {(r["dataset"], r["mode"], float(r["frac"]), r["method"]): r for r in read("robustness.jsonl")}
    for e in load_csv("ESM_1.csv"):
        k = (e["dataset"], e["mode"], float(e["frac"]), e["method"])
        if k not in got:
            add("MISSING", f"robustness {k}"); continue
        tol = TOL["robust_ca"] if e["method"] in CA_FAMILY else TOL["robust_base"]
        for metric, actual, expected in (
            ("acc", got[k]["acc"], float(e["acc"])),
            ("artifact share at 10%", got[k]["share"]["0.1"], float(e["artifact_share_at_0.1"])),
        ):
            add(verdict(actual - expected, tol),
                f"robustness {k} {metric}: got {actual:.4f}, expected {expected:.4f}")

    got = {(r["dataset"], r["variant"]): r for r in read("ablation.jsonl")}
    for e in load_csv("ablation.csv"):
        k = (e["dataset"], e["variant"])
        if k not in got:
            add("MISSING", f"ablation {k}"); continue
        for metric in ("acc", "nmi"):
            diff = got[k][metric] - float(e[metric])
            add(verdict(diff, TOL["crossalign"]), f"ablation {k} {metric}: got {got[k][metric]:.4f}, "
                                                  f"expected {float(e[metric]):.4f}")

    got = {(r["dataset"], int(r["k"])): r for r in read("sensitivity.jsonl")}
    for e in load_csv("sensitivity.csv"):
        k = (e["dataset"], int(e["k"]))
        if k not in got:
            add("MISSING", f"sensitivity {k}"); continue
        diff = got[k]["acc"] - float(e["acc"])
        add(verdict(diff, TOL["crossalign"]), f"sensitivity {k} acc: got {got[k]['acc']:.4f}, "
                                              f"expected {float(e['acc']):.4f}")

    head = (f"# Verification report\n\nPASS {counts['PASS']} | WARN {counts['WARN']} | FAIL {counts['FAIL']} | "
            f"MISSING {counts['MISSING']}\n\nTolerances (absolute, metric in [0, 1]): {TOL}\n\n")
    body = "\n".join(lines) if lines else "All compared values are within tolerance."
    with open(os.path.join(config.ROOT, "verify_report.md"), "w") as f:
        f.write(head + body + "\n")
    print(head + body)
    return int(bool(counts["FAIL"] or counts["MISSING"]))


if __name__ == "__main__":
    sys.exit(main())
