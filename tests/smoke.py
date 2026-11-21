"""End-to-end smoke test on tiny synthetic data (about 1-3 minutes)."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT = {  # name: (feature key, label key, samples in columns, view dims, text)
    "HW": ("X", "Y", False, [14, 10, 8], False),
    "OutScene": ("fea", "gt", False, [16, 8, 8], False),
    "ORL": ("X", "gt", True, [14, 12, 10], False),
    "Yale": ("X", "gt", True, [12, 12, 12], False),
    "BBCSport": ("X", "gt", True, [20, 16], True),
    "MNIST4": ("X", "Y", False, [12, 10, 10], False),
}


def synth(dims, text, rng, n=60, c=3):
    y = np.repeat(np.arange(c), n // c)
    views = []
    for d in dims:
        means = rng.rand(c, d) * 3
        X = means[y] + rng.randn(n, d)
        if text:
            X = rng.poisson(np.exp(np.clip(X, -2, 2)))
        views.append(X.astype(float))
    return views, y + 1


def write_data(folder):
    rng = np.random.RandomState(0)
    for name, (fk, lk, cols, dims, text) in LAYOUT.items():
        views, y = synth(dims, text, rng)
        cell = np.empty((1, len(views)), dtype=object)
        for v, X in enumerate(views):
            cell[0, v] = X.T if cols else X
        sio.savemat(os.path.join(folder, f"{name}.mat"), {fk: cell, lk: y.reshape(-1, 1)})


def check_constant_features():
    sys.path.insert(0, ROOT)
    from cam.crossalign import crossalign, greedy_alignment

    centred = np.array([[-1., 0., -1.], [1., 0., 1.], [-1., 0., 1.], [1., 0., -1.]])
    ranking, trace = greedy_alignment(centred, np.array([4., 0., 0.]), K=3)
    np.testing.assert_array_equal(ranking, [0, 2, 1])
    assert trace.size == 2 and np.all(np.isfinite(trace))

    ranking, trace = greedy_alignment(np.zeros((4, 3)), np.zeros(3), K=5)
    np.testing.assert_array_equal(ranking, [0, 1, 2])
    assert trace.size == 0

    rng = np.random.RandomState(0)
    views = [np.column_stack((rng.normal(size=(12, 2)), np.full(12, .1))),
             np.column_stack((np.zeros(12), rng.normal(size=12)))]
    for mode in ("align", "ratio", "relevance"):
        ranking = crossalign(views, 2, k=3, select=mode)
        np.testing.assert_array_equal(ranking[-2:], [2, 3])
        np.testing.assert_array_equal(np.sort(ranking), np.arange(5))


def main():
    check_constant_features()
    tmp = tempfile.mkdtemp(prefix="cam_smoke_")
    for sub in ("data", "results", "tables", "figures"):
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
    write_data(os.path.join(tmp, "data"))
    env = dict(os.environ, CAM_SMOKE="1", CAM_DATA=os.path.join(tmp, "data"),
               CAM_RESULTS=os.path.join(tmp, "results"), CAM_TABLES=os.path.join(tmp, "tables"),
               CAM_FIGURES=os.path.join(tmp, "figures"))
    stages = ["s0_check_data", "s1_tune_baselines", "s2_main_eval", "s3_robustness", "s4_ablation",
              "s5_sensitivity", "s6_runtime", "s7_report"]
    for s in stages:
        print(f"--- {s}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "stages", f"{s}.py")], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            print(f"SMOKE TEST FAILED at {s}")
            sys.exit(1)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)")
    needed = ["main_acc.md", "main_nmi.md", "ablation.md", "robustness_acc.md", "sensitivity.md", "runtime.md",
              "summary.json"]
    missing = [f for f in needed if not os.path.exists(os.path.join(tmp, "tables", f))]
    if missing:
        print("missing outputs:", missing)
        print("SMOKE TEST FAILED at outputs")
        sys.exit(1)
    print(f"SMOKE TEST PASSED (scratch folder: {tmp})")


if __name__ == "__main__":
    main()
