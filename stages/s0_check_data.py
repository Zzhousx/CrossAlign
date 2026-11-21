"""Stage 0: confirm that every dataset loads, and record its size (results/datasets.json)."""
import json

from common import config, log, path, args
from cam.data import describe

if __name__ == "__main__":
    a = args()
    rows = []
    for ds in a.datasets:
        d = describe(ds)
        rows.append(d)
        log(f"{ds:9s} n={d['n']:5d} V={d['V']} D={d['D']:6d} c={d['c']:3d} dims={d['dims']} "
            f"| smallest budget {int(0.1 * d['D'])} features for {d['c']} clusters")
        assert int(0.1 * d["D"]) >= d["c"], f"{ds}: smallest budget below c"
    with open(path("datasets.json"), "w") as f:
        json.dump(rows, f, indent=1)
    log(f"OK: {len(rows)} datasets; written {path('datasets.json')}")
