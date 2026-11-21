"""Run the benchmark pipeline."""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGES = ["s0_check_data", "s1_tune_baselines", "s2_main_eval", "s3_robustness", "s4_ablation",
          "s5_sensitivity", "s6_runtime", "s7_report", "target_diagnostics", "bbc_case_study",
          "manuscript_plots", "plot_target_diagnostics", "s8_verify"]


def main():
    wanted = sys.argv[1:]
    todo = [s for s in STAGES if not wanted or any(s.startswith(w) for w in wanted)]
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(var, "2")
    print("Numerical threads: " + ", ".join(f"{key}={env[key]}" for key in
          ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), flush=True)
    for s in todo:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {s} ...", flush=True)
        with open(os.path.join(ROOT, "logs", f"{s}.log"), "a") as log:
            r = subprocess.run([sys.executable, os.path.join(ROOT, "stages", f"{s}.py")], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, env=env)
        mins = (time.time() - t0) / 60
        if r.returncode != 0:
            print(f"    {s} exited with code {r.returncode} after {mins:.1f} min; see logs/{s}.log", flush=True)
            sys.exit(r.returncode)
        else:
            print(f"    done in {mins:.1f} min", flush=True)


if __name__ == "__main__":
    main()
