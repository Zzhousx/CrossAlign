"""Fetch the six benchmark files from their public GitHub repositories and verify their SHA-256."""
import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC, RAW = os.path.join(ROOT, "data", "src"), os.path.join(ROOT, "data", "raw")

REPOS = {
    "JMVFG": ("https://github.com/huangdonghere/JMVFG", "db570360df6d55e313b297879801564441518e03"),
    "TBGL-MVC": ("https://github.com/xdweixia/TBGL-MVC", "a4bdd9baf4475a94d506191750fb638f45c86407"),
    "FastMICE": ("https://github.com/huangdonghere/FastMICE", "7feccc0cd035812e16da78265464838b3c2db282"),
    "2019-ICCV-RMSL": ("https://github.com/XLearning-SCU/2019-ICCV-RMSL", "293c61bd5885b87614fce21593b322cee53eb448"),
    "DMF_MVC": ("https://github.com/hdzhao/DMF_MVC", "4bc8aebae740e7ff0e1e27d7f3609dfdc2f35fee"),
}

FILES = {  # target name: (repository, path inside repository, sha256)
    "HW.mat": ("JMVFG", "data_HW.mat", "a3c82ea6e0817853fc2f752852c287683b565165859699329c07f8424beb7670"),
    "OutScene.mat": ("FastMICE", "data_OutScene.mat", "4a8cf3ab114a0f988772893ed4261bfea86fb78bed3bf0709cb39ee5e28a26f9"),
    "ORL.mat": ("2019-ICCV-RMSL", "dataset/ORL_mtv.mat", "30902bfb39210839c7bfdfaa0946fd8dc53eddbaf5a9de5a6f5d5cd2c8a75fcb"),
    "Yale.mat": ("DMF_MVC", "data/yale_mtv.mat", "3ba47c3cd20679af025b0011e33f63975e9939ce294df7f807be01d814c4c9e9"),
    "BBCSport.mat": ("2019-ICCV-RMSL", "dataset/bbcsport_2view.mat",
                     "5d1c51961e0c126e589b7be44181679fa0b70095fb8c27c89a67292e5575ffa4"),
    "MNIST4.mat": ("TBGL-MVC", "dataset/mnist4.mat", "dc3dd7c000c5c44b43d1ea09498b01cac5af0a6276bdfd0872cede987516bd90"),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a, cwd=None):
    subprocess.run(["git", *a], cwd=cwd, check=True)


def main():
    os.makedirs(SRC, exist_ok=True); os.makedirs(RAW, exist_ok=True)
    need = {repo for name, (repo, _, digest) in FILES.items()
            if not (os.path.exists(os.path.join(RAW, name)) and sha256(os.path.join(RAW, name)) == digest)}
    for repo in sorted(need):
        url, commit = REPOS[repo]
        dest = os.path.join(SRC, repo)
        if not os.path.isdir(os.path.join(dest, ".git")):
            print(f"cloning {url}", flush=True)
            git("clone", "--quiet", url, dest)
        git("checkout", "--quiet", commit, cwd=dest)
    bad = 0
    for name, (repo, rel, digest) in FILES.items():
        target = os.path.join(RAW, name)
        if not os.path.exists(target) or sha256(target) != digest:
            src = os.path.join(SRC, repo, rel)
            if os.path.exists(src):
                shutil.copyfile(src, target)
        ok = os.path.exists(target) and sha256(target) == digest
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name:14s} {digest[:16]}...", flush=True)
    if bad:
        print(f"{bad} file(s) failed verification.")
        sys.exit(1)
    print("All six datasets are verified.")


if __name__ == "__main__":
    main()
