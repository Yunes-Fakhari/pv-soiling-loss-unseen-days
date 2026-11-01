# -*- coding: utf-8 -*-
"""Train the MobileNet v1 network of the paper (C-Mixup recipe, Section II-C/D).

Runs (each: MobileNet v1 192x192, image + irradiance, 10 epochs, SWA of epochs 8-10, one network):
  final_s42     12 days (training + validation days), seed 42  -> the network used in the paper
  final_s7      12 days, seed 7      } extra seeds, to measure the spread of the training
  final_s123    12 days, seed 123    }
  val10_s42     10 training days only, seed 42 -> its predictions on the validation days
                (runs/val10_s42/predictions/val_swa.csv) can set the combination weights

  python train_network.py --runs final_s42 --data-root path/to/dataset_raw
  python train_network.py --smoke                 # 10-minute test on 48 images
Results: runs/<name>/ (model/, predictions/, tables/, metrics.json). A finished run is never repeated.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ne import experiments as X  # noqa: E402
from ne.common import fmt_hours, now, safe_console, save_json  # noqa: E402

CMIXUP = {"cmixup": {"alpha": 2.0, "bandwidth": 0.05}}   # partner probability exp(-(y_i - y_j)^2 / 0.005)
RUNS = {"final_s42": ("splits_v5_trainval", 42), "final_s7": ("splits_v5_trainval", 7),
        "final_s123": ("splits_v5_trainval", 123), "val10_s42": ("splits_v5", 42)}


def main():
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["final_s42"], choices=list(RUNS))
    ap.add_argument("--data-root", default=None, help="folder that contains PanelImages/")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rd = HERE / ("runs_smoke" if a.smoke else "runs")
    for name in a.runs:
        d = rd / name
        if (d / "DONE.json").exists():
            print(f"{name}: already done"); continue
        split, seed = RUNS[name]
        cfg = X.config_for(CMIXUP); cfg["split"] = split; cfg["seed"] = seed
        if a.smoke:
            cfg = X.smoke_overrides(cfg)
        d.mkdir(parents=True, exist_ok=True)
        save_json({"name": name, "description": f"C-Mixup, {split}, seed {seed}", "config": cfg}, d / "request.json")
        cmd = [sys.executable, "-m", "ne.train_one", "--run-dir", str(d)] + (["--data-root", a.data_root] if a.data_root else [])
        t0 = time.time(); print(f"[{now()}] start {name}", flush=True)
        r = subprocess.call(cmd, cwd=str(HERE), env=dict(os.environ, PYTHONIOENCODING="utf-8:replace", TF_CPP_MIN_LOG_LEVEL="2"))
        print(f"[{now()}] {name}: {'ok' if r == 0 else 'FAILED'} after {fmt_hours(time.time() - t0)}", flush=True)


if __name__ == "__main__":
    main()
