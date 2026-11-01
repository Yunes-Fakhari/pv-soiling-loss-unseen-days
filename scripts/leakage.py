# -*- coding: utf-8 -*-
"""Leakage of image-level splits (Fig. 2 and Table IV of the paper): rules that only COPY a training label.

  random split   SoilingEdge-style random split of single images (72% training, 20% test) over all images:
                 (a) copy the label of the training image taken closest in time
                 (b) copy the label of the most similar training image, found from the pixels alone (no timestamps)
  day split      the same rules with the test days of the paper (June 15, 24, 30) kept apart
  by loss range  12 training and validation days; 20% of the images of each 20-point loss range drawn at random as a
                 test set; rule (b) gives a mean MSE of 0.0018 over ten draws (0.0012-0.0026), as in the paper

  python scripts/leakage.py --data-root path/to/dataset_raw
No model is trained: the numbers show how far a rule that learns nothing gets on each split.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"solar_\w{3}_(\w{3})_(\d+)_(\d+)__(\d+)__(\d+)_(\d{4})_L_([0-9.eE-]+)_I_")
MON = {"Jun": 6, "Jul": 7, "May": 5}


def all_images(data_root):
    rows = []
    for f in sorted((Path(data_root) / "PanelImages").glob("*.jpg")):
        m = PAT.match(f.name)
        if m:
            mo, d, h, mi, s, yr, loss = m.groups()
            rows.append((f"PanelImages/{f.name}", pd.Timestamp(int(yr), MON[mo], int(d), int(h), int(mi), int(s)), float(loss)))
    df = pd.DataFrame(rows, columns=["relpath", "datetime", "loss"])
    df["date"] = df.datetime.dt.strftime("%Y-%m-%d")
    return df


def thumbs(data_root, relpaths):
    return np.stack([np.asarray(Image.open(Path(data_root) / r).convert("RGB").resize((32, 32)), np.float32).ravel() / 255
                     for r in relpaths])


def nearest_time(tr, te):
    t_tr = tr.datetime.values.astype("int64"); o = np.argsort(t_tr); ts = t_tr[o]
    q = te.datetime.values.astype("int64"); k = np.clip(np.searchsorted(ts, q), 1, len(ts) - 1)
    pick = np.where(np.abs(ts[k] - q) < np.abs(ts[k - 1] - q), o[k], o[k - 1])
    return tr.loss.values[pick]


def nearest_pixels(Xtr, ytr, Xte, chunk=500):
    sq = (Xtr ** 2).sum(1); out = []
    for i in range(0, len(Xte), chunk):
        D = (Xte[i:i + chunk] ** 2).sum(1)[:, None] + sq[None] - 2 * Xte[i:i + chunk] @ Xtr.T
        out.append(ytr[D.argmin(1)])
    return np.concatenate(out)


def report(name, p, y):
    e = p - y
    r = {"split": name, "MAE": np.abs(e).mean(), "MSE": (e ** 2).mean(), "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()}
    print(f"{name:45s} MAE {r['MAE']:.3f}  MSE {r['MSE']:.4f}  R2 {r['R2']:.3f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n-test-pixels", type=int, default=1000, help="test images used for the pixel rule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=10, help="random draws of the split by loss range")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); res = []
    df = all_images(a.data_root); print(f"{len(df)} images")
    # random split of single images
    perm = rng.permutation(len(df)); n = len(df)
    tr, te = df.iloc[perm[:int(0.72 * n)]], df.iloc[perm[int(0.80 * n):]]
    res.append(report("random split, nearest time", nearest_time(tr, te), te.loss.values))
    sub = te.iloc[rng.choice(len(te), min(a.n_test_pixels, len(te)), replace=False)]
    res.append(report("random split, most similar pixels", nearest_pixels(thumbs(a.data_root, tr.relpath), tr.loss.values,
                                                                         thumbs(a.data_root, sub.relpath)), sub.loss.values))
    # day split of the paper
    test_days = {"2017-06-15", "2017-06-24", "2017-06-30"}; val_days = {"2017-06-22", "2017-06-27"}
    tr, te = df[~df.date.isin(test_days | val_days)], df[df.date.isin(test_days)]
    res.append(report("day split, nearest time", nearest_time(tr, te), te.loss.values))
    sub = te.iloc[rng.choice(len(te), min(a.n_test_pixels, len(te)), replace=False)]
    res.append(report("day split, most similar pixels", nearest_pixels(thumbs(a.data_root, tr.relpath), tr.loss.values,
                                                                      thumbs(a.data_root, sub.relpath)), sub.loss.values))
    # random within loss ranges, 12 training + validation days (one image per 30 s, labels as in splits/)
    d12 = pd.read_csv(ROOT / "splits" / "splits_v5_trainval" / "train.csv").drop_duplicates("relpath").reset_index(drop=True)
    X = thumbs(a.data_root, d12.relpath); y = d12.loss.values; b = np.digitize(y, [.2, .4, .6, .8])
    r2 = np.random.default_rng(a.seed); draws = []
    for _ in range(a.draws):                      # the MSE varies from draw to draw, so several draws are averaged
        te = np.zeros(len(y), bool)
        for q in range(5):
            i = np.where(b == q)[0]; te[r2.choice(i, int(round(0.2 * len(i))), replace=False)] = True
        draws.append(report("12 days, random within 20-point loss ranges, pixels", nearest_pixels(X[~te], y[~te], X[te]), y[te]))
    m = pd.DataFrame(draws)
    print(f"mean over {a.draws} draws: MSE {m.MSE.mean():.4f} (range {m.MSE.min():.4f}-{m.MSE.max():.4f})")
    res.append({"split": f"12 days, random within loss ranges, pixels, mean of {a.draws} draws",
                "MAE": m.MAE.mean(), "MSE": m.MSE.mean(), "R2": m.R2.mean()})
    out = ROOT / "outputs" / "leakage"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_csv(out / "copy_rules.csv", index=False); print("written", out / "copy_rules.csv")


if __name__ == "__main__":
    main()
