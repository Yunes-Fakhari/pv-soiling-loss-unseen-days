# -*- coding: utf-8 -*-
"""Index-to-loss formula, eq. (3): L_f = a I + b I sigma + c, least squares, limited to [0, 1].

Fitted on the 12 training and validation days (9,878 images) for the final method, and on the 10 training days
(8,390 images) for setting the combination weights. Prints the coefficients and the leave-one-day-out MAE.
  python scripts/fit_formula.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PF = pd.read_csv(ROOT / "splits" / "panel_features.csv").drop_duplicates("relpath").set_index("relpath")


def design(df):
    p = PF.loc[df.relpath]
    return np.c_[p.panel_index, p.panel_index * p.sun, np.ones(len(df))]


def fit(df):
    return np.linalg.lstsq(design(df), df.loss.values, rcond=None)[0]


def predict(coef, df):
    return np.clip(design(df) @ coef, 0, 1)


def lodo_mae(df):
    err = np.zeros(len(df))
    for d in df.date.unique():
        k = (df.date == d).values
        err[k] = np.abs(predict(fit(df[~k]), df[k]) - df.loss.values[k])
    return err.mean()


if __name__ == "__main__":
    out = {}
    for name, split in [("12_days", "splits_v5_trainval"), ("10_days", "splits_v5")]:
        tr = pd.read_csv(ROOT / "splits" / split / "train.csv")
        c = fit(tr)
        out[name] = {"a_I": c[0], "b_I_sigma": c[1], "c": c[2], "images": len(tr), "lodo_MAE": lodo_mae(tr)}
        print(f"{name}: L_f = {c[0]:.3f} I + {c[1]:.3f} I*sigma + {c[2]:.3f}   "
              f"({len(tr)} images, leave-one-day-out MAE {out[name]['lodo_MAE']:.3f})")
    (ROOT / "outputs" / "formula").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "outputs" / "formula" / "coefficients.json", "w"), indent=2)
