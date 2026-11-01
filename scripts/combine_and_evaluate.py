# -*- coding: utf-8 -*-
"""Combination weights (Section II-G) and every test number of the paper (Section III).

1. Weights: a network trained on the 10 training days and the formula fitted on the same days predict the two
   validation days; each estimate is weighted by the inverse of its MSE there (network 0.0275, formula 0.0129,
   network weight 0.32, rounded to 0.3). The test days are not used for this.
2. Proposed method, eq. (4): L = 0.3 L_net + 0.7 L_f, with the final network (12 days) and the formula fitted on
   the 12 days.
3. Metrics on the full and the clean test set, per test day, a bootstrap over half-hour blocks, and the error
   analysis of the hardest images.
  python scripts/combine_and_evaluate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_formula import PF, fit, predict  # noqa: E402

S, O = ROOT / "splits", ROOT / "outputs"


def metrics(p, y):
    e = p - y; b = np.digitize(y, [.2, .4, .6, .8])
    return {"MSE": (e ** 2).mean(), "MAE": np.abs(e).mean(), "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "bMAE": np.mean([np.abs(e)[b == q].mean() for q in range(5)]), "bias": e.mean(),
            "within_0.15_%": 100 * (np.abs(e) <= 0.15).mean()}


def main():
    # ---- 1. weights from the validation days
    tr10, val = pd.read_csv(S / "splits_v5" / "train.csv"), pd.read_csv(S / "splits_v5" / "val.csv")
    nv = pd.read_csv(O / "validation_network_10days" / "val_predictions.csv").set_index("relpath").loc[val.relpath]
    yv = val.loss.values
    mse_net = ((nv.y_pred.values - yv) ** 2).mean()
    mse_f = ((predict(fit(tr10), val) - yv) ** 2).mean()
    w_raw = (1 / mse_net) / (1 / mse_net + 1 / mse_f); w = round(w_raw, 1)
    print(f"validation MSE: network {mse_net:.4f}, formula {mse_f:.4f} -> network weight {w_raw:.3f}, rounded to {w}")

    # ---- 2. proposed method on the test days
    tr12 = pd.read_csv(S / "splits_v5_trainval" / "train.csv")
    test = pd.read_csv(S / "splits_v5_trainval" / "test.csv").drop_duplicates("relpath")
    clean = set(pd.read_csv(S / "splits_v5_trainval" / "test_clean.csv").relpath)
    net = pd.read_csv(O / "final_network" / "test_predictions.csv").drop_duplicates("relpath").set_index("relpath")
    y = test.loss.values; L_net = net.loc[test.relpath, "pred"].values
    L_f = predict(fit(tr12), test); L = np.clip(w * L_net + (1 - w) * L_f, 0, 1)
    k = test.relpath.isin(clean).values
    pr = test[["relpath", "date", "datetime"]].assign(y_true=y, L_net=L_net, I=PF.loc[test.relpath, "panel_index"].values,
                                                       L_f=L_f, L=L, in_clean_test=k)
    (O / "proposed_method").mkdir(parents=True, exist_ok=True)
    pr.to_csv(O / "proposed_method" / "test_predictions.csv", index=False)

    res = {"weights": {"val_MSE_network": mse_net, "val_MSE_formula": mse_f, "network_weight_raw": w_raw,
                       "network_weight": w, "formula_weight": round(1 - w, 1)}}
    for nm, m in [("full", np.ones(len(y), bool)), ("clean", k)]:
        res[nm] = {"network": metrics(L_net[m], y[m]), "formula": metrics(L_f[m], y[m]), "proposed": metrics(L[m], y[m])}
        r = 1 - res[nm]["proposed"]["MSE"] / res[nm]["network"]["MSE"]
        res[nm]["MSE_reduction_%"] = 100 * r
        print(f"{nm:5s} test: network MSE {res[nm]['network']['MSE']:.4f} | formula {res[nm]['formula']['MSE']:.4f} | "
              f"proposed {res[nm]['proposed']['MSE']:.4f} (-{100 * r:.0f}%), R2 {res[nm]['proposed']['R2']:.3f}, "
              f"MAE {res[nm]['proposed']['MAE']:.3f}, bMAE {res[nm]['proposed']['bMAE']:.3f}, "
              f"within 0.15: {res[nm]['proposed']['within_0.15_%']:.1f}%")
    en, ep = (L_net - y) ** 2, (L - y) ** 2
    res["per_day_MSE"] = {d: {"network": en[test.date.values == d].mean(), "proposed": ep[test.date.values == d].mean()}
                          for d in sorted(test.date.unique())}
    for d, v in res["per_day_MSE"].items():
        print(f"  {d}: network {v['network']:.4f} -> proposed {v['proposed']:.4f}")
    # bootstrap over half-hour blocks (consecutive images are correlated)
    t = pd.to_datetime(test.datetime)
    blk = (test.date + "_" + (t.dt.hour * 2 + t.dt.minute // 30).astype(str)).values
    ub = np.unique(blk); idx = {b: np.where(blk == b)[0] for b in ub}; rng = np.random.default_rng(0); boot = []
    for _ in range(5000):
        s = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]); boot.append(1 - ep[s].mean() / en[s].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5]) * 100
    res["bootstrap"] = {"blocks": len(ub), "MSE_reduction_95%_interval": [lo, hi]}
    print(f"  MSE reduction, 95% interval over {len(ub)} half-hour blocks: {lo:.0f}-{hi:.0f}%")
    res["other_weights_full_MSE"] = {f"{a:.1f}": (((np.clip(a * L_net + (1 - a) * L_f, 0, 1) - y) ** 2).mean())
                                     for a in (0.3, 0.4, 0.5, 0.6, 0.7)}
    # ---- 3. hardest images: loss much larger than the soiled area
    if "soiled_area" in PF:
        area = PF.loc[test.relpath, "soiled_area"].values
        hard = k & ((y - area) > 0.3); big = k & (np.abs(L - y) > 0.10)
        mins = (t.dt.hour * 60 + t.dt.minute).values
        noon = k & (test.date.values == "2017-06-30") & (mins >= 12 * 60 + 27) & (mins <= 13 * 60 + 37)
        day30 = k & (test.date.values == "2017-06-30")
        res["hard_cases"] = {"images_loss_above_area_by_0.3": int(hard.sum()), "share_of_clean_%": 100 * hard.mean() / k.mean(),
                             "share_of_squared_error_%": 100 * ep[hard].sum() / ep[k].sum(),
                             "share_of_errors_above_0.10_%": 100 * (hard & big).sum() / big.sum(),
                             "june30_1227_1337": {"images": int(noon.sum()), "of_clean_images_that_day": int(day30.sum()),
                                                  "soiled_area": area[noon].mean(), "true_loss": y[noon].mean(),
                                                  "prediction": L[noon].mean(),
                                                  "share_of_day_squared_error_%": 100 * ep[noon].sum() / ep[day30].sum()}}
        print("  hardest images:", json.dumps(res["hard_cases"], default=lambda v: round(float(v), 3)))
    json.dump(res, open(O / "proposed_method" / "metrics.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
