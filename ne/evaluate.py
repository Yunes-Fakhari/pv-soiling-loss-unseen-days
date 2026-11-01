# -*- coding: utf-8 -*-
"""Metrics (same definitions as the paper), prediction tables and figures of one run."""
import time

import numpy as np
import pandas as pd

EDGES = [0.2, 0.4, 0.6, 0.8]                       # 5 loss ranges: 0-20, 20-40, 40-60, 60-80, 80-100 %
RANGES = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def metrics(y, p, min_count=1):
    """MSE, RMSE, MAE, R2, bias (mean of prediction - truth), bMAE (mean of the 5 per-range MAEs),
    share of images with |error| <= 0.05 / 0.10 / 0.15, and per-range MAE / bias / count."""
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    e = p - y
    b = np.digitize(y, EDGES)
    per_mae, per_bias, per_n = [], [], []
    for k in range(5):
        m = b == k
        per_n.append(int(m.sum()))
        per_mae.append(float(np.abs(e[m]).mean()) if m.sum() >= min_count else float("nan"))
        per_bias.append(float(e[m].mean()) if m.sum() >= min_count else float("nan"))
    ss = ((y - y.mean()) ** 2).sum()
    valid = [v for v in per_mae if not np.isnan(v)]
    out = {"n": int(len(y)), "MSE": float((e ** 2).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
           "MAE": float(np.abs(e).mean()), "R2": float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"),
           "bias": float(e.mean()), "bMAE": float(np.mean(valid)) if valid else float("nan"),
           "within_0.05": float((np.abs(e) <= 0.05).mean() * 100),
           "within_0.10": float((np.abs(e) <= 0.10).mean() * 100),
           "within_0.15": float((np.abs(e) <= 0.15).mean() * 100)}
    for k in range(5):
        out[f"MAE_{RANGES[k]}"] = per_mae[k]
        out[f"bias_{RANGES[k]}"] = per_bias[k]
        out[f"n_{RANGES[k]}"] = per_n[k]
    return out


def per_day(df, pred):
    rows = []
    d = df.assign(pred=pred, err=pred - df["loss"].to_numpy())
    for day, g in d.groupby("date"):
        e = g["err"].to_numpy()
        rows.append({"date": day, "n": len(g), "MSE": float((e ** 2).mean()), "MAE": float(np.abs(e).mean()),
                     "bias": float(e.mean()), "within_0.15": float((np.abs(e) <= 0.15).mean() * 100)})
    return pd.DataFrame(rows)


_FWD = {}


def predict(model, ds):
    import tensorflow as tf
    if id(model) not in _FWD:                       # one compiled function per model (no retracing)
        @tf.function(reduce_retracing=True)
        def fwd(img, irr):
            return model({"image": img, "irr": irr}, training=False)
        _FWD[id(model)] = (model, fwd)
    fwd = _FWD[id(model)][1]
    out = [fwd(img, irr).numpy().ravel() for img, irr in ds]
    return np.concatenate(out) if out else np.zeros(0)


def latency_ms(model, pipe, df, n=60, warmup=10):
    """Mean time per image, batch size 1, one forward pass (CPU)."""
    ds = pipe.eval(df.head(n + warmup), batch=1)
    times = []
    for i, (img, irr) in enumerate(ds):
        t0 = time.perf_counter()
        model({"image": img, "irr": irr}, training=False).numpy()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return round(1000 * float(np.mean(times)), 2) if times else float("nan")


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_training(hist, path, title):
    plt = _plt()
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
    e = hist["epoch"]
    ax[0].plot(e, hist["train_loss"], "o-", ms=3)
    ax[0].set_title("Training loss")
    ax[1].plot(e, hist["val_MSE"], "o-", ms=3, color="#c05000")
    ax[1].set_title("Validation MSE")
    ax[2].plot(e, hist["val_bMAE"], "o-", ms=3, color="#2a7")
    ax[2].plot(e, hist["val_MAE"], "s--", ms=3, color="#777", label="MAE")
    ax[2].legend(["bMAE", "MAE"], frameon=False)
    ax[2].set_title("Validation bMAE / MAE")
    ax[3].plot(e, hist["lr"], "o-", ms=3, color="#555")
    ax[3].set_title("Learning rate")
    ax[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for a in ax:
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_results(df, pred, path_prefix, title):
    """Scatter, error histogram, per-range MAE/bias and time series of the given test set."""
    plt = _plt()
    y = df["loss"].to_numpy()
    e = pred - y
    # scatter + error histogram + cumulative
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].fill_between([0, 1], [-0.15, 0.85], [0.15, 1.15], color="#eee", lw=0)
    ax[0].scatter(y, pred, s=3, alpha=0.4)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="True power loss", ylabel="Predicted", title="Prediction vs truth")
    ax[1].hist(e, bins=60, range=(-0.6, 0.6), color="#4a7")
    ax[1].set(xlabel="Prediction error", ylabel="Images", title="Error distribution")
    xs = np.linspace(0, 0.4, 200)
    ax[2].plot(xs, [(np.abs(e) <= x).mean() * 100 for x in xs])
    for t in (0.05, 0.10, 0.15):
        v = (np.abs(e) <= t).mean() * 100
        ax[2].plot([t], [v], "o", color="#8C4A00", ms=3)
        ax[2].annotate(f"{v:.1f}% within {t:.2f}", (t, v), xytext=(t + 0.03, v - 8), fontsize=7)
    ax[2].set(xlabel="Absolute error", ylabel="Images within error (%)", ylim=(0, 102), title="Cumulative")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{path_prefix}_accuracy.png", dpi=150)
    plt.close(fig)
    # per range
    m = metrics(y, pred)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].bar(RANGES, [m[f"MAE_{r}"] for r in RANGES], color="#47a")
    ax[0].set_title("MAE per loss range")
    ax[1].bar(RANGES, [m[f"bias_{r}"] for r in RANGES], color="#a64")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_title("Bias per loss range (pred - true)")
    for a in ax:
        a.tick_params(axis="x", labelsize=7)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{path_prefix}_per_range.png", dpi=150)
    plt.close(fig)
    # time series per day
    days = sorted(df["date"].unique())
    fig, ax = plt.subplots(1, len(days), figsize=(4.2 * len(days), 3), squeeze=False)
    for a, day in zip(ax[0], days):
        g = df["date"] == day
        t = pd.to_datetime(df.loc[g, "datetime"])
        hrs = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
        a.plot(hrs, y[g.to_numpy()], ".", ms=2, color="#999", label="True")
        a.plot(hrs, pred[g.to_numpy()], ".", ms=2, color="#c05000", label="Predicted")
        a.set(title=str(day), xlabel="Hour", ylim=(-0.02, 1.02))
        a.grid(alpha=0.3)
    ax[0][0].set_ylabel("Power loss")
    ax[0][0].legend(frameon=False, markerscale=4)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{path_prefix}_timeseries.png", dpi=150)
    plt.close(fig)
