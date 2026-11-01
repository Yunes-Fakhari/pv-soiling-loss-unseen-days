# -*- coding: utf-8 -*-
"""Trains and evaluates ONE experiment. Started by run_all.py as a separate process:
    python -m ne.train_one --run-dir runs/<name>
The folder must contain request.json (the config). Everything this run produces goes into that folder."""
import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from .common import SPLITS, Tee, safe_console, environment_info, find_data_root, keep_awake, load_json, now, save_json, set_seed
from .data import Pipeline, keep_existing, load_split, sample_weights
from .evaluate import RANGES, latency_ms, metrics, per_day, plot_results, plot_training, predict
from .losses import balanced_mse_bmc, dldl_v2, ranksim, weighted_huber, weighted_mse
from .model import average_weights, build, lr_groups


def lr_at(cfg, epoch, step, steps):
    """Cosine decay per epoch, exactly as the paper runs: epoch 1 = lr ... epoch N = min_lr.
    Optional linear warm-up (per step) during the first warmup_epochs, from lr/10 up to lr."""
    n, hi, lo = int(cfg["epochs"]), float(cfg["lr"]), float(cfg["min_lr"])
    base = lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * epoch / max(n - 1, 1)))
    W = int(cfg.get("warmup_epochs", 0))
    if epoch < W:
        frac = (epoch * steps + step + 1) / float(W * steps)
        return hi * (0.1 + 0.9 * frac)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    out = Path(args.run_dir).resolve()
    req = load_json(out / "request.json")
    cfg = req["config"]
    name = req["name"]
    for sub in ("model", "predictions", "tables", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    safe_console()
    log = open(out / "console.log", "w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)
    keep_awake(True)
    tf.get_logger().setLevel("ERROR")
    print(f"[{now()}] Run {name}: {req.get('description', '')}")
    print("Config:", {k: v for k, v in cfg.items()})

    set_seed(int(cfg["seed"]))
    data_root = find_data_root(args.data_root)
    parts = load_split(cfg["split"], cfg.get("max_images"))
    if cfg.get("max_images"):
        parts = keep_existing(parts, data_root)
    for k, v in parts.items():
        print(f"  {k:10s} {len(v):6d} images, {v['date'].nunique():2d} days")
    train_df = parts["train"]
    y_train = train_df["loss"].to_numpy(np.float32)
    w_train = sample_weights(y_train, cfg)
    pipe = Pipeline(cfg, data_root)
    cov_head = bool(cfg.get("coverage_head"))
    cov_all, cov_train = {}, None
    if cov_head:
        from .coverage import get_coverage
        cov_all = get_coverage(cfg["split"], parts, data_root, int(cfg["img_size"]))
        cov_train = np.array([cov_all[p] for p in train_df["relpath"]], np.float32)
        ok = cov_train >= 0
        print(f"Soiled-area ratio of the training images: mean {cov_train[ok].mean() * 100:.1f}%, "
              f"{(~ok).sum()} too dark (ignored); correlation with power loss "
              f"{np.corrcoef(cov_train[ok], y_train[ok])[0, 1]:.2f}")

    aux = cfg.get("aux_targets")
    if aux:
        if cov_head:
            raise ValueError("aux_targets and coverage_head cannot be used together")
        # training-only targets measured by image processing (ne/README: aux_targets.csv):
        # 36 values = soiling of each 32 x 32 block of the 192 px image (-1 = block outside the panel),
        # 1 value = regional soiling index of the panel. Never used at prediction time.
        at = pd.read_csv(SPLITS / aux["file"]).set_index("relpath")
        hcols = [c for c in at.columns if c.startswith("h")]
        tab = at[hcols + ["regional_index"]].astype(np.float32)
        cov_train = np.full((len(train_df), len(hcols) + 1), -1.0, np.float32)
        known = train_df["relpath"].isin(tab.index).to_numpy()
        cov_train[known] = tab.loc[train_df["relpath"][known]].to_numpy()
        NH = len(hcols)
        print(f"Extra training targets: {known.sum()} of {len(train_df)} images; heatmap blocks {NH}; "
              f"regional index vs power loss correlation "
              f"{np.corrcoef(cov_train[known, -1], y_train[known])[0, 1]:.2f}")

    # ---------------- model, teacher, optimisers
    infer, train_net, full = build(cfg, imagenet=not cfg.get("no_imagenet", False))
    print(f"Parameters: {infer.count_params() / 1e6:.3f} M")
    teacher = None
    if cfg.get("distill"):
        tpath = Path(req["teacher_path"])
        teacher = tf.keras.models.load_model(tpath, compile=False)
        teacher.trainable = False
        print(f"Teacher (frozen, used only during training): {tpath}")
    groups = lr_groups(train_net, cfg)                      # all trainable weights (incl. a second head)
    opts = [tf.keras.optimizers.Adam(learning_rate=float(cfg["lr"]) * f) for f, _ in groups]
    gvars = [v for _, vs in groups for v in vs]
    sizes = [len(vs) for _, vs in groups]
    if cfg.get("llrd"):
        print("Layer-wise LR groups (factor, variables):", [(round(f, 4), len(v)) for f, v in groups])
    sigma = None
    if cfg["loss"] == "bmc":
        sigma = tf.Variable(float(cfg["bmc"]["init_sigma"]), dtype=tf.float32, name="bmc_sigma")
        sig_opt = tf.keras.optimizers.Adam(learning_rate=float(cfg["bmc"]["sigma_lr"]))
    loss_kind = cfg["loss"]
    rk = cfg.get("ranksim")
    ds_cfg = cfg.get("distill")

    def main_loss(y, o, w):
        p = o["pred"]
        if loss_kind == "mse":
            return weighted_mse(y, p, w)
        if loss_kind == "huber":
            return weighted_huber(y, p, w, float(cfg["huber_delta"]))
        if loss_kind == "bmc":
            return balanced_mse_bmc(y, p, sigma)
        if loss_kind == "dldl":
            d = cfg["dldl"]
            return dldl_v2(y, o["probs"], p, w, int(d["bins"]), float(d["sigma"]), float(d["l1_weight"]))
        raise ValueError(loss_kind)

    @tf.function(reduce_retracing=True)
    def train_step(img, irr, y, w, c):
        x = {"image": img, "irr": irr}
        with tf.GradientTape(persistent=sigma is not None) as tape:
            o = train_net(x, training=True)
            loss = main_loss(y, o, w)
            if ds_cfg:
                t = tf.stop_gradient(teacher(x, training=False))
                a = float(ds_cfg["alpha"])
                loss = (1 - a) * loss + a * weighted_mse(t, o["pred"], w)
            reg = tf.constant(0.0)
            if rk:
                reg = float(rk["gamma"]) * ranksim(o["feat"], y, float(rk["lam"]))
            if cov_head:                                  # second task: soiled-area ratio (MSE, known images only)
                valid = tf.cast(c >= 0, tf.float32)
                cov_loss = tf.reduce_sum(valid * tf.square(o["cov"] - c)) / tf.maximum(tf.reduce_sum(valid), 1.0)
                reg = reg + float(cfg.get("coverage_weight", 1.0)) * cov_loss
            if aux:                                       # extra training-only targets (masked MSE, -1 = unknown)
                if "heat" in o:
                    th = c[:, :NH]
                    vh = tf.cast(th >= 0, tf.float32)
                    hl = tf.reduce_sum(vh * tf.square(o["heat"] - th)) / tf.maximum(tf.reduce_sum(vh), 1.0)
                    reg = reg + float(aux.get("heat_weight", 0)) * hl
                if "aux_idx" in o:
                    ti = c[:, NH:NH + 1]
                    vi = tf.cast(ti >= 0, tf.float32)
                    il = tf.reduce_sum(vi * tf.square(o["aux_idx"] - ti)) / tf.maximum(tf.reduce_sum(vi), 1.0)
                    reg = reg + float(aux.get("index_weight", 0)) * il
            total = loss + reg
        grads = tape.gradient(total, gvars)
        i = 0
        for opt, n in zip(opts, sizes):
            opt.apply_gradients(zip(grads[i:i + n], gvars[i:i + n]))
            i += n
        if sigma is not None:
            sig_opt.apply_gradients([(tape.gradient(total, sigma), sigma)])
        return total, reg

    # ---------------- train
    E = int(cfg["epochs"])
    keep_from = E - int(cfg["swa_last"]) + 1                  # epochs averaged by SWA
    steps = int(math.ceil(len(train_df) / int(cfg["batch_size"])))
    val_ds = pipe.eval(parts["val"])
    y_val = parts["val"]["loss"].to_numpy()
    hist = []
    t_start = time.perf_counter()
    for ep in range(E):
        t0 = time.perf_counter()
        ds = pipe.train(train_df, w_train, ep, cov_train)
        tot, regs, n = 0.0, 0.0, 0
        for s, (img, irr, y, w, c) in enumerate(ds):
            lr = lr_at(cfg, ep, s, steps)
            for (f, _), opt in zip(groups, opts):
                opt.learning_rate.assign(lr * f)
            l, r = train_step(img, irr, y, w, c)
            tot += float(l)
            regs += float(r)
            n += 1
            if (s + 1) % 25 == 0 or s + 1 == steps:
                el = time.perf_counter() - t0
                print(f"  epoch {ep + 1}/{E} step {s + 1}/{steps}  loss {tot / n:.5f}  lr {lr:.2e}  "
                      f"{el / (s + 1):.2f} s/step  epoch ETA {el / (s + 1) * (steps - s - 1) / 60:.1f} min",
                      flush=True)
        train_time = time.perf_counter() - t0
        pv = predict(infer, val_ds)
        mv = metrics(y_val, pv)
        row = {"epoch": ep + 1, "lr": lr_at(cfg, ep, steps - 1, steps), "train_loss": tot / max(n, 1),
               "train_reg": regs / max(n, 1), "val_MSE": mv["MSE"], "val_MAE": mv["MAE"], "val_bMAE": mv["bMAE"],
               "val_R2": mv["R2"], **{f"val_MAE_{r}": mv[f"MAE_{r}"] for r in RANGES},
               "epoch_time_s": round(train_time, 1), "val_time_s": round(time.perf_counter() - t0 - train_time, 1)}
        if sigma is not None:
            row["bmc_sigma"] = float(sigma.numpy())
        hist.append(row)
        pd.DataFrame(hist).to_csv(out / "tables" / "history.csv", index=False)
        print(f"[{now()}] epoch {ep + 1}: train loss {row['train_loss']:.5f} | val MSE {mv['MSE']:.5f} "
              f"MAE {mv['MAE']:.4f} bMAE {mv['bMAE']:.4f} | {train_time / 60:.1f} min", flush=True)
        if ep + 1 >= keep_from:
            infer.save(out / "model" / f"epoch_{ep + 1:02d}.keras")
            if full is not None:
                full.save(out / "model" / f"epoch_{ep + 1:02d}_two_outputs.keras")
    train_minutes = (time.perf_counter() - t_start) / 60
    hist = pd.DataFrame(hist)
    plot_training(hist, out / "figures" / "training_curves.png", f"{name}")

    # ---------------- the two evaluated models: last epoch and SWA (one network each)
    last = out / "model" / f"epoch_{E:02d}.keras"
    swa_paths = [out / "model" / f"epoch_{e:02d}.keras" for e in range(keep_from, E + 1)]
    swa = average_weights(swa_paths, cfg)
    swa.save(out / "model" / "swa.keras")
    models = {"last": tf.keras.models.load_model(last, compile=False), "swa": swa}

    test_df, clean_df = parts["test"], parts["test_clean"]
    clean_set = set(clean_df["relpath"])
    sets = {"val": parts["val"], "test_full": test_df, "test_clean": clean_df}
    allm, rows = {}, []
    for mk, m in models.items():
        preds = {}
        for sk in ("val", "test_full"):
            preds[sk] = predict(m, pipe.eval(sets[sk]))
        preds["test_clean"] = preds["test_full"][test_df["relpath"].isin(clean_set).to_numpy()]
        cdf = test_df[test_df["relpath"].isin(clean_set)].reset_index(drop=True)
        for sk, df in (("val", parts["val"]), ("test_full", test_df), ("test_clean", cdf)):
            mm = metrics(df["loss"].to_numpy(), preds[sk])
            allm[f"{mk}/{sk}"] = mm
            rows.append({"model": mk, "set": sk, **mm})
        pdf = parts["val"].assign(pred=preds["val"])
        pdf.to_csv(out / "predictions" / f"val_{mk}.csv", index=False)
        tdf = test_df.assign(pred=preds["test_full"], in_clean_test=test_df["relpath"].isin(clean_set))
        tdf.to_csv(out / "predictions" / f"test_{mk}.csv", index=False)
        per_day(cdf, preds["test_clean"]).to_csv(out / "tables" / f"per_day_test_clean_{mk}.csv", index=False)
        per_day(test_df, preds["test_full"]).to_csv(out / "tables" / f"per_day_test_full_{mk}.csv", index=False)
        plot_results(cdf, preds["test_clean"], str(out / "figures" / f"test_clean_{mk}"),
                     f"{name} - {mk} model - clean test days")
        plot_results(parts["val"], preds["val"], str(out / "figures" / f"val_{mk}"),
                     f"{name} - {mk} model - validation days")
    pd.DataFrame(rows).to_csv(out / "tables" / "metrics.csv", index=False)
    cov_info = None
    if full is not None:
        cov_info = evaluate_coverage(out, cfg, keep_from, E, pipe, test_df, clean_set, cov_all)
    lat = latency_ms(models["swa"], pipe, clean_df)
    info = {"name": name, "description": req.get("description", ""), "finished": now(),
            "train_minutes": round(train_minutes, 1),
            "mean_epoch_seconds": float(hist["epoch_time_s"].mean()),
            "train_images": int(len(train_df)), "train_days": int(train_df["date"].nunique()),
            "latency_ms_per_image_cpu_batch1": lat, "parameters": int(infer.count_params()),
            "env": environment_info(), "metrics": allm}
    if sigma is not None:
        info["bmc_sigma_final"] = float(sigma.numpy())
    if cov_info is not None:
        info["soiled_area"] = cov_info
    save_json(info, out / "metrics.json")
    write_summary(out, name, req, cfg, info, hist)
    # keep only what is needed: the averaged model, the last epoch and the epochs it was made from
    save_json({"finished": now()}, out / "DONE.json")
    print(f"[{now()}] Run {name} finished in {train_minutes:.1f} min. Results: {out}")
    keep_awake(False)


def evaluate_coverage(out, cfg, keep_from, E, pipe, test_df, clean_set, cov_all):
    """SWA of the two-output models; how well the soiled-area ratio is predicted on the test days."""
    paths = [out / "model" / f"epoch_{e:02d}_two_outputs.keras" for e in range(keep_from, E + 1)]
    full = average_weights(paths, cfg)
    full.save(out / "model" / "swa_two_outputs.keras")

    @tf.function(reduce_retracing=True)
    def fwd(img, irr):
        o = full({"image": img, "irr": irr}, training=False)
        return o["power_loss"], o["soiled_area"]
    pl, ca = [], []
    for img, irr in pipe.eval(test_df):
        a, b = fwd(img, irr)
        pl.append(a.numpy().ravel())
        ca.append(b.numpy().ravel())
    pl, ca = np.concatenate(pl), np.concatenate(ca)
    true = np.array([cov_all.get(p, -1.0) for p in test_df["relpath"]], np.float32)
    clean = test_df["relpath"].isin(clean_set).to_numpy()
    test_df.assign(pred=pl, soiled_area_pred=ca, soiled_area_measured=true, in_clean_test=clean).to_csv(
        out / "predictions" / "test_swa_two_outputs.csv", index=False)
    res = {}
    for k, m in (("test_full", np.ones(len(true), bool)), ("test_clean", clean)):
        ok = m & (true >= 0)
        e = ca[ok] - true[ok]
        res[k] = {"n": int(ok.sum()), "MAE_area": float(np.abs(e).mean()),
                  "within_5_points": float((np.abs(e) <= 0.05).mean() * 100),
                  "corr_pred_vs_measured_area": float(np.corrcoef(ca[ok], true[ok])[0, 1]),
                  "corr_pred_area_vs_power_loss": float(np.corrcoef(ca[ok], test_df["loss"].to_numpy()[ok])[0, 1])}
    print("Soiled-area ratio on the test days:", res)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ok = clean & (true >= 0)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        ax[0].scatter(true[ok] * 100, ca[ok] * 100, s=3, alpha=0.4)
        ax[0].plot([0, 100], [0, 100], "k--", lw=0.8)
        ax[0].set(xlabel="Measured soiled area (%)", ylabel="Predicted soiled area (%)", title="Soiled area, clean test")
        ax[1].scatter(ca[ok] * 100, test_df["loss"].to_numpy()[ok] * 100, s=3, alpha=0.4, color="#c05000")
        ax[1].set(xlabel="Predicted soiled area (%)", ylabel="True power loss (%)", title="Area vs power loss")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "figures" / "soiled_area.png", dpi=150)
        plt.close(fig)
    except Exception as ex:
        print("soiled-area figure failed:", ex)
    return res


def write_summary(out, name, req, cfg, info, hist):
    m = info["metrics"]
    L = [f"# {name}", "", req.get("description", ""), "",
         f"- Training images: {info['train_images']} ({info['train_days']} days), epochs: {cfg['epochs']}, "
         f"training time: {info['train_minutes']} min ({info['mean_epoch_seconds'] / 60:.1f} min/epoch)",
         f"- One network, one forward pass per image; CPU time per image (batch 1): "
         f"{info['latency_ms_per_image_cpu_batch1']} ms",
         f"- SWA = mean weights of epochs {cfg['epochs'] - cfg['swa_last'] + 1}-{cfg['epochs']}", "",
         "| model | set | n | MSE | MAE | R2 | bMAE | bias | within 0.05 | within 0.10 | within 0.15 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, v in m.items():
        mk, sk = key.split("/")
        L.append(f"| {mk} | {sk} | {v['n']} | {v['MSE']:.4f} | {v['MAE']:.4f} | {v['R2']:.3f} | {v['bMAE']:.4f} | "
                 f"{v['bias']:+.4f} | {v['within_0.05']:.1f}% | {v['within_0.10']:.1f}% | {v['within_0.15']:.1f}% |")
    L += ["", "Per loss range (SWA model, clean test): MAE / bias / images", "",
          "| range | MAE | bias | n |", "|---|---|---|---|"]
    v = m["swa/test_clean"]
    for r in RANGES:
        L.append(f"| {r} | {v['MAE_' + r]:.4f} | {v['bias_' + r]:+.4f} | {v['n_' + r]} |")
    if info.get("soiled_area"):
        a = info["soiled_area"]["test_clean"]
        L += ["", f"Soiled-area ratio (second output, clean test, {a['n']} images): MAE {a['MAE_area'] * 100:.1f} points, "
              f"{a['within_5_points']:.1f}% within 5 points, correlation with the measured area {a['corr_pred_vs_measured_area']:.2f}, "
              f"with the power loss {a['corr_pred_area_vs_power_loss']:.2f}. Figure: figures/soiled_area.png; "
              f"model with both outputs: model/swa_two_outputs.keras"]
    L += ["", "Training history: tables/history.csv, figures/training_curves.png",
          "Predictions: predictions/*.csv   Figures: figures/*.png   All numbers: metrics.json, tables/metrics.csv"]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
