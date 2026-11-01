# -*- coding: utf-8 -*-
"""Comparison of all finished runs: results/comparison.csv, comparison.md, figures, final_model.md."""
import numpy as np
import pandas as pd

from .common import load_json
from .evaluate import RANGES

# the model currently in the paper (seed 42, SWA, 12 days), for reference in final_model.md
PAPER_MODEL = {"test_clean": {"MSE": 0.0067, "MAE": 0.0554, "R2": 0.907, "within_0.15": 90.1, "bMAE": 0.104},
               "test_full": {"MSE": 0.0155, "MAE": 0.0758, "R2": 0.766, "within_0.15": 84.4}}


def collect(runs_dir):
    rows = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not (d / "DONE.json").exists():
            continue
        info = load_json(d / "metrics.json")
        m = info["metrics"]
        r = {"run": d.name, "description": info.get("description", ""), "train_days": info["train_days"],
             "train_images": info["train_images"], "minutes": info["train_minutes"],
             "ms_per_image": info.get("latency_ms_per_image_cpu_batch1")}
        for mk in ("swa", "last"):
            for sk in ("val", "test_clean", "test_full"):
                v = m[f"{mk}/{sk}"]
                for k in ("MSE", "MAE", "R2", "bMAE", "bias", "within_0.15"):
                    r[f"{mk}_{sk}_{k}"] = v[k]
        for rg in RANGES:
            r[f"swa_test_clean_MAE_{rg}"] = m["swa/test_clean"][f"MAE_{rg}"]
        rows.append(r)
    return pd.DataFrame(rows)


def write(runs_dir, res_dir, selection=None):
    res_dir.mkdir(parents=True, exist_ok=True)
    df = collect(runs_dir)
    if df.empty:
        return df
    ctrl = df[df.run == "00_control"]
    if len(ctrl):
        c = ctrl.iloc[0]
        for k in ("swa_val_bMAE", "swa_val_MSE", "swa_test_clean_MSE"):
            df[f"{k}_vs_control_%"] = 100 * (df[k] / c[k] - 1)
    df.to_csv(res_dir / "comparison.csv", index=False)

    L = ["# Comparison of all runs", "",
         "All numbers: ONE network (SWA of the last 3 epochs), one forward pass per image.",
         "Runs 00-08 are trained on the 10 training days; runs 09-10 on 12 days (train + validation).",
         "**Choices are made on the validation days only** (columns 'val'); the test columns are reported, "
         "never used to choose.", "",
         "| run | val bMAE | val MSE | vs control (val bMAE) | test-clean MSE | test-clean MAE | test-clean R2 | "
         "test-clean within 0.15 | test-full MSE | min |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        vs = r.get("swa_val_bMAE_vs_control_%", np.nan)
        vs = "-" if r.run == "00_control" or pd.isna(vs) else f"{vs:+.1f}%"
        if r.train_days > 10:
            vs = "(val seen in training)"
        L.append(f"| {r.run} | {r.swa_val_bMAE:.4f} | {r.swa_val_MSE:.4f} | {vs} | {r.swa_test_clean_MSE:.4f} | "
                 f"{r.swa_test_clean_MAE:.4f} | {r.swa_test_clean_R2:.3f} | {r['swa_test_clean_within_0.15']:.1f}% | "
                 f"{r.swa_test_full_MSE:.4f} | {r.minutes:.0f} |")
    if selection:
        L += ["", "## Selection", "", f"Rule: {selection['rule']}", ""]
        for n, why in selection.get("decisions", {}).items():
            L.append(f"- **{n}**: {why}")
        L += ["", f"Winners: {', '.join(selection.get('winners', [])) or 'none'}",
              f"Configuration used for the final 12-day model: {selection.get('final_from', '-')}"]
    L += ["", "Each run folder (runs/<name>/) has summary.md, metrics.json, tables/, figures/, predictions/, model/."]
    (res_dir / "comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    plot(df, res_dir)
    return df


def plot(df, res_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.train_days <= 10]
    if d.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, col, title in zip(ax, ("swa_val_bMAE", "swa_val_MSE", "swa_test_clean_MSE"),
                             ("Validation bMAE (used to choose)", "Validation MSE", "Clean-test MSE (reported only)")):
        colors = ["#888" if r == "00_control" else "#47a" for r in d.run]
        a.barh(d.run, d[col], color=colors)
        if (d.run == "00_control").any():
            a.axvline(float(d.loc[d.run == "00_control", col].iloc[0]), color="k", ls="--", lw=0.8)
        a.set_title(title, fontsize=9)
        a.invert_yaxis()
        a.grid(axis="x", alpha=0.3)
        a.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(res_dir / "comparison_runs.png", dpi=150)
    plt.close(fig)


def final_report(run_dir, res_dir, selection):
    info = load_json(run_dir / "metrics.json")
    m = info["metrics"]
    q = None
    if (run_dir / "tflite" / "quantization.json").exists():
        q = load_json(run_dir / "tflite" / "quantization.json")
    c, f = m["swa/test_clean"], m["swa/test_full"]
    L = ["# Final model (12 days, one network, SWA epochs 8-10)", "",
         f"Configuration: {selection.get('final_from', '-')} -> changes: {selection.get('final_changes', {})}", "",
         "| set | MSE | MAE | R2 | bMAE | bias | within 0.05 | within 0.10 | within 0.15 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, v in (("clean test", c), ("full test", f)):
        L.append(f"| {name} | {v['MSE']:.4f} | {v['MAE']:.4f} | {v['R2']:.3f} | {v['bMAE']:.4f} | {v['bias']:+.4f} | "
                 f"{v['within_0.05']:.1f}% | {v['within_0.10']:.1f}% | {v['within_0.15']:.1f}% |")
    L += ["", "Model in the paper now (same days, measured earlier): clean MSE 0.0067, MAE 0.0554, R2 0.907, "
          "within 0.15 90.1%, bMAE 0.104; full MSE 0.0155.", "",
          "Per loss range (clean test):", "", "| range | MAE | bias | n |", "|---|---|---|---|"]
    for r in RANGES:
        L.append(f"| {r} | {c['MAE_' + r]:.4f} | {c['bias_' + r]:+.4f} | {c['n_' + r]} |")
    pdays = run_dir / "tables" / "per_day_test_clean_swa.csv"
    if pdays.exists():
        pdd = pd.read_csv(pdays)
        L += ["", "Per day (clean test):", "", "| day | n | MSE | MAE | within 0.15 |", "|---|---|---|---|---|"]
        for _, r in pdd.iterrows():
            L.append(f"| {r.date} | {r.n} | {r.MSE:.4f} | {r.MAE:.4f} | {r['within_0.15']:.1f}% |")
    L += ["", f"CPU time per image (Keras, batch 1): {info.get('latency_ms_per_image_cpu_batch1')} ms"]
    if q:
        L += ["", "int8 quantisation: see runs/" + run_dir.name + "/tflite/quantization.md", ""]
        L += (run_dir / "tflite" / "quantization.md").read_text(encoding="utf-8").splitlines()[2:]
    (res_dir / "final_model.md").write_text("\n".join(L) + "\n", encoding="utf-8")
