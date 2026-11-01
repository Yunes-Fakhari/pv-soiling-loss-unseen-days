# -*- coding: utf-8 -*-
"""int8 post-training quantisation (TensorFlow Lite) of the final model + CPU speed measurement.
    python -m ne.quantize --run-dir runs/10_final_12days
Writes <run>/tflite/: model_float32.tflite, model_int8.tflite, quantization.json, quantization.md and
the predictions of both TFLite models on the test days.
int8 = weights and activations in 8-bit integers, calibrated on 300 training images (inputs/outputs stay float)."""
import argparse
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

from . import model as _layers  # noqa: F401  (registers SoiledArea)
from .common import Tee, safe_console, environment_info, find_data_root, load_json, now, save_json
from .data import Pipeline, keep_existing, load_split
from .evaluate import metrics, predict


def interpreter(content, threads):
    try:
        from ai_edge_litert.interpreter import Interpreter          # new name of the TFLite runtime
    except Exception:
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_content=content, num_threads=threads)
    it.allocate_tensors()
    return it.get_signature_runner()


def run_tflite(runner, ds, limit=None):
    preds, times = [], []
    for i, (img, irr) in enumerate(ds):
        if limit is not None and i >= limit:
            break
        x = {"image": img.numpy().astype(np.float32), "irr": irr.numpy().reshape(1, -1).astype(np.float32)}
        t0 = time.perf_counter()
        o = runner(**x)
        times.append(time.perf_counter() - t0)
        preds.append(float(np.ravel(next(iter(o.values())))[0]))
    t = times[10:] if len(times) > 20 else times
    return np.array(preds), 1000 * float(np.mean(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    out = run / "tflite"
    out.mkdir(exist_ok=True)
    safe_console()
    log = open(out / "console.log", "w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)
    tf.get_logger().setLevel("ERROR")
    req = load_json(run / "request.json")
    cfg = req["config"]
    data_root = find_data_root(args.data_root)
    parts = load_split(cfg["split"], cfg.get("max_images"))
    if cfg.get("max_images"):
        parts = keep_existing(parts, data_root)
    pipe = Pipeline(cfg, data_root)
    keras_path = run / "model" / "swa.keras"
    model = tf.keras.models.load_model(keras_path, compile=False)
    size = int(cfg["img_size"])

    sm_dir = Path(tempfile.mkdtemp(prefix="soiling_savedmodel_"))
    model.export(str(sm_dir), format="tf_saved_model", verbose=False,
                 input_signature=[{"image": tf.TensorSpec([1, size, size, 3], tf.float32, name="image"),
                                   "irr": tf.TensorSpec([1, 1 + len(cfg.get("panel_features") or [])], tf.float32, name="irr")}])
    rep = parts["train"].sample(n=min(300, len(parts["train"])), random_state=0)

    def representative():
        for img, irr in pipe.eval(rep, batch=1):
            yield {"image": img.numpy(), "irr": irr.numpy().reshape(1, -1)}

    res = {"started": now(), "env": environment_info(), "calibration_images": int(len(rep))}
    content = {}
    for kind in ("float32", "int8"):
        conv = tf.lite.TFLiteConverter.from_saved_model(str(sm_dir))
        if kind == "int8":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = representative
            # int8 everywhere it is supported; the few ops without an int8 kernel (e.g. the comparisons of the
            # soiled-area feature) stay in float instead of crashing
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        data = conv.convert()
        (out / f"model_{kind}.tflite").write_bytes(data)
        content[kind] = data
        res[f"size_MB_{kind}"] = round(len(data) / 1e6, 2)
        print(f"{kind}: model_{kind}.tflite {len(data) / 1e6:.2f} MB")
    shutil.rmtree(sm_dir, ignore_errors=True)
    res["size_MB_keras"] = round(keras_path.stat().st_size / 1e6, 2)

    test_df, clean = parts["test"], set(parts["test_clean"]["relpath"])
    mask = test_df["relpath"].isin(clean).to_numpy()
    y = test_df["loss"].to_numpy()
    one = lambda: pipe.eval(test_df, batch=1)
    p_keras = predict(model, pipe.eval(test_df))
    res["keras_float"] = {"test_full": metrics(y, p_keras), "test_clean": metrics(y[mask], p_keras[mask])}
    cpu = os.cpu_count() or 1
    res["threads_all"] = cpu
    for kind, data in content.items():
        pred, ms1 = run_tflite(interpreter(data, 1), one())
        _, msn = run_tflite(interpreter(data, cpu), one(), limit=200)
        res[f"tflite_{kind}"] = {"test_full": metrics(y, pred), "test_clean": metrics(y[mask], pred[mask]),
                                 "ms_per_image_1_thread": round(ms1, 2), "ms_per_image_all_threads": round(msn, 2),
                                 "max_abs_diff_vs_keras": float(np.max(np.abs(pred - p_keras)))}
        test_df.assign(pred=pred, in_clean_test=mask).to_csv(out / f"test_predictions_{kind}.csv", index=False)
        print(f"{kind}: clean-test MSE {res[f'tflite_{kind}']['test_clean']['MSE']:.5f} | "
              f"{ms1:.1f} ms/image (1 thread), {msn:.1f} ms/image ({cpu} threads)")
    t = []
    for i, (img, irr) in enumerate(one()):            # Keras float model, batch 1
        if i >= 70:
            break
        t0 = time.perf_counter()
        model({"image": img, "irr": irr}, training=False).numpy()
        if i >= 10:
            t.append(time.perf_counter() - t0)
    res["keras_ms_per_image"] = round(1000 * float(np.mean(t)), 2)
    save_json(res, out / "quantization.json")
    k, f32, i8 = res["keras_float"]["test_clean"], res["tflite_float32"], res["tflite_int8"]
    L = ["# int8 quantisation of the final model", "",
         "| model | size (MB) | clean-test MSE | clean-test MAE | ms/image, 1 thread | ms/image, all threads |",
         "|---|---|---|---|---|---|",
         f"| Keras float32 | {res['size_MB_keras']} | {k['MSE']:.4f} | {k['MAE']:.4f} | {res['keras_ms_per_image']} | - |",
         f"| TFLite float32 | {res['size_MB_float32']} | {f32['test_clean']['MSE']:.4f} | {f32['test_clean']['MAE']:.4f} | "
         f"{f32['ms_per_image_1_thread']} | {f32['ms_per_image_all_threads']} |",
         f"| TFLite int8 | {res['size_MB_int8']} | {i8['test_clean']['MSE']:.4f} | {i8['test_clean']['MAE']:.4f} | "
         f"{i8['ms_per_image_1_thread']} | {i8['ms_per_image_all_threads']} |", "",
         f"CPU: {res['env'].get('processor', '')} ({cpu} threads). One image per call (batch 1). "
         f"int8 calibrated on {res['calibration_images']} training images."]
    (out / "quantization.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    save_json({"finished": now()}, out / "DONE.json")
    print("\n".join(L))


if __name__ == "__main__":
    main()
