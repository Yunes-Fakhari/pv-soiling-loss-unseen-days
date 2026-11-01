# -*- coding: utf-8 -*-
"""Paths, logging, seeding and small helpers shared by every stage."""
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # repository root
SPLITS = HERE / "splits"
WEIGHT_FILE = "mobilenet_1_0_192_tf_no_top.h5"      # ImageNet MobileNet v1 (alpha 1.0, 192x192), no top


def runs_dir(smoke=False):
    return HERE / ("runs_smoke" if smoke else "runs")


def results_dir(smoke=False):
    return HERE / ("results_smoke" if smoke else "results")


def find_data_root(explicit=None):
    """Folder that contains PanelImages/ (the DeepSolarEye images)."""
    cands = [explicit, os.environ.get("SOILING_DATA_ROOT"),
             HERE.parent / "dataset_raw", HERE / "dataset_raw", HERE.parent.parent / "dataset_raw"]
    for c in cands:
        if c and (Path(c) / "PanelImages").is_dir():
            return Path(c).resolve()
    raise FileNotFoundError(
        "Could not find the image folder (dataset_raw/PanelImages).\n"
        "Put the DeepSolarEye images in dataset_raw/PanelImages next to this repository, or run with\n"
        "  --data-root \"D:\\path\\to\\dataset_raw\"")


def ensure_imagenet_weights():
    """Copy the local ImageNet weight file into the Keras cache so no download is needed."""
    cache = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras")) / "models"
    dst = cache / WEIGHT_FILE
    if dst.exists():
        return str(dst)
    for src in (HERE / "pretrained" / WEIGHT_FILE, HERE.parent / "pretrained" / WEIGHT_FILE):
        if src.exists():
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"ImageNet weights copied to the Keras cache: {dst}")
            return str(dst)
    print("NOTE: local ImageNet weights not found; Keras will try to download them.")
    return None


def set_seed(seed):
    import numpy as np
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


class Tee:
    """Write console output to the screen and to a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def keep_awake(on):
    """Stop Windows from going to sleep while training runs (no effect on other systems)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        flags = 0x80000000 | (0x00000001 if on else 0)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def environment_info():
    import tensorflow as tf
    info = {"python": platform.python_version(), "tensorflow": tf.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "gpus": [g.name for g in tf.config.list_physical_devices("GPU")]}
    return info


def fmt_hours(seconds):
    m = int(round(seconds / 60))
    return f"{m // 60}h{m % 60:02d}m"


def safe_console():
    """Never crash on characters the Windows console cannot show (e.g. Persian folder names)."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")
