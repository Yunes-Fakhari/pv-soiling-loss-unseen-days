# -*- coding: utf-8 -*-
"""Splits, LDS weights, C-Mixup partners and the tf.data pipelines.

Image handling is identical to the paper code (mobilenet_soilingedge/src/data.py, v6 = whole image):
decode JPEG -> 192x192 -> round -> random h/v flip + brightness +-10 % (training only) -> x/127.5 - 1.
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from .common import SPLITS

AUTOTUNE = tf.data.AUTOTUNE


def load_split(name, max_images=None):
    folder = SPLITS / name
    parts = {}
    for p in ("train", "val", "test", "test_clean"):
        df = pd.read_csv(folder / f"{p}.csv")
        if max_images:                                  # smoke test only: a few images spread over the days
            df = df.iloc[np.linspace(0, len(df) - 1, min(max_images, len(df))).astype(int)].reset_index(drop=True)
        parts[p] = df
    return parts


def keep_existing(parts, data_root):
    """Smoke test only: drop rows whose image file is missing."""
    out = {}
    for k, df in parts.items():
        ok = [os.path.exists(os.path.join(str(data_root), p)) for p in df["relpath"]]
        out[k] = df[ok].reset_index(drop=True)
    return out


def lds_weights(loss, bins=50, sigma=2.0, power=1.0, clip=5.0):
    """Label Distribution Smoothing (Yang et al., ICML 2021); same code as the paper runs."""
    y = np.clip(np.asarray(loss, float), 0.0, 1.0)
    idx = np.minimum((y * bins).astype(int), bins - 1)
    hist = np.bincount(idx, minlength=bins).astype(float)
    half = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    smooth = np.convolve(np.pad(hist, half, mode="reflect"), k / k.sum(), mode="valid")
    w = (1.0 / np.maximum(smooth[idx], 1e-6)) ** power
    w = np.minimum(w / w.mean(), clip)
    return (w / w.mean()).astype(np.float32)


def sample_weights(y, cfg):
    if not cfg.get("lds"):
        return np.ones(len(y), np.float32)
    L = cfg["lds"]
    return lds_weights(y, int(L["bins"]), float(L["sigma"]), float(L["power"]), float(L["clip"]))


_FEATURES = {}


def side_inputs(df, cfg):
    """Numbers that go next to the image: the irradiance, plus (recipe cmixup_feat) panel features measured
    by image processing from the same image and its time: splits/panel_features.csv (see ne/panel_features.py)."""
    cols = list(cfg.get("panel_features") or [])
    x = [df["irradiance"].to_numpy(np.float32).reshape(-1, 1)]
    if cols:
        if "table" not in _FEATURES:
            _FEATURES["table"] = pd.read_csv(SPLITS / "panel_features.csv").set_index("relpath")
        t = _FEATURES["table"]
        missing = ~df["relpath"].isin(t.index)
        if missing.any():
            raise ValueError(f"{int(missing.sum())} images have no panel features in splits/panel_features.csv")
        x.append(t.loc[df["relpath"], cols].to_numpy(np.float32))
    return np.concatenate(x, axis=1)


def cmixup_partners(y, bandwidth, rng, chunk=512):
    """C-Mixup sampling (Yao et al., NeurIPS 2022): partner j of image i is drawn with probability
    proportional to exp(-(y_i - y_j)^2 / (2 bandwidth^2)) over the whole training set."""
    y = np.asarray(y, np.float64)
    n = len(y)
    out = np.empty(n, np.int64)
    for s in range(0, n, chunk):
        d = y[s:s + chunk, None] - y[None, :]
        p = np.exp(-0.5 * (d / bandwidth) ** 2)
        c = np.cumsum(p, axis=1)
        u = rng.random(len(d)) * c[:, -1]
        out[s:s + chunk] = np.minimum((c < u[:, None]).sum(axis=1), n - 1)
    return out


def _py_read(path):
    with open(path.decode("utf-8"), "rb") as f:
        return f.read()


_READER = {}


def choose_reader(sample_path):
    """TensorFlow's reader cannot open some non-English Windows paths; Python's open() always can."""
    if sample_path in _READER:
        return _READER[sample_path]
    try:
        tf.io.read_file(sample_path).numpy()
        mode = "tf"
    except Exception:
        mode = "python"
        print("NOTE: using the Python file reader (non-English characters in the image path).")
    _READER[sample_path] = mode
    return mode


PANEL_CORNERS = [(80, 21), (186, 35), (160, 164), (17, 145)]   # TL, TR, BR, BL of the soiled panel (fixed camera)


def panel_mask(size=192, margin_px=3):
    """Binary mask of the soiled panel (1 inside, 0 outside), widened by ~3 px so the frame is kept.
    The camera never moves, so the same mask fits every image. Used for training, testing and prediction."""
    from PIL import Image, ImageDraw, ImageFilter
    s = size / 192.0
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).polygon([(x * s, y * s) for x, y in PANEL_CORNERS], fill=255)
    m = m.filter(ImageFilter.MaxFilter(2 * margin_px + 1))
    return tf.constant((np.asarray(m, np.float32) / 255.0)[..., None])


class Pipeline:
    def __init__(self, cfg, data_root):
        self.cfg = cfg
        self.root = str(data_root)
        self.size = int(cfg["img_size"])
        self.aug = cfg["aug"]
        self.mask = panel_mask(self.size) if cfg.get("panel_mask") else None

    def paths(self, df):
        return np.array([os.path.join(self.root, p) for p in df["relpath"]])

    def _decode(self, reader):
        size = self.size

        def dec(path):
            if reader == "tf":
                raw = tf.io.read_file(path)
            else:
                raw = tf.numpy_function(_py_read, [path], tf.string)
                raw.set_shape([])
            img = tf.io.decode_image(raw, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.image.resize(img, (size, size))
            img = tf.round(img)
            img.set_shape([size, size, 3])
            if self.mask is not None:                        # keep the soiled panel, grey everywhere else
                img = img * self.mask + 127.5 * (1.0 - self.mask)
            return img                                       # float32, 0..255
        return dec

    def _augment(self, img):
        a = self.aug
        if a.get("hflip"):
            img = tf.image.random_flip_left_right(img)
        if a.get("vflip"):
            img = tf.image.random_flip_up_down(img)
        if float(a.get("brightness", 0)) > 0:
            img = tf.image.random_brightness(img, float(a["brightness"]) * 255.0)
            img = tf.clip_by_value(img, 0.0, 255.0)
        return img

    @staticmethod
    def _prep(img):
        return img / 127.5 - 1.0

    def train(self, df, weights, epoch, cov=None):
        """New shuffle (and new C-Mixup partners) every epoch, reproducible from seed + epoch."""
        cfg = self.cfg
        rng = np.random.default_rng(int(cfg["seed"]) * 1000 + epoch)
        n = len(df)
        order = rng.permutation(n)
        P = self.paths(df)
        irr = side_inputs(df, self.cfg)
        y = df["loss"].to_numpy(np.float32).reshape(-1, 1)
        w = np.asarray(weights, np.float32)
        c = (np.full(n, -1.0, np.float32) if cov is None else np.asarray(cov, np.float32)).reshape(n, -1)
        dec = self._decode(choose_reader(P[0]))
        mix = cfg.get("cmixup")
        if mix:
            part = cmixup_partners(y.ravel(), float(mix["bandwidth"]), rng)[order]
            alpha = float(mix["alpha"])
            prob = float(mix.get("prob", 1.0))          # share of samples that are mixed (1.0 = all)
            no_mix_below = float(mix.get("no_mix_below", 0.0))   # images with a lower loss stay real (not mixed)
            ds = tf.data.Dataset.from_tensor_slices(
                (P[order], irr[order], y[order], w[order], c[order],
                 P[part], irr[part], y[part], w[part], c[part]))

            def load(p1, i1, y1, w1, c1, p2, i2, y2, w2, c2):
                g1 = tf.random.gamma([], alpha)
                g2 = tf.random.gamma([], alpha)
                lam = g1 / (g1 + g2)                     # lambda ~ Beta(alpha, alpha)
                if prob < 1.0:                           # the other samples stay real (lambda = 1)
                    lam = tf.where(tf.random.uniform([]) < prob, lam, 1.0)
                if no_mix_below > 0.0:                   # low-loss images keep their own image and exact label
                    lam = tf.where(tf.reduce_max(y1) < no_mix_below, 1.0, lam)
                img = lam * self._augment(dec(p1)) + (1 - lam) * self._augment(dec(p2))
                cm = tf.where((c1 >= 0) & (c2 >= 0), lam * c1 + (1 - lam) * c2, -1.0)   # -1 = unknown
                return (self._prep(img), lam * i1 + (1 - lam) * i2, lam * y1 + (1 - lam) * y2,
                        lam * w1 + (1 - lam) * w2, cm)
        else:
            ds = tf.data.Dataset.from_tensor_slices((P[order], irr[order], y[order], w[order], c[order]))

            def load(p, i, yy, ww, cc):
                return self._prep(self._augment(dec(p))), i, yy, ww, cc
        fdrop = float(cfg.get("feature_dropout", 0.0))
        if fdrop > 0 and cfg.get("panel_features"):
            # robustness: in this share of the training samples the panel index is hidden (-1 = unknown),
            # so the network cannot rely on it alone and must also read the image
            base_load = load

            def load(*a):
                img, i, yy, ww, cc = base_load(*a)
                hide = tf.random.uniform([]) < fdrop
                col = tf.one_hot(1, tf.shape(i)[0])
                i = tf.where(hide, i * (1 - col) - col, i)
                return img, i, yy, ww, cc
        ds = ds.map(load, num_parallel_calls=AUTOTUNE)
        return ds.batch(int(cfg["batch_size"])).prefetch(AUTOTUNE)

    def eval(self, df, batch=64):
        P = self.paths(df)
        irr = side_inputs(df, self.cfg)
        dec = self._decode(choose_reader(P[0]))
        ds = tf.data.Dataset.from_tensor_slices((P, irr))
        ds = ds.map(lambda p, i: (self._prep(dec(p)), i), num_parallel_calls=AUTOTUNE)
        return ds.batch(batch).prefetch(AUTOTUNE)
