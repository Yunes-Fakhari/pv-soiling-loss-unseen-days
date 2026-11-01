# -*- coding: utf-8 -*-
"""Regional soiling index I (Section II-E of the paper) for every image, from image processing only.

Writes splits/panel_features.csv with, for each image:
  panel_index   regional soiling index I (0..1), eq. in Section II-E
  sun           time-of-day factor sigma = max(0, sin(pi (h - 6) / 12.5))
  soiled_area   share of the panel area that is soiled (used only for the error analysis of Section III-B)

Usage:
  python scripts/compute_index.py --data-root path/to/dataset_raw          # all images listed in splits/
  python scripts/compute_index.py --image path/to/one_image.jpg           # one image
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ne import panel_features as pf  # noqa: E402


def soiled_area(rgb):
    """Share of the panel (1 px inside the fitted frame) that is soiled after the colour test and a 3x3 opening."""
    corners, _ = pf.fit_corners(rgb)
    a = rgb.astype(np.float32); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    V = a.max(-1); S = (V - a.min(-1)) / (V + 1e-6)
    sref = np.median(S[pf.poly(pf.NB0 + (corners - pf.C0).mean(0))])
    M = pf.poly(corners, 1)
    soiled = pf.opening(~((B >= R) & (B >= G) & (S > pf.CLEAN_SAT * sref)) & M) & M
    TL, TR, BR, BL = corners
    top = TL[None, None] + (TR - TL)[None, None] * pf._U[..., None]
    bot = BL[None, None] + (BR - BL)[None, None] * pf._U[..., None]
    xy = top + (bot - top) * pf._V[..., None]
    xi = np.clip(np.round(xy[..., 0]).astype(int), 0, 191).ravel()
    yi = np.clip(np.round(xy[..., 1]).astype(int), 0, 191).ravel()
    return float(soiled[yi, xi].sum() / max(M[yi, xi].sum(), 1))


def features(path, hour):
    rgb = np.asarray(Image.open(path).convert("RGB").resize((192, 192)))
    return pf.panel_index(rgb), pf.sun_factor(hour), soiled_area(rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", help="folder that contains PanelImages/")
    ap.add_argument("--image", help="one image instead of the whole data set")
    ap.add_argument("--out", default=str(ROOT / "splits" / "panel_features.csv"))
    a = ap.parse_args()
    if a.image:
        rgb = np.asarray(Image.open(a.image).convert("RGB").resize((192, 192)))
        print(f"regional soiling index I = {pf.panel_index(rgb):.4f}, soiled area = {soiled_area(rgb):.4f}")
        return
    rows = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "splits").glob("splits_*/*.csv"))])
    rows = rows.drop_duplicates("relpath")
    t = pd.to_datetime(rows.datetime)
    hours = (t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600).values
    out = []
    for k, (rel, h) in enumerate(zip(rows.relpath, hours)):
        out.append((rel, *features(Path(a.data_root) / rel, h)))
        if (k + 1) % 500 == 0:
            print(f"{k + 1} of {len(rows)} images")
    pd.DataFrame(out, columns=["relpath", "panel_index", "sun", "soiled_area"]).to_csv(a.out, index=False)
    print("written", a.out)


if __name__ == "__main__":
    main()
