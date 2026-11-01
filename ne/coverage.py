# -*- coding: utf-8 -*-
"""Soiled-area ratio of the panel, measured automatically from the image colours (no manual labels).

Inside the soiled panel (frame excluded), a pixel counts as CLEAN when it is clearly blue
(blue > red + 25 and blue > green + 5, on the 0-255 scale); every other pixel counts as COVERED
(brown dust, white powder, droppings, grey cover).  coverage = covered pixels / panel pixels.
Images that are too dark to judge (mean panel brightness < 40) get coverage = -1 and are ignored
in the training loss.  Results are cached in splits/<split>/coverage.csv.
"""
import os

import numpy as np
import pandas as pd

from .common import SPLITS

DARK = 40.0


def inner_panel_mask(size=192, shrink_px=4):
    from PIL import Image, ImageDraw, ImageFilter
    from .data import PANEL_CORNERS
    s = size / 192.0
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).polygon([(x * s, y * s) for x, y in PANEL_CORNERS], fill=255)
    m = m.filter(ImageFilter.MinFilter(2 * shrink_px + 1))
    return np.asarray(m) > 0


def coverage_of(rgb, mask):
    a = rgb.astype(np.float32)
    if a[mask].mean() < DARK:
        return -1.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    clean = (b > r + 25) & (b > g + 5)
    return float(1.0 - clean[mask].mean())


def coverage_map(rgb, mask):
    """Boolean map of covered pixels (for figures)."""
    a = rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return ~((b > r + 25) & (b > g + 5)) & mask


def get_coverage(split, parts, data_root, size=192):
    """Coverage of every image of the split (train, val, test), computed once and cached."""
    from PIL import Image
    cache = SPLITS / split / "coverage.csv"
    known = {}
    if cache.exists():
        c = pd.read_csv(cache)
        known = dict(zip(c["relpath"], c["coverage"]))
    allrel = pd.concat([p["relpath"] for p in parts.values()]).unique()
    todo = [r for r in allrel if r not in known]
    if todo:
        print(f"Measuring the soiled-area ratio of {len(todo)} images (once; cached in {cache.name}) ...")
        mask = inner_panel_mask(size)
        for i, rel in enumerate(todo):
            with Image.open(os.path.join(str(data_root), rel)) as im:
                rgb = np.asarray(im.convert("RGB").resize((size, size)))
            known[rel] = coverage_of(rgb, mask)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
        pd.DataFrame({"relpath": list(known), "coverage": list(known.values())}).to_csv(cache, index=False)
    return known
