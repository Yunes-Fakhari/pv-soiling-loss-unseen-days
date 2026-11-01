# -*- coding: utf-8 -*-
"""Panel features measured by image processing from ONE image (+ its time of day), used as extra network inputs
next to the irradiance (recipe cmixup_feat):
  panel_index  regional soiling index of the panel (0..1)
  sun          time-of-day factor sin(pi * (hour - 6) / 12.5), 0 at 6:00 and 18:30, 1 at about 12:15
Steps for panel_index:
  1. find the four edges of the soiled panel in the image (bright frame outside, dark panel inside)
  2. a pixel is clean when blue >= red, blue >= green and its colour saturation > 0.55 x that of the clean neighbour panel;
     soiled pixels 3 px or more inside the frame go through a 3 x 3 morphological opening (removes thin lines such as
     cell gaps); soiling found there is then grown outwards through soiled pixels up to 1 px from the frame, so dust
     lying against the frame is counted while the frame itself is not
  3. share of soiled area in each of the 72 cells (6 columns x 12 rows, 400 sample points per cell)
  4. a cell counts as fully blocked from 70 % soiled area on (min(1, share / 0.7))
  5. region = 2 cell columns; region value = mean of its 2 most soiled cells; panel_index = mean of the 3 regions
splits/panel_features.csv holds these two numbers for every image of the data set (computed with this file)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
C0 = np.array([(80, 21), (186, 35), (160, 164), (17, 145)], float)       # nominal corners TL, TR, BR, BL
NB0 = np.array([(175, 105), (190, 103), (190, 155), (168, 152)], float)  # clean neighbour panel (light reference)
NC, NR = 6, 12

def _bil(img, x, y):
    x = np.clip(x, 0, 190.999); y = np.clip(y, 0, 190.999); x0 = x.astype(int); y0 = y.astype(int); fx = x - x0; fy = y - y0
    return img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy

def _side(V, p, q, centre, d=2.0):
    t = np.linspace(0.08, 0.92, 60)[:, None]; pts = p + (q - p) * t
    n = np.array([-(q - p)[1], (q - p)[0]]); n /= np.linalg.norm(n)
    if np.dot(centre - (p + q) / 2, n) < 0: n = -n
    i = pts + d * n; o = pts - d * n
    return np.median(_bil(V, o[:, 0], o[:, 1]) - _bil(V, i[:, 0], i[:, 1]))   # bright frame outside, dark panel inside

def _side_many(V, p0, q0, n, A, B, centre, d=2.0):
    """scores of many candidate lines at once: endpoints p0 + a n, q0 + b n"""
    t = np.linspace(0.08, 0.92, 60)
    P_ = p0[None] + A[:, None] * n[None]; Q_ = q0[None] + B[:, None] * n[None]          # (K, 2)
    pts = P_[:, None, :] + (Q_ - P_)[:, None, :] * t[None, :, None]                      # (K, 60, 2)
    nn = np.stack([-(Q_ - P_)[:, 1], (Q_ - P_)[:, 0]], 1); nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    sgn = np.sign(((centre[None] - (P_ + Q_) / 2) * nn).sum(1)); nn *= sgn[:, None]
    i = pts + d * nn[:, None]; o = pts - d * nn[:, None]
    return np.median(_bil(V, o[..., 0], o[..., 1]) - _bil(V, i[..., 0], i[..., 1]), axis=1)

def fit_corners(rgb, search=8):
    V = np.asarray(Image.fromarray(rgb).convert("L").filter(ImageFilter.GaussianBlur(0.8))).astype(np.float32)
    centre = C0.mean(0); lines = []
    for k in range(4):
        p0, q0 = C0[k], C0[(k + 1) % 4]
        n = np.array([-(q0 - p0)[1], (q0 - p0)[0]]); n /= np.linalg.norm(n)
        g = np.arange(-search, search + .01, 2.0); A, B = [x.ravel() for x in np.meshgrid(g, g, indexing="ij")]
        s = _side_many(V, p0, q0, n, A, B, centre); j = int(np.argmax(s)); a0, b0, best = A[j], B[j], s[j]
        g2a = np.arange(a0 - 1.5, a0 + 1.51, 0.5); g2b = np.arange(b0 - 1.5, b0 + 1.51, 0.5)
        A2, B2 = [x.ravel() for x in np.meshgrid(g2a, g2b, indexing="ij")]
        s2 = _side_many(V, p0, q0, n, A2, B2, centre); j2 = int(np.argmax(s2))
        if s2[j2] > best: a0, b0, best = A2[j2], B2[j2], s2[j2]
        lines.append((p0 + a0 * n, q0 + b0 * n, best))
    def inter(l1, l2):
        p1, q1 = l1[0], l1[1]; p2, q2 = l2[0], l2[1]
        t = np.linalg.solve(np.array([q1 - p1, -(q2 - p2)]).T, p2 - p1); return p1 + t[0] * (q1 - p1)
    corners = np.array([inter(lines[(k - 1) % 4], lines[k]) for k in range(4)])
    if np.abs(corners - C0).max() > 14: corners = C0.copy()              # fit failed: keep the nominal corners
    return corners, [l[2] for l in lines]

def poly(pts, shrink=0):
    m = Image.new("L", (192, 192), 0); ImageDraw.Draw(m).polygon([tuple(p) for p in pts], fill=255)
    if shrink: m = m.filter(ImageFilter.MinFilter(2 * shrink + 1))
    return np.asarray(m) > 0

def opening(m):
    e = np.ones_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1): e &= np.roll(np.roll(m, dy, 0), dx, 1)
    d = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1): d |= np.roll(np.roll(e, dy, 0), dx, 1)
    return d


CLEAN_SAT, SHRINK, FULL_AT, GROW = 0.55, 3, 0.7, 5


def _dilate(m):
    d = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1): d |= np.roll(np.roll(m, dy, 0), dx, 1)
    return d
_U, _V = np.meshgrid((np.arange(120) + .5) / 120, (np.arange(240) + .5) / 240)   # 20 x 20 sample points per cell
_CELL = (np.minimum((_V * NR).astype(int), NR - 1) * NC + np.minimum((_U * NC).astype(int), NC - 1)).ravel()


def cell_soiled_share(rgb):
    """rgb: 192 x 192 x 3 uint8 -> (12, 6) share of soiled area per cell, fitted corners."""
    corners, _ = fit_corners(rgb)
    a = rgb.astype(np.float32); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    V = a.max(-1); S = (V - a.min(-1)) / (V + 1e-6)
    NB = poly(NB0 + (corners - C0).mean(0)); sref = np.median(S[NB])
    raw = ~((B >= R) & (B >= G) & (S > CLEAN_SAT * sref))
    M3 = poly(corners, SHRINK); M = poly(corners, 1)
    soiled = opening(raw & M3) & M3                          # safe soiling, away from the frame
    full = raw & M
    for _ in range(GROW):                                     # grow it to the frame through soiled pixels only
        soiled = _dilate(soiled) & full
    TL, TR, BR, BL = corners
    top = TL[None, None] + (TR - TL)[None, None] * _U[..., None]; bot = BL[None, None] + (BR - BL)[None, None] * _U[..., None]
    xy = top + (bot - top) * _V[..., None]
    xi = np.clip(np.round(xy[..., 0]).astype(int), 0, 191).ravel(); yi = np.clip(np.round(xy[..., 1]).astype(int), 0, 191).ravel()
    n = np.bincount(_CELL[M[yi, xi]], minlength=72); s = np.bincount(_CELL[soiled[yi, xi]], minlength=72)
    return (s / np.maximum(n, 1)).reshape(NR, NC)


def panel_index(rgb):
    Y = np.minimum(1.0, cell_soiled_share(rgb) / FULL_AT)
    return float(np.mean([np.sort(Y[:, 2 * g:2 * g + 2].ravel())[-2:].mean() for g in range(3)]))


def sun_factor(hour_decimal):
    return float(max(0.0, np.sin(np.pi * (hour_decimal - 6.0) / 12.5)))


if __name__ == "__main__":
    import sys
    from PIL import Image
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB").resize((192, 192)))
    print("panel_index", round(panel_index(img), 4))
