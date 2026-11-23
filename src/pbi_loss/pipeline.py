import cv2
import numpy as np
from PIL import Image, ImageFilter


def _bilinear(image, x, y):
    x, y = np.clip(x, 0, 190.999), np.clip(y, 0, 190.999)
    ix, iy = x.astype(int), y.astype(int)
    fx, fy = x - ix, y - iy
    return (image[iy, ix] * (1-fx) * (1-fy)
            + image[iy, ix+1] * fx * (1-fy)
            + image[iy+1, ix] * (1-fx) * fy
            + image[iy+1, ix+1] * fx * fy)


def _line_scores(gray, p, q, normal, offsets_a, offsets_b, centre):
    start = p + offsets_a[:, None] * normal
    end = q + offsets_b[:, None] * normal
    t = np.linspace(0.08, 0.92, 60)
    points = start[:, None] + (end-start)[:, None] * t[None, :, None]
    normals = np.stack([-(end-start)[:, 1], (end-start)[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals *= np.sign(((centre-(start+end)/2)*normals).sum(1))[:, None]
    inside, outside = points + 2*normals[:, None], points - 2*normals[:, None]
    return np.median(_bilinear(gray, outside[..., 0], outside[..., 1])
                     - _bilinear(gray, inside[..., 0], inside[..., 1]), axis=1)


def fit_corners(rgb, nominal):
    gray = np.asarray(Image.fromarray(rgb).convert('L').filter(
        ImageFilter.GaussianBlur(0.8)), dtype=np.float32)
    nominal = np.asarray(nominal, dtype=float)
    if nominal.shape != (4, 2) or not np.isfinite(nominal).all():
        raise ValueError('corners must contain four finite (x, y) pairs: TL, TR, BR, BL')
    centre, lines = nominal.mean(0), []
    for k in range(4):
        p, q = nominal[k], nominal[(k+1) % 4]
        normal = np.array([-(q-p)[1], (q-p)[0]])
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError('Adjacent nominal corners must differ')
        normal /= norm
        grid = np.arange(-8, 8.01, 2.0)
        a, b = [v.ravel() for v in np.meshgrid(grid, grid, indexing='ij')]
        scores = _line_scores(gray, p, q, normal, a, b, centre)
        j = int(np.argmax(scores))
        a0, b0, best = a[j], b[j], scores[j]
        aa = np.arange(a0-1.5, a0+1.51, 0.5)
        bb = np.arange(b0-1.5, b0+1.51, 0.5)
        a, b = [v.ravel() for v in np.meshgrid(aa, bb, indexing='ij')]
        scores = _line_scores(gray, p, q, normal, a, b, centre)
        j = int(np.argmax(scores))
        if scores[j] > best:
            a0, b0 = a[j], b[j]
        lines.append((p+a0*normal, q+b0*normal))
    try:
        corners = []
        for k in range(4):
            p, q = lines[(k-1) % 4]
            r, s = lines[k]
            t = np.linalg.solve(np.array([q-p, -(s-r)]).T, r-p)
            corners.append(p+t[0]*(q-p))
        corners = np.asarray(corners)
        failed = not np.isfinite(corners).all() or np.abs(corners-nominal).max() > 14
    except np.linalg.LinAlgError:
        corners, failed = nominal.copy(), True
    return (nominal.copy() if failed else corners), bool(failed)


def regional_index(shares, full_blockage=0.7, top_cells=2):
    shares = np.asarray(shares, dtype=float)
    if shares.shape[-1] != 72 or not np.isfinite(shares).all():
        raise ValueError('Expected 72 finite cell shares in row-major order')
    if ((shares < 0) | (shares > 1)).any() or not 0 < full_blockage <= 1:
        raise ValueError('Cell shares and full_blockage must lie in [0, 1]')
    if not isinstance(top_cells, int) or not 1 <= top_cells <= 24:
        raise ValueError('top_cells must be an integer between 1 and 24')
    grid = np.minimum(1, shares/full_blockage).reshape(shares.shape[:-1] + (12, 6))
    regions = [np.sort(grid[..., 2*r:2*r+2].reshape(shares.shape[:-1] + (24,)), axis=-1)
               [..., -top_cells:].mean(-1) for r in range(3)]
    return np.mean(regions, axis=0)


def extract_features(path, config, return_images=False):
    if config['size'] != 192:
        raise ValueError('The published grid and localization use size=192')
    with Image.open(path) as image:
        if image.size != (192, 192):
            raise ValueError(f'Expected a 192x192 full camera frame, got {image.size}: {path}')
        rgb = np.asarray(image.convert('RGB'))
    corners, fallback = fit_corners(rgb, config['corners'])
    target = np.array([[0, 0], [191, 0], [191, 191], [0, 191]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    crop = cv2.warpPerspective(rgb, matrix, (192, 192), flags=cv2.INTER_LINEAR)
    enhanced = crop.copy()
    if config['clahe']:
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=config['clahe_clip'],
                                     tileGridSize=tuple(config['clahe_tiles'])).apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    values = enhanced.astype(np.float32)
    beta = (values[..., 2]-np.maximum(values[..., 0], values[..., 1]))/(values.max(-1)+1)
    margin = config['margin']
    if not isinstance(margin, int) or not 0 <= margin < 16:
        raise ValueError('margin must be an integer from 0 to 15')
    valid = np.zeros((192, 192), dtype=bool)
    valid[margin:192-margin, margin:192-margin] = True
    reference = float(np.percentile(beta[valid], 95))
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(f'No positive blueness reference; image is outside the method assumptions: {path}')
    degree = np.clip(1-beta/reference, 0, 1)
    opened = cv2.morphologyEx(degree, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    threshold = config['mask_threshold']
    if not 0 < threshold <= 1:
        raise ValueError('mask_threshold must lie in (0, 1]')
    mask = opened >= threshold
    def cells(array):
        return array.reshape(12, 16, 6, 32).sum(axis=(1, 3)).ravel()
    shares = cells(mask & valid)/cells(valid)
    if config['feature_precision'] == 'float16':
        shares = shares.astype(np.float16).astype(float)
    elif config['feature_precision'] != 'float64':
        raise ValueError('feature_precision must be float16 or float64')
    index = float(regional_index(shares, config['full_blockage'], config['top_cells']))
    result = {'shares': shares, 'pbi': index, 'beta95': reference,
              'corners': corners.tolist(), 'corner_fallback': fallback}
    if return_images:
        result.update(crop=crop, enhanced=enhanced, mask=mask, degree=degree)
    return result
