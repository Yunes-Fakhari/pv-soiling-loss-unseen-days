from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import hashlib
import json
import re
import cv2
import numpy as np
import pandas as pd

from .config import write_json
from .pipeline import extract_features, regional_index

CELL_COLUMNS = [f'm{i:02d}' for i in range(72)]
PATTERN = re.compile(r'^solar_[A-Za-z]{3}_([A-Za-z]{3})_(\d+)_(\d+)__(\d+)__(\d+)_(\d{4})_L_([-+\d.eE]+)_I_([-+\d.eE]+)$')
MONTHS = {m: i for i, m in enumerate(
    ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'), 1)}


def parse_filename(path):
    match = PATTERN.match(Path(path).stem)
    if not match:
        raise ValueError(f'Unrecognized DeepSolarEye filename: {path}')
    month, day, hour, minute, second, year, loss, irradiance = match.groups()
    timestamp = pd.Timestamp(year=int(year), month=MONTHS[month], day=int(day),
                             hour=int(hour), minute=int(minute), second=int(second))
    return {'datetime': timestamp, 'loss': float(loss), 'irradiance': float(irradiance)}


def normalize_metadata(frame):
    frame = frame.copy()
    required = {'relpath', 'datetime', 'loss'}
    if required - set(frame.columns):
        raise ValueError(f'Missing metadata fields: {sorted(required-set(frame.columns))}')
    frame['relpath'] = frame.relpath.astype(str).str.replace('\\', '/', regex=False)
    if frame.relpath.duplicated().any():
        raise ValueError('Image paths must be unique')
    frame['datetime'] = pd.to_datetime(frame.datetime, errors='raise')
    if frame.datetime.isna().any() or frame.datetime.dt.tz is not None:
        raise ValueError('Use complete, timezone-naive local acquisition timestamps')
    frame['loss'] = pd.to_numeric(frame.loss, errors='raise')
    if not np.isfinite(frame.loss).all() or not frame.loss.between(0, 1).all():
        raise ValueError('Loss labels must be finite and in [0, 1]')
    frame['date'] = frame.datetime.dt.strftime('%Y-%m-%d')
    frame['seconds_of_day'] = (frame.datetime.dt.hour*3600
                               + frame.datetime.dt.minute*60 + frame.datetime.dt.second)
    frame['hour_decimal'] = frame.seconds_of_day/3600
    return frame.sort_values(['datetime', 'relpath'], kind='stable').reset_index(drop=True)


def scan_dataset(root):
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths:
        raise ValueError(f'No images found under {root}')
    rows = [{'relpath': p.relative_to(root).as_posix(), **parse_filename(p)} for p in paths]
    return normalize_metadata(pd.DataFrame(rows))


def subsample(frame, seconds=30):
    if not isinstance(seconds, int) or seconds < 1:
        raise ValueError('Sampling interval must be a positive integer')
    frame = normalize_metadata(frame)
    key = frame.date + ':' + (frame.seconds_of_day//seconds).astype(str)
    return frame.loc[~key.duplicated()].reset_index(drop=True)


def diagnostic_subset(sampled, full_metadata):
    stable = np.ones(len(sampled), dtype=bool)
    for day, selected in sampled.groupby('date', sort=False):
        full = full_metadata.loc[full_metadata.date == day].sort_values('datetime')
        t = full.seconds_of_day.to_numpy()
        labels = full.loss.to_numpy()
        for i in selected.index:
            time = sampled.at[i, 'seconds_of_day']
            lo, hi = np.searchsorted(t, [time-90, time+90], side='left')
            hi = np.searchsorted(t, time+90, side='right')
            median = float(np.median(labels[lo:hi]))
            a, b = np.searchsorted(t, time-180), np.searchsorted(t, time-30, side='right')
            c, d = np.searchsorted(t, time+30), np.searchsorted(t, time+180, side='right')
            transition = b-a >= 3 and d-c >= 3 and abs(np.median(labels[c:d])-np.median(labels[a:b])) > 0.25
            stable[i] = abs(sampled.at[i, 'loss']-median) <= 0.05 and not transition
    return stable


def _extract(relpath, root, processing):
    path = (Path(root)/relpath).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Image path leaves the dataset root: {relpath}')
    try:
        result = extract_features(path, processing)
    except Exception as exc:
        raise RuntimeError(f'Feature extraction failed for {relpath}: {exc}') from exc
    return [*result['shares'], result['pbi'], result['beta95'], result['corner_fallback']]


def extract_dataset(frame, root, processing, workers=1):
    if workers < 1:
        raise ValueError('workers must be positive')
    work = partial(_extract, root=str(Path(root).resolve()), processing=processing)
    columns = CELL_COLUMNS + ['pbi', 'beta95', 'corner_fallback']
    cv2.setNumThreads(1)
    if workers == 1:
        iterator = map(work, frame.relpath)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=cv2.setNumThreads, initargs=(1,))
        iterator = pool.map(work, frame.relpath, chunksize=32)
    rows = []
    try:
        for i, row in enumerate(iterator, 1):
            rows.append(row)
            if i % 500 == 0 or i == len(frame):
                print(f'Processed {i}/{len(frame)} images', flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows, columns=columns)], axis=1)


def feature_signature(processing):
    values = {k: v for k, v in processing.items() if k not in {'full_blockage', 'top_cells'}}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def save_features(frame, path, processing, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    write_json(str(path)+'.meta.json', {'schema_version': 1, 'processing': processing,
               'feature_signature': feature_signature(processing), **provenance})


def load_features(path, processing):
    path = Path(path)
    sidecar = Path(str(path)+'.meta.json')
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    if metadata['feature_signature'] != feature_signature(processing):
        raise ValueError('Processing settings differ from the feature cache; recompute features or select its configuration')
    frame = normalize_metadata(pd.read_csv(path))
    if set(CELL_COLUMNS)-set(frame.columns):
        raise ValueError('Feature file must contain m00 through m71')
    frame['pbi'] = regional_index(frame[CELL_COLUMNS].to_numpy(),
                                  processing['full_blockage'], processing['top_cells'])
    if 'stable' in frame:
        if frame.stable.dtype != bool:
            converted = frame.stable.astype(str).str.lower().map({'true': True, 'false': False})
            if converted.isna().any():
                raise ValueError('stable must contain only true/false values')
            frame['stable'] = converted.astype(bool)
    return frame
