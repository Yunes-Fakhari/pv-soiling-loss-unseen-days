from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from .config import write_json


def sun_factor(hours, start=6.0, duration=12.5):
    hours = np.asarray(hours, dtype=float)
    if not np.isfinite(hours).all() or ((hours < 0) | (hours >= 24)).any() or duration <= 0:
        raise ValueError('hours must be finite local hours in [0, 24); duration must be positive')
    return np.maximum(0, np.sin(np.pi*(hours-start)/duration))


def design(index, sun, exponent=0.7):
    index, sun = np.broadcast_arrays(np.asarray(index, float), np.asarray(sun, float))
    if not np.isfinite(index).all() or not np.isfinite(sun).all():
        raise ValueError('Inputs must be finite')
    if ((index < 0) | (index > 1) | (sun < 0) | (sun > 1)).any() or exponent <= 0:
        raise ValueError('PBI and solar factor must be in [0, 1]; exponent must be positive')
    q = index.ravel()**exponent
    return np.column_stack([q, q*sun.ravel(), np.ones(q.size)])


@dataclass
class LossModel:
    coefficients: list
    exponent: float = 0.7
    sun_start: float = 6.0
    sun_duration: float = 12.5

    @classmethod
    def fit(cls, index, hours, labels, exponent=0.7, sun_start=6.0, sun_duration=12.5):
        labels = np.asarray(labels, dtype=float)
        x = design(index, sun_factor(hours, sun_start, sun_duration), exponent)
        if labels.ndim != 1 or len(labels) != len(x) or len(labels) < 3:
            raise ValueError('Training requires at least three aligned samples')
        if not np.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any():
            raise ValueError('Training labels must be finite losses in [0, 1]')
        coefficients, _, rank, _ = np.linalg.lstsq(x, labels, rcond=None)
        if rank != 3:
            raise ValueError('Training predictors are rank deficient; use more varied images and times')
        return cls(coefficients.tolist(), exponent, sun_start, sun_duration)

    def predict(self, index, hours):
        x = design(index, sun_factor(hours, self.sun_start, self.sun_duration), self.exponent)
        return np.clip(x @ np.asarray(self.coefficients), 0, 1)

    def save(self, path, processing, provenance):
        write_json(path, {'schema_version': 1, 'method': 'PBI-Loss',
                          'coefficients': dict(zip(('a', 'b', 'c'), self.coefficients)),
                          'model': {'exponent': self.exponent, 'sun_start': self.sun_start,
                                    'sun_duration': self.sun_duration},
                          'processing': processing, 'training': provenance})

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('method') != 'PBI-Loss' or data.get('schema_version') != 1:
            raise ValueError('Unsupported model format')
        coefficients = [data['coefficients'][k] for k in ('a', 'b', 'c')]
        if not np.isfinite(coefficients).all():
            raise ValueError('Model coefficients must be finite')
        return cls(coefficients, **data['model']), data
