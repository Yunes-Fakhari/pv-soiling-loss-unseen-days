import json
from copy import deepcopy
from pathlib import Path

DEFAULTS = {
    'dataset_root': 'data/raw',
    'output_dir': 'data/processed',
    'sampling_seconds': 30,
    'processing': {
        'corners': [[80, 21], [186, 35], [160, 164], [17, 145]],
        'size': 192, 'margin': 6, 'clahe': True, 'clahe_clip': 2.0,
        'clahe_tiles': [4, 4], 'mask_threshold': 0.7,
        'full_blockage': 0.7, 'top_cells': 2,
        'feature_precision': 'float16',
    },
    'model': {'exponent': 0.7, 'sun_start': 6.0, 'sun_duration': 12.5},
    'seeds': [42, 7, 123],
    'test_days': ['2017-06-13', '2017-06-15', '2017-06-22'],
}


def load_config(path):
    path = Path(path).resolve()
    config = deepcopy(DEFAULTS)
    supplied = json.loads(path.read_text(encoding='utf-8'))
    for key, value in supplied.items():
        if key not in config:
            raise ValueError(f'Unknown configuration key: {key}')
        if isinstance(config[key], dict):
            unknown = set(value) - set(config[key])
            if unknown:
                raise ValueError(f'Unknown {key} options: {sorted(unknown)}')
            config[key].update(value)
        else:
            config[key] = value
    for key in ('dataset_root', 'output_dir'):
        value = Path(config[key]).expanduser()
        config[key] = str(value if value.is_absolute() else path.parent.parent / value)
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
