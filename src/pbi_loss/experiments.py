from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

from .config import write_json
from .metrics import regression_metrics
from .model import LossModel
from .protocols import make_folds


def run_protocol(frame, config, output, protocol, seed=42, test_days=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = np.full(len(frame), np.nan)
    tested = np.zeros(len(frame), dtype=bool)
    fold_rows = []
    for name, train, test in make_folds(frame, protocol, seed, test_days):
        model = LossModel.fit(frame.pbi.to_numpy()[train], frame.hour_decimal.to_numpy()[train],
                              frame.loss.to_numpy()[train], **config['model'])
        predictions[test] = model.predict(frame.pbi.to_numpy()[test], frame.hour_decimal.to_numpy()[test])
        tested[test] = True
        train_days = sorted(frame.iloc[train].date.unique().tolist())
        held_days = sorted(frame.iloc[test].date.unique().tolist())
        provenance = {'protocol': protocol, 'seed': seed, 'train_days': train_days,
                      'test_days': held_days, 'train_n': len(train), 'test_n': len(test),
                      'training_paths_sha256': hashlib.sha256('\n'.join(frame.iloc[train].relpath).encode()).hexdigest()}
        model.save(output/'models'/f'{name}.json', config['processing'], provenance)
        split = frame[['relpath', 'date']].copy()
        split['subset'] = 'unused'
        split.loc[train, 'subset'], split.loc[test, 'subset'] = 'train', 'test'
        (output/'splits').mkdir(exist_ok=True)
        split.to_csv(output/'splits'/f'{name}.csv.gz', index=False)
        fold_rows.append({'fold': name, **provenance,
                          **dict(zip(('a', 'b', 'c'), model.coefficients))})
    result = frame.loc[tested, ['relpath', 'datetime', 'date', 'loss', 'pbi']].copy()
    result['prediction'] = predictions[tested]
    if 'stable' in frame:
        result['stable'] = frame.loc[tested, 'stable'].to_numpy()
    result.to_csv(output/'predictions.csv.gz', index=False)
    records = [{'protocol': protocol, 'seed': seed if protocol != 'lodo' else None,
                'subset': 'all', **regression_metrics(result.loss, result.prediction)}]
    if 'stable' in result and result.stable.any():
        subset = result.loc[result.stable]
        records.append({'protocol': protocol, 'seed': seed if protocol != 'lodo' else None,
                        'subset': 'clean_diagnostic', **regression_metrics(subset.loss, subset.prediction)})
    write_json(output/'metrics.json', records)
    write_json(output/'folds.json', fold_rows)
    return records


def summarize(records):
    rows = []
    for (protocol, subset), group in pd.DataFrame(records).groupby(['protocol', 'subset'], sort=False):
        row = {'protocol': protocol, 'subset': subset, 'runs': len(group)}
        for key in ('MSE', 'RMSE', 'MAE', 'R2'):
            row[key] = float(group[key].mean())
            row[key+'_SD'] = float(group[key].std(ddof=1)) if len(group) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)
