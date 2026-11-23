import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from pbi_loss.config import load_config
from pbi_loss.data import CELL_COLUMNS, load_features
from pbi_loss.model import sun_factor
from pbi_loss.pipeline import regional_index
from pbi_loss.protocols import make_folds


def mse_for(frame, index, settings, protocol, seed, exponent=0.7, include_sun=True):
    q = index**exponent
    phi = sun_factor(frame.hour_decimal, settings['sun_start'], settings['sun_duration'])
    x = np.column_stack([q, q*phi, np.ones(len(q))]) if include_sun else np.column_stack([q, np.ones(len(q))])
    labels, errors = frame.loss.to_numpy(), []
    for _, train, test in make_folds(frame, protocol, seed):
        coefficients = np.linalg.lstsq(x[train], labels[train], rcond=None)[0]
        errors.extend((np.clip(x[test]@coefficients, 0, 1)-labels[test])**2)
    return float(np.mean(errors))


def main():
    parser = argparse.ArgumentParser(description='Reproduce component ablations with identical data splits')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/paper.json')
    parser.add_argument('--features', type=Path, default=ROOT/'data/processed/features.csv.gz')
    parser.add_argument('--no-clahe-features', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/ablation.csv')
    args = parser.parse_args()
    config = load_config(args.config)
    frame = load_features(args.features, config['processing'])
    shares = frame[CELL_COLUMNS].to_numpy()
    u = np.minimum(1, shares/config['processing']['full_blockage'])
    exponent = config['model']['exponent']
    variants = {
        'PBI-Loss (full)': (frame.pbi.to_numpy(), exponent, True),
        'Mean of all 72 cells': (u.mean(1), exponent, True),
        'Two worst cells globally': (np.sort(u, axis=1)[:, -2:].mean(1), exponent, True),
        'One cell per region': (regional_index(shares, config['processing']['full_blockage'], 1), exponent, True),
        'Linear in PBI': (frame.pbi.to_numpy(), 1.0, True),
        'Without solar-time factor': (frame.pbi.to_numpy(), exponent, False),
    }
    if args.no_clahe_features:
        processing = deepcopy(config['processing'])
        processing['clahe'] = False
        other = load_features(args.no_clahe_features, processing)
        if not frame[['relpath', 'loss', 'datetime']].equals(other[['relpath', 'loss', 'datetime']]):
            raise ValueError('The no-CLAHE features must use exactly the same images, timestamps and labels')
        variants['Without CLAHE'] = (other.pbi.to_numpy(), exponent, True)
    rows = []
    for name, (index, power, use_sun) in variants.items():
        row = {'variant': name}
        for protocol in ('lodo', 'random_days', 'images'):
            seeds = [42] if protocol == 'lodo' else config['seeds']
            row[protocol] = float(np.mean([mse_for(frame, index, config['model'], protocol, seed,
                                                  power, use_sun) for seed in seeds]))
        rows.append(row)
    result = pd.DataFrame(rows)
    for protocol in ('lodo', 'random_days', 'images'):
        result[protocol+'_change_percent'] = 100*(result[protocol]/result.iloc[0][protocol]-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    if not args.no_clahe_features:
        print('Without CLAHE was not run; provide --no-clahe-features to include that row.')


if __name__ == '__main__':
    main()
