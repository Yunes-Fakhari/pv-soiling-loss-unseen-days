import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Compare reproduced scores with the manuscript reference')
    parser.add_argument('--summary', type=Path, default=ROOT/'outputs/paper/metrics_summary.csv')
    parser.add_argument('--reference', type=Path, default=ROOT/'results/paper/expected_metrics.csv')
    parser.add_argument('--atol', type=float, default=1e-10)
    args = parser.parse_args()
    actual = pd.read_csv(args.summary).set_index(['protocol', 'subset'])
    reference = pd.read_csv(args.reference).set_index(['protocol', 'subset'])
    if set(actual.index) != set(reference.index):
        raise ValueError('The set of protocols/subsets does not match the reference')
    columns = ['MSE', 'RMSE', 'MAE', 'R2']
    actual = actual.loc[reference.index, columns]
    np.testing.assert_allclose(actual.to_numpy(), reference[columns].to_numpy(), atol=args.atol, rtol=0)
    print(f'PASS: {len(reference)*len(columns)} metric values match, atol={args.atol:g}')


if __name__ == '__main__':
    main()
