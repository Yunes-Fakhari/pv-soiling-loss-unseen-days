import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import pandas as pd
from pbi_loss.config import write_json
from pbi_loss.metrics import regression_metrics


def main():
    parser = argparse.ArgumentParser(description='Recalculate metrics from saved held-out predictions')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/evaluation.json')
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    results = {'all': regression_metrics(frame.loss, frame.prediction)}
    if 'stable' in frame:
        stable = frame.stable.astype(str).str.lower().map({'true': True, 'false': False})
        if stable.isna().any():
            raise ValueError('stable must contain true/false')
        subset = frame.loc[stable.astype(bool)]
        if len(subset):
            results['clean_diagnostic'] = regression_metrics(subset.loss, subset.prediction)
    write_json(args.output, results)
    print(results)


if __name__ == '__main__':
    main()
