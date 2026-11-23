import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pbi_loss.config import load_config
from pbi_loss.data import load_features
from pbi_loss.experiments import run_protocol
from pbi_loss.model import LossModel


def main():
    parser = argparse.ArgumentParser(description='Fit the three PBI-Loss coefficients')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/paper.json')
    parser.add_argument('--features', type=Path, default=ROOT/'data/processed/features.csv.gz')
    parser.add_argument('--protocol', choices=['random_days', 'lodo', 'images', 'fixed_days', 'deployment'], default='random_days')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-days', nargs='+', help='YYYY-MM-DD dates for fixed_days')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/train')
    args = parser.parse_args()
    config = load_config(args.config)
    frame = load_features(args.features, config['processing'])
    if args.protocol == 'deployment':
        model = LossModel.fit(frame.pbi, frame.hour_decimal, frame.loss, **config['model'])
        model.save(args.output/'deployment_model.json', config['processing'],
                   {'protocol': 'deployment', 'train_n': len(frame),
                    'train_days': sorted(frame.date.unique().tolist()),
                    'test_days': [], 'held_out_evaluation': False})
        print(f'Deployment model fitted on all {len(frame)} frames; no held-out score is produced.')
    else:
        records = run_protocol(frame, config, args.output, args.protocol, args.seed,
                               args.test_days or config['test_days'])
        for record in records:
            print(record)


if __name__ == '__main__':
    main()
