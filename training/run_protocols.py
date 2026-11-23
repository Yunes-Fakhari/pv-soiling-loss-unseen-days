import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import pandas as pd
from pbi_loss.config import load_config, write_json
from pbi_loss.data import load_features
from pbi_loss.experiments import run_protocol, summarize


def main():
    parser = argparse.ArgumentParser(description='Run all manuscript evaluation protocols')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/paper.json')
    parser.add_argument('--features', type=Path, default=ROOT/'data/processed/features.csv.gz')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/paper')
    args = parser.parse_args()
    config = load_config(args.config)
    frame = load_features(args.features, config['processing'])
    records = run_protocol(frame, config, args.output/'lodo', 'lodo')
    for protocol in ('random_days', 'images'):
        for seed in config['seeds']:
            records.extend(run_protocol(frame, config, args.output/protocol/f'seed_{seed}', protocol, seed))
    records.extend(run_protocol(frame, config, args.output/'fixed_days', 'fixed_days',
                                 test_days=config['test_days']))
    pd.DataFrame(records).to_csv(args.output/'metrics_per_run.csv', index=False)
    summary = summarize(records)
    summary.to_csv(args.output/'metrics_summary.csv', index=False)
    write_json(args.output/'run_config.json', config)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
