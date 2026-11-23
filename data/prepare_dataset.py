import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import pandas as pd
from pbi_loss.config import load_config, write_json
from pbi_loss.data import (scan_dataset, normalize_metadata, subsample, diagnostic_subset,
                           extract_dataset, save_features)


def main():
    parser = argparse.ArgumentParser(description='Build metadata and PBI-Loss features from raw images')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/paper.json')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--metadata', type=Path, help='Optional CSV with relpath, datetime, loss')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    root = (args.data_root or Path(config['dataset_root'])).resolve()
    output = args.output_dir or Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    full = normalize_metadata(pd.read_csv(args.metadata)) if args.metadata else scan_dataset(root)
    sampled = subsample(full, config['sampling_seconds'])
    sampled['stable'] = diagnostic_subset(sampled, full)
    full.to_csv(output/'metadata_all.csv.gz', index=False)
    sampled.to_csv(output/'metadata_sampled.csv.gz', index=False)
    print(f'{len(full)} raw frames; {len(sampled)} sampled; {sampled.date.nunique()} days', flush=True)
    features = extract_dataset(sampled, root, config['processing'], args.workers)
    provenance = {'source': 'raw_images', 'raw_n': len(full), 'sampled_n': len(sampled),
                  'diagnostic_n': int(sampled.stable.sum()), 'sampling_seconds': config['sampling_seconds']}
    save_features(features, output/'features.csv.gz', config['processing'], provenance)
    write_json(output/'preparation.json', provenance)
    print(f'Features saved to {output / "features.csv.gz"}')


if __name__ == '__main__':
    main()
