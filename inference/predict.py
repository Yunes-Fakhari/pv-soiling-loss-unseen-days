import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from PIL import Image
from pbi_loss.config import write_json
from pbi_loss.model import LossModel
from pbi_loss.pipeline import extract_features


def main():
    parser = argparse.ArgumentParser(description='Estimate loss from one full camera frame and local acquisition time')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    time = parser.add_mutually_exclusive_group(required=True)
    time.add_argument('--hour', type=float, help='Local decimal hour, e.g. 10.75')
    time.add_argument('--timestamp', help='Local ISO timestamp, e.g. 2017-06-22T10:45:00')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/prediction.json')
    parser.add_argument('--visuals', type=Path, help='Optional folder for crop, enhanced image and mask')
    args = parser.parse_args()
    if args.timestamp:
        timestamp = datetime.fromisoformat(args.timestamp)
        if timestamp.tzinfo is not None:
            raise ValueError('Supply local camera time without a timezone offset')
        hour = timestamp.hour + timestamp.minute/60 + timestamp.second/3600
    else:
        hour = args.hour
    model, metadata = LossModel.load(args.model)
    features = extract_features(args.image, metadata['processing'], return_images=bool(args.visuals))
    prediction = float(model.predict([features['pbi']], [hour])[0])
    output = {'image': str(args.image), 'hour': hour, 'pbi': features['pbi'],
              'predicted_relative_loss': prediction, 'predicted_loss_percent': 100*prediction,
              'beta95': features['beta95'], 'corner_fallback': features['corner_fallback'],
              'corners': features['corners']}
    write_json(args.output, output)
    if args.visuals:
        args.visuals.mkdir(parents=True, exist_ok=True)
        Image.fromarray(features['crop']).save(args.visuals/'rectified.png')
        Image.fromarray(features['enhanced']).save(args.visuals/'clahe.png')
        Image.fromarray(features['mask'].astype(np.uint8)*255).save(args.visuals/'soiling_mask.png')
    print(f'PBI={features["pbi"]:.6f}; predicted relative loss={prediction:.6f} ({100*prediction:.2f}%)')


if __name__ == '__main__':
    main()
