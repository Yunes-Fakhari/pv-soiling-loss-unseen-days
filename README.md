# PBI-Loss

Image processing and interpretable regression for photovoltaic soiling power-loss estimation. The method rectifies one camera frame, applies CLAHE, segments blueness loss, aggregates blockage over three substring regions, and estimates relative power loss:

```text
loss = clip(a * PBI^0.7 + b * PBI^0.7 * sun(hour) + c, 0, 1)
sun(hour) = max(0, sin(pi * (hour - 6) / 12.5))
```

Training fits only `a`, `b`, and `c` by ordinary least squares. No neural network or GPU is required. The archived MobileNet comparison model is separate from this implementation.

[راهنمای فارسی](docs/README_FA.md)

## Repository layout

```text
configs/             Dataset paths and published method settings
data/                Dataset preparation script
  raw/               Put the original PanelImages directory here
  processed/         Generated metadata and image features
  reference/         Archived cell shares for numerical reproduction
src/pbi_loss/        Rectification, segmentation, regional index and regression
training/            Coefficient fitting and manuscript protocols
evaluation/          Error metrics and reproduction checks
inference/           Prediction for a new image and timestamp
models/              Two fitted, small JSON coefficient files
results/paper/       Expected manuscript metrics and verification record
tests/               Numerical, data-split and inference checks
outputs/             Generated models, splits, predictions and metrics
docs/                Dataset format and reproduction details
```

## Install

Use Python 3.10 or newer. Run commands from the repository root.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements-tested.txt` records the dependency versions used for the supplied verification. For an exact environment, install that file instead of `requirements.txt`.

## Dataset and image processing

Download DeepSolarEye from the [official project](https://deep-solar-eye.github.io/). Extract the full 45,754-image dataset as:

```text
data/raw/PanelImages/solar_Tue_Jun_13_9__46__49_2017_L_0.0474843723155_I_0.296733333333.jpg
```

Run:

```bash
python data/prepare_dataset.py --workers 4
```

For a dataset stored elsewhere:

```bash
python data/prepare_dataset.py --data-root "D:/Datasets/DeepSolarEye" --workers 4
```

Alternatively, edit `dataset_root` in `configs/paper.json`. Relative configuration paths are resolved against the repository root; command-line paths are resolved against the current working directory. The dataset root must contain the paths recorded in the metadata, normally `PanelImages/`.

Preparation selects the first frame in each 30-second window and produces 11,632 frames from the complete dataset. It saves metadata, cell shares and processing provenance to `data/processed/`. Images are processed as full 192×192 camera frames, not pre-cropped panel images. Irradiance labels are retained in metadata but are not used by PBI-Loss.

## Train

Fit on 12 randomly selected training days and evaluate on the other three:

```bash
python training/train.py --protocol random_days --seed 42 --output outputs/random_days_42
```

Other protocols:

```bash
python training/train.py --protocol lodo --output outputs/lodo
python training/train.py --protocol images --seed 42 --output outputs/images_42
python training/train.py --protocol fixed_days --test-days 2017-06-13 2017-06-15 2017-06-22 --output outputs/fixed_days
```

Each run saves fitted JSON models, exact training/test image manifests, held-out predictions, MSE, RMSE, MAE and R². LODO fits 15 models and pools predictions for the 15 excluded days. Repeated image/day scores are calculated per seed before averaging.

To fit a deployment model on all available labeled images:

```bash
python training/train.py --protocol deployment --output outputs/deployment
```

This last command has no held-out test set and produces no test accuracy claim.

## Reproduce the manuscript

From newly processed images:

```bash
python training/run_protocols.py --output outputs/paper
python evaluation/verify_paper.py --summary outputs/paper/metrics_summary.csv --atol 0.00005
```

For a quick numerical reproduction without downloading images, use the supplied archived cell-share file:

```bash
python training/run_protocols.py --features data/reference/paper_features.csv.gz --output outputs/reference
python evaluation/verify_paper.py --summary outputs/reference/metrics_summary.csv
```

The archive contains 72 cell shares per sampled image, timestamps, loss labels and the diagnostic-subset flag. It contains no images or learned predictions. Training and prediction are recalculated. This numerical check alone does not exercise raw-image preprocessing. See [reproduction details](docs/REPRODUCIBILITY.md) for precision, subset definitions and the distinction between these checks.

## Predict one image

```bash
python inference/predict.py --model outputs/random_days_42/models/seed_42.json --image "path/to/frame.jpg" --hour 10.75 --output outputs/prediction.json --visuals outputs/example
```

`--timestamp 2017-06-22T10:45:00` can replace `--hour`. Supply local acquisition time. Prediction reads the pixels and explicit time, not the loss or irradiance embedded in an image filename. The saved model carries the processing settings used during training.

For a first check without fitting a model, use `--model models/random_days_seed42.json`. `models/all_days.json` is fitted on all sampled data and carries no held-out accuracy claim.

The default corners and substring layout describe the DeepSolarEye camera view. A different camera or module requires an appropriate geometric setup and independent evaluation. A nonpositive blue reference raises an error instead of returning an unsupported loss estimate.

## Evaluate saved predictions and run checks

```bash
python evaluation/evaluate.py --predictions outputs/lodo/predictions.csv.gz --output outputs/lodo/recomputed_metrics.json
python -m unittest discover -s tests -v
```

## Dataset citation

S. Mehta, A. P. Azad, S. A. Chemmengath, V. Raykar, and S. Kalyanaraman, “DeepSolarEye: Power loss prediction and weakly supervised soiling localization via fully convolutional networks for solar panels,” WACV, pp. 333–342, 2018. DOI: [10.1109/WACV.2018.00043](https://doi.org/10.1109/WACV.2018.00043).

Dataset images are obtained separately from the original authors and remain subject to their distribution terms.
