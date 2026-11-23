# Reproduction details

The default configuration implements the fixed PBI-Loss method used for the manuscript's primary scores. Training uses ordinary least squares on `[PBI^0.7, PBI^0.7 * sun(hour), 1]`; predictions are clipped to [0, 1]. Coefficients are fitted separately inside each training split.

## Processing convention

- Full input frame: 192×192 pixels; nominal corners in `configs/paper.json`.
- Frame localization: Pillow grayscale and Gaussian blur 0.8, followed by the two-stage endpoint search described in the manuscript.
- Perspective mapping: corner destinations (0,0), (191,0), (191,191), (0,191), OpenCV linear interpolation.
- CLAHE: 8-bit OpenCV CIELAB lightness, clip limit 2, 4×4 tiles.
- Blue excess: `(B - max(R,G)) / (max(R,G,B) + 1)` in floating point.
- Reference: the 95th percentile excluding the six-pixel margin.
- Morphology: grayscale opening of the loss-degree map with the 3×3 elliptical element, followed by threshold ≥0.7. At a fixed threshold this is equivalent to opening the binary mask.
- Cell shares: soiled pixels divided by valid pixels of each 16-row ×32-column cell, excluding the margin.
- The historical cell-share archive used float16 storage. The default rounds shares to float16 before returning to float64 for regression, preserving that convention. The RGB and degree calculations do not use float16.

The six fixed module columns form three adjacent pairs. Each region contains 24 cells; its score is the mean of the two largest `min(1, cell_share/0.7)` values. PBI is the mean of the three regional scores. This implementation uses no clean neighboring panel.

## Exact splits

Dates are sorted before applying NumPy `default_rng(seed).permutation`. Random-day splits hold out the first three permuted dates. Image splits hold out the first `round(0.2 * N)` permuted chronological rows.

| Protocol | Test days |
|---|---|
| Random days, seed 42 | June 22, 23, 25 |
| Random days, seed 7 | June 16, 22, 26 |
| Random days, seed 123 | June 15, 23, 30 |
| Explicit comparison dates | June 13, 15, 22 |

LODO pools all held-out predictions before calculating metrics. For the three-seed protocols, each metric is calculated per seed before averaging; therefore mean RMSE is not defined as the square root of mean MSE. MSE standard deviations use `ddof=1`.

`results/paper/expected_metrics.csv` records the previously audited manuscript values. The archived cell-share reproduction should match within 1e−10. Raw-image preprocessing can differ slightly across imaging-library versions; compare those scores at manuscript precision, with an initial absolute tolerance of 5e−5. The tested versions are pinned separately.

## Component ablations

```bash
python evaluation/ablation.py --features data/reference/paper_features.csv.gz --output outputs/ablation.csv
```

The six variants available from the primary shares are recalculated with the same split rules. To include the no-CLAHE row, generate its separate cache:

```bash
python data/prepare_dataset.py --config configs/no_clahe.json --data-root "D:/Datasets/DeepSolarEye" --workers 4
python evaluation/ablation.py --no-clahe-features data/processed_no_clahe/features.csv.gz --output outputs/ablation.csv
```

The historical exploratory mask-selection search is not rerun by these fixed-method reproduction commands. Some fixed constants were developed using the same acquisition days, as stated in the manuscript. These runs establish computational reproduction, not independent site or module transfer.

The explicit-date protocol reproduces the PBI-Loss row. The archived MobileNet experiment used the same dates but a different selection of frames and auxiliary inputs; its code and training are not part of this package.
