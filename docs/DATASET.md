# Dataset format

Download the original dataset through https://deep-solar-eye.github.io/. Do not use rectified crops in place of the full camera frames.

The default root is `data/raw/`, with images under `PanelImages/`. Nested directories are supported. If using an existing metadata CSV, all `relpath` entries are relative to the `--data-root` directory.

Automatic parsing reads local acquisition time, relative loss `L`, and irradiance `I` from the original filename. `I` is not a model input. Custom metadata must contain:

```csv
relpath,datetime,loss
PanelImages/frame_0001.jpg,2017-06-13 09:00:00,0.12
PanelImages/frame_0002.jpg,2017-06-13 09:00:05,0.13
```

Custom files must be 192×192 RGB-compatible images of the same full camera view. Relative loss is on the 0–1 scale, so a 12% loss is `0.12`. Use local, timezone-naive timestamps. Paths must be unique.

```bash
python data/prepare_dataset.py --data-root "D:/Datasets/DeepSolarEye" --metadata path/to/metadata.csv --workers 4
```

Preparation sorts all frames by timestamp and relative path, then selects the first frame of each 30-second window separately per acquisition day. The complete dataset produces 45,754 raw and 11,632 selected frames on 15 days.

`features.csv.gz` contains metadata and `m00`–`m71`: the masked fractions in 72 cells ordered row by row. `pbi` is the regional index. `beta95` and `corner_fallback` are diagnostic fields. A JSON sidecar records processing settings and their signature; training rejects an incompatible feature cache.

The supplied `data/reference/paper_features.csv.gz` contains archived cell shares, not images. Its `stable` flag was computed using the full original label series. It permits coefficient fitting and metric reproduction without raw-image access.

The diagnostic `stable` flag keeps a label only if its difference from the median within ±90 seconds is at most 0.05 and the medians before/after the image do not differ by more than 0.25. The before/after windows are [t−180, t−30] and [t+30, t+180] seconds, with at least three labels in each. Labels themselves are not replaced. Since this diagnostic uses test labels, its scores are not the primary deployment benchmark.
