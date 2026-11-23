# Archived primary features

`paper_features.csv.gz` contains the archived 72 float16 cell shares per selected image, relative paths, timestamps, loss/irradiance metadata, and the label-selected diagnostic flag. It contains 11,632 rows on 15 acquisition days, including 9,902 diagnostic rows. Irradiance is not used by PBI-Loss.

The adjacent `.meta.json` file identifies the processing settings. These features reproduce the numerical primary results without distributing the images. They do not replace testing `data/prepare_dataset.py` on the original dataset.
