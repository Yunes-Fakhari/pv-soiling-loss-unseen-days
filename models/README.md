# Fitted coefficient files

`random_days_seed42.json` was fitted on 12 acquisition days using the archived primary cell shares. Its held-out dates are June 22, 23 and 25, 2017. It is included for the single-image example.

`all_days.json` was fitted on all 11,632 sampled images for deployment experiments. It has no held-out test score.

Each file stores the three coefficients, exponent, solar-time constants, processing settings and training provenance. These small JSON files are regression models, not neural-network weights. Retrain with `training/train.py` when changing the data or processing assumptions.
