# Verification records

`expected_metrics.csv` contains the previously audited primary manuscript metrics. `archived_features_metrics.csv` and `raw_image_metrics.csv` were recalculated using this package; `verification.json` records their differences and the validation scope.

The numerical archive reproduces all 32 primary metric values within 1e−10. The complete raw-image run was checked on all 11,632 selected frames, including the same 9,902 diagnostic frames, against an absolute metric tolerance of 5e−5.

`raw_image_ablation.csv` contains the newly recalculated component scores with full precision. Some historical rounded entries of manuscript Table II differ in their last digits, especially the random-day one-cell variant and the no-CLAHE row. The code reports its computed values rather than forcing historical table values. Use this CSV when checking or updating those entries.

No dataset images, MobileNet weights, or generated training directories are included in this release.
