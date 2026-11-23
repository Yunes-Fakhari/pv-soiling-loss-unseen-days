Place the extracted DeepSolarEye `PanelImages/` directory here, or supply another dataset root with `--data-root`.

Expected structure:

```text
data/raw/PanelImages/solar_Tue_Jun_13_9__46__49_2017_L_0.0474843723155_I_0.296733333333.jpg
```

Official dataset and download link: https://deep-solar-eye.github.io/

Images are not redistributed in this repository. Follow the dataset authors' terms and cite DeepSolarEye. Images must retain their original filenames for automatic metadata extraction. Alternatively, provide a CSV with relative `relpath`, local `datetime`, and `loss` columns using `--metadata`.
