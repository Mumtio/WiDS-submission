# Model Artifacts

This solution was developed as a single Kaggle notebook workflow.

The complete model ensemble is trained inside:

```text
wids-datathon-2026-5th-place-solution.ipynb
```

A Python script version is also included:

```text
wids-datathon-2026-5th-place-solution.py
```

The notebook/script trains the full survival-analysis ensemble from scratch and generates the final submission file.

## Model Components

The solution uses an ensemble of the following model families:

```text
Gradient Boosting Survival Analysis
Cox Proportional Hazards Survival Model
Random Survival Forest
XGBoost AFT Survival Model, if available
LightGBM timing/calibration models, if available
Two-stage logistic + isotonic calibration models
OOF zone-based blending and optional meta-stacking
```

## Serialized Model File

The original workflow primarily trains the complete ensemble inside the notebook/script.

If a serialized model bundle is created, it should be saved at:

```text
models/model_bundle.pkl
```

At the time of this archive, the reproducible workflow is:

```text
Run wids-datathon-2026-5th-place-solution.ipynb from top to bottom
```

or:

```bash
python wids-datathon-2026-5th-place-solution.py
```

This retrains the models and writes the final prediction file.

## Required Data

The official Kaggle competition data is not included in this archive.

The notebook expects:

```text
/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26/train.csv
/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26/test.csv
```

These files must be obtained directly from the official Kaggle competition page.

## Output

The final generated prediction file is:

```text
submission.csv
```



## External Data

No external data was used.

Only the official Kaggle competition train and test files were used.
