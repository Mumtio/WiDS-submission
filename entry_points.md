# Entry Points

This solution was developed as a single Kaggle notebook.

The main solution file is:

```text
wids-datathon-2026-5th-place-solution.ipynb
```

A Python script version is also included:

```text
wids-datathon-2026-5th-place-solution.py
```

The solution trains the full model ensemble and generates the final submission in one run.

---

# 1. Kaggle Notebook Entry Point

Recommended reproduction method:

```text
Run all cells in wids-datathon-2026-5th-place-solution.ipynb from top to bottom.
```

The notebook will:

1. Install/import required packages.
2. Load the official Kaggle competition data.
3. Engineer wildfire survival-analysis features.
4. Train the survival ensemble using cross-validation.
5. Generate out-of-fold diagnostics.
6. Create predictions for the test set.
7. Save the final submission file.

Expected output:

```text
submission.csv
```

In the Kaggle environment, the output file is written to:

```text
/kaggle/working/submission.csv
```

---

# 2. Python Script Entry Point

The notebook was also exported as a Python script.

To run the full training and prediction pipeline from the command line:

```bash
python wids-datathon-2026-5th-place-solution.py
```

This command performs both training and prediction.

Expected output:

```text
submission.csv
```

---

# 3. Data Input

The official Kaggle competition data is not included in this archive.

The code expects the competition files to be available at:

```text
/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26/train.csv
/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26/test.csv
```

The required input files are:

```text
train.csv
test.csv
```

The main configuration in the code uses:

```python
data_dir = Path("/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26")
train_name = "train.csv"
test_name = "test.csv"
```

---

# 4. Local Run Instructions

If running locally instead of Kaggle:

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Place the official Kaggle competition files in the expected data directory, or update the `data_dir` path in the configuration.

3. Run:

```bash
python wids-datathon-2026-5th-place-solution.py
```

---

# 5. Main Function

The script entry point is:

```python
if __name__ == "__main__":
    main()
```

The `main()` function calls:

```python
build_and_write_submission(CFG)
```

This function runs the complete pipeline and writes the final submission file.

---

# 6. Output

The final generated file is:

```text
submission.csv
```

Expected columns:

```text
event_id
prob_12h
prob_24h
prob_48h
prob_72h
```

---

# 7. Notes

This solution does not use external data.

The complete model is trained inside the notebook/script. The workflow does not require a separate preprocessing script, training script, or prediction script.

Running the notebook/script may overwrite:

```text
submission.csv
```

No original Kaggle competition data files are modified.
