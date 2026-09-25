# Reproduction guide

The workflow progresses from seven MIMIC-IV tables to a shared cohort, baseline models, mitigation policies and aggregate evaluation. Run commands from the repository root. Examples use PowerShell and keep the prepared patient-level data outside the repository.

## Environment and data

The recorded evaluation environment is Python 3.11.15. Direct dependencies are pinned in [`requirements.txt`](../../requirements.txt), including PyTorch 2.11.0 and Fairlearn 0.13.0. The original PyTorch build was `2.11.0+cu128`; choose a compatible CPU/CUDA build for the machine used.

```powershell
python -m pip install -r requirements.txt
python -m pytest
$MIMIC_DIR = "D:\data\mimiciv\3.1"
$DATA_DIR = "D:\data\mimic4-ihm-prepared"
```

The raw-data root must contain:

```text
hosp/patients.csv[.gz]       hosp/admissions.csv[.gz]
hosp/d_labitems.csv[.gz]     hosp/labevents.csv[.gz]
icu/icustays.csv[.gz]       icu/d_items.csv[.gz]
icu/chartevents.csv[.gz]
```

Access is obtained through [PhysioNet's MIMIC-IV 3.1 record](https://physionet.org/content/mimiciv/3.1/). Preserve the source files; the pipeline can read gzip directly. Event ingestion uses chunks of 1,000,000 rows, an on-disk SQLite database and memory-mapped arrays. This documentation does not prescribe an unmeasured runtime or minimum disk estimate.

## 1. Build the cohort and features

Input: the seven raw tables. Command:

```powershell
python -m mimic4_ihm.build_dataset --mimic-root "$MIMIC_DIR" --output-dir "$DATA_DIR" --seed 49297
```

Outputs include `metadata.csv`, `lstm_X.npy`, `lr_X_raw.npy`, `y.npy`, `split.npy`, `config.json`, `split_manifest.csv`, `events.sqlite` and the audits described in [data cleaning](data_cleaning.md). On the formal full-data run, the arrays have shapes `(36648, 48, 76)` and `(36648, 714)`.

`--max-stays` and `--subject-list` are optional subset controls. Omit them for the formal cohort. Existing prepared files are protected unless `--overwrite` is explicitly supplied. A new output directory is preferable when comparing a different preparation rule.

## 2. Fit and evaluate the baseline models

Input: the prepared directory. Commands:

```powershell
python -m mimic4_ihm.train --model logistic_regression --data-dir "$DATA_DIR" --output-dir "$DATA_DIR/lr"
python -m mimic4_ihm.train --model cw_lstm --data-dir "$DATA_DIR" --output-dir "$DATA_DIR/cw_lstm" --device cuda --epochs 100
python -m mimic4_ihm.evaluate --metadata "$DATA_DIR/metadata.csv" --predictions "lr=$DATA_DIR/lr/predictions.csv" "cw_lstm=$DATA_DIR/cw_lstm/predictions.csv" --output-dir "$DATA_DIR/evaluation" --bootstrap-iterations 10000 --subgroup-bootstrap-iterations 10000 --seed 49297
```

Training writes model artifacts, environment records, metrics and split-labelled predictions into each model directory. The evaluator writes overall, subgroup and calibration summaries to `evaluation/`. This baseline evaluator uses the historical stay-level bootstrap convention; patient-cluster comparisons are generated in step 5.

Use `--device cpu` for LSTM if CUDA is unavailable. The recorded results use the frozen configuration; changing a device or library build may affect numerical reproducibility.

## 3. Fit the four reweighed models

Input: prepared arrays and baseline prediction files. Command:

```powershell
python -m fairness_study.src.run_training --data-dir "$DATA_DIR" --device cuda
```

This runs LR and LSTM for insurance and ethnicity sequentially. Outputs are under `fairness_study/experiments/reweight/<attribute>/<model>/`, including cell-weight audits, sample weights, configurations, predictions and fitted models. The two LSTM fits run for 100 epochs and select checkpoints by unweighted validation BCE.

## 4. Fit decision policies

Input: metadata, both baseline prediction files and the four reweighed prediction files. Command:

```powershell
python -m fairness_study.src.postprocess --data-dir "$DATA_DIR" --study-dir fairness_study
```

The script selects global thresholds and fits Fairlearn policies on validation data. It writes five policy conditions for each model and target, together with fitted rules, under `fairness_study/experiments/`. Policy records contain mortality risk, alert probability and a seeded binary decision. These patient-level records stay local.

## 5. Evaluate frozen policies

Input: the completed policy exports. Command:

```powershell
python -m fairness_study.src.evaluate --study-dir fairness_study --iterations 10000 --seed 49297
```

The evaluator writes group metrics, disparity measures, paired intervals and calibration exports to `fairness_study/results/`, and figures to `results/plots/` and `fairness_study/figures/`. It uses patient-cluster bootstrap sampling with all policies paired within each replicate.

Running this step replaces same-named local result tables. The checked-in calibration table contains only the overall display subset; fresh evaluation can generate finer group-level exports. Review those outputs against [data management](data_management.md) before publishing any updates.

## 6. Plot aggregate results

To redraw the checked-in figure set without patient data:

```powershell
python -m fairness_study.src.redraw_figures
python -m fairness_study.src.baseline_figure_revised
```

The first command reads the included baseline summaries, policy tables, intervals and epoch-level training losses. The second regenerates the revised English baseline subgroup layout. Outputs go to `fairness_study/figures/`; caches go to `fairness_study/tem/`. Chinese figure variants require a suitable CJK font.

For a newly rerun cohort, the baseline figure interface accepts fresh baseline evaluation outputs:

```powershell
python -m fairness_study.src.baseline_figures --evaluation-dir "$DATA_DIR/evaluation" --cohort-flow "$DATA_DIR/cohort_flow.csv" --metadata "$DATA_DIR/metadata.csv"
```

The all-figure redraw helper is an entry point for the distributed aggregate snapshot. Its `results/baseline/` inputs and `training_curves.csv` must be replaced with matching aggregate exports before using it to depict a different run. The policy evaluator itself plots its newly generated mitigation results.

## Historical compatibility and tests

An early baseline LSTM train export used a shuffled loader, although its validation and test exports were aligned. `align_baseline_train.py` is a repair tool for that specific checkpoint, verified by its recorded SHA-256 hash. It saves an aligned copy under `experiments/baseline_alignment/`; postprocessing prefers that copy when present. Newly trained models use the corrected ordered export and do not need this repair.

`finish_experiments.py` chains the historical repair and evaluation steps. Use the explicit workflow above for a fresh run rather than applying the historical checkpoint constraint to a new model.

Tests use synthetic data and cover preparation, features, weighting, thresholds, randomized policies and cluster sampling. The optional upstream header test requires `MIMIC3_DISCRETIZER_CONFIG` to point to a separately obtained benchmark configuration file; otherwise it reports a skip. All other test inputs are supplied or generated locally.
