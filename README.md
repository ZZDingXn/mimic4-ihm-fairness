# Project Progress Report

The methodology described in the papers and the data requirements have now been substantially reviewed, and the project is moving into full-dataset training and evaluation. Rather than running the original end-to-end implementation, which was written for MIMIC-III, Keras, and an outdated dependency stack, this reproduction uses PyTorch and a data-cleaning pipeline adapted to the MIMIC-IV schema. The design choices are described below.

The shared preprocessing pipeline, both model-training workflows, and the unified evaluation framework have been implemented. A rehearsal on 500 real ICU stays completed two epochs of CUDA training for the channel-wise LSTM, logistic-regression fitting, inference on both CPU and CUDA, and a small-bootstrap evaluation. Full-dataset training and result analysis are still in progress.

## 1. Research Tasks

**Task 1 - Channel-wise LSTM:** Train a channel-wise LSTM without deep supervision using 17 time-series channels from the first 48 hours of each ICU stay.

**Task 2 - Logistic regression:** Replace the neural model with the L2-regularised logistic-regression baseline from the Harutyunyan benchmark while retaining exactly the same cohort, labels, and patient-level train/validation/test split. The logistic-regression model uses 714 features calculated from sparse events rather than a flattened version of the LSTM's `48 x 76` tensor. The complete evaluation from Task 1 is then repeated.

## 2. Data Relationships and Overall Workflow

The three key identifiers in MIMIC-IV are:

- `subject_id`: a patient;
- `hadm_id`: a hospital admission;
- `stay_id`: an ICU stay.

One patient may have multiple hospital admissions, and one hospital admission may contain multiple ICU stays. This project first joins ICU stays to hospital admissions by `subject_id + hadm_id`, and then joins patient demographics by `subject_id`. Bedside events are linked by `stay_id`. Hospital laboratory events are linked by `hadm_id` and then restricted to the first 48 hours of the corresponding ICU stay.

The evaluation module consumes only the shared `metadata.csv` and standardised `predictions.csv` files. This ensures that both models are evaluated on the same test set, with the same subgroup definitions and statistical procedures.

## 3. Data Requirements

The main pipeline reads seven tables from MIMIC-IV v3.1:

| Module | Table | Purpose |
|---|---|---|
| `hosp/` | `patients.csv[.gz]` | Sex, anchor age, and anchor year |
| `hosp/` | `admissions.csv[.gz]` | Admission times, mortality label, race, and insurance |
| `hosp/` | `d_labitems.csv[.gz]` | Laboratory item-ID dictionary audit |
| `hosp/` | `labevents.csv[.gz]` | Target laboratory events |
| `icu/` | `icustays.csv[.gz]` | ICU stay, care unit, admission/discharge times, and length of stay |
| `icu/` | `d_items.csv[.gz]` | ICU item-ID dictionary audit |
| `icu/` | `chartevents.csv[.gz]` | Bedside observations, GCS, height, weight, and other target events |

Both uncompressed `.csv` and compressed `.csv.gz` inputs are supported:

```text
<MIMIC_ROOT>/
|-- hosp/
|   |-- patients.csv[.gz]
|   |-- admissions.csv[.gz]
|   |-- d_labitems.csv[.gz]
|   `-- labevents.csv[.gz]
`-- icu/
    |-- icustays.csv[.gz]
    |-- d_items.csv[.gz]
    `-- chartevents.csv[.gz]
```

Users must obtain authorised access to MIMIC-IV v3.1 and verify their downloaded files independently.

## 4. Cohort and Data-Cleaning Design

### 4.1 Cohort selection

The cohort is constructed in the following order. `cohort_flow.csv` records the number entering, excluded from, and retained after every step:

1. Join `icustays`, `admissions`, and `patients`.
2. Retain hospital admissions containing exactly one ICU stay.
3. Require `first_careunit == last_careunit` to exclude transfers between ICU units.
4. Estimate age at ICU admission from the anchor information and retain adults aged 18 years or older.
5. Require an ICU length of stay of at least 48 hours.
6. Read target events only from `[0, 48h]` after ICU admission; pre-ICU events are excluded.
7. Require at least one successfully cleaned target-variable observation during the first 48 hours.

`--max-stays` creates an engineering subset. With a fixed random seed, patients are selected until at most the requested number of stays is reached. A patient is never split across subsets, and selection is not balanced by mortality, sex, race, or insurance.

### 4.2 In-hospital mortality label

The primary label is the MIMIC-IV `hospital_expire_flag`. Two additional audit labels are calculated:

- whether `deathtime` lies exactly between `admittime` and `dischtime`;
- whether `deathtime` lies within the same calendar-date interval.

Records in which the official label disagrees with the exact timestamp comparison are written to `label_audit.csv` instead of being overwritten. Among the complete set of preliminarily eligible stays, 316 such disagreements were observed. A common reason was that `dischtime` was recorded as 00:00 on the day of death while `deathtime` occurred later that day.

### 4.3 The 17 variables

The project retains the 17 channels defined by the benchmark:

`Capillary refill rate`, `Diastolic blood pressure`, `Fraction inspired oxygen`, `Glascow coma scale eye opening`, `Glascow coma scale motor response`, `Glascow coma scale total`, `Glascow coma scale verbal response`, `Glucose`, `Heart Rate`, `Height`, `Mean blood pressure`, `Oxygen saturation`, `Respiratory rate`, `Systolic blood pressure`, `Temperature`, `Weight`, and `pH`.

MIMIC-IV item-ID mappings are defined in `mimic4_ihm/constants.py`. Each preprocessing run produces `mapping_audit.csv`, which records the item ID, dictionary label, unit, number of observations found inside the time window, and number retained after cleaning. Right-side manual diastolic blood pressure (`itemid=227242`) is included in the `Diastolic blood pressure` channel.

### 4.4 Units and categorical values

Only narrow conversions with unambiguous meanings are applied:

- FiO2 percentages are converted to proportions;
- SpO2 values recorded on a 0-1 scale are converted to percentages;
- Fahrenheit is converted to Celsius;
- inches are converted to centimetres;
- pounds are converted to kilograms;
- capillary-refill and GCS values are mapped to the benchmark's canonical categories.

When GCS text is empty, a valid integer `valuenum` can be used as a fallback. GCS total is derived only when the eye, motor, and verbal components all occur at the same timestamp. Unrecognised categorical values and non-numeric laboratory values are written to `unknown_values.csv`; they are not filled using guessed values.

### 4.5 Extreme values

The primary analysis does not introduce physiological-range clipping that was not used in the paper's original processing path. Although the upstream benchmark contains a range-checking function, that function was not called by the episode-generation workflow. Extreme values are therefore retained and summarised by source, item ID, and channel in `continuous_value_audit.csv`, including count, minimum, maximum, and mean. If a range-based sensitivity analysis is required after the full-cohort audit, it can be run separately.

### 4.6 Large-table ingestion and disk-backed intermediate storage

`chartevents` and `labevents` are read in chunks of 1,000,000 rows by default. Each chunk is filtered by stay, item ID, and the 48-hour time window before retained events are written to SQLite. Events are then ordered by `stay_id, charttime, itemid` and encoded one stay at a time, avoiding an ever-growing in-memory DataFrame.

The LSTM and logistic-regression arrays are written as memory-mapped `.npy` files. This keeps memory use primarily bounded by the current input chunk, although sufficient disk capacity is still required for `events.sqlite` and the output arrays.

## 5. Model Inputs

### 5.1 LSTM input: `(N, 48, 76)`

- Each stay is discretised into 48 one-hour bins; the last observation within a channel and hour overwrites earlier observations.
- Continuous variables occupy one value column, while categorical variables are one-hot encoded using the benchmark categories.
- Each of the 17 variables receives a separate observation mask.
- A predefined normal value is used before the first observation, followed by previous-value imputation.
- The mask always indicates whether a value was genuinely observed during that hour, allowing the model to distinguish observations from imputations.
- Continuous-column means and standard deviations are fitted on the training split only and then applied to validation and test.
- The final header must match all 76 fields in the original benchmark configuration.

### 5.2 Logistic-regression input: `(N, 714)`

The logistic-regression workflow independently extracts `17 x 7 x 6 = 714` features from the cleaned sparse events. For each variable, it considers the full 100%, first 10/25/50%, and last 10/25/50% of the relative observation period and calculates `min, max, mean, std, skew, len` in every interval.

Intervals without observations, and intervals in which skew is undefined, produce structural `NaN` values. A mean imputer and `StandardScaler` are fitted on the training split only and then applied to validation and test. The logistic-regression workflow does not reuse the previously imputed LSTM tensor, so the two paper-defined methods remain separate.

## 6. Patient Splitting and Demographic Subgroups

The default random seed is `49297`:

- approximately 15% of patients are assigned to test;
- approximately 18% of the remaining patients are assigned to validation;
- all remaining patients are assigned to training.

Splitting is performed by `subject_id`, so all stays belonging to one patient remain in a single split. The split is not stratified by mortality or protected attributes. The LSTM checkpoint is selected using validation loss, while the logistic-regression imputer and scaler see training data only. The test set is not used for model selection.

The demographic subgroups are:

| Attribute | Groups |
|---|---|
| Gender | Female, Male; unknown values are mapped to Other |
| Age | 18-29, 30-49, 50-69, 70-89, 90+ |
| Ethnicity | Asian, Black, Hispanic, White, Other |
| Insurance | Medicare, Medicaid, Private, Other |

Raw `race` and `insurance` values are retained alongside the standardised groups. Every raw value, mapped group, mapping rule, and count is written to `demographic_mapping_audit.csv`. Single-class subgroups are not deleted or merged: undefined AUROC/AUPRC values are emitted as `NA/null`, while sample and event counts are retained.

## 7. Model Definitions

### 7.1 Channel-wise LSTM

- Each variable's value/one-hot representation and mask are passed to an independent bidirectional LSTM.
- Each channel produces an 8-dimensional output.
- The 17 channel outputs are concatenated and passed to a 32-unit main LSTM.
- The final time step of the main LSTM is used for binary classification.
- `dim=8`, `size_coef=4`, dropout 0.3, batch size 8.
- Adam optimiser, learning rate `1e-3`, `beta1=0.9`.
- `BCEWithLogitsLoss`.
- No deep supervision, target replication, or class weighting.
- Up to 100 epochs by default, with checkpoint selection based on the lowest validation loss.

The PyTorch implementation preserves the architecture and study workflow but does not aim for bitwise equivalence with the historical Keras implementation.

### 7.2 Logistic regression

- L2 regularisation;
- `C=0.001`;
- `random_state=42`;
- `lbfgs` solver;
- up to 1,000 iterations;
- the training-only imputer, scaler, and model are saved together.

Both models produce:

```text
model, subject_id, hadm_id, stay_id, split, label, probability
```

## 8. Unified Evaluation

Evaluation uses only the test split. Before metrics are calculated, the program verifies that predictions and metadata contain the same stays, labels, and split assignments and that every probability lies in `[0, 1]`.

### 8.1 Overall and subgroup performance

- fixed classification threshold of 0.5;
- accuracy;
- event/non-event precision;
- event/non-event recall;
- confusion matrix;
- AUROC, AUPRC, and minpse;
- 10,000 bootstrap samples with empirical 95% intervals by default;
- sample size, mortality count, and event rate for every subgroup.

### 8.2 Calibration

- calibration-in-the-large;
- ten `q^(1/5)` exponential-quantile risk groups;
- mean predicted risk, observed mortality rate, and Wilson 95% interval for every group;
- LOWESS smoothing with span 0.5;
- outcome-stratified prediction distributions;
- PNG and SVG figures.

### 8.3 Model comparison

The LSTM and logistic-regression models are compared on identical test stays using paired bootstrap samples. Overall and subgroup differences in AUROC, AUPRC, and minpse are reported with intervals rather than presenting two unrelated metric tables.

The implementation is aligned with the Röösli reference source for the calibration-in-the-large figure definition, exponential-quantile boundaries, bootstrap sampling unit, and prediction-distribution display.

## 9. Code Structure

The core repository structure is:

```text
.
|-- mimic4_ihm/
|   |-- constants.py     # 17 channels, item IDs, categories, normal values, and the 76-field header
|   |-- prepare.py       # Cohort, labels, chunked ingestion, cleaning, SQLite, encoding, and audits
|   |-- features.py      # 714 logistic-regression features
|   |-- model.py         # PyTorch channel-wise LSTM
|   |-- train.py         # Training, checkpoints, environment snapshots, and standardised predictions
|   |-- metrics.py       # Threshold metrics, AUROC/AUPRC, and minpse
|   `-- evaluate.py      # Bootstrap, subgroups, fairness, calibration, and paired comparison
|-- tests/               # Unit, integration, and minimal end-to-end tests
|-- README.md
|-- requirements.txt
`-- pytest.ini
```

The data root, output directories, and device are provided through command-line arguments. The implementation contains no hard-coded Python interpreter or database path tied to a particular machine.

## 10. Outputs

### 10.1 Data preparation

| File | Contents |
|---|---|
| `events.sqlite` | Disk-backed intermediate store for target events in the first 48 hours |
| `metadata.csv` | IDs, labels, split, raw demographics, and standardised subgroups |
| `lstm_X.npy` | `(N, 48, 76)` LSTM input |
| `lr_X_raw.npy` | `(N, 714)` raw logistic-regression features |
| `cohort_flow.csv` | Cohort attrition |
| `coverage.csv` | Missingness and observed hours for the 17 variables |
| `mapping_audit.csv` | Item IDs, units, matched observations, and retained observations |
| `label_audit.csv` | Differences between the official label and timestamp comparisons |
| `demographic_mapping_audit.csv` | Demographic mappings and counts |
| `unknown_values.csv` | Unrecognised or non-numeric events |
| `continuous_value_audit.csv` | Continuous-variable extreme-value audit |
| `resource_usage.csv` | Per-chunk row counts and RSS/VMS |
| `config.json` | Header, feature names, normalisation statistics, and seed |
| `provenance.json` | Command, input files, Git commit, and software environment |

### 10.2 Models and evaluation

- LSTM: `best_model.pt`, `training_log.csv`, `predictions.csv`, `metrics.json`, and `environment.json`;
- logistic regression: `model.pkl`, `predictions.csv`, `metrics.json`, and `environment.json`;
- evaluation: overall/subgroup metrics, calibration data, paired model differences, PNG/SVG figures, and an evaluation manifest.

SQLite databases, arrays, metadata, model weights, and patient-level predictions may contain controlled or linkable information and are therefore not included in the public repository.

## 11. Running the Project

Run the modules from the repository root with `python -m mimic4_ihm...`.

## 12. Completed Real-Subset Rehearsal

The 500-stay rehearsal contained 500 stays from 432 patients, with 70 in-hospital deaths. The training, validation, and test splits had no patients in common.

Engineering acceptance results:

- the LSTM input had shape `(500, 48, 76)` with no `NaN` or infinite values;
- the raw logistic-regression features had shape `(500, 714)` and were all finite after training-only imputation and scaling;
- `itemid=227242` occurred three times in the subset and all three observations were retained;
- after adding the GCS `valuenum` fallback, the number of unknown GCS observations fell to zero;
- both prediction files contained 500 rows and agreed with metadata on stay, label, and split;
- the LSTM completed two epochs of CUDA training and inference on both CPU and CUDA;
- logistic-regression fitting and prediction completed successfully;
- single-class subgroups were retained with undefined metrics instead of causing failure or being removed;
- the formal local test suite recorded `18 passed`.

These results demonstrate that the large-file ingestion, cleaning, feature extraction, training, and evaluation paths can run end to end.

## 13. Public-Repository Scope

Included:

- the `mimic4_ihm/` core implementation;
- `tests/` and test configuration;
- `requirements.txt`;
- this reviewed English README;
- aggregate experiment records that contain no patient-level information, raw values, or controlled data.

Not included:

- MIMIC source CSV/CSV.GZ files;
- SQLite databases, NumPy arrays, metadata, patient-level predictions, or model weights;
- local copies of papers, project briefs, or third-party reference directories;
- the historical Demo project;
- internal working reports, logs, or machine-specific configuration;
- caches, virtual environments, or temporary files.
