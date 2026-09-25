# Data cleaning and feature construction

This document describes the implemented MIMIC-IV 3.1 pipeline, from cohort joins to model-ready arrays. Rules are tied to the current code so that a reproduction can distinguish data selection, value conversion and missing-value handling.

The main implementation is [`prepare.py`](../../mimic4_ihm/prepare.py), with mappings in [`constants.py`](../../mimic4_ihm/constants.py) and LR summaries in [`features.py`](../../mimic4_ihm/features.py). The benchmark motivates the 17-variable representation; MIMIC-IV item mappings, joins and label reconciliation are project adaptations.

## 1. Records, joins and cohort selection

`subject_id` identifies a patient, `hadm_id` a hospital admission, and `stay_id` an ICU stay. ICU stays are inner-joined to admissions on both patient and admission identifiers, then to patients on the patient identifier. Both joins require many-to-one relationships on the right-hand side.

The pipeline applies these filters in order:

1. Retain hospital admissions with exactly one ICU stay among the linked records.
2. Require `first_careunit == last_careunit`.
3. Compute age as `anchor_age + year(intime) - anchor_year` and retain age at least 18.
4. Require the supplied ICU `los` field to be at least 2.0 days.
5. Apply an optional explicit patient list or engineering subset, when requested.
6. After event extraction, retain stays with at least one usable observation from a mapped variable.

The formal run used the full eligible cohort rather than an engineering subset. The careunit check compares admission and discharge units; it does not inspect intermediate transfers. The length-of-stay filter uses the supplied `los`, rather than recomputing it from `outtime - intime`.

The resulting metadata are sorted by patient, ICU admission time and stay identifier. Patient-level splitting then assigns 15% of patients to test and 18% of the remaining patients to validation, using seed 49297 and rounded counts. All eligible stays belonging to a patient receive the same split.

## 2. Mortality labels and predictor timing

The binary label is one when recorded `deathtime` lies within the inclusive hospital interval `[admittime, dischtime]`. Missing death times receive label zero. Both the official `hospital_expire_flag` and a date-only comparison are retained for reconciliation.

The frozen cohort contains 4,469 exact-window events, compared with 4,785 official death flags; 316 records differ. `label_audit.csv` records these disagreements locally, including identifiers and timestamps. It is a patient-level audit and is not distributed with the repository.

Predictors use the inclusive interval `[intime, intime + 48 hours]`. A separate timing audit found 152 labeled deaths at or before its endpoint, including 24 in test. The reported analysis preserves that cohort. Consequently, the task is retrospective in-hospital mortality modelling rather than a strict forecast among patients confirmed alive at hour 48.

## 3. Event selection and numeric parsing

`chartevents` is matched by `stay_id`; `labevents` is matched by `hadm_id`, which maps to one retained ICU stay. Both sources are filtered by mapped item IDs and the inclusive predictor interval. A chart event is not joined solely by patient identity, and a laboratory result from another part of the admission is excluded by time.

For continuous values, the cleaner first parses `valuenum`; if it is missing or cannot be parsed, it tries `value`. If both attempts fail, the row is omitted and its reason is counted. Laboratory glucose, oxygen saturation and pH follow the same parsing order.

The ingestion code reads identifiers, `charttime`, `itemid`, `value` and `valuenum`. It does not use `storetime`, measurement-quality flags or `valueuom` to resolve conflicts. Unit conversions are selected by item ID or the value rules below.

| Variable or item | Implemented conversion | Example |
|---|---|---|
| Chart FiO2 | Divide by 100 when the parsed value is greater than 1 | 40 becomes 0.40 |
| Chart or laboratory oxygen saturation | Multiply by 100 when the value is at most 1 | 0.97 becomes 97 |
| Temperature, item 223761 | `(value - 32) × 5 / 9` | 98.6 becomes 37 °C |
| Height, item 226707 | `round(value × 2.54)` | 70 inches becomes 178 cm |
| Weight, item 226531 | `value × 0.453592` | 180 pounds becomes 81.64656 kg |
| Other mapped numeric items | Keep the parsed value | No additional conversion |

These are explicit arithmetic rules, not a clinical plausibility filter. The cleaner does not clip, winsorize or remove numeric observations using physiological bounds. In particular, the saturation rule also applies to zero or negative values because its condition is `value <= 1`. Extremes remain visible in the continuous-value audit.

Numeric parsing rejects missing or unparseable values, but ingestion has no general `isfinite` check. LR's continuous-event conversion later excludes non-finite numbers; LSTM staging does not define an equivalent universal rejection rule. The formal arrays were checked for finiteness, but this should not be described as a broader cleaning rule implemented at ingestion.

## 4. Categorical variables and Glasgow Coma Scale

Capillary refill is encoded as `0.0` for `Normal <3 Seconds`, `Normal <3 secs` or `Brisk`, and `1.0` for `Abnormal >3 Seconds`, `Abnormal >3 secs` or `Delayed`. Other strings are omitted and counted in `unknown_values.csv`.

Glasgow Coma Scale (GCS) components are matched to the explicit text dictionaries in `constants.py`. The cleaner accepts the defined raw labels and their canonical equivalents. Otherwise, it tries a numeric component score from `valuenum`, then `value`, and accepts a score only if it matches the relevant dictionary. Matching is case-sensitive after surrounding whitespace is stripped.

Eye opening uses scores 1–4, motor response 1–6 and verbal response 1–5. Intubated verbal responses map to `1.0 ET/Trach` with score 1. Numeric verbal score 1 resolves to the first matching dictionary entry, `1 No Response`; it does not reconstruct intubation status from the number alone.

GCS total is derived only when all three component scores occur at the **same exact timestamp**. Components from different timestamps are not carried forward to construct a total. When repeated components share a timestamp, the last processed component supplies its score. Code and saved feature names retain the historical spelling `Glascow` for compatibility; the prose uses Glasgow.

## 5. Repeated observations and hourly bins

Events are stored without a deduplication pass. The database query orders them by `stay_id`, `charttime_ns` and `itemid`; the derived GCS total is inserted with item ID −1 and then sorted by timestamp and item ID.

For LSTM, each observation replaces any earlier value for its channel in the same hourly bin. Thus the last processed observation is retained, rather than an hourly mean. At a shared timestamp, different item IDs follow item-ID order. Exact ties on timestamp and item ID have no further documented source-priority rule; chart and laboratory duplicates are not explicitly reconciled by clinical source preference.

The bin calculation subtracts a tolerance of 10⁻⁶ hours before taking the integer part. An observation at an exact positive hour boundary is therefore assigned to the preceding bin. Hour 0 enters the first bin, and hour 48 enters the last bin. The tolerance also affects observations extremely close to a boundary.

LR receives the accepted sparse events before hourly replacement, so repeated measurements can contribute separately to its summaries and observation counts. This distinction is part of the intended difference between the two feature representations.

## 6. Missing values and scaling

LSTM uses 48 hourly steps with 59 encoded values and 17 channel masks. Each categorical channel is one-hot encoded. An observed bin has mask 1; an unobserved bin has mask 0, even after filling.

Within each stay and channel, missing bins carry the previous value forward. Before the first observation, the pipeline uses the fixed benchmark defaults listed below. These initial values are predefined constants rather than estimates from the cohort.

| Channel | Initial value |
|---|---|
| Capillary refill rate | `0.0` |
| Diastolic blood pressure | `59.0` |
| Fraction inspired oxygen | `0.21` |
| Glasgow coma scale eye opening | `4 Spontaneously` |
| Glasgow coma scale motor response | `6 Obeys Commands` |
| Glasgow coma scale total | `15` |
| Glasgow coma scale verbal response | `5 Oriented` |
| Glucose | `128.0` |
| Heart Rate | `86.0` |
| Height | `170.0` |
| Mean blood pressure | `77.0` |
| Oxygen saturation | `98.0` |
| Respiratory rate | `19.0` |
| Systolic blood pressure | `118.0` |
| Temperature | `36.6` |
| Weight | `81.0` |
| pH | `7.4` |

Continuous LSTM features are standardized using all 48 encoded time steps from **training stays only**, including filled values. Variance is computed as `E[x²] − E[x]²` with a floor of 10⁻¹⁴. The resulting means and standard deviations are applied to all splits; one-hot values and masks are unchanged.

LR computes minimum, maximum, mean, population standard deviation, skewness and observation count across seven subperiods. For each variable, the periods cover all observations, the first 10%, 25% and 50%, and the last 10%, 25% and 50% of its observed first-to-last time span. They are not fixed fractions of the full 48-hour window. Boundary comparisons use a 10⁻⁶-hour tolerance.

An empty LR subperiod yields six missing features, including a missing count; undefined skewness also remains missing. The training pipeline fits mean imputation with `keep_empty_features=True`, then standardization, on train only. This produces 714 LR features while preserving columns that are entirely missing in train.

## 7. Demographic mappings

Demographic groups support evaluation and reweighing. They are not added to the 17 physiological input channels.

| Attribute | Mapping |
|---|---|
| Ethnicity | Uppercase source text; check `HISPANIC`, `LATINO` or `SOUTH AMERICAN` first, then `ASIAN`, `BLACK`, and `WHITE`; otherwise `Other` |
| Insurance | Strip and lowercase; exact matches for Medicare, Medicaid and Private; otherwise `Other` |
| Gender | Uppercase; `F` → Female and `M` → Male; otherwise `Other` |
| Age | 18–29, 30–49, 50–69, 70–89 and 90+ using the calculated age |

Missing ethnicity or insurance entries fall into `Other`, rather than being dropped. The supplied mappings are retained across training, validation, test and mitigation.

## 8. Item mappings and audit outputs

The table below is generated from the current mapping constants. `Derived` means that no raw item is used directly for that channel.

| Channel | Chart item IDs | Laboratory item IDs |
|---|---|---|
| Capillary refill rate | 224308, 223951 | — |
| Diastolic blood pressure | 220051, 220180, 224643, 225310, 227242 | — |
| Fraction inspired oxygen | 223835 | — |
| Glasgow coma scale eye opening | 220739 | — |
| Glasgow coma scale motor response | 223901 | — |
| Glasgow coma scale total | Derived | — |
| Glasgow coma scale verbal response | 223900 | — |
| Glucose | 220621, 225664, 226537 | 50809, 50931 |
| Heart Rate | 220045 | — |
| Height | 226707, 226730 | — |
| Mean blood pressure | 220052, 220181, 224322, 225312 | — |
| Oxygen saturation | 220227, 220277 | 50817 |
| Respiratory rate | 220210, 224422, 224689, 224690 | — |
| Systolic blood pressure | 220050, 220179, 224167, 225309, 227243 | — |
| Temperature | 223761, 223762 | — |
| Weight | 224639, 226512, 226531 | — |
| pH | 220274, 223830 | 50820, 50831 |

The local preparation step writes the following audits:

| Output | What it records |
|---|---|
| `cohort_flow.csv` | Counts before and after each selection step |
| `mapping_audit.csv` | Item dictionary information, events seen in-window, and events retained after cleaning |
| `unknown_values.csv` | Rejected category or parsing reasons and their counts; reason strings may include raw values |
| `continuous_value_audit.csv` | Continuous-value counts, minima, maxima and means from staged events |
| `coverage.csv` | Channel-level absence, full 48-bin coverage and observed-hour counts before filling |
| `demographic_mapping_audit.csv` | Raw demographic labels and resulting group counts |
| `label_audit.csv`, `split_manifest.csv` | Patient-level label reconciliation and split membership |
| `config.json`, `provenance.json` | Encodings, normalization parameters, input provenance and environment |

Audits are local research outputs. Their contents require review before release, particularly identifiers, timestamps, raw-value strings and fitted preprocessing parameters. See [data management](data_management.md).
