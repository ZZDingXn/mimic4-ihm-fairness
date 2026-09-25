# Study methods

The study compares predictive performance, group detection rates and probability calibration in MIMIC-IV 3.1. It extends a MIMIC-III benchmark-based evaluation through dataset adaptation and two established mitigation mechanisms: training reweighing and constrained threshold postprocessing.

## Cohort and prediction task

The analysis includes 36,648 ICU stays from 31,459 patients, with 4,469 exact-window mortality events. A positive label denotes a recorded death between hospital admission and discharge, inclusive. Predictors cover the first 48 ICU hours. Selection, measurement conversion and feature handling are specified in [data cleaning](data_cleaning.md).

| Split | Stays | Patients | Deaths |
|---|---:|---:|---:|
| Train | 25,502 | 21,927 | 3,072 |
| Validation | 5,610 | 4,813 | 712 |
| Test | 5,536 | 4,719 | 685 |

The split uses seed 49297 at the patient level, without outcome stratification. Insurance and ethnicity are separate mitigation targets; gender and age are also evaluated. The existing group mappings, including `Other`, are preserved.

## Baseline models and selection

LR uses 714 sparse-event features with train-fitted mean imputation and standardization. Its classifier uses L2 regularization, `C=0.001`, `solver="lbfgs"`, `max_iter=1000` and `random_state=42`.

The channel-wise LSTM uses 48 steps and 76 encoded columns, with `dim=8`, `size_coef=4`, dropout 0.3 and batch size 8. Training uses Adam with learning rate 0.001, betas (0.9, 0.999), seed 49297 and 100 epochs. The checkpoint is the strict minimum of unweighted validation binary cross-entropy (BCE); an exact tie retains the earlier epoch. The reweighed LSTM runs selected epochs 22 for insurance and 14 for ethnicity.

## Fairness target and comparison design

For group g, true-positive rate (TPR) is the proportion of deaths receiving a high-risk alert. The primary disparity is the maximum group TPR minus the minimum group TPR. Worst-group TPR is reported alongside the gap to distinguish narrowing disparities from improving detection.

| Policy key | Model or rule | Selection data |
|---|---|---|
| `baseline_05` | Original model, threshold 0.5 | Fixed rule |
| `global_threshold` | Original model, common threshold maximizing balanced accuracy | Validation |
| `reweight` | Reweighed model, optimized common threshold | Train weights; validation checkpoint and threshold |
| `opportunity` | Original model, TPR-parity threshold policy | Validation |
| `equalized_odds` | Original model, TPR- and FPR-parity threshold policy | Validation |

There are 20 model–target–policy conditions. Global thresholds enumerate all attainable partitions; ties select the higher threshold. Balanced accuracy averages sensitivity and specificity. It is a common research utility objective here, rather than a clinically established error-cost model.

### Training reweighing

For attribute A and outcome Y, the training weight is

`w(a,y) = P_train(A=a) × P_train(Y=y) / P_train(A=a,Y=y)`.

Weights are normalized to mean one across training samples. LR receives sample weights during classifier fitting; LSTM multiplies each example's BCE by its weight and takes the batch mean. The existing preprocessing, optimizer and sampling scheme remain in place. Weight-cell counts and values are available in the two `weight_audit_*.csv` tables.

This follows the reweighing approach of [Kamiran and Calders](https://doi.org/10.1007/s10115-011-0463-8). Reweighing changes group–outcome contributions to training; its effects on TPR disparity and calibration are measured empirically.

### Threshold postprocessing

Fairlearn 0.13.0 `ThresholdOptimizer` uses `constraints="true_positive_rate_parity"` or `"equalized_odds"`, `objective="balanced_accuracy_score"`, `grid_size=1000`, `flip=False`, `prefit=True`, `predict_method="predict_proba"` and `tol=None`.

The optimizer fits validation scores and uses the target attribute at prediction time. A fitted policy can randomize between thresholds. The implementation saves both the probability of an alert and one realized binary decision using seed 49297. Validation constraints need not hold exactly in test.

Primary policy metrics use expected confusion counts. For alert probability q, expected TP is the sum of q over deaths, and expected FP is the sum over non-deaths; FN and TN use 1−q. PPV is the ratio of expected TP to expected alerts, rather than the average PPV over random policy realizations. `MetricFrame` supplies grouped point estimates.

This decision-level approach follows [Hardt, Price and Srebro](https://arxiv.org/abs/1610.02413), implemented through [Fairlearn's versioned documentation](https://fairlearn.org/v0.13/). Original mortality risks, alert probabilities and realized decisions remain distinct outputs. Risk metrics for threshold-only policies use the original model's mortality scores.

## Outcomes and uncertainty

The primary comparison is the change in TPR gap relative to `global_threshold`, accompanied by group TPR and worst-group TPR. Supporting decision metrics include FPR, PPV, balanced accuracy, alert rate, and false positives and negatives per 1,000 stays. Equalized-odds gap is `max(TPR gap, FPR gap)`.

Risk-score metrics are AUROC, trapezoidal AUPRC, Brier score, mean predicted minus observed mortality, and calibration curves. AUPRC is calculated by trapezoidal integration of the precision–recall curve, not by average precision. Group event rates accompany its interpretation. Mean bias describes average risk agreement, while Brier score reflects more than calibration alone.

New comparisons use 10,000 paired patient-cluster bootstrap samples with seed 49297. Sampling a patient includes all their stays, with multiplicity when sampled repeatedly. Each replicate uses the same sampled patients across policies. Models and fitted rules stay fixed, and 95% intervals are percentile intervals. Undefined metrics and valid replicate counts are retained.

Historical baseline figures retain their original 10,000-replicate stay-level intervals. Baselines participating in the new comparisons have patient-cluster intervals recomputed alongside the mitigation methods. The [figure guide](results_guide.md) identifies the convention for each display.

## Interpretation and provenance

The baseline test results were available before the mitigation protocol was frozen on 23 September 2026. Validation data were reused for model and policy selection. This is a retrospective extension with fixed evaluation rules, rather than an untouched-test prospective study.

The timing audit identifies 152 deaths at or before the input-window endpoint. This limits interpretation as prediction of future events after hour 48. Results come from one dataset and one training seed per configuration; bootstrap intervals cover test sampling rather than the full variability of model development. Clinical deployment, external validation and causal fairness are outside the completed analysis.

The [archived protocol](../old/protocol_2026-09-23.md) preserves the previously distributed record, including its execution notes. This reader-facing description reorganizes the scientific methods without changing experimental rules. AI tools assisted code development, documentation and manuscript editing; their involvement does not establish clinical validity or independent replication.
