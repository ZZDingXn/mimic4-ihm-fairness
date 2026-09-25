# Results and figure guide

The distributed results describe the frozen MIMIC-IV 3.1 cohort and its 5,536-stay test set, including 685 deaths from 4,719 patients. Tables are under [`fairness_study/results/`](../../fairness_study/results/); figures are under [`fairness_study/figures/`](../../fairness_study/figures/).

## Reading the policy comparisons

`model` is `lr` or `cw_lstm`. `target_attribute` identifies the attribute used for mitigation, while `attribute` identifies the grouping used for evaluation. These may differ: an insurance-targeted policy can still be evaluated across ethnicity, age or gender.

`method` uses the five keys in the [methods document](methods.md): `baseline_05`, `global_threshold`, `reweight`, `opportunity` and `equalized_odds`. `overall` rows represent the full test set; group rows describe the named subgroup. The original fixed/global policies occur in both target blocks to support the same comparison structure.

Decision metrics for randomized postprocessing are calculated from expected confusion counts. Consequently, `tp`, `fp`, `tn` and `fn` may be fractional. `n` and `events` remain observed stay and death counts. Risk-score metrics use mortality probabilities, including for threshold-only policies.

| Quantity | Definition and units |
|---|---|
| `tpr`, `fpr`, `ppv` | TP/(TP+FN), FP/(FP+TN), TP/(TP+FP); proportions |
| `balanced_accuracy` | Mean of TPR and specificity; proportion |
| `selection_rate` | Expected alerts divided by stays; proportion |
| `false_positives_per_1000`, `false_negatives_per_1000` | Expected count divided by all stays in the row, multiplied by 1,000 |
| `tpr_gap`, `fpr_gap`, `selection_rate_gap` | Maximum minus minimum group rate |
| `worst_group_tpr` | Minimum group TPR |
| `equalized_odds_gap` | Maximum of TPR gap and FPR gap |
| `auroc`, `auprc` | ROC area and trapezoidal precision–recall area |
| `brier_score` | Mean squared error of mortality probabilities |
| `mean_calibration_bias` | Mean predicted mortality probability minus observed mortality |

Proportions are stored on the 0–1 scale. Multiplication by 100 gives percentages or percentage-point differences, as appropriate. A negative disparity change represents a smaller gap; a negative worst-group TPR change represents lower detection. Undefined estimates remain missing, with status or valid-bootstrap counts providing context.

## Table inventory

| File | Statistical unit and purpose |
|---|---|
| `policy_metrics.csv` | Model × target × policy × evaluation group; point estimates and group counts |
| `fairness_gaps.csv` | Model × target × policy × evaluation attribute; gap and worst-group measures |
| `bootstrap_confidence_intervals.csv` | One metric interval per policy and evaluation group/attribute; `lower_95`, `upper_95`, `valid_iterations` |
| `paired_differences_vs_global_threshold.csv` | Policy value, matched global-threshold value, difference and paired 95% interval |
| `model_comparison_patient_cluster.csv` | Paired original-model discrimination comparison under patient-cluster resampling |
| `report_comparison_summary.csv` | Twelve target-specific mitigation comparisons; point changes in gap, worst-group TPR, balanced accuracy and FPR |
| `baseline_summary.csv` | Original test AUROC, AUPRC and 0.5-threshold recall, with historical interval provenance |
| `cohort_counts.csv`, `cohort_demographics.csv` | Total cohort size and group counts by split |
| `calibration_bins.csv` | Overall model-risk calibration bins and fixed-grid smoothed display values |
| `training_curves.csv` | Reweighed LSTM target, epoch, weighted training loss, unweighted validation loss and selected epoch |
| `weight_audit_insurance_group.csv`, `weight_audit_ethnicity_group.csv` | Train group × outcome counts, marginal/joint probabilities and weights |
| `training_configuration.json` | Parameters for the four reweighed training fits |
| `evaluation_manifest.json` | Evaluation seed, replicate count, software record and probability/count conventions |
| `temporal_label_audit.json` | Aggregate label-timing counts |
| `public_package_manifest.json` | SHA-256 inventory of the public snapshot, excluding the manifest itself |
| `baseline/` | Historical overall, subgroup, mean-calibration, cohort-flow and cohort-summary inputs |

New policy comparisons use 10,000 patient-cluster bootstrap samples, paired across methods. Historical files under `baseline/` retain stay-level bootstrap intervals. The baseline recall row in `baseline_summary.csv` has no historical confidence interval. These conventions are attached to the relevant figures below.

## Figure guide

File stems below have `_en` and `_zh` suffixes and PNG/PDF formats. The revised baseline subgroup figure is English-only and also includes SVG.

### Cohort flow: `cohort_flow`

Generated after cohort construction from `baseline/cohort_flow.csv`. The diagram follows linked stays through admission, careunit, age, length-of-stay and observation filters. Counts are stays; patient totals are reported separately in the cohort tables. The historical label “no ICU unit transfer” refers to equality of the first and last careunit fields.

### Baseline performance: `baseline_performance`

Generated from `baseline/overall_metrics.json` for the two original models on test. It compares AUROC, trapezoidal AUPRC and mortality recall at threshold 0.5. AUROC and AUPRC intervals use the original 10,000 stay-level bootstrap samples; recall has no added interval. Read recall alongside discrimination to distinguish ranking performance from the chosen decision threshold.

### Baseline subgroups: `baseline_subgroups`

Generated from `baseline/subgroup_metrics.csv` and `baseline/calibration_in_large.csv`. The upper panels show AUROC by insurance and ethnicity; lower panels show mean predicted minus observed risk in percentage points. Negative bias indicates average underestimation. Intervals are historical stay-level bootstrap intervals. Counts and event rates are available in the subgroup source table.

`revised/baseline_subgroups_en` uses the same estimates and intervals with a revised label and panel layout. The revision changes presentation, not the underlying comparison. Mean bias summarizes average agreement and should be read alongside the calibration curves.

### Detection and false-positive rates: `group_tpr`, `group_fpr`

Generated after freezing all policies, using `policy_metrics.csv` and `bootstrap_confidence_intervals.csv`. LSTM appears above LR; ethnicity is on the left and insurance on the right. Colors identify the five policies. TPR divides by deaths within a group, while FPR divides by non-deaths. Error bars use patient-cluster bootstrap intervals.

These plots show whether a narrowing gap accompanies higher detection in the worst-performing group and how the false-positive burden changes. Randomized policies use expected counts. Rates describe the test set after validation fitting, where exact parity is not guaranteed.

### Fairness and utility: `fairness_utility_tradeoff`

Generated by joining `fairness_gaps.csv` with overall policy metrics. The horizontal axis shows TPR gap and the vertical axis balanced accuracy. Each panel contains the predefined policy points, without confidence intervals or a fitted frontier. Inspect each panel's axis range when comparing targets or models.

### Overall calibration: `overall_calibration`

Generated from the overall subset of `calibration_bins.csv`, comparing original and reweighed model risk scores. Ten bins use exponential quantile boundaries q^(1/5). LOWESS uses `frac=0.5`, `it=0`; exported display values lie on a fixed 0.005-risk grid within observed support. The public file omits subgroup calibration exports and individual risk coordinates.

The diagonal represents agreement between predicted risk and observed mortality. Curves above it indicate underestimation. The display has no uncertainty band; bin-level Wilson limits are retained as fields in the source table. Threshold-only policies share the original mortality scores, so they do not create separate risk calibration curves.

### Training curves: `training_curves`

Generated from epoch-level summaries of the two 100-epoch reweighed LSTM fits. Each panel compares weighted training BCE with unweighted validation BCE; dashed lines mark selected epochs 22 and 14 for insurance and ethnicity. The different loss weightings mean the vertical separation is not a conventional train–validation generalization gap. The figure represents one training seed per target.

## Rebuilding and extending displays

Use `python -m fairness_study.src.redraw_figures` for the distributed figure set and `python -m fairness_study.src.baseline_figure_revised` for the revised layout. The [reproduction guide](reproduction.md) distinguishes these aggregate redraws from evaluating newly trained models. Preserve the metric definitions and uncertainty convention when adding a display from these tables.
