# Frozen mitigation protocol

Frozen on 2026-09-23, before mitigation test evaluation. This is an extension after inspection of the original baseline test results, not an untouched-test prospective study. Original validation data are reused for checkpoint and decision selection.

## Questions and data

RQ1: Which original MIMIC-III findings recur in MIMIC-IV? RQ2: Do LR and channel-wise LSTM agree? RQ3: What fairness/utility/calibration tradeoffs arise from reweighing and threshold postprocessing?

Source: `<DATA_DIR>`. Preserve cohort (36,648 stays; 31,459 patients), exact admission-to-discharge death label, original patient split and demographic mapping. Unit is ICU stay; resampling unit is patient. Predictors cover the first 48 ICU hours; not immediate admission triage. Baseline test: 5,536 stays and 685 deaths. Keep source arrays/models/predictions unchanged.

Insurance and ethnicity are separate mitigation targets (`insurance_group`, `ethnicity_group`), including Other. Evaluate all existing gender, age, ethnicity and insurance groups. Do not claim simultaneous fairness or causal discrimination.

## Fixed methods

For both LR and LSTM, compare baseline 0.5; validation-optimal global threshold; reweighed model plus validation-optimal global threshold; baseline plus TPR parity; baseline plus equalized odds. Do not stack reweighing and postprocessing. Global thresholds maximize validation balanced accuracy over all achievable partitions; exact ties select the higher threshold. Predicted high-risk is positive; actual in-hospital death is the event.

Reweighing: training-only w(a,y)=P(a)P(y)/P(a,y), normalized to sample mean one, no clipping, strength search, resampling or additional class weights. Audit group-by-label support; unsupported necessary cells stop the corresponding fit rather than silently merge groups. LR retains train-only mean imputation and standardization, L2 C=.001, lbfgs, max_iter=1000, LR random_state=42. LSTM retains 17-channel architecture, dim=8, size_coef=4, dropout=.3, Adam lr=.001/betas=(.9,.999), batch_size=8, seed=49297, 100 epochs, no early stopping; weighted per-item BCE batch mean. Select strict minimum unweighted validation BCE, earliest exact tie. Save all-split predictions in metadata alignment. No multi-seed or hyperparameter search.

Fairlearn 0.13.0 ThresholdOptimizer: constraints true_positive_rate_parity or equalized_odds, objective balanced_accuracy_score, grid_size=1000, flip=False, prefit=True, no tolerance relaxation. Fit validation scores only. Save interpolation rules and evaluate frozen rules on test. Primary policy metrics use expected confusion counts from P(alert=1); PPV is the ratio of expected counts, not mean PPV over random draws. Also export one realized decision with random_state=49297. Keep mortality probability distinct from alert probability and binary output. MetricFrame supplies point-estimate group audits.

## Outcomes and inference

Primary: group TPR, minimum TPR and max-minus-min TPR; paired changes relative to baseline with optimized global threshold. Secondary: FPR, PPV, balanced accuracy, selection rate, expected confusion counts, FN/FP per 1000 stays, FPR gap and max(TPR gap,FPR gap). Probability evaluation uses original risk scores: AUROC, trapezoidal PR AUC (not average precision), Brier score, mean predicted minus observed risk and calibration curves. Threshold-only policies do not change these risk scores. Report n, events, event rate and undefined metrics; do not delete small groups.

10,000 patient-cluster bootstrap replicates, seed 49297, paired across policies; sampled patients contribute all their stays with multiplicity. Fixed trained models and rules, no refitting inside bootstrap. Percentile 95% intervals; report valid replicate counts and NA for absent/one-class groups. Recompute baseline uncertainty with the same cluster scheme for new comparisons; retain original row-bootstrap findings separately. No clinical acceptance threshold, test-driven model choice or confirmatory multiplicity claims. Inspect both absolute outcomes and disparities, retain every planned method including unfavorable results. Intervals do not capture training-seed variability.

## Implementation and reporting

Primary agent freezes shared interfaces, reviews code, schedules serial GPU runs and signs off outputs. Three subagents implement reweighing, postprocessing and evaluation respectively; no additional repeated review rounds. Necessary tests cover alignment/leakage, weights, thresholds, expected counts, reproducibility and cluster resampling. Keep implementation direct rather than build generic defensive frameworks.

Use PR_latex_template layout, natbib/BibTeX author-year references, no ten-page cap. English Research Report — Review Draft first; translate its checked content to Chinese with matching equations, numbers, references and figure/table numbering. Author/supervisor remain explicit placeholders. Include factual AI declaration. Compile both PDFs, render and visually inspect every page. No automatic publication or upload. Patient-level outputs and temporary files stay ignored by Git.

Sections: abstract (written last), AI declaration, introduction, related work, data/methods, results, discussion, conclusion, appendices/references. Main figures cover cohort, baseline performance, group discrimination/calibration, both targets, TPR/FPR and utility tradeoffs. The 16-group full audit goes in appendices without hiding major negative results. Cite primary research; distinguish internal evidence documents from literature. External validation, deployment, XAI, causal fairness and additional training seeds are outside this run.

Public packaging note: the original local data path has been replaced with <DATA_DIR>; the experimental rules are unchanged.
