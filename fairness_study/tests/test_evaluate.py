from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

from fairness_study.src.evaluate import Policy, _risk_auc_pr_draws, bootstrap_draws, evaluate, expected_confusion, risk_metrics


def _policy_frame() -> pd.DataFrame:
    rows = []
    groups = {
        "gender_group": ("Female", "Male"), "age_group": ("18-29", "30-49"),
        "ethnicity_group": ("Asian", "White"), "insurance_group": ("Medicare", "Private"),
    }
    for subject in range(1, 9):
        label = int(subject % 3 == 0)
        rows.append({
            "subject_id": subject, "stay_id": 100 + subject, "split": "test", "label": label,
            "risk_probability": 0.85 if label else 0.12 + subject / 100,
            "decision_probability": 0.7 if label else 0.15, "prediction": int(label),
            **{column: values[subject % 2] for column, values in groups.items()},
        })
    return pd.DataFrame(rows)


def _write_policies(study: Path) -> None:
    frame = _policy_frame()
    for model in ("lr", "cw_lstm"):
        for attribute in ("insurance_group", "ethnicity_group"):
            for method in ("baseline_05", "global_threshold", "reweight", "opportunity", "equalized_odds"):
                path = study / "experiments" / "policies" / model / attribute / f"{method}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                value = frame.copy()
                if method == "opportunity":
                    value.loc[value[attribute].eq(value[attribute].iloc[0]), "decision_probability"] = 0.55
                value.to_csv(path, index=False)


def test_expected_confusion_uses_policy_probability() -> None:
    confusion = expected_confusion(np.array([1, 0]), np.array([0.25, 0.4]))
    assert np.allclose(confusion, [0.6, 0.4, 0.75, 0.25])


def test_risk_metrics_has_trapezoid_auprc() -> None:
    metrics = risk_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert metrics["auroc"] == 1.0
    assert 0.9 <= metrics["auprc"] <= 1.0


def test_weighted_risk_draws_match_explicit_resampling_with_ties_and_missing_maximum() -> None:
    y = np.array([1, 0, 1, 0])
    risk = np.array([0.99, 0.80, 0.80, 0.15])
    # Subject 1 has two stays.  The bootstrap omits subject 0, whose stay has
    # the maximum risk, which previously produced a spurious NaN PR curve.
    subject_codes = np.array([0, 1, 1, 2])
    counts = np.array([[0, 2, 1]], dtype=np.int16)
    aucs, aprcs = _risk_auc_pr_draws(y, risk, subject_codes, counts, np.ones(len(y), dtype=bool))
    weights = counts[0, subject_codes]
    selected = np.repeat(np.arange(len(y)), weights)
    precision, recall, _ = precision_recall_curve(y[selected], risk[selected])
    assert np.isclose(aucs[0], roc_auc_score(y[selected], risk[selected]))
    assert np.isclose(aprcs[0], np.trapezoid(precision[::-1], recall[::-1]))


def test_cluster_draws_keep_multiple_stays_and_invalidate_gap_when_a_group_event_disappears() -> None:
    frame = pd.DataFrame({
        "subject_id": [1, 1, 2], "stay_id": [11, 12, 21], "split": ["test"] * 3,
        "label": [1, 1, 1], "risk_probability": [0.8, 0.7, 0.6],
        "decision_probability": [0.4, 0.6, 0.3], "prediction": [0, 1, 0],
        "gender_group": ["Female", "Female", "Male"], "age_group": ["18-29"] * 3,
        "ethnicity_group": ["Asian"] * 3, "insurance_group": ["Medicare"] * 3,
    })
    policy = Policy("lr", "gender_group", "global_threshold", Path("synthetic.csv"), frame)
    # First replicate draws subject 1 twice: both of that patient's stays must
    # contribute, producing TPR = (2 * (.4 + .6)) / 4 = .5.
    draws = bootstrap_draws(policy, np.array([[2, 0], [0, 2], [1, 1]], dtype=np.int16), {})
    assert draws[("overall", "tpr")][0] == 0.5
    # Second replicate has no Female death.  It must invalidate the gender gap
    # instead of computing a gap only from the remaining Male group.
    assert np.isnan(draws[("gender_group", "tpr_gap")][1])
    assert np.isnan(draws[("gender_group", "selection_rate_gap")][1])
    assert np.isclose(draws[("gender_group", "selection_rate_gap")][2], 0.2)


def test_evaluation_writes_point_bootstrap_and_publication_outputs(tmp_path: Path) -> None:
    _write_policies(tmp_path)
    outputs = evaluate(tmp_path, iterations=30, seed=7)
    metrics = pd.read_csv(outputs["metrics"])
    assert {"overall", "gender_group", "age_group", "ethnicity_group", "insurance_group"}.issubset(metrics.attribute)
    assert len(metrics[metrics.attribute.eq("gender_group")].group.unique()) == 2
    assert {"tpr", "fpr", "ppv", "balanced_accuracy", "selection_rate"}.issubset(metrics.columns)
    confidence = pd.read_csv(outputs["confidence"])
    assert (confidence.valid_iterations > 0).any()
    assert (confidence.valid_iterations == 0).any(), "absent groups must remain explicitly undefined"
    assert "selection_rate_gap" in pd.read_csv(outputs["gaps"]).columns
    comparison = pd.read_csv(outputs["model_comparison"])
    assert np.allclose(comparison[["lstm_minus_lr", "lower_95", "upper_95"]], 0), "identical risk vectors must have zero paired differences"
    assert (tmp_path / "results" / "plots" / "group_tpr_en.png").stat().st_size > 1000
    assert (tmp_path / "results" / "plots" / "fairness_utility_tradeoff_zh.pdf").stat().st_size > 1000
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["iterations"] == 30
