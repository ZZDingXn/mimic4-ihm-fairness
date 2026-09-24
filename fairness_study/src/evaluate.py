"""Evaluation of frozen fairness-policy outputs.

The evaluator treats ``decision_probability`` as the probability that a
policy raises an alert.  This matters for Fairlearn policies: their random
thresholding is evaluated through its expected confusion matrix, rather than
through one incidental draw of binary alerts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

# Fairlearn is kept locally for this study so the research environment does
# not silently depend on an unrelated global installation.
_LOCAL_FAIRLEARN = Path(__file__).resolve().parents[1] / "tem" / "python_packages"
if _LOCAL_FAIRLEARN.is_dir() and str(_LOCAL_FAIRLEARN) not in sys.path:
    sys.path.insert(0, str(_LOCAL_FAIRLEARN))
from fairlearn.metrics import MetricFrame  # noqa: E402


ATTRIBUTES: dict[str, tuple[str, ...]] = {
    # The frozen study mapping has two sex groups; the other 14 categories
    # produce the planned total of 16 reported demographic groups.
    "gender_group": ("Female", "Male"),
    "age_group": ("18-29", "30-49", "50-69", "70-89", "90+"),
    "ethnicity_group": ("Asian", "Black", "Hispanic", "White", "Other"),
    "insurance_group": ("Medicare", "Medicaid", "Private", "Other"),
}
REQUIRED_COLUMNS = {
    "subject_id", "stay_id", "split", "label", "risk_probability",
    "decision_probability", "prediction", *ATTRIBUTES,
}
CONFUSION_COLUMNS = ("tn", "fp", "fn", "tp")
RATE_COLUMNS = ("tpr", "fpr", "ppv", "balanced_accuracy", "selection_rate")
DECISION_RATE_COLUMNS = (*RATE_COLUMNS, "false_positives_per_1000", "false_negatives_per_1000")


@dataclass(frozen=True)
class Policy:
    model: str
    target_attribute: str
    method: str
    path: Path
    data: pd.DataFrame

    @property
    def name(self) -> str:
        return f"{self.model}/{self.target_attribute}/{self.method}"


def _ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray | float:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    answer = np.full(np.broadcast(numerator_array, denominator_array).shape, np.nan, dtype=float)
    np.divide(numerator_array, denominator_array, out=answer, where=denominator_array != 0)
    return float(answer) if answer.ndim == 0 else answer


def expected_confusion(y: np.ndarray, decision_probability: np.ndarray) -> np.ndarray:
    """Return expected TN, FP, FN and TP for binary labels and alert chances."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(decision_probability, dtype=float)
    return np.array(((1 - y) @ (1 - q), (1 - y) @ q, y @ (1 - q), y @ q), dtype=float)


def decision_metrics(confusion: np.ndarray, n: float | None = None) -> dict[str, float]:
    tn, fp, fn, tp = np.asarray(confusion, dtype=float)
    total = tn + fp + fn + tp if n is None else float(n)
    tpr = _ratio(tp, tp + fn)
    fpr = _ratio(fp, fp + tn)
    tnr = _ratio(tn, tn + fp)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "tpr": tpr,
        "fpr": fpr,
        "ppv": _ratio(tp, tp + fp),
        "balanced_accuracy": (tpr + tnr) / 2 if np.isfinite(tpr) and np.isfinite(tnr) else np.nan,
        "selection_rate": _ratio(tp + fp, total),
        "false_positives_per_1000": _ratio(fp * 1000, total),
        "false_negatives_per_1000": _ratio(fn * 1000, total),
    }


def risk_metrics(y: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    risk = np.asarray(risk, dtype=float)
    value: dict[str, float] = {
        "auroc": np.nan,
        "auprc": np.nan,
        "brier_score": float(np.mean((risk - y) ** 2)) if len(y) else np.nan,
        "mean_calibration_bias": float(np.mean(risk - y)) if len(y) else np.nan,
    }
    if len(y) and np.unique(y).size == 2:
        value["auroc"] = float(roc_auc_score(y, risk))
        # The study baseline used a trapezoidal PR area rather than sklearn's
        # average precision (a step-function summary).
        precision, recall, _ = precision_recall_curve(y, risk)
        value["auprc"] = float(np.trapezoid(precision[::-1], recall[::-1]))
    return value


def wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    proportion = events / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return float(max(0, center - radius)), float(min(1, center + radius))


def _metric_frame_audit(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Use MetricFrame for the point group audit with expected-alert metrics."""
    def metric(y_true: np.ndarray, alert_probability: np.ndarray) -> float:
        return float(expected_confusion(y_true, alert_probability)[3])

    # MetricFrame is deliberately retained as an audit layer.  Detailed values
    # below come from the same expected confusion formula, which also supports
    # all required rates without converting randomized policies to one draw.
    audit = MetricFrame(
        metrics={"expected_tp": metric},
        y_true=frame["label"].to_numpy(int),
        y_pred=frame["decision_probability"].to_numpy(float),
        sensitive_features=frame[column].astype("string").fillna("Missing"),
    ).by_group
    return audit.rename_axis("group").reset_index()


def metric_rows(policy: Policy) -> pd.DataFrame:
    frame = policy.data
    rows: list[dict[str, object]] = []
    overall_confusion = expected_confusion(frame.label.to_numpy(), frame.decision_probability.to_numpy())
    rows.append({
        "model": policy.model, "target_attribute": policy.target_attribute, "method": policy.method,
        "attribute": "overall", "group": "Overall", "status": "available", "n": len(frame),
        "events": int(frame.label.sum()), **decision_metrics(overall_confusion, len(frame)),
        **risk_metrics(frame.label.to_numpy(), frame.risk_probability.to_numpy()),
    })
    for column, ordered_groups in ATTRIBUTES.items():
        audit = _metric_frame_audit(frame, column).set_index("group")["expected_tp"].to_dict()
        observed = frame[column].astype("string").fillna("Missing")
        groups = list(ordered_groups) + ["Missing"] if (observed == "Missing").any() else list(ordered_groups)
        for group in groups:
            selected = frame.loc[observed.eq(group)]
            if selected.empty:
                rows.append({
                    "model": policy.model, "target_attribute": policy.target_attribute, "method": policy.method,
                    "attribute": column, "group": group, "status": "absent", "n": 0, "events": 0,
                    **{key: np.nan for key in (*CONFUSION_COLUMNS, *RATE_COLUMNS,
                                                  "false_positives_per_1000", "false_negatives_per_1000",
                                                  "auroc", "auprc", "brier_score", "mean_calibration_bias")},
                })
                continue
            confusion = expected_confusion(selected.label.to_numpy(), selected.decision_probability.to_numpy())
            # This guards the audit wiring without altering the policy metric.
            if not np.isclose(confusion[3], audit[group]):
                raise AssertionError(f"MetricFrame audit disagrees for {column}={group}")
            rows.append({
                "model": policy.model, "target_attribute": policy.target_attribute, "method": policy.method,
                "attribute": column, "group": group, "status": "available", "n": len(selected),
                "events": int(selected.label.sum()), **decision_metrics(confusion, len(selected)),
                **risk_metrics(selected.label.to_numpy(), selected.risk_probability.to_numpy()),
            })
    return pd.DataFrame(rows)


def gap_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, subset in metrics[metrics.attribute.ne("overall")].groupby(
        ["model", "target_attribute", "method", "attribute"], sort=False
    ):
        available = subset[subset.status.eq("available")]
        # A gap is not allowed to become artificially smaller merely because a
        # group has no observed event, or no predicted positives, in a sample.
        tpr = available.tpr
        fpr = available.fpr
        selection = available.selection_rate
        tpr_defined = len(tpr) >= 2 and tpr.notna().all()
        fpr_defined = len(fpr) >= 2 and fpr.notna().all()
        rows.append(dict(zip(("model", "target_attribute", "method", "attribute"), keys)) | {
            "groups_with_defined_tpr": int(tpr.notna().sum()), "groups_with_defined_fpr": int(fpr.notna().sum()),
            "tpr_gap": float(tpr.max() - tpr.min()) if tpr_defined else np.nan,
            "fpr_gap": float(fpr.max() - fpr.min()) if fpr_defined else np.nan,
            "equalized_odds_gap": float(max(tpr.max() - tpr.min(), fpr.max() - fpr.min()))
                if tpr_defined and fpr_defined else np.nan,
            "worst_group_tpr": float(tpr.min()) if tpr_defined else np.nan,
            "selection_rate_gap": float(selection.max()-selection.min()) if len(selection)>=2 and selection.notna().all() else np.nan,
        })
    return pd.DataFrame(rows)


def calibration_bins(policy: Policy, bins: int = 10) -> pd.DataFrame:
    frame = policy.data
    rows: list[dict[str, object]] = []
    for attribute, group in [("overall", "Overall"), *[(a, g) for a, gs in ATTRIBUTES.items() for g in gs]]:
        selected = frame if attribute == "overall" else frame.loc[frame[attribute].eq(group)]
        if selected.empty:
            continue
        # Retain the previously declared exponential quantiles q^(1/5).
        selected = selected.sort_values("risk_probability", kind="stable")
        boundaries = np.power(np.linspace(0.0, 1.0, bins + 1), 1.0 / 5.0)
        risk_boundaries = np.quantile(selected.risk_probability.to_numpy(float), boundaries)
        bin_id = np.minimum(np.searchsorted(risk_boundaries[1:], selected.risk_probability.to_numpy(float), side="right") + 1, bins)
        for number in np.unique(bin_id):
            part = selected.iloc[bin_id == number]
            rows.append({
                "model": policy.model, "target_attribute": policy.target_attribute, "method": policy.method,
                "attribute": attribute, "group": group, "kind": "bin", "bin": int(number), "n": len(part),
                "events": int(part.label.sum()), "mean_risk": float(part.risk_probability.mean()),
                "observed_rate": float(part.label.mean()),
                "wilson_lower": wilson_interval(int(part.label.sum()), len(part))[0],
                "wilson_upper": wilson_interval(int(part.label.sum()), len(part))[1],
            })
        # LOWESS is a display-only calibration smoother.  It is calculated on
        # the original score/label pairs and does not affect reported metrics.
        from statsmodels.nonparametric.smoothers_lowess import lowess
        if len(selected) >= 2 and selected.risk_probability.nunique() >= 2:
            smooth = lowess(selected.label.to_numpy(float), selected.risk_probability.to_numpy(float), frac=0.5, it=0, return_sorted=True)
            # Export the display curve on a fixed grid, not one row for each
            # patient's original risk score.  Risk metrics remain unchanged.
            grid = np.linspace(0.0, 1.0, 201)
            grid = grid[(grid >= smooth[0, 0]) & (grid <= smooth[-1, 0])]
            for point in zip(grid, np.interp(grid, smooth[:, 0], smooth[:, 1])):
                rows.append({
                    "model": policy.model, "target_attribute": policy.target_attribute, "method": policy.method,
                    "attribute": attribute, "group": group, "kind": "lowess", "bin": np.nan, "n": np.nan,
                    "events": np.nan, "mean_risk": float(point[0]), "observed_rate": float(point[1]),
                    "wilson_lower": np.nan, "wilson_upper": np.nan,
                })
    return pd.DataFrame(rows)


def _subject_contributions(frame: pd.DataFrame, attribute: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Subject-level expected-confusion contributions, by group when requested."""
    groups = ("Overall",) if attribute is None else ATTRIBUTES[attribute]
    subject_codes, subjects = pd.factorize(frame.subject_id, sort=True)
    out = np.zeros((len(subjects), len(groups), 4), dtype=float)
    y = frame.label.to_numpy(float)
    q = frame.decision_probability.to_numpy(float)
    values = np.column_stack(((1-y)*(1-q), (1-y)*q, y*(1-q), y*q))
    group_values = np.full(len(frame), -1, dtype=int)
    if attribute is None:
        group_values.fill(0)
    else:
        lookup = {value: i for i, value in enumerate(groups)}
        group_values = frame[attribute].map(lookup).fillna(-1).to_numpy(int)
    valid = group_values >= 0
    for outcome in range(4):
        np.add.at(out[:, :, outcome], (subject_codes[valid], group_values[valid]), values[valid, outcome])
    return np.asarray(subjects), out


def _bootstrap_counts(n_subjects: int, iterations: int, rng: np.random.Generator) -> np.ndarray:
    # A count matrix allows all policies to use exactly the same patient draws.
    counts = np.empty((iterations, n_subjects), dtype=np.int16)
    for row in range(iterations):
        counts[row] = np.bincount(rng.integers(n_subjects, size=n_subjects), minlength=n_subjects)
    return counts


def _bootstrap_summary(values: np.ndarray, names: Iterable[str]) -> pd.DataFrame:
    lower: list[float] = []
    upper: list[float] = []
    valid = np.sum(np.isfinite(values), axis=0)
    for index in range(values.shape[1]):
        column = values[:, index]
        finite = column[np.isfinite(column)]
        lower.append(float(np.percentile(finite, 2.5)) if len(finite) else np.nan)
        upper.append(float(np.percentile(finite, 97.5)) if len(finite) else np.nan)
    return pd.DataFrame({
        "metric": list(names), "lower_95": lower,
        "upper_95": upper, "valid_iterations": valid,
    })


def _risk_auc_pr_draws(
    y: np.ndarray, risk: np.ndarray, subject_codes: np.ndarray, counts: np.ndarray, mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized weighted-rank cluster AUROC and trapezoidal AUPRC draws."""
    aucs = np.full(len(counts), np.nan, dtype=float)
    aprcs = np.full(len(counts), np.nan, dtype=float)
    y = y[mask]
    risk = risk[mask]
    codes = subject_codes[mask]
    if not len(y):
        return aucs, aprcs
    order = np.argsort(risk, kind="stable")
    y = y[order]
    codes = codes[order]
    sorted_risk = risk[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_risk)) + 1]
    positive = y.astype(float)
    negative = 1.0 - positive
    # Work in blocks: this keeps the test cohort bootstrap below a few MB while
    # avoiding 10,000 Python/sklearn metric calls for every subgroup.
    for start in range(0, len(counts), 128):
        stop = min(start + 128, len(counts))
        weights = counts[start:stop, codes].astype(float)
        pos_group = np.add.reduceat(weights * positive, starts, axis=1)
        neg_group = np.add.reduceat(weights * negative, starts, axis=1)
        total_pos = pos_group.sum(axis=1)
        total_neg = neg_group.sum(axis=1)
        valid = (total_pos > 0) & (total_neg > 0)
        preceding_neg = np.cumsum(neg_group, axis=1) - neg_group
        numerator = (pos_group * (preceding_neg + 0.5 * neg_group)).sum(axis=1)
        block_auc = _ratio(numerator, total_pos * total_neg)
        # Precision--recall thresholds are traversed from high to low risk.
        descending_pos = pos_group[:, ::-1]
        descending_neg = neg_group[:, ::-1]
        cumulative_pos = np.cumsum(descending_pos, axis=1)
        cumulative_total = np.cumsum(descending_pos + descending_neg, axis=1)
        precision = _ratio(cumulative_pos, cumulative_total)
        # A resample may omit the maximum-score group.  Its first represented
        # PR point is (recall=0, precision=1), rather than an undefined value.
        precision = np.where(cumulative_total == 0, 1.0, precision)
        recall = _ratio(cumulative_pos, total_pos[:, None])
        block_aprc = np.trapezoid(np.column_stack((np.ones(len(weights)), precision)),
                                  np.column_stack((np.zeros(len(weights)), recall)), axis=1)
        aucs[start:stop] = np.where(valid, block_auc, np.nan)
        aprcs[start:stop] = np.where(valid, block_aprc, np.nan)
    return aucs, aprcs


def bootstrap_draws(
    policy: Policy, counts: np.ndarray, risk_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[str, str], np.ndarray]:
    """Vectorized cluster-bootstrap metric draws keyed by (attribute, metric)."""
    draws: dict[tuple[str, str], np.ndarray] = {}
    frame = policy.data
    subject_codes, subjects = pd.factorize(frame.subject_id, sort=True)
    if len(subjects) != counts.shape[1]:
        raise ValueError("Policies must share the same test subjects for paired bootstrap")
    y = frame.label.to_numpy(float)
    risk = frame.risk_probability.to_numpy(float)
    # These two probability metrics have additive subject contributions, unlike
    # AUROC/AUPRC, and therefore support the full 10,000 cluster resamples.
    risk_contribution = np.zeros((len(subjects), 3), dtype=float)
    np.add.at(risk_contribution[:, 0], subject_codes, (risk - y) ** 2)
    np.add.at(risk_contribution[:, 1], subject_codes, risk - y)
    np.add.at(risk_contribution[:, 2], subject_codes, 1.0)
    risk_total = counts @ risk_contribution
    draws[("overall", "brier_score")] = _ratio(risk_total[:, 0], risk_total[:, 2])
    draws[("overall", "mean_calibration_bias")] = _ratio(risk_total[:, 1], risk_total[:, 2])
    subjects, overall = _subject_contributions(policy.data, None)
    if len(subjects) != counts.shape[1]:
        raise ValueError("Policies must share the same test subjects for paired bootstrap")
    overall_confusion = counts @ overall[:, 0, :]
    overall_metrics = np.column_stack((
        _ratio(overall_confusion[:, 3], overall_confusion[:, 3] + overall_confusion[:, 2]),
        _ratio(overall_confusion[:, 1], overall_confusion[:, 1] + overall_confusion[:, 0]),
        _ratio(overall_confusion[:, 3], overall_confusion[:, 3] + overall_confusion[:, 1]),
        (_ratio(overall_confusion[:, 3], overall_confusion[:, 3] + overall_confusion[:, 2]) +
         _ratio(overall_confusion[:, 0], overall_confusion[:, 0] + overall_confusion[:, 1])) / 2,
        _ratio(overall_confusion[:, 3] + overall_confusion[:, 1], overall_confusion.sum(axis=1)),
    ))
    overall_with_burden = np.column_stack((overall_metrics,
        _ratio(overall_confusion[:, 1] * 1000, overall_confusion.sum(axis=1)),
        _ratio(overall_confusion[:, 2] * 1000, overall_confusion.sum(axis=1))))
    for index, metric in enumerate(DECISION_RATE_COLUMNS):
        draws[("overall", metric)] = overall_with_burden[:, index]
    for attribute in ATTRIBUTES:
        _, contribution = _subject_contributions(policy.data, attribute)
        matrices = (counts @ contribution.reshape(len(subjects), -1)).reshape(len(counts), len(ATTRIBUTES[attribute]), 4)
        tpr = _ratio(matrices[:, :, 3], matrices[:, :, 3] + matrices[:, :, 2])
        fpr = _ratio(matrices[:, :, 1], matrices[:, :, 1] + matrices[:, :, 0])
        active = np.array([frame[attribute].eq(group).any() for group in ATTRIBUTES[attribute]])
        active_tpr = tpr[:, active]
        active_fpr = fpr[:, active]
        selection = _ratio(matrices[:, :, 3]+matrices[:, :, 1], matrices.sum(axis=2))[:, active]
        selection_gap = np.where(np.isfinite(selection).all(axis=1), selection.max(axis=1)-selection.min(axis=1), np.nan)
        tpr_present = np.isfinite(active_tpr).all(axis=1) if active.any() else np.zeros(len(counts), dtype=bool)
        fpr_present = np.isfinite(active_fpr).all(axis=1) if active.any() else np.zeros(len(counts), dtype=bool)
        min_tpr = np.where(tpr_present, active_tpr.min(axis=1), np.nan)
        max_tpr = np.where(tpr_present, active_tpr.max(axis=1), np.nan)
        min_fpr = np.where(fpr_present, active_fpr.min(axis=1), np.nan)
        max_fpr = np.where(fpr_present, active_fpr.max(axis=1), np.nan)
        tpr_gap = max_tpr - min_tpr
        fpr_gap = max_fpr - min_fpr
        gaps = np.column_stack((tpr_gap, fpr_gap,
                                np.where(tpr_present & fpr_present, np.maximum(tpr_gap, fpr_gap), np.nan),
                                min_tpr, selection_gap))
        for index, metric in enumerate(("tpr_gap", "fpr_gap", "equalized_odds_gap", "worst_group_tpr", "selection_rate_gap")):
            draws[(attribute, metric)] = gaps[:, index]
        # All subgroup decision metrics use expected confusion counts and the
        # same cluster draws.  Empty frozen groups remain explicit all-NA rows.
        group_values = frame[attribute]
        for group_index, group in enumerate(ATTRIBUTES[attribute]):
            confusion = matrices[:, group_index, :]
            total = confusion.sum(axis=1)
            group_metrics = np.column_stack((
                _ratio(confusion[:, 3], confusion[:, 3] + confusion[:, 2]),
                _ratio(confusion[:, 1], confusion[:, 1] + confusion[:, 0]),
                _ratio(confusion[:, 3], confusion[:, 3] + confusion[:, 1]),
                (_ratio(confusion[:, 3], confusion[:, 3] + confusion[:, 2]) +
                 _ratio(confusion[:, 0], confusion[:, 0] + confusion[:, 1])) / 2,
                _ratio(confusion[:, 3] + confusion[:, 1], total),
                _ratio(confusion[:, 1] * 1000, total),
                _ratio(confusion[:, 2] * 1000, total),
            ))
            for index, metric in enumerate((*RATE_COLUMNS, "false_positives_per_1000", "false_negatives_per_1000")):
                draws[(f"{attribute}:{group}", metric)] = group_metrics[:, index]
            mask = group_values.eq(group).to_numpy()
            group_risk = np.zeros((len(subjects), 3), dtype=float)
            np.add.at(group_risk[:, 0], subject_codes[mask], ((risk - y) ** 2)[mask])
            np.add.at(group_risk[:, 1], subject_codes[mask], (risk - y)[mask])
            np.add.at(group_risk[:, 2], subject_codes[mask], 1.0)
            total = counts @ group_risk
            draws[(f"{attribute}:{group}", "brier_score")] = _ratio(total[:, 0], total[:, 2])
            draws[(f"{attribute}:{group}", "mean_calibration_bias")] = _ratio(total[:, 1], total[:, 2])
            key_source = np.ascontiguousarray(np.column_stack((subject_codes, y, risk, mask))).view(np.uint8)
            cache_key = hashlib.sha256(key_source).hexdigest()
            if cache_key not in risk_cache:
                risk_cache[cache_key] = _risk_auc_pr_draws(y, risk, subject_codes, counts, mask)
            draws[(f"{attribute}:{group}", "auroc")], draws[(f"{attribute}:{group}", "auprc")] = risk_cache[cache_key]
    overall_key = hashlib.sha256(np.ascontiguousarray(np.column_stack((subject_codes, y, risk))).view(np.uint8)).hexdigest()
    if overall_key not in risk_cache:
        risk_cache[overall_key] = _risk_auc_pr_draws(y, risk, subject_codes, counts, np.ones(len(frame), dtype=bool))
    draws[("overall", "auroc")], draws[("overall", "auprc")] = risk_cache[overall_key]
    return draws


def bootstrap_policy(
    policy: Policy, counts: np.ndarray, risk_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Cluster bootstrap CIs for decision and additive probability metrics."""
    draws = bootstrap_draws(policy, counts, risk_cache)
    rows: list[pd.DataFrame] = []
    for (location, metric), values in draws.items():
        if ":" in location:
            attribute, group = location.split(":", 1)
        elif location == "overall":
            attribute, group = "overall", "Overall"
        else:
            attribute, group = location, "All groups"
        summary = _bootstrap_summary(values.reshape(-1, 1), (metric,))
        summary.insert(0, "attribute", attribute)
        summary.insert(0, "group", group)
        rows.append(summary)
    answer = pd.concat(rows, ignore_index=True)
    answer.insert(0, "method", policy.method)
    answer.insert(0, "target_attribute", policy.target_attribute)
    answer.insert(0, "model", policy.model)
    return answer, draws


def load_policies(study_dir: Path) -> list[Policy]:
    root = study_dir / "experiments" / "policies"
    if not root.is_dir():
        raise FileNotFoundError(f"Policy directory does not exist: {root}")
    policies: list[Policy] = []
    for path in sorted(root.glob("*/*/*.csv")):
        relative = path.relative_to(root).parts
        if len(relative) != 3:
            continue
        model, target_attribute, filename = relative
        method = Path(filename).stem
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        if frame.stay_id.duplicated().any():
            raise ValueError(f"{path} contains duplicate stay_id values")
        if not frame.label.isin([0, 1]).all():
            raise ValueError(f"{path} labels must be binary")
        for column in ("risk_probability", "decision_probability"):
            if not frame[column].between(0, 1).all():
                raise ValueError(f"{path} has {column} outside [0, 1]")
        test = frame.loc[frame.split.eq("test")].copy()
        if test.empty:
            raise ValueError(f"{path} has no test rows")
        policies.append(Policy(model, target_attribute, method, path, test.sort_values("stay_id").reset_index(drop=True)))
    if not policies:
        raise FileNotFoundError(f"No policy CSVs found below {root}")
    return policies


def _validate_pairing(policies: list[Policy]) -> np.ndarray:
    first = policies[0].data[["subject_id", "stay_id", "label"]]
    for policy in policies[1:]:
        candidate = policy.data[["subject_id", "stay_id", "label"]]
        if not first.equals(candidate):
            raise ValueError(f"{policy.path} does not have the same sorted test stays, subjects, and labels")
    return first.subject_id.to_numpy()


def evaluate(study_dir: Path, iterations: int = 10_000, seed: int = 49_297) -> dict[str, Path]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    policies = load_policies(study_dir)
    subject_ids = _validate_pairing(policies)
    _, unique_subjects = pd.factorize(subject_ids, sort=True)
    counts = _bootstrap_counts(len(unique_subjects), iterations, np.random.default_rng(seed))
    all_metrics = pd.concat([metric_rows(policy) for policy in policies], ignore_index=True)
    gaps = gap_rows(all_metrics)
    cals = pd.concat([calibration_bins(policy) for policy in policies], ignore_index=True)
    risk_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bootstrap = [bootstrap_policy(policy, counts, risk_cache) for policy in policies]
    confidence = pd.concat([item[0] for item in bootstrap], ignore_index=True)
    policy_draws = {policy.name: item[1] for policy, item in zip(policies, bootstrap)}

    # Differences are calculated from the *same* patient-resampling draws,
    # rather than subtracting two marginal confidence intervals.
    point_decision = all_metrics.melt(
        id_vars=["model", "target_attribute", "method", "attribute", "group", "status"],
        value_vars=[*DECISION_RATE_COLUMNS, "auroc", "auprc", "brier_score", "mean_calibration_bias"],
        var_name="metric", value_name="value",
    )
    point_gaps = gaps.melt(
        id_vars=["model", "target_attribute", "method", "attribute"],
        value_vars=["tpr_gap", "fpr_gap", "equalized_odds_gap", "worst_group_tpr", "selection_rate_gap"],
        var_name="metric", value_name="value",
    ).assign(group="All groups", status="available")
    point = pd.concat([point_decision, point_gaps], ignore_index=True)
    paired_rows: list[dict[str, object]] = []
    indexed_points = point.set_index(["model", "target_attribute", "method", "attribute", "group", "metric"])["value"]
    for policy in policies:
        if policy.method == "global_threshold":
            continue
        global_name = f"{policy.model}/{policy.target_attribute}/global_threshold"
        baseline = policy_draws.get(global_name)
        if baseline is None:
            continue
        for (location, metric), values in policy_draws[policy.name].items():
            if (location, metric) not in baseline:
                continue
            if ":" in location:
                attribute, group = location.split(":", 1)
            elif location == "overall":
                attribute, group = "overall", "Overall"
            else:
                attribute, group = location, "All groups"
            diff = values - baseline[(location, metric)]
            current = indexed_points.get((policy.model, policy.target_attribute, policy.method, attribute, group, metric), np.nan)
            reference = indexed_points.get((policy.model, policy.target_attribute, "global_threshold", attribute, group, metric), np.nan)
            finite_diff = diff[np.isfinite(diff)]
            paired_rows.append({
                "model": policy.model, "target_attribute": policy.target_attribute, "method": policy.method,
                "attribute": attribute, "group": group, "metric": metric,
                "value": current, "global_threshold_value": reference,
                "difference_vs_global_threshold": current - reference,
                "lower_95": float(np.percentile(finite_diff, 2.5)) if len(finite_diff) else np.nan,
                "upper_95": float(np.percentile(finite_diff, 97.5)) if len(finite_diff) else np.nan,
                "valid_iterations": int(len(finite_diff)),
            })
    paired = pd.DataFrame(paired_rows)
    model_rows = []
    for metric in ("auroc", "auprc"):
        lr_draws = policy_draws["lr/insurance_group/baseline_05"][("overall", metric)]
        lstm_draws = policy_draws["cw_lstm/insurance_group/baseline_05"][("overall", metric)]
        difference = lstm_draws - lr_draws
        finite = difference[np.isfinite(difference)]
        lr_value = indexed_points.loc[("lr", "insurance_group", "baseline_05", "overall", "Overall", metric)]
        lstm_value = indexed_points.loc[("cw_lstm", "insurance_group", "baseline_05", "overall", "Overall", metric)]
        model_rows.append({"metric": metric, "lr": lr_value, "cw_lstm": lstm_value,
                           "lstm_minus_lr": lstm_value-lr_value,
                           "lower_95": float(np.percentile(finite, 2.5)),
                           "upper_95": float(np.percentile(finite, 97.5)),
                           "valid_iterations": len(finite)})
    model_comparison = pd.DataFrame(model_rows)

    result_dir = study_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics": result_dir / "policy_metrics.csv", "gaps": result_dir / "fairness_gaps.csv",
        "calibration": result_dir / "calibration_bins.csv", "confidence": result_dir / "bootstrap_confidence_intervals.csv",
        "paired": result_dir / "paired_differences_vs_global_threshold.csv", "manifest": result_dir / "evaluation_manifest.json",
        "schema": result_dir / "evaluation_table_schema.json",
        "model_comparison": result_dir / "model_comparison_patient_cluster.csv",
    }
    all_metrics.to_csv(outputs["metrics"], index=False)
    gaps.to_csv(outputs["gaps"], index=False)
    cals.to_csv(outputs["calibration"], index=False)
    confidence.to_csv(outputs["confidence"], index=False)
    paired.to_csv(outputs["paired"], index=False)
    model_comparison.to_csv(outputs["model_comparison"], index=False)
    outputs["schema"].write_text(json.dumps({
        "policy_metrics.csv": {"grain": "policy x reported attribute x group", "columns": list(all_metrics.columns)},
        "fairness_gaps.csv": {"grain": "policy x attribute", "columns": list(gaps.columns)},
        "calibration_bins.csv": {"grain": "policy x attribute x group x bin or LOWESS point", "columns": list(cals.columns)},
        "bootstrap_confidence_intervals.csv": {"grain": "policy x attribute x group x metric", "columns": list(confidence.columns)},
        "paired_differences_vs_global_threshold.csv": {"grain": "non-global policy x attribute x group x metric", "columns": list(paired.columns)},
        "model_comparison_patient_cluster.csv": {"grain": "baseline probability metric", "columns": list(model_comparison.columns)},
    }, indent=2), encoding="utf-8")
    outputs["manifest"].write_text(json.dumps({
        "iterations": iterations, "seed": seed, "policy_files": [str(p.path) for p in policies],
        "test_stays": int(len(policies[0].data)), "test_subjects": int(len(unique_subjects)),
        "decision_evaluation": "expected confusion counts from decision_probability",
        "risk_evaluation": "risk_probability; AUROC and trapezoidal AUPRC",
        "calibration_curve_export": "LOWESS display values interpolated on a fixed 0.005 risk grid within observed support; no row-per-patient risk coordinates exported",
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
    }, indent=2), encoding="utf-8")
    from .plots import write_publication_plots
    write_publication_plots(all_metrics, gaps, cals, confidence, result_dir / "plots", study_dir / "figures", study_dir / "tem")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen MIMIC-IV fairness policies")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=49_297)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args.study_dir, args.iterations, args.seed)


if __name__ == "__main__":
    main()
