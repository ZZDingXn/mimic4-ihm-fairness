"""Freeze validation-selected decision policies for the fairness study.

The module deliberately consumes saved probabilities rather than model files.  That
keeps policy selection separate from training and makes the train/validation/test
boundary auditable: only validation rows are passed to threshold selection or
Fairlearn's optimizer; all rows are scored only after a policy has been frozen.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


GROUP_COLUMNS = ("gender_group", "age_group", "ethnicity_group", "insurance_group")
MODELS = ("lr", "cw_lstm")
ATTRIBUTES = ("insurance_group", "ethnicity_group")
RANDOM_STATE = 49297


class FrozenProbabilityEstimator(ClassifierMixin, BaseEstimator):
    """A fitted sklearn-shaped estimator whose sole feature is a saved risk score."""

    def __init__(self) -> None:
        # ``ThresholdOptimizer(prefit=True)`` checks for a fitted sklearn
        # estimator. No model is fitted here; classes_ records the frozen binary
        # score interface represented by the one-column input.
        self.classes_ = np.asarray([0, 1])

    def fit(self, X: Any, y: Any = None) -> "FrozenProbabilityEstimator":
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("FrozenProbabilityEstimator expects one score column")
        probability = values[:, 0]
        return np.column_stack((1.0 - probability, probability))


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def load_metadata(data_dir: Path) -> pd.DataFrame:
    metadata_path = data_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    _require_columns(metadata, {"subject_id", "stay_id", "split", "label", *GROUP_COLUMNS}, str(metadata_path))
    if metadata["stay_id"].duplicated().any():
        raise ValueError(f"{metadata_path} contains duplicated stay_id values")
    if not metadata["split"].isin(("train", "validation", "test")).all():
        raise ValueError(f"{metadata_path} contains an unsupported split")
    if not metadata["label"].isin((0, 1)).all():
        raise ValueError(f"{metadata_path} labels must be binary")
    return metadata


def load_predictions(metadata: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Load a probability file and prove it is a one-to-one metadata match."""
    prediction = pd.read_csv(path)
    _require_columns(prediction, {"stay_id", "split", "label", "probability"}, str(path))
    if prediction["stay_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated stay_id values")
    expected = metadata[["stay_id", "label", "split"]]
    joined = expected.merge(
        prediction[["stay_id", "label", "split", "probability"]],
        on="stay_id",
        how="outer",
        suffixes=("_metadata", "_prediction"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"{path} does not contain exactly the metadata stays")
    if not joined["label_metadata"].eq(joined["label_prediction"]).all():
        raise ValueError(f"{path} labels disagree with metadata")
    if not joined["split_metadata"].eq(joined["split_prediction"]).all():
        raise ValueError(f"{path} splits disagree with metadata")
    probability = joined["probability"].to_numpy(float)
    if not np.isfinite(probability).all() or not ((0.0 <= probability) & (probability <= 1.0)).all():
        raise ValueError(f"{path} contains invalid probabilities")
    return metadata.merge(prediction[["stay_id", "probability"]], on="stay_id", how="left", validate="one_to_one")


def balanced_accuracy(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    positives = y == 1
    negatives = ~positives
    if not positives.any() or not negatives.any():
        raise ValueError("Balanced accuracy requires both label classes")
    return float(0.5 * (prediction[positives].mean() + (1 - prediction[negatives]).mean()))


def select_global_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Choose the validation BA-optimal score cutoff, resolving ties upward."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y) != len(probability):
        raise ValueError("Labels and probabilities must have the same length")
    if not np.isfinite(probability).all():
        raise ValueError("Probabilities must be finite")
    # With the decision convention score >= threshold, every observed score is
    # an achievable change point; include +inf for the all-negative rule.
    candidates = np.unique(np.append(probability, np.inf))
    best_threshold = float(candidates[0])
    best_score = -np.inf
    for threshold in candidates:
        score = balanced_accuracy(y, probability >= threshold)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and threshold > best_threshold):
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, float(best_score)


def _policy_frame(metadata: pd.DataFrame, risk: np.ndarray, decision_probability: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    result = metadata[["subject_id", "stay_id", "split", "label", *GROUP_COLUMNS]].copy()
    result["risk_probability"] = np.asarray(risk, dtype=float)
    result["decision_probability"] = np.asarray(decision_probability, dtype=float)
    result["prediction"] = np.asarray(prediction, dtype=int)
    ordered = ["subject_id", "stay_id", "split", "label", "risk_probability", "decision_probability", "prediction", *GROUP_COLUMNS]
    return result[ordered]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "operator") and hasattr(value, "threshold"):
        return {"operator": value.operator, "threshold": _json_safe(value.threshold)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_policy(
    study_dir: Path,
    model: str,
    attribute: str,
    method: str,
    frame: pd.DataFrame,
    rule: dict[str, Any],
    source: Path,
) -> Path:
    output_dir = study_dir / "experiments" / "policies" / model / attribute
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{method}.csv"
    frame.to_csv(output, index=False)
    (output_dir / f"{method}_rules.json").write_text(json.dumps(_json_safe(rule), indent=2), encoding="utf-8")
    config = {
        "method": method,
        "model": model,
        "sensitive_attribute": attribute,
        "source_predictions": str(source.resolve()),
        "selection_split": "validation",
        "random_state_for_realized_predictions": RANDOM_STATE,
        "decision_probability": "expected probability of a positive alert",
    }
    (output_dir / f"{method}_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output


def _fit_fairlearn_policy(validation: pd.DataFrame, all_rows: pd.DataFrame, attribute: str, constraint: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import fairlearn
        from fairlearn.postprocessing import ThresholdOptimizer
    except ImportError as error:  # pragma: no cover - exercised only without project dependency
        raise RuntimeError("Fairlearn 0.13.0 is required for fairness postprocessing") from error
    if fairlearn.__version__ != "0.13.0":
        raise RuntimeError(f"Fairlearn 0.13.0 is required, found {fairlearn.__version__}")

    estimator = FrozenProbabilityEstimator()
    optimizer = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraint,
        objective="balanced_accuracy_score",
        grid_size=1000,
        flip=False,
        prefit=True,
        predict_method="predict_proba",
        tol=None,
    )
    validation_scores = validation[["probability"]].to_numpy(float)
    optimizer.fit(validation_scores, validation["label"].to_numpy(int), sensitive_features=validation[attribute])
    scores = all_rows[["probability"]].to_numpy(float)
    sensitive = all_rows[attribute]
    decision_probability = optimizer._pmf_predict(scores, sensitive_features=sensitive)[:, 1]
    prediction = optimizer.predict(scores, sensitive_features=sensitive, random_state=RANDOM_STATE)
    rule = {
        "constraint": constraint,
        "objective": "balanced_accuracy_score",
        "grid_size": 1000,
        "flip": False,
        "prefit": True,
        "predict_method": "predict_proba",
        "tol": None,
        "interpolation": optimizer.interpolated_thresholder_.interpolation_dict,
    }
    return decision_probability, prediction, rule


def _baseline_source(data_dir: Path, model: str, study_dir: Path | None = None) -> Path:
    """Return the approved baseline prediction source for a model.

    The original LSTM export shuffled train rows, while its validation and test
    rows remain valid.  A separately regenerated, alignment-verified file is
    preferred when it exists; the original artifact is never overwritten.
    """
    if model == "cw_lstm" and study_dir is not None:
        aligned = study_dir / "experiments" / "baseline_alignment" / "cw_lstm" / "predictions.csv"
        if aligned.is_file():
            return aligned
    return data_dir / model / "predictions.csv"


def _reweight_source(study_dir: Path, attribute: str, model: str) -> Path:
    return study_dir / "experiments" / "reweight" / attribute / model / "predictions.csv"


def build_policies(data_dir: Path, study_dir: Path = Path("fairness_study"), baseline_only: bool = False) -> list[Path]:
    """Build all prespecified policy files and return their paths.

    ``baseline_only`` intentionally leaves out reweighted models while still
    creating every baseline decision policy, allowing the first review pass to
    finish before GPU retraining artifacts exist.
    """
    data_dir, study_dir = Path(data_dir), Path(study_dir)
    metadata = load_metadata(data_dir)
    outputs: list[Path] = []
    for model in MODELS:
        baseline_path = _baseline_source(data_dir, model, study_dir)
        baseline = load_predictions(metadata, baseline_path)
        validation = baseline[baseline["split"].eq("validation")].copy()
        threshold, ba = select_global_threshold(validation["label"].to_numpy(int), validation["probability"].to_numpy(float))
        risk = baseline["probability"].to_numpy(float)
        for attribute in ATTRIBUTES:
            baseline_probability = (risk >= 0.5).astype(float)
            outputs.append(_write_policy(study_dir, model, attribute, "baseline_05", _policy_frame(metadata, risk, baseline_probability, baseline_probability.astype(int)), {"threshold": 0.5, "comparison": ">="}, baseline_path))
            global_probability = (risk >= threshold).astype(float)
            outputs.append(_write_policy(study_dir, model, attribute, "global_threshold", _policy_frame(metadata, risk, global_probability, global_probability.astype(int)), {"threshold": threshold, "comparison": ">=", "validation_balanced_accuracy": ba}, baseline_path))
            for method, constraint in (("opportunity", "true_positive_rate_parity"), ("equalized_odds", "equalized_odds")):
                probability, prediction, rule = _fit_fairlearn_policy(validation, baseline, attribute, constraint)
                outputs.append(_write_policy(study_dir, model, attribute, method, _policy_frame(metadata, risk, probability, prediction), rule, baseline_path))

            if not baseline_only:
                reweight_path = _reweight_source(study_dir, attribute, model)
                reweighted = load_predictions(metadata, reweight_path)
                reweighted_validation = reweighted[reweighted["split"].eq("validation")]
                reweight_threshold, reweight_ba = select_global_threshold(reweighted_validation["label"].to_numpy(int), reweighted_validation["probability"].to_numpy(float))
                reweight_risk = reweighted["probability"].to_numpy(float)
                reweight_probability = (reweight_risk >= reweight_threshold).astype(float)
                outputs.append(_write_policy(study_dir, model, attribute, "reweight", _policy_frame(metadata, reweight_risk, reweight_probability, reweight_probability.astype(int)), {"threshold": reweight_threshold, "comparison": ">=", "validation_balanced_accuracy": reweight_ba}, reweight_path))

    manifest = {
        "data_dir": str(data_dir.resolve()),
        "baseline_only": baseline_only,
        "models": list(MODELS),
        "attributes": list(ATTRIBUTES),
        "fairlearn_required_version": "0.13.0",
        "baseline_prediction_sources": {
            model: str(_baseline_source(data_dir, model, study_dir).resolve()) for model in MODELS
        },
        "cw_lstm_train_alignment": {
            "original_issue": "The original saved train predictions were shuffled relative to metadata.",
            "policy": "Prefer experiments/baseline_alignment/cw_lstm/predictions.csv when present; retain the original artifact unchanged.",
            "validation_and_test_scores": "Unchanged by the alignment correction.",
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    manifest_path = study_dir / "experiments" / "policies" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze validation-selected fairness decision policies")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing metadata.csv and baseline model folders")
    parser.add_argument("--study-dir", type=Path, default=Path("fairness_study"))
    parser.add_argument("--baseline-only", action="store_true", help="Skip reweighted-model policies before their predictions exist")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = build_policies(args.data_dir, args.study_dir, baseline_only=args.baseline_only)
    print(json.dumps({"policy_files": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
