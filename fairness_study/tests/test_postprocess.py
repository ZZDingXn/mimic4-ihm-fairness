from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from postprocess import (  # noqa: E402
    FrozenProbabilityEstimator,
    _fit_fairlearn_policy,
    balanced_accuracy,
    load_predictions,
    select_global_threshold,
)


def test_global_threshold_prefers_higher_cutoff_on_tie() -> None:
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    threshold, score = select_global_threshold(labels, scores)
    assert threshold == pytest.approx(0.9)
    assert score == pytest.approx(0.75)


def test_balanced_accuracy_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both label classes"):
        balanced_accuracy(np.array([1, 1]), np.array([1, 0]))


def test_saved_predictions_must_match_metadata(tmp_path: Path) -> None:
    metadata = pd.DataFrame({
        "subject_id": [1, 2], "stay_id": [10, 11], "split": ["train", "validation"], "label": [0, 1],
    })
    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"stay_id": [10, 11], "split": ["train", "test"], "label": [0, 1], "probability": [0.1, 0.8]}).to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="splits disagree"):
        load_predictions(metadata, invalid)


def test_frozen_probability_estimator_preserves_scores() -> None:
    estimator = FrozenProbabilityEstimator()
    scores = np.array([[0.2], [0.8]])
    np.testing.assert_allclose(estimator.predict_proba(scores), [[0.8, 0.2], [0.2, 0.8]])


def test_fairlearn_expected_probabilities_and_realized_predictions_are_reproducible() -> None:
    pytest.importorskip("fairlearn")
    frame = pd.DataFrame(
        {
            "probability": [0.95, 0.80, 0.65, 0.35, 0.20, 0.05, 0.90, 0.70, 0.55, 0.45, 0.25, 0.10],
            "label": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            "insurance_group": ["Private"] * 6 + ["Medicare"] * 6,
        }
    )
    expected_a, realized_a, _ = _fit_fairlearn_policy(
        frame, frame, "insurance_group", "true_positive_rate_parity"
    )
    expected_b, realized_b, _ = _fit_fairlearn_policy(
        frame, frame, "insurance_group", "true_positive_rate_parity"
    )
    assert np.all((0.0 <= expected_a) & (expected_a <= 1.0))
    np.testing.assert_allclose(expected_a, expected_b)
    np.testing.assert_array_equal(realized_a, realized_b)
