from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mimic4_ihm.evaluate import (
    evaluate,
    exponential_calibration_bins,
    load_prediction,
    wilson_interval,
)


def _metadata(rows: int = 60) -> pd.DataFrame:
    index = np.arange(rows)
    split = np.where(index < 36, "train", np.where(index < 45, "validation", "test"))
    label = (index % 5 == 0).astype(int)
    return pd.DataFrame(
        {
            "subject_id": 10_000 + index,
            "hadm_id": 20_000 + index,
            "stay_id": 30_000 + index,
            "split": split,
            "label": label,
            "gender_group": np.where(index % 3 == 0, "Female", "Male"),
            "age_group": np.asarray(["18-29", "30-49", "50-69", "70-89", "90+"])[index % 5],
            "ethnicity_group": np.asarray(["Asian", "Black", "Hispanic", "White", "Other"])[index % 5],
            "insurance_group": np.asarray(["Medicare", "Medicaid", "Private", "Other"])[index % 4],
        }
    )


def _prediction(metadata: pd.DataFrame, scale: float) -> pd.DataFrame:
    probability = np.clip(0.04 + scale * metadata["label"].to_numpy() + np.arange(len(metadata)) / 500, 0, 1)
    return metadata[["subject_id", "hadm_id", "stay_id", "split", "label"]].assign(
        model="synthetic", probability=probability
    )


def test_wilson_and_exponential_bins_cover_every_observation() -> None:
    lower, upper = wilson_interval(2, 10)
    assert 0 <= lower < 0.2 < upper <= 1
    y = np.asarray([0, 1] * 10)
    p = np.linspace(0.01, 0.99, len(y))
    groups = exponential_calibration_bins(y, p)
    assert groups["n"].sum() == len(y)
    assert groups["risk_group"].is_monotonic_increasing


def test_evaluation_writes_metrics_calibration_plots_and_paired_differences(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata_path = tmp_path / "metadata.csv"
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    metadata.to_csv(metadata_path, index=False)
    _prediction(metadata, 0.72).to_csv(first_path, index=False)
    _prediction(metadata, 0.40).to_csv(second_path, index=False)
    output = tmp_path / "evaluation"
    evaluate(
        argparse.Namespace(
            metadata=metadata_path,
            predictions=[f"lstm={first_path}", f"lr={second_path}"],
            output_dir=output,
            bootstrap_iterations=20,
            subgroup_bootstrap_iterations=10,
            seed=49297,
        )
    )
    metrics = json.loads((output / "overall_metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"lstm", "lr"}
    comparison = pd.read_csv(output / "model_comparison.csv")
    assert len(comparison) > 3
    assert {"overall", "gender", "age", "ethnicity", "insurance"}.issubset(comparison["attribute"])
    assert len(pd.read_csv(output / "calibration_in_large.csv")) > 2
    assert (output / "calibration_lstm.svg").read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert (output / "calibration_lstm.png").stat().st_size > 1000
    assert (output / "calibration_lstm_gender_Female.png").stat().st_size > 1000
    subgroup = pd.read_csv(output / "subgroup_metrics.csv")
    assert subgroup["auroc"].isna().any(), "single-class subgroups must be retained with NA discrimination"


def test_prediction_label_mismatch_is_rejected(tmp_path: Path) -> None:
    metadata = _metadata(10)
    prediction = _prediction(metadata, 0.5)
    prediction.loc[0, "label"] = 1 - prediction.loc[0, "label"]
    path = tmp_path / "bad.csv"
    prediction.to_csv(path, index=False)
    with pytest.raises(ValueError, match="labels disagree"):
        load_prediction(metadata, path, "bad")
