from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from mimic4_ihm.constants import build_header
from mimic4_ihm.model import ChannelWiseLSTM
from mimic4_ihm.train import MemmapDataset, predict_lstm, train_lstm
from fairness_study.src.reweight import compute_training_weights, run


def test_train_only_reweighting_has_mean_one_and_expected_joint_mass() -> None:
    metadata = pd.DataFrame(
        {
            "split": ["train"] * 8 + ["validation", "test"],
            "label": [0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
            "insurance_group": ["Private", "Private", "Medicare", "Private", "Medicare", "Medicare", "Medicare", "Private", "Other", "Other"],
        }
    )
    weights, audit = compute_training_weights(metadata, "insurance_group")
    train = metadata["split"].eq("train")
    assert np.isclose(weights[train].mean(), 1.0)
    assert np.all(weights[~train] == 1.0)
    assert set(audit.columns) >= {"insurance_group", "label", "n", "weight", "normalized_weight"}
    weighted = metadata.loc[train, ["insurance_group", "label"]].copy()
    weighted["weight"] = weights[train]
    group_label_mass = weighted.groupby(["insurance_group", "label"])["weight"].sum() / weights[train].sum()
    assert np.allclose(group_label_mass.to_numpy(), 0.25)


def test_reweighting_rejects_target_groups_without_both_labels() -> None:
    metadata = pd.DataFrame(
        {
            "split": ["train"] * 4,
            "label": [0, 0, 1, 1],
            "insurance_group": ["Private", "Private", "Medicare", "Medicare"],
        }
    )
    with pytest.raises(ValueError, match="need both labels 0 and 1"):
        compute_training_weights(metadata, "insurance_group")


def test_reweighted_lr_writes_auditable_outputs() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / "data"
        output = root / "output"
        data.mkdir()
        rng = np.random.default_rng(12)
        np.save(data / "lr_X_raw.npy", rng.normal(size=(16, 714)).astype(np.float32))
        metadata = pd.DataFrame(
            {
                "subject_id": np.arange(16),
                "hadm_id": np.arange(100, 116),
                "stay_id": np.arange(200, 216),
                "split": ["train"] * 12 + ["validation"] * 2 + ["test"] * 2,
                "label": [0, 1] * 8,
                "insurance_group": ["Private", "Private", "Medicare", "Medicare"] * 4,
                "ethnicity_group": ["White"] * 16,
            }
        )
        metadata.to_csv(data / "metadata.csv", index=False)
        args = argparse.Namespace(
            attribute="insurance_group", model="logistic_regression", data_dir=data, output_dir=output,
            seed=49297, device="cpu", epochs=100, patience=0, batch_size=8, dim=8, size_coef=4.0,
            dropout=0.3, learning_rate=1e-3, c=0.001, lr_random_state=42, max_iter=1000,
        )
        run(args)
        assert len(pd.read_csv(output / "predictions.csv")) == 16
        assert len(pd.read_csv(output / "training_weights.csv")) == 16
        assert (output / "weight_audit.csv").is_file()
        assert json.loads((output / "reweight_config.json").read_text(encoding="utf-8"))["attribute"] == "insurance_group"


def test_lstm_training_predictions_follow_metadata_order_after_shuffled_training() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data, output = root / "data", root / "output"
        data.mkdir()
        features = np.random.default_rng(4).normal(size=(12, 48, 76)).astype(np.float32)
        np.save(data / "lstm_X.npy", features)
        metadata = pd.DataFrame(
            {
                "subject_id": np.arange(12), "hadm_id": np.arange(100, 112),
                "stay_id": np.arange(200, 212), "split": ["train"] * 8 + ["validation"] * 2 + ["test"] * 2,
                "label": [0, 1] * 6,
            }
        )
        _, channel_indices, _ = build_header()
        args = argparse.Namespace(
            data_dir=data, output_dir=output, seed=49297, device="cpu", batch_size=3, dim=2,
            size_coef=1.0, dropout=0.0, learning_rate=1e-3, epochs=1, patience=0,
        )
        train_lstm(args, metadata, {"channel_indices": channel_indices}, sample_weights=np.ones(12))
        checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
        model = ChannelWiseLSTM(channel_indices, dim=2, size_coef=1.0, dropout=0.0)
        model.load_state_dict(checkpoint["model_state"])
        indices = np.flatnonzero(metadata["split"].eq("train"))
        _, expected, _ = predict_lstm(
            model, DataLoader(MemmapDataset(data / "lstm_X.npy", metadata["label"].to_numpy(), indices), batch_size=3), torch.device("cpu")
        )
        observed = pd.read_csv(output / "predictions.csv").query("split == 'train'")["probability"].to_numpy()
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)
