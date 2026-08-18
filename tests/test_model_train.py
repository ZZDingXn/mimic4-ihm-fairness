from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mimic4_ihm.constants import build_header
from mimic4_ihm.metrics import binary_metrics
from mimic4_ihm.model import ChannelWiseLSTM
from mimic4_ihm.train import train_logistic_regression, train_lstm


class ModelTests(unittest.TestCase):
    def test_channel_wise_forward_and_reload(self) -> None:
        _, channel_indices, _ = build_header()
        torch.manual_seed(1)
        model = ChannelWiseLSTM(channel_indices, dim=8, size_coef=4.0, dropout=0.0)
        model.eval()
        features = torch.randn(3, 48, 76)
        first = model(features)
        clone = ChannelWiseLSTM(channel_indices, dim=8, size_coef=4.0, dropout=0.0)
        clone.load_state_dict(model.state_dict())
        clone.eval()
        second = clone(features)
        self.assertEqual(first.shape, (3,))
        torch.testing.assert_close(first, second)

    def test_binary_metrics_match_original_definitions(self) -> None:
        result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.2]))
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tp"], 1)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertIsNotNone(result["auroc"])
        self.assertIsNotNone(result["auprc"])


class LogisticTrainingTests(unittest.TestCase):
    def test_lr_uses_shared_metadata_and_writes_unified_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "output"
            data.mkdir()
            rng = np.random.default_rng(3)
            matrix = rng.normal(size=(30, 714)).astype(np.float32)
            matrix[::3, 10:20] = np.nan
            np.save(data / "lr_X_raw.npy", matrix)
            split = np.asarray(["train"] * 20 + ["validation"] * 5 + ["test"] * 5)
            labels = np.asarray([0, 1] * 15)
            metadata = pd.DataFrame(
                {
                    "subject_id": np.arange(30),
                    "hadm_id": np.arange(100, 130),
                    "stay_id": np.arange(200, 230),
                    "split": split,
                    "label": labels,
                }
            )
            args = argparse.Namespace(
                data_dir=data,
                output_dir=output,
                c=0.001,
                lr_random_state=42,
                max_iter=1000,
            )
            train_logistic_regression(args, metadata)
            predictions = pd.read_csv(output / "predictions.csv")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(predictions), len(metadata))
            self.assertEqual(
                list(predictions.columns),
                ["model", "subject_id", "hadm_id", "stay_id", "split", "label", "probability"],
            )
            self.assertTrue(predictions["probability"].between(0, 1).all())
            self.assertEqual(metrics["model"], "logistic_regression")


class LstmTrainingTests(unittest.TestCase):
    def test_one_epoch_cpu_training_writes_checkpoint_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "output"
            data.mkdir()
            rng = np.random.default_rng(9)
            np.save(data / "lstm_X.npy", rng.normal(size=(18, 48, 76)).astype(np.float32))
            metadata = pd.DataFrame(
                {
                    "subject_id": np.arange(18),
                    "hadm_id": np.arange(100, 118),
                    "stay_id": np.arange(200, 218),
                    "split": ["train"] * 10 + ["validation"] * 4 + ["test"] * 4,
                    "label": [0, 1] * 9,
                }
            )
            _, channel_indices, _ = build_header()
            args = argparse.Namespace(
                data_dir=data,
                output_dir=output,
                seed=49297,
                device="cpu",
                batch_size=4,
                dim=2,
                size_coef=1.0,
                dropout=0.0,
                learning_rate=1e-3,
                epochs=1,
                patience=0,
            )
            train_lstm(args, metadata, {"channel_indices": channel_indices})
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(len(predictions), 18)
            self.assertTrue((output / "best_model.pt").is_file())
            self.assertTrue((output / "training_log.csv").is_file())


if __name__ == "__main__":
    unittest.main()
