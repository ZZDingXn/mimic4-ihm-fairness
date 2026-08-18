from __future__ import annotations

import argparse
import csv
import json
import pickle
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import binary_metrics
from .model import ChannelWiseLSTM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MemmapDataset(Dataset):
    def __init__(self, path: Path, labels: np.ndarray, indices: np.ndarray) -> None:
        self.path = path
        self.matrix = np.load(path, mmap_mode="r")
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        features = np.asarray(self.matrix[index], dtype=np.float32).copy()
        return torch.from_numpy(features), torch.tensor(self.labels[index], dtype=torch.float32)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but the current PyTorch environment cannot use it")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_lstm(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    total_loss = 0.0
    for features, target in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(features)
        total_loss += float(criterion(logits, target).item())
        labels.append(target.cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(labels)
    p = np.concatenate(probabilities)
    return y, p, total_loss / len(y)


def write_predictions(
    output: Path,
    model_name: str,
    metadata: pd.DataFrame,
    probability_by_index: dict[int, float],
) -> None:
    frame = metadata[["subject_id", "hadm_id", "stay_id", "split", "label"]].copy()
    frame["probability"] = [probability_by_index[index] for index in frame.index]
    frame.insert(0, "model", model_name)
    frame.to_csv(output / "predictions.csv", index=False)


def environment_snapshot(device: str | None = None) -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "selected_device": device,
    }


def train_lstm(args: argparse.Namespace, metadata: pd.DataFrame, config: dict) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    labels = metadata["label"].to_numpy(np.float32)
    split = metadata["split"].to_numpy(str)
    generator = torch.Generator().manual_seed(args.seed)
    loaders: dict[str, DataLoader] = {}
    for name in ("train", "validation", "test"):
        indices = np.flatnonzero(split == name)
        loaders[name] = DataLoader(
            MemmapDataset(args.data_dir / "lstm_X.npy", labels, indices),
            batch_size=args.batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            pin_memory=device.type == "cuda",
            num_workers=0,
        )

    model = ChannelWiseLSTM(
        config["channel_indices"],
        dim=args.dim,
        size_coef=args.size_coef,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.csv"
    best_path = args.output_dir / "best_model.pt"
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "validation_loss", "validation_auroc", "validation_auprc"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            seen = 0
            for features, target in loaders["train"]:
                features = features.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(target)
                seen += len(target)
            validation_y, validation_p, validation_loss = predict_lstm(model, loaders["validation"], device)
            validation_metrics = binary_metrics(validation_y, validation_p)
            row = {
                "epoch": epoch,
                "train_loss": total_loss / seen,
                "validation_loss": validation_loss,
                "validation_auroc": validation_metrics["auroc"],
                "validation_auprc": validation_metrics["auprc"],
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if validation_loss < best_validation_loss - 1e-8:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "config": config,
                        "args": vars(args),
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1
                if args.patience and epochs_without_improvement >= args.patience:
                    break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    probability_by_index: dict[int, float] = {}
    metrics: dict[str, object] = {"model": "cw_lstm", "best_epoch": best_epoch}
    for name, loader in loaders.items():
        y, p, loss = predict_lstm(model, loader, device)
        indices = np.flatnonzero(split == name)
        probability_by_index.update({int(index): float(probability) for index, probability in zip(indices, p)})
        metrics[name] = {"loss": loss, **binary_metrics(y, p)}
    write_predictions(args.output_dir, "cw_lstm", metadata, probability_by_index)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment_snapshot(str(device)), indent=2), encoding="utf-8"
    )


def train_logistic_regression(args: argparse.Namespace, metadata: pd.DataFrame) -> None:
    matrix = np.load(args.data_dir / "lr_X_raw.npy", mmap_mode="r")
    split = metadata["split"].to_numpy(str)
    labels = metadata["label"].to_numpy(int)
    train_indices = np.flatnonzero(split == "train")
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    scaler = StandardScaler()
    train_x = imputer.fit_transform(np.asarray(matrix[train_indices]))
    train_x = scaler.fit_transform(train_x)
    model = LogisticRegression(
        C=args.c,
        l1_ratio=0.0,
        random_state=args.lr_random_state,
        max_iter=args.max_iter,
        solver="lbfgs",
    )
    model.fit(train_x, labels[train_indices])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability_by_index: dict[int, float] = {}
    metrics: dict[str, object] = {"model": "logistic_regression", "C": args.c}
    for name in ("train", "validation", "test"):
        indices = np.flatnonzero(split == name)
        features = scaler.transform(imputer.transform(np.asarray(matrix[indices])))
        probability = model.predict_proba(features)[:, 1]
        probability_by_index.update(
            {int(index): float(value) for index, value in zip(indices, probability)}
        )
        metrics[name] = binary_metrics(labels[indices], probability)
    write_predictions(args.output_dir, "logistic_regression", metadata, probability_by_index)
    with (args.output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"imputer": imputer, "scaler": scaler, "model": model}, handle)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment_snapshot("cpu"), indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the two prespecified IHM models")
    parser.add_argument("--model", choices=["cw_lstm", "logistic_regression"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=49297)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--size-coef", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--c", type=float, default=0.001)
    parser.add_argument("--lr-random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = pd.read_csv(args.data_dir / "metadata.csv")
    if not metadata.index.equals(pd.RangeIndex(len(metadata))):
        metadata = metadata.reset_index(drop=True)
    config = json.loads((args.data_dir / "config.json").read_text(encoding="utf-8"))
    if args.model == "cw_lstm":
        train_lstm(args, metadata, config)
    else:
        train_logistic_regression(args, metadata)


if __name__ == "__main__":
    main()
