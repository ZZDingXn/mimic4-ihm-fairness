"""Train prespecified IHM models with train-only group-by-label reweighting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimic4_ihm.train import environment_snapshot, train_logistic_regression, train_lstm


ATTRIBUTES = ("insurance_group", "ethnicity_group")
MODELS = ("logistic_regression", "cw_lstm")


def compute_training_weights(
    metadata: pd.DataFrame, attribute: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return mean-one weights, calculated exclusively from training rows."""
    if attribute not in ATTRIBUTES:
        raise ValueError(f"attribute must be one of {ATTRIBUTES}")
    required = {"split", "label", attribute}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")

    train = metadata.loc[metadata["split"].eq("train"), [attribute, "label"]].copy()
    if train.empty:
        raise ValueError("metadata contains no training rows")
    train[attribute] = train[attribute].astype(str)
    train["label"] = train["label"].astype(int)
    counts = train.groupby([attribute, "label"], sort=True).size().rename("n").reset_index()
    group_label_counts = (
        counts.pivot(index=attribute, columns="label", values="n")
        .reindex(columns=[0, 1], fill_value=0)
        .fillna(0)
    )
    incomplete_groups = group_label_counts.index[(group_label_counts == 0).any(axis=1)].tolist()
    if incomplete_groups:
        raise ValueError(
            f"training rows for {attribute} need both labels 0 and 1 in every group; "
            f"missing label cells for: {incomplete_groups}"
        )
    group_probability = train[attribute].value_counts(normalize=True)
    label_probability = train["label"].value_counts(normalize=True)
    counts["group_probability"] = counts[attribute].map(group_probability)
    counts["label_probability"] = counts["label"].map(label_probability)
    counts["joint_probability"] = counts["n"] / len(train)
    counts["weight"] = (
        counts["group_probability"] * counts["label_probability"] / counts["joint_probability"]
    )
    lookup = counts.set_index([attribute, "label"])["weight"]
    weights = np.ones(len(metadata), dtype=np.float64)
    train_indices = metadata.index[metadata["split"].eq("train")]
    weights[train_indices] = [
        lookup.loc[(str(metadata.at[index, attribute]), int(metadata.at[index, "label"]))]
        for index in train_indices
    ]
    normalization = weights[train_indices].mean()
    weights[train_indices] /= normalization
    # The formula has expected weight one; this final normalization removes only
    # floating-point drift and is deliberately retained in the audit.
    counts["normalized_weight"] = counts["weight"] / normalization
    return weights, counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribute", choices=ATTRIBUTES, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
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


def run(args: argparse.Namespace) -> None:
    metadata = pd.read_csv(args.data_dir / "metadata.csv").reset_index(drop=True)
    weights, audit = compute_training_weights(metadata, args.attribute)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_frame = metadata[["subject_id", "hadm_id", "stay_id", "split", "label", args.attribute]].copy()
    weights_frame["sample_weight"] = weights
    weights_frame.to_csv(args.output_dir / "training_weights.csv", index=False)
    audit.to_csv(args.output_dir / "weight_audit.csv", index=False)
    config = {
        "model": args.model,
        "attribute": args.attribute,
        "weight_formula": "P_train(attribute) * P_train(label) / P_train(attribute, label)",
        "weight_normalization": "training-row mean equals 1",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "reweight_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.model == "cw_lstm":
        model_config = json.loads((args.data_dir / "config.json").read_text(encoding="utf-8"))
        train_lstm(args, metadata, model_config, sample_weights=weights)
    else:
        train_logistic_regression(args, metadata, sample_weights=weights)
    (args.output_dir / "reweight_environment.json").write_text(
        json.dumps(environment_snapshot(None if args.model == "logistic_regression" else args.device), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
