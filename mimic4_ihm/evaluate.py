from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "mimic4_ihm_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels
from sklearn import metrics as sk_metrics
from statsmodels.nonparametric.smoothers_lowess import lowess as statsmodels_lowess

from .metrics import binary_metrics


GROUP_COLUMNS = {
    "gender": "gender_group",
    "age": "age_group",
    "ethnicity": "ethnicity_group",
    "insurance": "insurance_group",
}
GROUP_ORDER = {
    "gender": ["Female", "Male", "Other"],
    "age": ["18-29", "30-49", "50-69", "70-89", "90+"],
    "ethnicity": ["Asian", "Black", "Hispanic", "White", "Other"],
    "insurance": ["Medicare", "Medicaid", "Private", "Other"],
}


def parse_prediction_specs(specifications: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Prediction specification must be name=path, got {specification!r}")
        name, raw_path = specification.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Prediction model name is empty or duplicated: {name!r}")
        result[name] = Path(raw_path)
    if not result:
        raise ValueError("At least one prediction file is required")
    return result


def load_prediction(metadata: pd.DataFrame, path: Path, requested_name: str) -> pd.DataFrame:
    prediction = pd.read_csv(path)
    required = {"stay_id", "label", "split", "probability"}
    missing = required.difference(prediction.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if prediction["stay_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated stay_id values")
    expected = metadata[["stay_id", "label", "split"]].copy()
    merged = expected.merge(
        prediction[["stay_id", "label", "split", "probability"]],
        on="stay_id",
        how="outer",
        suffixes=("_metadata", "_prediction"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{path} does not contain exactly the metadata stays")
    if not merged["label_metadata"].eq(merged["label_prediction"]).all():
        raise ValueError(f"{path} labels disagree with metadata")
    if not merged["split_metadata"].eq(merged["split_prediction"]).all():
        raise ValueError(f"{path} splits disagree with metadata")
    if not merged["probability"].between(0, 1).all():
        raise ValueError(f"{path} contains probabilities outside [0, 1]")
    return metadata.merge(
        prediction[["stay_id", "probability"]], on="stay_id", how="left", validate="one_to_one"
    ).assign(model=requested_name)


def metric_triplet(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    if np.unique(y).size < 2:
        return {"auroc": None, "auprc": None, "minpse": None}
    precision, recall, _ = sk_metrics.precision_recall_curve(y, p)
    return {
        "auroc": float(sk_metrics.roc_auc_score(y, p)),
        "auprc": float(sk_metrics.auc(recall, precision)),
        "minpse": float(max(min(x, z) for x, z in zip(precision, recall))),
    }


def bootstrap_metric_intervals(
    y: np.ndarray,
    p: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    point = metric_triplet(y, p)
    samples: dict[str, list[float]] = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        indices = rng.integers(0, len(y), size=len(y))
        result = metric_triplet(y[indices], p[indices])
        for name, value in result.items():
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
    summary: dict[str, dict[str, float | int | None]] = {}
    for name, value in point.items():
        values = np.asarray(samples[name], dtype=float)
        summary[name] = {
            "value": value,
            "mean": float(values.mean()) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "lower_95": float(np.percentile(values, 2.5)) if values.size else None,
            "upper_95": float(np.percentile(values, 97.5)) if values.size else None,
            "valid_iterations": int(values.size),
            "requested_iterations": int(iterations),
        }
    return summary


def paired_bootstrap_difference(
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    iterations: int,
    seed: int,
) -> list[dict[str, float | int | None]]:
    rng = np.random.default_rng(seed)
    point_first = metric_triplet(y, first)
    point_second = metric_triplet(y, second)
    samples: dict[str, list[float]] = {"auroc": [], "auprc": [], "minpse": []}
    for _ in range(iterations):
        indices = rng.integers(0, len(y), size=len(y))
        first_metric = metric_triplet(y[indices], first[indices])
        second_metric = metric_triplet(y[indices], second[indices])
        for name in samples:
            if first_metric[name] is not None and second_metric[name] is not None:
                samples[name].append(float(first_metric[name] - second_metric[name]))
    rows = []
    for name, values in samples.items():
        array = np.asarray(values, dtype=float)
        point = None
        if point_first[name] is not None and point_second[name] is not None:
            point = float(point_first[name] - point_second[name])
        rows.append(
            {
                "metric": name,
                "difference": point,
                "lower_95": float(np.percentile(array, 2.5)) if array.size else None,
                "upper_95": float(np.percentile(array, 97.5)) if array.size else None,
                "valid_iterations": int(array.size),
            }
        )
    return rows


def wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = events / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def exponential_calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    if len(y) == 0:
        return pd.DataFrame()
    order = np.argsort(p, kind="stable")
    rank_fraction = (np.arange(len(y)) + 1) / len(y)
    boundaries = np.power(np.linspace(0.0, 1.0, bins + 1), 1.0 / 5.0)
    assignments = np.searchsorted(boundaries[1:], rank_fraction, side="left")
    rows = []
    for group in range(bins):
        selected = order[assignments == group]
        if len(selected) == 0:
            continue
        events = int(y[selected].sum())
        lower, upper = wilson_interval(events, len(selected))
        rows.append(
            {
                "risk_group": group + 1,
                "n": int(len(selected)),
                "events": events,
                "mean_prediction": float(p[selected].mean()),
                "observed_rate": float(y[selected].mean()),
                "wilson_lower": lower,
                "wilson_upper": upper,
                "min_prediction": float(p[selected].min()),
                "max_prediction": float(p[selected].max()),
            }
        )
    return pd.DataFrame(rows)


def calibration_in_large(
    y: np.ndarray,
    p: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float | int | None]:
    observed = float(y.mean())
    predicted = float(p.mean())
    difference = predicted - observed
    ratio = predicted / observed if observed else None
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y), size=len(y))
        differences.append(float(p[indices].mean() - y[indices].mean()))
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "observed_rate": observed,
        "mean_prediction": predicted,
        "prediction_minus_observed": difference,
        "prediction_to_observed_ratio": ratio,
        "difference_lower_95": float(np.percentile(differences, 2.5)),
        "difference_upper_95": float(np.percentile(differences, 97.5)),
    }


def calibration_plot(
    output_stem: Path,
    model: str,
    bins: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1.5, label="Ideal")
    if not bins.empty:
        lower = np.maximum(
            bins["observed_rate"].to_numpy(float) - bins["wilson_lower"].to_numpy(float), 0.0
        )
        upper = np.maximum(
            bins["wilson_upper"].to_numpy(float) - bins["observed_rate"].to_numpy(float), 0.0
        )
        axis.errorbar(
            bins["mean_prediction"].to_numpy(float),
            bins["observed_rate"].to_numpy(float),
            yerr=np.vstack([lower, upper]),
            fmt="o",
            color="#1f77b4",
            capsize=3,
            label="Exponential-quantile groups (Wilson 95% CI)",
        )
    if len(y) >= 2 and np.unique(p).size >= 2:
        smooth = statsmodels_lowess(y, p, frac=0.5, it=0, return_sorted=True)
        axis.plot(
            smooth[:, 0],
            np.clip(smooth[:, 1], 0, 1),
            color="#d62728",
            linewidth=2.2,
            label="LOWESS (span 0.5)",
        )
    for outcome, color, offset, label in (
        (0, "#4c78a8", -0.035, "Non-event predictions"),
        (1, "#e45756", -0.018, "Event predictions"),
    ):
        values = p[y == outcome]
        if len(values) > 500:
            values = values[np.linspace(0, len(values) - 1, 500).astype(int)]
        axis.plot(values, np.full(len(values), offset), "|", color=color, alpha=0.55, label=label)
    axis.set(xlim=(0, 1), ylim=(-0.05, 1), xlabel="Predicted in-hospital mortality risk", ylabel="Observed mortality rate")
    axis.set_title(f"{model} calibration")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", fontsize=8)
    figure.savefig(output_stem.with_suffix(".svg"))
    figure.savefig(output_stem.with_suffix(".png"), dpi=180)
    plt.close(figure)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "group"


def descriptive_tables(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            {
                "stays": len(metadata),
                "subjects": metadata["subject_id"].nunique(),
                "events": int(metadata["label"].sum()),
                "event_rate": float(metadata["label"].mean()),
            }
        ]
    )
    rows = []
    for attribute, column in GROUP_COLUMNS.items():
        for group in GROUP_ORDER[attribute]:
            selected = metadata[metadata[column].eq(group)]
            if selected.empty:
                continue
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "stays": len(selected),
                    "subjects": selected["subject_id"].nunique(),
                    "events": int(selected["label"].sum()),
                    "event_rate": float(selected["label"].mean()),
                }
            )
    return overall, pd.DataFrame(rows)


def evaluate(args: argparse.Namespace) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    required_metadata = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "split",
        "label",
        *GROUP_COLUMNS.values(),
    }
    missing = required_metadata.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    specifications = parse_prediction_specs(args.predictions)
    frames = {name: load_prediction(metadata, path, name) for name, path in specifications.items()}
    overall_cohort, subgroup_cohort = descriptive_tables(metadata)
    overall_cohort.to_csv(output / "cohort_summary.csv", index=False)
    subgroup_cohort.to_csv(output / "subgroup_counts.csv", index=False)

    metrics_json: dict[str, object] = {}
    subgroup_rows = []
    calibration_rows = []
    calibration_bin_frames = []
    test_predictions: dict[str, pd.DataFrame] = {}
    for model_index, (model, frame) in enumerate(frames.items()):
        test = frame[frame["split"].eq("test")].sort_values("stay_id").reset_index(drop=True)
        test_predictions[model] = test
        y = test["label"].to_numpy(int)
        p = test["probability"].to_numpy(float)
        metrics_json[model] = {
            "test": binary_metrics(y, p),
            "bootstrap": bootstrap_metric_intervals(
                y, p, args.bootstrap_iterations, args.seed + model_index
            ),
        }
        overall_calibration = calibration_in_large(
            y, p, args.bootstrap_iterations, args.seed + 10_000 + model_index
        )
        calibration_rows.append({"model": model, "attribute": "overall", "group": "Overall", **overall_calibration})
        bins = exponential_calibration_bins(y, p)
        bins.insert(0, "group", "Overall")
        bins.insert(0, "attribute", "overall")
        bins.insert(0, "model", model)
        calibration_bin_frames.append(bins)
        calibration_plot(output / f"calibration_{model}", model, bins, y, p)

        for attribute, column in GROUP_COLUMNS.items():
            for group in GROUP_ORDER[attribute]:
                selected = test[test[column].eq(group)]
                if selected.empty:
                    continue
                subgroup_y = selected["label"].to_numpy(int)
                subgroup_p = selected["probability"].to_numpy(float)
                result = binary_metrics(subgroup_y, subgroup_p)
                intervals = bootstrap_metric_intervals(
                    subgroup_y,
                    subgroup_p,
                    args.subgroup_bootstrap_iterations,
                    args.seed + model_index * 1000 + len(subgroup_rows),
                )
                subgroup_rows.append(
                    {
                        "model": model,
                        "attribute": attribute,
                        "group": group,
                        **result,
                        "auroc_lower_95": intervals["auroc"]["lower_95"],
                        "auroc_upper_95": intervals["auroc"]["upper_95"],
                        "auprc_lower_95": intervals["auprc"]["lower_95"],
                        "auprc_upper_95": intervals["auprc"]["upper_95"],
                    }
                )
                calibration_rows.append(
                    {
                        "model": model,
                        "attribute": attribute,
                        "group": group,
                        **calibration_in_large(
                            subgroup_y,
                            subgroup_p,
                            args.subgroup_bootstrap_iterations,
                            args.seed + 20_000 + len(calibration_rows),
                        ),
                    }
                )
                subgroup_bins = exponential_calibration_bins(subgroup_y, subgroup_p)
                subgroup_bins.insert(0, "group", group)
                subgroup_bins.insert(0, "attribute", attribute)
                subgroup_bins.insert(0, "model", model)
                calibration_bin_frames.append(subgroup_bins)
                calibration_plot(
                    output / f"calibration_{safe_filename(model)}_{attribute}_{safe_filename(group)}",
                    f"{model}: {attribute}={group}",
                    subgroup_bins,
                    subgroup_y,
                    subgroup_p,
                )

    comparison_rows = []
    model_names = list(test_predictions)
    for first_index in range(len(model_names)):
        for second_index in range(first_index + 1, len(model_names)):
            first_name = model_names[first_index]
            second_name = model_names[second_index]
            first = test_predictions[first_name]
            second = test_predictions[second_name]
            if not first["stay_id"].equals(second["stay_id"]) or not first["label"].equals(second["label"]):
                raise AssertionError("Paired model comparison requires identical sorted test stays and labels")
            rows = paired_bootstrap_difference(
                first["label"].to_numpy(int),
                first["probability"].to_numpy(float),
                second["probability"].to_numpy(float),
                args.bootstrap_iterations,
                args.seed + 30_000,
            )
            comparison_rows.extend(
                [
                    {
                        "first_model": first_name,
                        "second_model": second_name,
                        "attribute": "overall",
                        "group": "Overall",
                        **row,
                    }
                    for row in rows
                ]
            )
            for attribute, column in GROUP_COLUMNS.items():
                for group in GROUP_ORDER[attribute]:
                    mask = first[column].eq(group).to_numpy()
                    if not mask.any():
                        continue
                    subgroup_rows_for_pair = paired_bootstrap_difference(
                        first.loc[mask, "label"].to_numpy(int),
                        first.loc[mask, "probability"].to_numpy(float),
                        second.loc[mask, "probability"].to_numpy(float),
                        args.subgroup_bootstrap_iterations,
                        args.seed + 40_000 + len(comparison_rows),
                    )
                    comparison_rows.extend(
                        [
                            {
                                "first_model": first_name,
                                "second_model": second_name,
                                "attribute": attribute,
                                "group": group,
                                **row,
                            }
                            for row in subgroup_rows_for_pair
                        ]
                    )

    (output / "overall_metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    pd.DataFrame(subgroup_rows).to_csv(output / "subgroup_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_in_large.csv", index=False)
    pd.concat(calibration_bin_frames, ignore_index=True).to_csv(output / "calibration_bins.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "model_comparison.csv", index=False)
    coverage_source = args.metadata.parent / "coverage.csv"
    if coverage_source.is_file():
        pd.read_csv(coverage_source).to_csv(output / "missingness_report.csv", index=False)
    manifest = {
        "metadata": str(args.metadata.resolve()),
        "predictions": {name: str(path.resolve()) for name, path in specifications.items()},
        "bootstrap_iterations": args.bootstrap_iterations,
        "subgroup_bootstrap_iterations": args.subgroup_bootstrap_iterations,
        "seed": args.seed,
        "threshold": 0.5,
        "calibration_quantiles": "q^(1/5)",
        "lowess_span": 0.5,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib_config_dir": str(MPL_CONFIG_DIR),
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate IHM performance, calibration, and subgroup fairness")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", nargs="+", required=True, help="One or more name=predictions.csv specifications")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--subgroup-bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=49297)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
