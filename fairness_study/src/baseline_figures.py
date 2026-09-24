"""Create report-ready baseline figures from aggregate evaluation artefacts only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tem" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_language(language: str) -> None:
    plt.rcParams["font.size"] = 11
    if language == "zh":
        installed = {font.name for font in font_manager.fontManager.ttflist}
        preferred = next((name for name in ("SimSun", "SimHei") if name in installed), None)
        if preferred is None:
            raise RuntimeError("Chinese figures require an installed SimSun or SimHei font")
        plt.rcParams.update({"font.family": preferred, "axes.unicode_minus": False})
    else:
        plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def labels(language: str) -> dict[str, str]:
    if language == "zh":
        return {
            "lr": "逻辑回归", "cw_lstm": "通道 LSTM", "auroc": "AUROC", "auprc": "AUPRC",
            "recall": "死亡召回率（阈值=0.5）", "performance": "测试集基线表现",
            "flow": "队列筛选流程", "remaining": "保留 ICU stays", "stage": "筛选阶段",
            "subgroup": "基线亚组判别与平均校准偏差", "calibration": "平均预测 - 观察死亡率（百分点）",
            "discrimination": "AUROC", "insurance": "保险", "ethnicity": "族裔",
            "prior": "既往基线：stay 行级 bootstrap 95% CI", "source": "仅使用既有汇总评价产物",
        }
    return {
        "lr": "Logistic regression", "cw_lstm": "Channel-wise LSTM", "auroc": "AUROC", "auprc": "AUPRC",
        "recall": "Mortality recall (threshold = 0.5)", "performance": "Baseline test performance",
        "flow": "Cohort selection flow", "remaining": "Retained ICU stays", "stage": "Selection stage",
        "subgroup": "Baseline subgroup discrimination and calibration-in-the-large",
        "calibration": "Mean predicted - observed mortality (percentage points)", "discrimination": "AUROC",
        "insurance": "Insurance", "ethnicity": "Ethnicity",
        "prior": "Prior baseline: stay-level bootstrap 95% CI", "source": "Existing aggregate evaluation artefacts only",
    }


def model_label(model: str, text: dict[str, str]) -> str:
    return text[model]


def save_pair(figure: plt.Figure, output: Path) -> None:
    figure.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_cohort_flow(flow: pd.DataFrame, language: str, output: Path) -> None:
    set_language(language)
    text = labels(language)
    stage = flow["stage"].astype(str).tolist()
    retained = flow["retained"].to_numpy()
    excluded = flow["excluded"].to_numpy()
    translated = {
        "linked ICU stays": "关联 ICU stays", "one ICU stay per hospital admission": "排除含多次 ICU 住院的入院记录",
        "no ICU unit transfer": "首末 ICU 单元相同", "adult age at least 18": "年龄至少 18 岁",
        "ICU length of stay at least 48 hours": "ICU 住院至少 48 小时",
        "deterministic study subset": "确定性研究子集",
        "at least one usable observation in first 48 hours": "前 48 小时至少一项可用观测",
    }
    english = {
        "one ICU stay per hospital admission": "Admissions with exactly one ICU stay",
        "no ICU unit transfer": "Same first and last ICU care unit",
    }
    stage = [(translated if language == "zh" else english).get(value, value) for value in stage]
    figure, axis = plt.subplots(figsize=(9, 5.7))
    y = np.arange(len(stage))
    axis.barh(y, retained, color="#2474a6")
    axis.set_yticks(y, stage)
    axis.invert_yaxis()
    axis.set_xlabel(text["remaining"])
    axis.set_title(text["flow"])
    axis.grid(axis="x", alpha=0.22)
    for index, (value, removed) in enumerate(zip(retained, excluded)):
        annotation = f"{value:,}" if removed == 0 else f"{value:,}  (-{removed:,})"
        axis.text(value + retained.max() * 0.012, index, annotation, va="center", fontsize=9)
    axis.set_xlim(0, retained.max() * 1.22)
    figure.text(0.5, 0.01, text["source"], ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_pair(figure, output)


def plot_performance(overall: dict[str, object], language: str, output: Path) -> pd.DataFrame:
    set_language(language)
    text = labels(language)
    rows: list[dict[str, object]] = []
    metrics = [("auroc", text["auroc"]), ("auprc", text["auprc"]), ("recall_event", text["recall"])]
    models = ("lr", "cw_lstm")
    for model in models:
        test = overall[model]["test"]
        bootstrap = overall[model]["bootstrap"]
        for metric, display in metrics:
            interval = bootstrap.get(metric)
            rows.append({
                "model": model, "metric": metric, "metric_label": display, "value": test[metric],
                "lower_95": None if interval is None else interval["lower_95"],
                "upper_95": None if interval is None else interval["upper_95"],
                "n": test["n"], "events": test["events"],
                "ci_method": "prior stay-level bootstrap (10,000 iterations)" if interval else "not estimated in prior baseline",
            })
    summary = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(8.5, 5.3))
    x = np.arange(len(metrics))
    width = 0.32
    colors = {"lr": "#4f7cac", "cw_lstm": "#c75b39"}
    for offset, model in zip((-width / 2, width / 2), models):
        selected = summary.loc[summary["model"].eq(model)].reset_index(drop=True)
        values = selected["value"].to_numpy(float)
        lower = selected["lower_95"].to_numpy(float)
        upper = selected["upper_95"].to_numpy(float)
        yerr = np.vstack([values - np.nan_to_num(lower, nan=values), np.nan_to_num(upper, nan=values) - values])
        axis.bar(x + offset, values, width, label=model_label(model, text), color=colors[model], yerr=yerr, capsize=3)
        for position, value, upper_bound in zip(x + offset, values, upper):
            label_height = upper_bound if np.isfinite(upper_bound) else value
            axis.text(position, label_height + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    axis.set_xticks(x, [display for _, display in metrics])
    axis.set_ylabel("Value" if language == "en" else "数值")
    axis.set_ylim(0, 1.03)
    axis.set_title(text["performance"])
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    figure.text(0.5, 0.01, text["prior"] + "; " + text["source"], ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save_pair(figure, output)
    return summary


def plot_subgroups(metrics: pd.DataFrame, calibration: pd.DataFrame, language: str, output: Path) -> None:
    set_language(language)
    text = labels(language)
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 7), constrained_layout=True)
    colors = {"lr": "#4f7cac", "cw_lstm": "#c75b39"}
    for column, attribute in enumerate(("insurance", "ethnicity")):
        groups = metrics.loc[metrics["attribute"].eq(attribute), "group"].drop_duplicates().tolist()
        for row, measure in enumerate(("auroc", "calibration")):
            axis = axes[row, column]
            x = np.arange(len(groups))
            for offset, model in zip((-0.12, 0.12), ("lr", "cw_lstm")):
                if measure == "auroc":
                    selected = metrics.loc[(metrics["attribute"].eq(attribute)) & (metrics["model"].eq(model))].set_index("group").loc[groups]
                    value = selected["auroc"].to_numpy(float)
                    lower = selected["auroc_lower_95"].to_numpy(float)
                    upper = selected["auroc_upper_95"].to_numpy(float)
                    axis.errorbar(x + offset, value, yerr=np.vstack([value - lower, upper - value]), fmt="o", color=colors[model], capsize=3, label=model_label(model, text))
                    axis.set_ylabel(text["discrimination"])
                    axis.set_ylim(0.65, 1.01)
                else:
                    selected = calibration.loc[(calibration["attribute"].eq(attribute)) & (calibration["model"].eq(model))].set_index("group").loc[groups]
                    value = 100 * selected["prediction_minus_observed"].to_numpy(float)
                    lower = 100 * selected["difference_lower_95"].to_numpy(float)
                    upper = 100 * selected["difference_upper_95"].to_numpy(float)
                    axis.errorbar(x + offset, value, yerr=np.vstack([value - lower, upper - value]), fmt="o", color=colors[model], capsize=3, label=model_label(model, text))
                    axis.axhline(0, color="black", linewidth=0.8)
                    axis.set_ylabel(text["calibration"])
                axis.set_xticks(x, groups, rotation=25, ha="right")
                axis.grid(axis="y", alpha=0.22)
                axis.set_title(text[attribute])
            if row == 0:
                axis.legend(frameon=False, loc="lower left")
    figure.suptitle(text["subgroup"], fontsize=14)
    figure.text(0.5, 0.005, text["prior"] + "; " + text["source"], ha="center", fontsize=8)
    save_pair(figure, output)


def cohort_demographics(metadata: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the frozen metadata into the report's all/train/validation/test table."""
    attributes = ("gender_group", "age_group", "ethnicity_group", "insurance_group")
    rows: list[pd.DataFrame] = []
    for attribute in attributes:
        for split in ("all", "train", "validation", "test"):
            selected = metadata if split == "all" else metadata.loc[metadata["split"].eq(split)]
            grouped = selected.groupby(attribute, sort=True).agg(
                stays=("stay_id", "size"),
                patients=("subject_id", "nunique"),
                deaths=("label", "sum"),
                event_rate=("label", "mean"),
            ).reset_index().rename(columns={attribute: "group"})
            grouped.insert(0, "attribute", attribute)
            grouped.insert(2, "split", split)
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def run(
    evaluation_dir: Path,
    cohort_flow_path: Path,
    metadata_path: Path,
    results_dir: Path,
    figures_dir: Path,
) -> None:
    overall = json.loads((evaluation_dir / "overall_metrics.json").read_text(encoding="utf-8"))
    subgroup_metrics = pd.read_csv(evaluation_dir / "subgroup_metrics.csv")
    calibration = pd.read_csv(evaluation_dir / "calibration_in_large.csv")
    flow = pd.read_csv(cohort_flow_path)
    cohort_summary = pd.read_csv(evaluation_dir / "cohort_summary.csv")
    metadata = pd.read_csv(metadata_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    counts = cohort_summary.melt(var_name="measure", value_name="value")
    counts.to_csv(results_dir / "cohort_counts.csv", index=False)
    cohort_demographics(metadata).to_csv(results_dir / "cohort_demographics.csv", index=False)
    for language in ("en", "zh"):
        suffix = f"_{language}"
        plot_cohort_flow(flow, language, figures_dir / f"cohort_flow{suffix}")
        summary = plot_performance(overall, language, figures_dir / f"baseline_performance{suffix}")
        plot_subgroups(subgroup_metrics, calibration, language, figures_dir / f"baseline_subgroups{suffix}")
        if language == "en":
            summary.to_csv(results_dir / "baseline_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--cohort-flow", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures")
    args = parser.parse_args()
    run(args.evaluation_dir, args.cohort_flow, args.metadata, args.results_dir, args.figures_dir)


if __name__ == "__main__":
    main()
