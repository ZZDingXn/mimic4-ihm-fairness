"""Publication figures with separate English and Chinese variants."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ['baseline_05', 'global_threshold', 'reweight', 'opportunity', 'equalized_odds']
METHOD_LABELS = {
    'en': ['Fixed 0.5', 'Global threshold', 'Reweighing', 'Equal opportunity', 'Equalized odds'],
    'zh': ['固定0.5', '统一阈值', '重加权', '机会均等', '均等化赔率'],
}

def _panel_label(model: str, target: str, language: str) -> str:
    return ('LSTM' if model == 'cw_lstm' else 'LR') + ' / ' + ({'insurance_group': '保险', 'ethnicity_group': '族裔'}[target] if language == 'zh' else target.replace('_group', '').title())


def _setup(tem_dir: Path) -> tuple[object, object]:
    tem_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(tem_dir / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


def _save(figure: object, stem: str, result_dir: Path, report_dir: Path) -> None:
    for directory in (result_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight")
        figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")


def _panel_pairs(metrics: pd.DataFrame) -> list[tuple[str, str]]:
    return list(metrics[["model", "target_attribute"]].drop_duplicates().itertuples(index=False, name=None))


def group_rates(metrics: pd.DataFrame, confidence: pd.DataFrame, metric: str, language: str, result_dir: Path, report_dir: Path, plt: object) -> None:
    frame = metrics[(metrics.attribute.ne("overall")) & metrics.status.eq("available")].copy()
    panels = _panel_pairs(frame)
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 7))
    figure.subplots_adjust(top=.88,bottom=.11,left=.09,right=.98,wspace=.30,hspace=.42)
    title = {"en": {"tpr": "True-positive rate", "fpr": "False-positive rate"},
             "zh": {"tpr": "真正率", "fpr": "假正率"}}[language][metric]
    methods = METHODS
    for axis, pair in zip(axes.flat, panels):
        model, target = pair
        part = frame[(frame.model.eq(model)) & (frame.target_attribute.eq(target)) & frame.attribute.eq(target)]
        pivot = part.pivot_table(index="group", columns="method", values=metric).reindex(columns=methods)
        pivot.plot(kind="bar", ax=axis, width=0.82, legend=False)
        intervals=confidence[(confidence.model==model)&(confidence.target_attribute==target)&(confidence.attribute==target)&(confidence.metric==metric)]
        for method,container in zip(methods,axis.containers):
            for group,bar in zip(pivot.index,container):
                row=intervals[(intervals.method==method)&(intervals.group==group)].iloc[0]
                x=bar.get_x()+bar.get_width()/2
                axis.vlines(x,row.lower_95,row.upper_95,color='0.2',linewidth=.65)
                axis.hlines([row.lower_95,row.upper_95],x-.018,x+.018,color='0.2',linewidth=.65)
        axis.set(title=_panel_label(model, target, language), xlabel="", ylabel=title, ylim=(0, 1))
        axis.tick_params(axis="x", rotation=30, labelsize=10); axis.grid(axis="y", alpha=0.25)
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, METHOD_LABELS[language], loc='upper center', bbox_to_anchor=(.5,1.0), ncol=3, fontsize=10)
    _save(figure, f"group_{metric}_{language}", result_dir, report_dir)
    plt.close(figure)


def tradeoff(gaps: pd.DataFrame, metrics: pd.DataFrame, language: str, result_dir: Path, report_dir: Path, plt: object) -> None:
    overall = metrics[metrics.attribute.eq("overall")][["model", "target_attribute", "method", "balanced_accuracy"]]
    frame = gaps[gaps.attribute.eq(gaps.target_attribute)].merge(overall, on=["model", "target_attribute", "method"])
    panels = _panel_pairs(frame)
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 7), constrained_layout=True)
    x_label = "TPR gap (lower is better)" if language == "en" else "TPR 差距（越低越好）"
    y_label = "Balanced accuracy" if language == "en" else "平衡准确率"
    for axis, pair in zip(axes.flat, panels):
        model, target = pair
        part = frame[(frame.model.eq(model)) & (frame.target_attribute.eq(target))]
        for i, method in enumerate(METHODS):
            row = part[part.method.eq(method)].iloc[0]
            axis.scatter([row.tpr_gap], [row.balanced_accuracy], s=60, color=f'C{i}', marker=['o','s','^','D','P'][i], label=METHOD_LABELS[language][i])
        axis.set(title=_panel_label(model, target, language), xlabel=x_label, ylabel=y_label)
        axis.margins(x=.12,y=.12)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=10,loc='best')
    _save(figure, f"fairness_utility_tradeoff_{language}", result_dir, report_dir)
    plt.close(figure)


def calibration(calibration_bins: pd.DataFrame, language: str, result_dir: Path, report_dir: Path, plt: object) -> None:
    overall = calibration_bins[(calibration_bins.attribute.eq("overall")) & (calibration_bins.group.eq("Overall"))].copy()
    baseline_all = overall[overall.method.eq("baseline_05")]
    selected_baseline = baseline_all[["model", "target_attribute"]].drop_duplicates().sort_values("target_attribute").drop_duplicates("model")
    base = baseline_all.merge(selected_baseline, on=["model", "target_attribute"], how="inner")
    reweighted = overall[overall.method.eq("reweight")]
    frame = pd.concat((base, reweighted), ignore_index=True)
    figure, axes = plt.subplots(1,2,figsize=(8.5,4.5), constrained_layout=True)
    for axis,model in zip(axes,['lr','cw_lstm']):
        axis.plot((0, 1), (0, 1), "--", color="0.45", label="Ideal" if language == "en" else "理想校准")
        for i,(keys,part) in enumerate(frame[frame.model.eq(model)].groupby(['target_attribute','method'],sort=False)):
            target,method=keys
            label=('原模型' if language=='zh' else 'Baseline') if method=='baseline_05' else ({'insurance_group':'保险重加权','ethnicity_group':'族裔重加权'}[target] if language=='zh' else target.replace('_group','').title()+' reweighed')
            bins=part[part.kind.eq('bin')]
            axis.plot(bins.mean_risk,bins.observed_rate,'o',color=f'C{i}',markersize=3,label=label)
            smooth=part[part.kind.eq('lowess')]
            axis.plot(smooth.mean_risk,smooth.observed_rate,color=f'C{i}',linewidth=1.3)
        axis.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted mortality risk' if language=='en' else '预测死亡风险',ylabel='Observed mortality' if language=='en' else '观察死亡率',title='LSTM' if model=='cw_lstm' else 'LR')
        axis.grid(alpha=.25);axis.legend(fontsize=10,loc='upper left')
    _save(figure, f"overall_calibration_{language}", result_dir, report_dir)
    plt.close(figure)


def write_publication_plots(
    metrics: pd.DataFrame, gaps: pd.DataFrame, calibration_bins: pd.DataFrame, confidence: pd.DataFrame,
    result_dir: Path, report_dir: Path, tem_dir: Path,
) -> None:
    matplotlib, plt = _setup(tem_dir)
    matplotlib.rcParams["font.size"] = 11
    for language in ("en", "zh"):
        if language == "zh":
            matplotlib.rcParams["font.sans-serif"] = ["SimHei", "SimSun", "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
        else:
            matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        group_rates(metrics, confidence, "tpr", language, result_dir, report_dir, plt)
        group_rates(metrics, confidence, "fpr", language, result_dir, report_dir, plt)
        tradeoff(gaps, metrics, language, result_dir, report_dir, plt)
        calibration(calibration_bins, language, result_dir, report_dir, plt)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Redraw figures from completed aggregate results")
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parents[1])
    study = parser.parse_args().study_dir
    results = study / "results"
    write_publication_plots(
        pd.read_csv(results / "policy_metrics.csv"),
        pd.read_csv(results / "fairness_gaps.csv"),
        pd.read_csv(results / "calibration_bins.csv"),
        pd.read_csv(results / "bootstrap_confidence_intervals.csv"),
        results / "plots", study / "figures", study / "tem",
    )
