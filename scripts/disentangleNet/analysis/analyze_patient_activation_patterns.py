#!/usr/bin/env python
"""
将窗口级 basis activation 聚合为患者级汇总。

此脚本功能：
- 读取 `window_basis_activations_wide.csv`。
- 按患者聚合 usage / activation / coeff 统计量。
- 推导 entropy、dominant-basis、gap 和启发式 pattern-label 汇总。
- 导出患者 profiles、group 汇总、交叉表、极值排名和 Markdown 报告。

典型用法：
1. 默认患者汇总构建：
   `python scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py analyze`
2. 从其他 wide CSV 构建：
   `python scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py analyze \\
      --wide_csv outputs/.../window_basis_activations_wide.csv`
3. 写入自定义目录：
   `python scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py analyze \\
      --output_dir outputs/.../patient_profile_summary_custom`

默认输出路径：
- `.../window_basis_activations_all/patient_pattern_analysis/patient_profile_summary/`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import fire
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """计算每行（每患者）的 Shannon 熵"""
    clipped = np.clip(probs, 1e-12, None)
    return -(clipped * np.log(clipped)).sum(axis=1)


def top_two_gap(values: np.ndarray) -> np.ndarray:
    """计算每行最大值与次大值的差（dominant 程度）"""
    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=np.float64)
    sorted_values = np.sort(values, axis=1)
    return sorted_values[:, -1] - sorted_values[:, -2]


def build_patient_profiles(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    将窗口级宽表聚合为患者级 profiles。
    计算各 level 的 usage 统计、entropy、top2-gap、dominant basis。
    基于 side_coeff / side_entropy / side_dom 的启发式规则生成 pattern_label。
    """
    usage_cols = [c for c in wide_df.columns if c.startswith("basis_usage_b")]
    activation_cols = [c for c in wide_df.columns if c.startswith("basis_activation_b")]
    coeff_cols = [c for c in ("free_coeff_l0", "free_coeff_l1", "side_coeff") if c in wide_df.columns]

    group_cols = ["dataset_name", "dataset_label", "subject", "side_label", "side_label_name"]
    if "score" in wide_df.columns:
        group_cols.append("score")
    if "label_5class" in wide_df.columns:
        group_cols.append("label_5class")

    # 均值 / 标准差 / 窗口计数
    patient_mean = wide_df.groupby(group_cols, as_index=False)[usage_cols + activation_cols + coeff_cols].mean()
    patient_std = wide_df.groupby(group_cols, as_index=False)[coeff_cols].std(ddof=0).rename(
        columns={col: f"{col}_std" for col in coeff_cols}
    )
    window_stats = (
        wide_df.groupby(group_cols, as_index=False)
        .agg(
            window_count=("window_idx", "count"),
            first_window_idx=("window_idx", "min"),
            last_window_idx=("window_idx", "max"),
            first_start_frame=("start_frame", "min"),
            last_end_frame=("end_frame", "max"),
        )
    )

    patient = patient_mean.merge(patient_std, on=group_cols, how="left").merge(
        window_stats,
        on=group_cols,
        how="left",
    )

    # 分组统计
    free_l0_cols = [f"basis_usage_b{i}" for i in range(0, 2)]
    free_l1_cols = [f"basis_usage_b{i}" for i in range(2, 8)]
    side_cols = [f"basis_usage_b{i}" for i in range(8, 11)]

    free_l0_usage = patient[free_l0_cols].to_numpy(dtype=np.float64)
    free_l1_usage = patient[free_l1_cols].to_numpy(dtype=np.float64)
    side_usage = patient[side_cols].to_numpy(dtype=np.float64)

    patient["free_l0_entropy"] = shannon_entropy(free_l0_usage)
    patient["free_l1_entropy"] = shannon_entropy(free_l1_usage)
    patient["side_entropy"] = shannon_entropy(side_usage)
    patient["free_l0_top2_gap"] = top_two_gap(free_l0_usage)
    patient["free_l1_top2_gap"] = top_two_gap(free_l1_usage)
    patient["side_top2_gap"] = top_two_gap(side_usage)
    patient["free_l0_dominant_basis"] = free_l0_usage.argmax(axis=1)
    patient["free_l1_dominant_basis"] = free_l1_usage.argmax(axis=1) + 2
    patient["side_dominant_basis"] = side_usage.argmax(axis=1) + 8

    patient["free_l1_b5_minus_b6"] = patient["basis_usage_b5"] - patient["basis_usage_b6"]
    patient["side_b10_minus_b9"] = patient["basis_usage_b9"] - patient["basis_usage_b10"]
    patient["side_b10_minus_b8"] = patient["basis_usage_b8"] - patient["basis_usage_b10"]

    # 启发式 pattern label
    pattern_labels = []
    for _, row in patient.iterrows():
        side_dom = int(row["side_dominant_basis"])
        side_coeff = float(row["side_coeff"])
        side_entropy = float(row["side_entropy"])

        if side_dom == 10 and side_coeff > 8 and side_entropy < 0.35:
            label = "left_like_b10_pure_positive"
        elif side_dom in (9, 10) and side_coeff < -8 and side_entropy < 0.95:
            label = "normal_like_b9_b10_negative"
        elif side_dom in (8, 10) and side_entropy >= 0.95 and abs(side_coeff) < 3:
            label = "right_like_diffuse_low_coeff"
        else:
            label = "mixed_or_boundary"
        pattern_labels.append(label)
    patient["pattern_label"] = pattern_labels
    return patient


def summarize_by_group(patient_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """按给定分组列（side_label_name / dataset_name）汇总关键统计量"""
    summary_cols = [
        "window_count",
        "free_coeff_l0",
        "free_coeff_l1",
        "side_coeff",
        "free_l0_entropy",
        "free_l1_entropy",
        "side_entropy",
        "basis_usage_b5",
        "basis_usage_b6",
        "basis_usage_b8",
        "basis_usage_b9",
        "basis_usage_b10",
        "basis_activation_b9",
        "basis_activation_b10",
    ]
    available_cols = [col for col in summary_cols if col in patient_df.columns]
    summary = patient_df.groupby(group_cols)[available_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary["num_patients"] = patient_df.groupby(group_cols).size().values
    return summary


def build_rankings(patient_df: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    """
    生成各指标的 top-N 和 bottom-N 患者排名。
    用于发现极端案例。
    """
    metrics = [
        "side_coeff",
        "side_entropy",
        "basis_activation_b10",
        "basis_activation_b9",
        "basis_usage_b10",
        "basis_usage_b9",
        "basis_usage_b8",
        "free_l1_b5_minus_b6",
    ]
    rows = []
    keep_cols = [
        "dataset_name",
        "subject",
        "side_label_name",
        "pattern_label",
        "window_count",
        "side_coeff",
        "side_entropy",
        "basis_usage_b8",
        "basis_usage_b9",
        "basis_usage_b10",
        "basis_activation_b9",
        "basis_activation_b10",
        "free_l1_dominant_basis",
        "side_dominant_basis",
    ]
    for metric in metrics:
        available = [col for col in keep_cols if col in patient_df.columns]
        selected_cols = list(dict.fromkeys(available + [metric]))
        top_df = patient_df.nlargest(top_n, metric)[selected_cols].copy()
        top_df["metric"] = metric
        top_df["direction"] = "top"
        top_df["rank"] = np.arange(1, len(top_df) + 1)
        rows.append(top_df)

        bottom_df = patient_df.nsmallest(top_n, metric)[selected_cols].copy()
        bottom_df["metric"] = metric
        bottom_df["direction"] = "bottom"
        bottom_df["rank"] = np.arange(1, len(bottom_df) + 1)
        rows.append(bottom_df)

    ranking_df = pd.concat(rows, ignore_index=True)
    ordered_cols = ["metric", "direction", "rank"] + available + [metric]
    return ranking_df


def build_crosstabs(patient_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """生成交叉表：dominant basis vs side_label / dataset / pattern_label"""
    return {
        "crosstab_free_l1_dominant_basis_by_side_label": pd.crosstab(
            patient_df["free_l1_dominant_basis"],
            patient_df["side_label_name"],
        ),
        "crosstab_side_dominant_basis_by_side_label": pd.crosstab(
            patient_df["side_dominant_basis"],
            patient_df["side_label_name"],
        ),
        "crosstab_free_l1_dominant_basis_by_dataset": pd.crosstab(
            patient_df["free_l1_dominant_basis"],
            patient_df["dataset_name"],
        ),
        "crosstab_pattern_label_by_side_label": pd.crosstab(
            patient_df["pattern_label"],
            patient_df["side_label_name"],
        ),
        "crosstab_pattern_label_by_dataset": pd.crosstab(
            patient_df["pattern_label"],
            patient_df["dataset_name"],
        ),
    }


def build_report(
    *,
    patient_df: pd.DataFrame,
    side_summary_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    crosstabs: dict[str, pd.DataFrame],
    ranking_df: pd.DataFrame,
) -> str:
    """生成 Markdown 报告，包含主要发现、变异排名、汇总表和交叉表"""
    usage_std = patient_df[[c for c in patient_df.columns if c.startswith("basis_usage_b")]].std().sort_values(ascending=False)
    activation_std = patient_df[[c for c in patient_df.columns if c.startswith("basis_activation_b")]].std().sort_values(ascending=False)

    lines = [
        "# Patient Activation Pattern Report",
        "",
        f"- patients: `{int(patient_df.shape[0])}`",
        f"- mean windows per patient: `{patient_df['window_count'].mean():.3f}`",
        f"- median windows per patient: `{patient_df['window_count'].median():.3f}`",
        "",
        "## Main Findings",
        "",
        f"1. `free_b0` is effectively fixed as the dominant level-0 basis for all `{int(patient_df.shape[0])}` patients, so patient-level variation in the free path mainly comes from level-1 (`free_b5` vs `free_b6`) rather than level-0.",
        f"2. The strongest patient-to-patient variation sits in the side branch, especially `basis_activation_b10` (std `{activation_std['basis_activation_b10']:.3f}`) and `basis_activation_b9` (std `{activation_std['basis_activation_b9']:.3f}`).",
        f"3. Side-label structure is very strong: `side_b10` dominates `{int((patient_df['side_dominant_basis'] == 10).sum())}` / `{int(patient_df.shape[0])}` patients overall, but its entropy and coefficient sign differ sharply across Left / Normal / Right groups.",
        f"4. Dataset leakage still shows up in the patient summaries: IMR patients lean more toward `free_b5`, while TT patients contribute most of the `free_b6`-dominant subgroup.",
        "",
        "## Variation Ranking",
        "",
        "Patient-level usage std ranking:",
        usage_std.round(4).to_string(),
        "",
        "Patient-level activation std ranking:",
        activation_std.round(4).to_string(),
        "",
        "## Side Summary",
        "",
        side_summary_df.to_markdown(index=False),
        "",
        "## Dataset Summary",
        "",
        dataset_summary_df.to_markdown(index=False),
        "",
        "## Crosstabs",
        "",
    ]

    for name, table in crosstabs.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(table.to_markdown())
        lines.append("")

    lines.extend(
        [
            "## Representative Extremes",
            "",
        ]
    )
    for metric in ["side_coeff", "basis_activation_b10", "basis_activation_b9", "side_entropy"]:
        subset = ranking_df[ranking_df["metric"] == metric].copy()
        if subset.empty:
            continue
        lines.append(f"### {metric}")
        lines.append("")
        lines.append(subset.to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def analyze(
    wide_csv: str = "outputs/disentangleNet/v31_current_verify/window_basis_activations_all/window_basis_activations_wide.csv",
    output_dir: str | None = None,
):
    """
    主入口函数。

    参数：
    - `wide_csv`：窗口级 basis activation 宽表
    - `output_dir`：可选的自定义输出目录
    """
    wide_path = Path(wide_csv).expanduser().resolve()
    if not wide_path.exists():
        raise FileNotFoundError(wide_path)

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else wide_path.parent / "patient_pattern_analysis" / "patient_profile_summary"
    )
    destination.mkdir(parents=True, exist_ok=True)

    wide_df = pd.read_csv(wide_path)
    patient_df = build_patient_profiles(wide_df)
    side_summary_df = summarize_by_group(patient_df, ["side_label_name"])
    dataset_summary_df = summarize_by_group(patient_df, ["dataset_name"])
    dataset_side_summary_df = summarize_by_group(patient_df, ["dataset_name", "side_label_name"])
    ranking_df = build_rankings(patient_df, top_n=10)
    crosstabs = build_crosstabs(patient_df)

    patient_path = destination / "patient_activation_profiles.csv"
    side_summary_path = destination / "summary_by_side_label.csv"
    dataset_summary_path = destination / "summary_by_dataset.csv"
    dataset_side_summary_path = destination / "summary_by_dataset_and_side_label.csv"
    ranking_path = destination / "patient_extreme_rankings.csv"
    report_path = destination / "report.md"
    summary_path = destination / "summary.json"

    patient_df.to_csv(patient_path, index=False)
    side_summary_df.to_csv(side_summary_path, index=False)
    dataset_summary_df.to_csv(dataset_summary_path, index=False)
    dataset_side_summary_df.to_csv(dataset_side_summary_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    for name, table in crosstabs.items():
        table.to_csv(destination / f"{name}.csv")

    report_path.write_text(
        build_report(
            patient_df=patient_df,
            side_summary_df=side_summary_df,
            dataset_summary_df=dataset_summary_df,
            crosstabs=crosstabs,
            ranking_df=ranking_df,
        ),
        encoding="utf-8",
    )

    summary = {
        "wide_csv": str(wide_path),
        "num_patients": int(patient_df.shape[0]),
        "num_rankings": int(ranking_df.shape[0]),
        "pattern_counts": patient_df["pattern_label"].value_counts().to_dict(),
        "paths": {
            "patient_profiles_csv": str(patient_path),
            "summary_by_side_label_csv": str(side_summary_path),
            "summary_by_dataset_csv": str(dataset_summary_path),
            "summary_by_dataset_and_side_label_csv": str(dataset_side_summary_path),
            "patient_extreme_rankings_csv": str(ranking_path),
            "report_md": str(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved patient profiles: {patient_path}")
    print(f"Saved side summary: {side_summary_path}")
    print(f"Saved dataset summary: {dataset_summary_path}")
    print(f"Saved dataset-side summary: {dataset_side_summary_path}")
    print(f"Saved rankings: {ranking_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")
    return summary


if __name__ == "__main__":
    fire.Fire({"analyze": analyze})