#!/usr/bin/env python
"""
Compare patient-level coactivation structure across classes.

What this script does:
- Read `patient_activation_profiles.csv`.
- Select either patient-mean `usage` or `activation` features.
- Compute class-wise correlation matrices.
- Run a permutation MANOVA-style omnibus test.
- Run pairwise permutation tests on correlation differences.
- Export heatmaps, delta heatmaps, CSV tables, and a Markdown report.

Typical usage:
1. Side-label coactivation on activations:
   `python scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py analyze \\
      --class_col side_label_name --feature_family activation`
2. Dataset coactivation on usage:
   `python scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py analyze \\
      --class_col dataset_name --feature_family usage`
3. Stronger permutation budget:
   `python scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py analyze \\
      --class_col label_5class --feature_family activation --n_perm 4000`

Default outputs:
- `.../patient_pattern_analysis/coactivation/by_<class_col>/<feature_family>/`
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
import sys

import fire
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    ordered = p_values[order]
    adjusted = ordered * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def corr_matrix(values: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(values, rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def permanova_pseudo_f(values: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    n_samples = values.shape[0]
    n_groups = len(unique_labels)
    grand_mean = values.mean(axis=0, keepdims=True)

    ss_between = 0.0
    ss_within = 0.0
    for label in unique_labels:
        subset = values[labels == label]
        group_mean = subset.mean(axis=0, keepdims=True)
        ss_between += float(subset.shape[0] * np.square(group_mean - grand_mean).sum())
        ss_within += float(np.square(subset - group_mean).sum())

    df_between = max(n_groups - 1, 1)
    df_within = max(n_samples - n_groups, 1)
    ms_between = ss_between / df_between
    ms_within = ss_within / max(df_within, 1)
    if ms_within <= 1e-12:
        return float("inf")
    return ms_between / ms_within


def permutation_permanova(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    observed = permanova_pseudo_f(values, labels)
    perm_stats = np.empty(n_perm, dtype=np.float64)
    for perm_idx in range(n_perm):
        permuted = rng.permutation(labels)
        perm_stats[perm_idx] = permanova_pseudo_f(values, permuted)
    p_value = (np.sum(perm_stats >= observed) + 1.0) / (n_perm + 1.0)
    return {
        "pseudo_f": float(observed),
        "p_value": float(p_value),
        "n_perm": int(n_perm),
    }


def permutation_corr_diff(
    values: np.ndarray,
    labels: np.ndarray,
    group_a: str,
    group_b: str,
    *,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mask = np.isin(labels, [group_a, group_b])
    subset_values = values[mask]
    subset_labels = labels[mask]

    values_a = subset_values[subset_labels == group_a]
    values_b = subset_values[subset_labels == group_b]
    corr_a = corr_matrix(values_a)
    corr_b = corr_matrix(values_b)
    delta = corr_a - corr_b

    n_a = values_a.shape[0]
    perm_exceed = np.zeros_like(delta, dtype=np.int64)
    for _ in range(n_perm):
        perm_idx = rng.permutation(subset_values.shape[0])
        perm_a = subset_values[perm_idx[:n_a]]
        perm_b = subset_values[perm_idx[n_a:]]
        perm_delta = corr_matrix(perm_a) - corr_matrix(perm_b)
        perm_exceed += (np.abs(perm_delta) >= np.abs(delta)).astype(np.int64)

    rows = []
    for i, j in combinations(range(values.shape[1]), 2):
        p_value = (perm_exceed[i, j] + 1.0) / (n_perm + 1.0)
        rows.append(
            {
                "comparison": f"{group_a}_vs_{group_b}",
                "basis_i": i,
                "basis_j": j,
                "corr_group_a": float(corr_a[i, j]),
                "corr_group_b": float(corr_b[i, j]),
                "delta_corr": float(delta[i, j]),
                "abs_delta_corr": float(abs(delta[i, j])),
                "p_value": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = benjamini_hochberg(result["p_value"].to_numpy(dtype=np.float64))
    result["significant_fdr_0_05"] = result["q_value"] <= 0.05
    return result.sort_values(["q_value", "abs_delta_corr"], ascending=[True, False]).reset_index(drop=True)


def matrix_to_long(
    matrix: np.ndarray,
    basis_names: list[str],
    *,
    value_name: str,
) -> pd.DataFrame:
    rows = []
    for i in range(len(basis_names)):
        for j in range(len(basis_names)):
            rows.append(
                {
                    "basis_i": i,
                    "basis_j": j,
                    "basis_i_name": basis_names[i],
                    "basis_j_name": basis_names[j],
                    value_name: float(matrix[i, j]),
                }
            )
    return pd.DataFrame(rows)


def plot_heatmap(
    matrix: np.ndarray,
    basis_names: list[str],
    *,
    title: str,
    output_path: Path,
    vmin: float,
    vmax: float,
    cmap: str,
    significance_mask: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(basis_names)))
    ax.set_yticks(np.arange(len(basis_names)))
    ax.set_xticklabels(basis_names, rotation=45, ha="right")
    ax.set_yticklabels(basis_names)
    ax.set_title(title)

    if significance_mask is not None:
        for i in range(significance_mask.shape[0]):
            for j in range(significance_mask.shape[1]):
                if i >= j:
                    continue
                if significance_mask[i, j]:
                    ax.scatter(j, i, s=28, c="black", marker="o")
                    ax.scatter(i, j, s=28, c="black", marker="o")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_focus_features(
    patient_df: pd.DataFrame,
    *,
    family: str,
    std_threshold: float,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    if family == "activation":
        cols = [col for col in patient_df.columns if col.startswith("basis_activation_b")]
        basis_names = [col.replace("basis_activation_", "") for col in cols]
    elif family == "usage":
        cols = [col for col in patient_df.columns if col.startswith("basis_usage_b")]
        basis_names = [col.replace("basis_usage_", "") for col in cols]
    else:
        raise ValueError(f"Unsupported feature family: {family!r}")

    values = patient_df[cols].to_numpy(dtype=np.float64)
    std = values.std(axis=0)
    keep_mask = std > std_threshold
    kept_cols = [col for col, keep in zip(cols, keep_mask) if keep]
    kept_basis_names = [name for name, keep in zip(basis_names, keep_mask) if keep]
    kept_values = patient_df[kept_cols].to_numpy(dtype=np.float64)
    return patient_df[kept_cols].copy(), kept_basis_names, kept_values


def build_class_corr_tables(
    patient_df: pd.DataFrame,
    values: np.ndarray,
    basis_names: list[str],
    *,
    class_col: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    matrices = {}
    long_rows = []
    labels = patient_df[class_col].to_numpy(dtype=object)
    for class_name in sorted(patient_df[class_col].unique().tolist()):
        class_values = values[labels == class_name]
        corr = corr_matrix(class_values)
        matrices[class_name] = corr
        table = matrix_to_long(corr, basis_names, value_name="corr")
        table.insert(0, class_col, class_name)
        long_rows.append(table)
    return matrices, pd.concat(long_rows, ignore_index=True)


def build_delta_matrix(
    pairwise_df: pd.DataFrame,
    basis_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    size = len(basis_names)
    delta = np.zeros((size, size), dtype=np.float64)
    sig = np.zeros((size, size), dtype=bool)
    for _, row in pairwise_df.iterrows():
        i = int(row["basis_i"])
        j = int(row["basis_j"])
        delta[i, j] = float(row["delta_corr"])
        delta[j, i] = float(row["delta_corr"])
        if bool(row["significant_fdr_0_05"]):
            sig[i, j] = True
            sig[j, i] = True
    return delta, sig


def build_report(
    *,
    family: str,
    basis_names: list[str],
    permanova_result: dict,
    pairwise_results: pd.DataFrame,
    class_corr_matrices: dict[str, np.ndarray],
) -> str:
    lines = [
        f"# {family.title()} Coactivation Report",
        "",
        f"- basis count analyzed: `{len(basis_names)}`",
        f"- basis names: `{', '.join(basis_names)}`",
        f"- omnibus permutation pseudo-F: `{permanova_result['pseudo_f']:.4f}`",
        f"- omnibus p-value: `{permanova_result['p_value']:.6f}`",
        "",
        "## Strongest Class-Specific Correlations",
        "",
    ]

    for class_name, corr in class_corr_matrices.items():
        rows = []
        for i, j in combinations(range(len(basis_names)), 2):
            rows.append((abs(corr[i, j]), corr[i, j], basis_names[i], basis_names[j]))
        rows.sort(reverse=True)
        lines.append(f"### {class_name}")
        lines.append("")
        for _, value, basis_i, basis_j in rows[:8]:
            lines.append(f"- `{basis_i}` vs `{basis_j}`: `{value:.3f}`")
        lines.append("")

    lines.append("## Significant Pairwise Correlation Differences")
    lines.append("")
    for comparison, subset in pairwise_results.groupby("comparison"):
        lines.append(f"### {comparison}")
        lines.append("")
        sig = subset[subset["significant_fdr_0_05"]].copy()
        if sig.empty:
            lines.append("- No basis-pair correlation differences survived FDR 0.05.")
            lines.append("")
            continue
        view = sig.head(12).copy()
        view["basis_i_name"] = view["basis_i"].map(lambda x: basis_names[int(x)])
        view["basis_j_name"] = view["basis_j"].map(lambda x: basis_names[int(x)])
        lines.append(
            view[
                [
                    "basis_i_name",
                    "basis_j_name",
                    "corr_group_a",
                    "corr_group_b",
                    "delta_corr",
                    "p_value",
                    "q_value",
                ]
            ].to_markdown(index=False)
        )
        lines.append("")
    return "\n".join(lines)


def analyze(
    patient_profiles_csv: str = "outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/patient_profile_summary/patient_activation_profiles.csv",
    output_dir: str | None = None,
    class_col: str = "side_label_name",
    feature_family: str = "activation",
    std_threshold: float = 1e-6,
    n_perm: int = 4000,
    seed: int = 42,
):
    """
    Main CLI entry for class-conditional coactivation analysis.

    Parameters:
    - `patient_profiles_csv`: patient-level feature table
    - `output_dir`: optional custom destination
    - `class_col`: grouping column such as `side_label_name`, `dataset_name`, `score`, `label_5class`
    - `feature_family`: `activation` or `usage`
    - `std_threshold`: drop near-constant basis before correlation analysis
    - `n_perm`, `seed`: permutation test controls
    """
    patient_path = Path(patient_profiles_csv).expanduser().resolve()
    if not patient_path.exists():
        raise FileNotFoundError(patient_path)

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else patient_path.parent.parent / "coactivation" / f"by_{class_col}" / feature_family
    )
    destination.mkdir(parents=True, exist_ok=True)

    patient_df = pd.read_csv(patient_path)
    features_df, basis_names, values = build_focus_features(
        patient_df,
        family=feature_family,
        std_threshold=std_threshold,
    )
    labels = patient_df[class_col].to_numpy(dtype=object)

    class_corr_matrices, class_corr_df = build_class_corr_tables(
        patient_df,
        values,
        basis_names,
        class_col=class_col,
    )
    permanova_result = permutation_permanova(
        values,
        labels,
        n_perm=n_perm,
        seed=seed,
    )

    pairwise_tables = []
    class_names = sorted(patient_df[class_col].unique().tolist())
    for idx_a in range(len(class_names)):
        for idx_b in range(idx_a + 1, len(class_names)):
            pairwise_tables.append(
                permutation_corr_diff(
                    values,
                    labels,
                    class_names[idx_a],
                    class_names[idx_b],
                    n_perm=n_perm,
                    seed=seed + idx_a * 101 + idx_b * 307,
                )
            )
    pairwise_df = pd.concat(pairwise_tables, ignore_index=True)

    basis_manifest_df = pd.DataFrame(
        {
            "basis_index": np.arange(len(basis_names), dtype=np.int64),
            "basis_name": basis_names,
            "global_std": values.std(axis=0),
            "global_mean": values.mean(axis=0),
        }
    )

    corr_dir = destination / "correlation_heatmaps"
    delta_dir = destination / "delta_heatmaps"
    corr_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    for class_name, corr in class_corr_matrices.items():
        plot_heatmap(
            corr,
            basis_names,
            title=f"{feature_family.title()} correlation | {class_name}",
            output_path=corr_dir / f"{class_name}_corr.png",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )

    for comparison, subset in pairwise_df.groupby("comparison"):
        delta_matrix, sig_mask = build_delta_matrix(subset, basis_names)
        vmax = max(float(np.abs(delta_matrix).max()), 1e-6)
        plot_heatmap(
            delta_matrix,
            basis_names,
            title=f"Delta corr | {comparison}",
            output_path=delta_dir / f"{comparison}_delta_corr.png",
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
            significance_mask=sig_mask,
        )

    mean_by_class = patient_df.groupby(class_col)[features_df.columns.tolist()].mean()
    mean_by_class.index.name = class_col
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mean_by_class.to_numpy(dtype=np.float64), aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(basis_names)))
    ax.set_xticklabels(basis_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(mean_by_class.shape[0]))
    ax.set_yticklabels(mean_by_class.index.tolist())
    ax.set_title(f"Mean {feature_family} by {class_col}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    mean_heatmap_path = destination / f"mean_{feature_family}_by_{class_col}.png"
    fig.savefig(mean_heatmap_path, dpi=220)
    plt.close(fig)

    basis_manifest_path = destination / "basis_manifest.csv"
    class_corr_path = destination / "class_correlation_long.csv"
    pairwise_path = destination / "pairwise_corr_diff_tests.csv"
    mean_by_class_path = destination / f"mean_{feature_family}_by_{class_col}.csv"
    report_path = destination / "report.md"
    summary_path = destination / "summary.json"

    basis_manifest_df.to_csv(basis_manifest_path, index=False)
    class_corr_df.to_csv(class_corr_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    mean_by_class.reset_index().to_csv(mean_by_class_path, index=False)
    report_path.write_text(
        build_report(
            family=feature_family,
            basis_names=basis_names,
            permanova_result=permanova_result,
            pairwise_results=pairwise_df,
            class_corr_matrices=class_corr_matrices,
        ),
        encoding="utf-8",
    )

    summary = {
        "patient_profiles_csv": str(patient_path),
        "class_col": class_col,
        "feature_family": feature_family,
        "basis_count": int(len(basis_names)),
        "basis_names": basis_names,
        "n_perm": int(n_perm),
        "permanova": permanova_result,
        "num_significant_pairwise_edges_fdr_0_05": int(pairwise_df["significant_fdr_0_05"].sum()),
        "paths": {
            "basis_manifest_csv": str(basis_manifest_path),
            "class_correlation_long_csv": str(class_corr_path),
            "pairwise_tests_csv": str(pairwise_path),
            "mean_by_class_csv": str(mean_by_class_path),
            "mean_heatmap_png": str(mean_heatmap_path),
            "corr_heatmaps_dir": str(corr_dir),
            "delta_heatmaps_dir": str(delta_dir),
            "report_md": str(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved basis manifest: {basis_manifest_path}")
    print(f"Saved class correlations: {class_corr_path}")
    print(f"Saved pairwise tests: {pairwise_path}")
    print(f"Saved class mean heatmap: {mean_heatmap_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")
    return summary


if __name__ == "__main__":
    fire.Fire({"analyze": analyze})
