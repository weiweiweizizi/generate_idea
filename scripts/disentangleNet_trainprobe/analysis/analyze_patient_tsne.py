#!/usr/bin/env python
"""
disentangleNet basis 表征的患者级 t-SNE 可视化。

此脚本功能：
- 读取 `patient_profile_summary/patient_activation_profiles.csv`。
- 构建三种特征族：
  1. `usage`：患者均值的 `basis_usage_b*`
  2. `activation`：患者均值的 `basis_activation_b*`
  3. `combined`：usage + activation 拼接
- 对每种特征族运行 2D 和 3D t-SNE。
- 导出嵌入 CSV 和五张图（按 dataset_name、side_label_name、score、label_5class 和 combined view）。
- 3D combined view 额外导出旋转 GIF。

典型用法：
1. 默认运行：
   `python scripts/disentangleNet_trainprobe/analysis/analyze_patient_tsne.py analyze`
2. 排除某些 basis（如排除 side basis）：
   `python scripts/disentangleNet_trainprobe/analysis/analyze_patient_tsne.py analyze \\
      --output_root outputs/.../patient_pattern_analysis/tsne/all/no_side \\
      --exclude_basis_indices 8,9,10`
3. 仅保留 side basis（排除 free basis）：
   `python scripts/disentangleNet_trainprobe/analysis/analyze_patient_tsne.py analyze \\
      --output_root outputs/.../patient_pattern_analysis/tsne/all/side_only \\
      --exclude_basis_indices 0,1,2,3,4,5,6,7`

结果输出位置：
- 顶层目录由 `--output_root` 控制。
- 每种特征族的结果写入：
  `<output_root>/<feature_family>/tsne_2d/`
  `<output_root>/<feature_family>/tsne_3d/`
- 每个目录包含：嵌入 CSV、四个分类着色图、一个 combined 图。
  `tsne_3d/` 额外包含一个 combined 旋转 GIF。
- 顶层目录还包含：`report.md` 和 `summary.json`。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import fire
import imageio.v2 as imageio
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# 颜色配置
DATASET_COLORS = {
    "IMR": "#1f77b4",
    "TT": "#d62728",
}

SIDE_COLORS = {
    "Left": "#2ca02c",
    "Normal": "#ff7f0e",
    "Right": "#9467bd",
}

SCORE_COLORS = {
    0: "#2c7fb8",
    1: "#fdae61",
    2: "#d7191c",
}

LABEL5_CMAP = plt.get_cmap("coolwarm")


def load_patient_profiles(patient_profiles_csv: Path) -> pd.DataFrame:
    """加载患者级特征表作为 t-SNE 输入"""
    df = pd.read_csv(patient_profiles_csv)
    required_cols = ["dataset_name", "subject", "side_label_name", "score", "label_5class"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in patient profile table: {missing}")
    return df


def canonicalize_subject(value) -> str:
    """规范化 subject id，使患者表和 fold manifest 能对齐"""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text:
        return text
    if text.isdigit():
        stripped = text.lstrip("0")
        return stripped or "0"
    return text


def build_oof_patient_profiles(
    patient_df: pd.DataFrame,
    *,
    subject_fold_assignments_csv: Path,
) -> tuple[pd.DataFrame, dict]:
    """通过拼接每个 fold 的验证子集构建 OOF 患者表"""
    fold_df = pd.read_csv(subject_fold_assignments_csv, dtype={"subject": str}).copy()
    if "subject" not in fold_df.columns or "fold" not in fold_df.columns:
        raise ValueError("subject_fold_assignments_csv must contain `subject` and `fold` columns")

    patient_df = patient_df.copy()
    patient_df["_subject_key"] = patient_df["subject"].map(canonicalize_subject)
    fold_df["_subject_key"] = fold_df["subject"].map(canonicalize_subject)

    fold_map = fold_df.drop_duplicates("_subject_key", keep="first")[["_subject_key", "fold"]]
    merged = patient_df.merge(fold_map, on="_subject_key", how="left", validate="one_to_one")
    if merged["fold"].isna().any():
        missing = merged.loc[merged["fold"].isna(), "subject"].astype(str).tolist()
        raise RuntimeError(f"Missing fold assignments for subjects: {missing[:10]}")

    merged["fold"] = merged["fold"].astype(int)
    merged["oof_role"] = "val"
    merged = merged.sort_values(["fold", "dataset_name", "subject"]).reset_index(drop=True)
    counts_by_fold = merged.groupby("fold")["subject"].count().sort_index().astype(int).to_dict()
    merged = merged.drop(columns=["_subject_key"])
    return merged, {
        "subject_fold_assignments_csv": str(subject_fold_assignments_csv.resolve()),
        "num_folds": int(len(counts_by_fold)),
        "num_patients_oof": int(merged.shape[0]),
        "patients_per_fold": {str(k): int(v) for k, v in counts_by_fold.items()},
    }


def parse_basis_index(col_name: str) -> int:
    """从 `basis_usage_b10` 风格列名提取末尾 basis 索引"""
    return int(col_name.rsplit("b", 1)[1])


def parse_exclude_basis_indices(exclude_basis_indices) -> list[int]:
    """将 Fire CLI 输入规范化为要排除的 basis 索引排序列表"""
    if exclude_basis_indices is None:
        return []
    if isinstance(exclude_basis_indices, str):
        if not exclude_basis_indices.strip():
            return []
        return sorted({int(part.strip()) for part in exclude_basis_indices.split(",") if part.strip()})
    if isinstance(exclude_basis_indices, (list, tuple, set)):
        return sorted({int(value) for value in exclude_basis_indices})
    return [int(exclude_basis_indices)]


def filter_basis_columns(cols: list[str], exclude_basis_indices: list[int]) -> list[str]:
    """删除所有 basis 索引在排除列表中的列"""
    if not exclude_basis_indices:
        return cols
    exclude_set = set(exclude_basis_indices)
    return [col for col in cols if parse_basis_index(col) not in exclude_set]


def feature_columns(
    patient_df: pd.DataFrame,
    feature_family: str,
    *,
    exclude_basis_indices: list[int],
) -> list[str]:
    """根据特征族选择 basis 特征列（支持可选排除）"""
    if feature_family == "usage":
        cols = [col for col in patient_df.columns if col.startswith("basis_usage_b")]
        return filter_basis_columns(cols, exclude_basis_indices)
    if feature_family == "activation":
        cols = [col for col in patient_df.columns if col.startswith("basis_activation_b")]
        return filter_basis_columns(cols, exclude_basis_indices)
    if feature_family == "combined":
        usage_cols = [col for col in patient_df.columns if col.startswith("basis_usage_b")]
        activation_cols = [col for col in patient_df.columns if col.startswith("basis_activation_b")]
        return filter_basis_columns(usage_cols, exclude_basis_indices) + filter_basis_columns(
            activation_cols,
            exclude_basis_indices,
        )
    raise ValueError(f"Unsupported feature_family: {feature_family!r}")


def tsne_perplexity(n_samples: int, requested: float) -> float:
    """将 perplexity 限制在安全范围内（不超过样本数 - 1）"""
    upper = max(5.0, min(float(requested), float(n_samples - 1)))
    adaptive = max(5.0, min(upper, float((n_samples - 1) // 3)))
    return adaptive


def compute_embedding(
    values: np.ndarray,
    *,
    n_components: int,
    perplexity: float,
    random_state: int,
) -> np.ndarray:
    """先标准化特征，再计算 t-SNE 嵌入"""
    scaled = StandardScaler().fit_transform(values)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    return tsne.fit_transform(scaled)


def categorical_handles(color_map: dict, marker: str = "o") -> list[Line2D]:
    """为类别变量生成 legend handles"""
    handles = []
    for label, color in color_map.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="white",
                label=str(label),
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
                linewidth=0,
            )
        )
    return handles


def numeric_handles(color_map: dict, marker: str = "o") -> list[Line2D]:
    """为数值变量（有序类别）生成 legend handles"""
    handles = []
    for label, color in sorted(color_map.items()):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="white",
                label=str(label),
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
                linewidth=0,
            )
        )
    return handles


def make_axes(n_components: int):
    """根据维数创建 matplotlib figure 和 axes（2D 或 3D）"""
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(8.6, 7.2))
    elif n_components == 3:
        fig = plt.figure(figsize=(8.8, 7.4))
        ax = fig.add_subplot(111, projection="3d")
    else:
        raise ValueError(f"Unsupported n_components: {n_components}")
    return fig, ax


def scatter_points(
    ax,
    embedding: np.ndarray,
    *,
    n_components: int,
    colors,
    markers,
    edgecolors,
    sizes,
    alpha: float = 0.9,
):
    """在 axes 上绘制散点（支持单 markers 或 per-point markers）"""
    if n_components == 2:
        if isinstance(markers, str):
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=colors,
                marker=markers,
                s=sizes,
                alpha=alpha,
                edgecolors=edgecolors,
                linewidths=0.5,
            )
        else:
            unique_markers = sorted(set(markers))
            for marker in unique_markers:
                mask = np.array(markers) == marker
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=np.asarray(colors)[mask],
                    marker=marker,
                    s=np.asarray(sizes)[mask],
                    alpha=alpha,
                    edgecolors=np.asarray(edgecolors)[mask],
                    linewidths=0.5,
                )
    else:
        if isinstance(markers, str):
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                embedding[:, 2],
                c=colors,
                marker=markers,
                s=sizes,
                alpha=alpha,
                edgecolors=edgecolors,
                linewidths=0.4,
                depthshade=False,
            )
        else:
            unique_markers = sorted(set(markers))
            for marker in unique_markers:
                mask = np.array(markers) == marker
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    embedding[mask, 2],
                    c=np.asarray(colors)[mask],
                    marker=marker,
                    s=np.asarray(sizes)[mask],
                    alpha=alpha,
                    edgecolors=np.asarray(edgecolors)[mask],
                    linewidths=0.4,
                    depthshade=False,
                )


def style_axes(ax, *, n_components: int, title: str):
    """设置 axes 的标题和坐标轴标签"""
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if n_components == 3:
        ax.set_zlabel("t-SNE 3")
    ax.grid(alpha=0.18)


def plot_group_colored(
    embedding: np.ndarray,
    patient_df: pd.DataFrame,
    *,
    n_components: int,
    group_col: str,
    output_path: Path,
    title: str,
) -> None:
    """按单一分组变量着色绘制嵌入散点图"""
    fig, ax = make_axes(n_components)

    if group_col == "dataset_name":
        color_map = DATASET_COLORS
        colors = patient_df[group_col].map(color_map).fillna("#7f7f7f").to_numpy()
        handles = categorical_handles(color_map)
    elif group_col == "side_label_name":
        color_map = SIDE_COLORS
        colors = patient_df[group_col].map(color_map).fillna("#7f7f7f").to_numpy()
        handles = categorical_handles(color_map)
    elif group_col == "score":
        color_map = SCORE_COLORS
        score_values = patient_df[group_col].astype(int)
        colors = score_values.map(color_map).fillna("#7f7f7f").to_numpy()
        handles = numeric_handles(color_map)
    elif group_col == "label_5class":
        values = patient_df[group_col].astype(float).to_numpy()
        norm = TwoSlopeNorm(vmin=0.0, vcenter=2.0, vmax=4.0)
        colors = LABEL5_CMAP(norm(values))
        handles = None
    else:
        raise ValueError(f"Unsupported group_col: {group_col!r}")

    scatter_points(
        ax,
        embedding,
        n_components=n_components,
        colors=colors,
        markers="o",
        edgecolors=np.array(["black"] * len(patient_df)),
        sizes=np.full(len(patient_df), 42.0),
        alpha=0.88,
    )
    style_axes(ax, n_components=n_components, title=title)

    if handles is not None:
        ax.legend(handles=handles, title=group_col, loc="best", frameon=True)
    else:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=LABEL5_CMAP)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("label_5class")
        cbar.set_ticks([0, 1, 2, 3, 4])

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_combined(
    embedding: np.ndarray,
    patient_df: pd.DataFrame,
    *,
    n_components: int,
    output_path: Path,
    title: str,
    gif_output_path: Path | None = None,
    rotation_elev: float = 22.0,
    rotation_frames: int = 36,
    rotation_duration: float = 0.12,
) -> None:
    """
    使用更丰富视觉编码绘制嵌入散点图。
    编码规则：
    - marker 形状：dataset_name（IMR=o，TT=^）
    - 点填充颜色：label_5class 中心为 2 的 diverging colormap
    - 点边框颜色：side_label_name
    3D 模式下若提供了 gif_output_path，还导出旋转 GIF。
    """
    fig, ax = make_axes(n_components)

    label_values = patient_df["label_5class"].astype(float).to_numpy()
    color_norm = TwoSlopeNorm(vmin=0.0, vcenter=2.0, vmax=4.0)
    colors = LABEL5_CMAP(color_norm(label_values))
    markers = patient_df["dataset_name"].map({"IMR": "o", "TT": "^"}).fillna("s").to_numpy()
    edgecolors = patient_df["side_label_name"].map(SIDE_COLORS).fillna("#333333").to_numpy()
    sizes = np.where(patient_df["dataset_name"].eq("TT").to_numpy(), 62.0, 46.0)

    scatter_points(
        ax,
        embedding,
        n_components=n_components,
        colors=colors,
        markers=markers,
        edgecolors=edgecolors,
        sizes=sizes,
        alpha=0.92,
    )
    style_axes(ax, n_components=n_components, title=title)

    # 双图例
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="IMR",
            markerfacecolor="#d9d9d9",
            markeredgecolor="black",
            markersize=8,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="white",
            label="TT",
            markerfacecolor="#d9d9d9",
            markeredgecolor="black",
            markersize=8,
            linewidth=0,
        ),
    ]
    legend_dataset = ax.legend(handles=dataset_handles, title="dataset_name", loc="upper left", frameon=True)
    ax.add_artist(legend_dataset)

    sm = plt.cm.ScalarMappable(norm=color_norm, cmap=LABEL5_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("label_5class")
    cbar.set_ticks([0, 1, 2, 3, 4])

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)

    # 3D 旋转 GIF
    if n_components == 3 and gif_output_path is not None:
        frames = []
        for azim in np.linspace(0.0, 360.0, rotation_frames, endpoint=False):
            ax.view_init(elev=rotation_elev, azim=float(azim))
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
            frames.append(frame)
        imageio.mimsave(gif_output_path, frames, duration=rotation_duration, loop=0)

    plt.close(fig)


def export_feature_family(
    patient_df: pd.DataFrame,
    *,
    feature_family: str,
    output_root: Path,
    perplexity: float,
    random_state: int,
    exclude_basis_indices: list[int],
) -> dict:
    """对一种特征族运行 2D/3D t-SNE，导出嵌入 CSV 和各种着色图"""
    cols = feature_columns(
        patient_df,
        feature_family,
        exclude_basis_indices=exclude_basis_indices,
    )
    values = patient_df[cols].to_numpy(dtype=np.float64)
    family_dir = output_root / feature_family
    family_dir.mkdir(parents=True, exist_ok=True)

    adaptive_perplexity = tsne_perplexity(values.shape[0], perplexity)
    family_summary = {
        "feature_family": feature_family,
        "feature_count": len(cols),
        "feature_columns": cols,
        "excluded_basis_indices": exclude_basis_indices,
        "n_samples": int(values.shape[0]),
        "perplexity_requested": float(perplexity),
        "perplexity_used": float(adaptive_perplexity),
        "random_state": int(random_state),
        "embeddings": {},
    }

    # 元数据列（用于嵌入 CSV 附带信息）
    base_meta_cols = [
        "dataset_name",
        "dataset_label",
        "subject",
        "side_label_name",
        "score",
        "label_5class",
        "pattern_label",
        "window_count",
    ]
    meta_cols = [col for col in base_meta_cols if col in patient_df.columns]

    for n_components in (2, 3):
        embedding = compute_embedding(
            values,
            n_components=n_components,
            perplexity=adaptive_perplexity,
            random_state=random_state,
        )
        dim_dir = family_dir / f"tsne_{n_components}d"
        dim_dir.mkdir(parents=True, exist_ok=True)

        # 嵌入 CSV
        embed_cols = [f"tsne_{n_components}d_{idx + 1}" for idx in range(n_components)]
        embed_df = patient_df[meta_cols].copy()
        for idx, col in enumerate(embed_cols):
            embed_df[col] = embedding[:, idx]
        embedding_csv = dim_dir / f"{feature_family}_tsne_{n_components}d_embeddings.csv"
        embed_df.to_csv(embedding_csv, index=False)

        # 按类别着色图
        plot_paths = {}
        for group_col in ("dataset_name", "side_label_name", "score", "label_5class"):
            output_path = dim_dir / f"{feature_family}_tsne_{n_components}d_by_{group_col}.png"
            plot_group_colored(
                embedding,
                patient_df,
                n_components=n_components,
                group_col=group_col,
                output_path=output_path,
                title=f"{feature_family} t-SNE ({n_components}D) colored by {group_col}",
            )
            plot_paths[group_col] = str(output_path.resolve())

        # combined 图（+ 可选 GIF）
        combined_path = dim_dir / f"{feature_family}_tsne_{n_components}d_combined.png"
        combined_gif_path = None
        if n_components == 3:
            combined_gif_path = dim_dir / f"{feature_family}_tsne_{n_components}d_combined.gif"
        plot_combined(
            embedding,
            patient_df,
            n_components=n_components,
            output_path=combined_path,
            title=f"{feature_family} t-SNE ({n_components}D) combined view",
            gif_output_path=combined_gif_path,
        )
        plot_paths["combined"] = str(combined_path.resolve())
        if combined_gif_path is not None:
            plot_paths["combined_gif"] = str(combined_gif_path.resolve())

        family_summary["embeddings"][f"tsne_{n_components}d"] = {
            "embedding_csv": str(embedding_csv.resolve()),
            "plot_paths": plot_paths,
        }

    return family_summary


def build_report(summary: dict) -> str:
    """生成机器可读的索引 Markdown 报告"""
    lines = [
        "# Patient-level t-SNE Report",
        "",
        f"- patient_profiles_csv: `{summary['patient_profiles_csv']}`",
        f"- output_root: `{summary['output_root']}`",
        f"- n_patients: `{summary['n_patients']}`",
        f"- feature_families: `{', '.join(summary['feature_families'])}`",
        f"- excluded_basis_indices: `{summary['excluded_basis_indices']}`",
        f"- plots_per_family: `10`",
        f"- total_plots: `{summary['total_plots']}`",
        "",
    ]
    if summary.get("oof_mode"):
        lines.extend(
            [
                "## OOF Mode",
                "",
                f"- subject_fold_assignments_csv: `{summary['oof_summary']['subject_fold_assignments_csv']}`",
                f"- num_folds: `{summary['oof_summary']['num_folds']}`",
                f"- num_patients_oof: `{summary['oof_summary']['num_patients_oof']}`",
                f"- patients_per_fold: `{summary['oof_summary']['patients_per_fold']}`",
                "",
            ]
        )
    lines.extend(["## Families", ""])
    for family in summary["families"]:
        lines.extend(
            [
                f"### {family['feature_family']}",
                "",
                f"- feature_count: `{family['feature_count']}`",
                f"- excluded_basis_indices: `{family['excluded_basis_indices']}`",
                f"- perplexity_used: `{family['perplexity_used']}`",
                f"- random_state: `{family['random_state']}`",
                f"- 2D embeddings: `{family['embeddings']['tsne_2d']['embedding_csv']}`",
                f"- 3D embeddings: `{family['embeddings']['tsne_3d']['embedding_csv']}`",
                f"- 3D combined GIF: `{family['embeddings']['tsne_3d']['plot_paths'].get('combined_gif', 'n/a')}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def analyze(
    patient_profiles_csv: str = (
        "outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/window_basis_activations_all/"
        "patient_pattern_analysis/patient_profile_summary/patient_activation_profiles.csv"
    ),
    output_root: str = (
        "outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/window_basis_activations_all/"
        "patient_pattern_analysis/tsne/all/all_basis"
    ),
    subject_fold_assignments_csv: str = "",
    exclude_basis_indices: str = "",
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """
    主入口函数。

    参数：
    - `patient_profiles_csv`：患者级输入表
    - `output_root`：所有图/嵌入的根目录
    - `subject_fold_assignments_csv`：可选的 k-fold manifest；
      若提供则构建 OOF 患者表（拼接每 fold 验证子集）
    - `exclude_basis_indices`：逗号分隔的待忽略 basis id
    - `perplexity`：请求的 t-SNE perplexity
    - `random_state`：固定随机种子以保证可复现性
    """
    patient_profiles_path = Path(patient_profiles_csv)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    patient_df = load_patient_profiles(patient_profiles_path)
    oof_summary = None
    if subject_fold_assignments_csv:
        raise ValueError(
            "Pseudo-OOF mode is deprecated. For strict OOF t-SNE, generate patient profiles "
            "from strict OOF window activations and pass that patient_profiles_csv directly."
        )
    excluded_basis = parse_exclude_basis_indices(exclude_basis_indices)

    family_summaries = []
    for feature_family in ("usage", "activation", "combined"):
        family_summaries.append(
            export_feature_family(
                patient_df,
                feature_family=feature_family,
                output_root=output_root_path,
                perplexity=perplexity,
                random_state=random_state,
                exclude_basis_indices=excluded_basis,
            )
        )

    summary = {
        "patient_profiles_csv": str(patient_profiles_path.resolve()),
        "output_root": str(output_root_path.resolve()),
        "n_patients": int(patient_df.shape[0]),
        "oof_mode": bool(oof_summary is not None),
        "oof_summary": oof_summary,
        "feature_families": ["usage", "activation", "combined"],
        "excluded_basis_indices": excluded_basis,
        "plots_per_family": 10,
        "total_plots": 30,
        "families": family_summaries,
    }

    (output_root_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (output_root_path / "report.md").write_text(build_report(summary), encoding="utf-8")


def main():
    fire.Fire({"analyze": analyze})


if __name__ == "__main__":
    main()
