#!/usr/bin/env python
"""
Patient-level t-SNE visualization for disentangleNet basis representations.

What this script is for:
- Read the patient-level summary table exported from
  `patient_profile_summary/patient_activation_profiles.csv`.
- Build three feature families:
  1. `usage`: patient-mean `basis_usage_b*`
  2. `activation`: patient-mean `basis_activation_b*`
  3. `combined`: concatenate usage + activation
- Run t-SNE in both 2D and 3D for each feature family.
- Export one embedding CSV plus five plots for each dimensionality:
  `dataset_name`, `side_label_name`, `score`, `label_5class`, and a combined view.
- For the 3D combined view, also export a rotating GIF.

How to use:
1. Default run:
   `python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze`
2. Exclude some basis indices, for example ignore side basis:
   `python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \\
      --output_root outputs/.../patient_pattern_analysis/tsne/all/no_side \\
      --exclude_basis_indices 8,9,10`
3. Keep only side basis by excluding free basis:
   `python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \\
      --output_root outputs/.../patient_pattern_analysis/tsne/all/side_only \\
      --exclude_basis_indices 0,1,2,3,4,5,6,7`

Where results go:
- The top-level output directory is controlled by `--output_root`.
- For each feature family, results are written to:
  `<output_root>/<feature_family>/tsne_2d/`
  `<output_root>/<feature_family>/tsne_3d/`
- Each of those folders contains:
  - one embedding CSV
  - four class-colored plots
  - one combined plot
  - in `tsne_3d/`, one extra combined rotating GIF
- The root output directory also contains:
  - `report.md`
  - `summary.json`
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
    """Load the patient-level feature table used as the t-SNE input."""
    df = pd.read_csv(patient_profiles_csv)
    required_cols = ["dataset_name", "subject", "side_label_name", "score", "label_5class"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in patient profile table: {missing}")
    return df


def parse_basis_index(col_name: str) -> int:
    """Extract the trailing basis index from `basis_usage_b10`-style columns."""
    return int(col_name.rsplit("b", 1)[1])


def parse_exclude_basis_indices(exclude_basis_indices) -> list[int]:
    """Normalize Fire CLI input into a sorted list of basis indices to exclude."""
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
    """Drop all columns whose basis index is in `exclude_basis_indices`."""
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
    """Select feature columns for one family after optional basis exclusion."""
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
    """Keep perplexity in a safe range for the current sample count."""
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
    """Standardize features first, then compute one t-SNE embedding."""
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
    """Plot one embedding with color determined by a single grouping variable."""
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
    Plot one embedding with a denser visual encoding.

    Encoding:
    - marker shape: dataset_name (`IMR=o`, `TT=^`)
    - point fill color: label_5class diverging colormap centered at 2
    - point edge color: side_label_name
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
    """Run 2D/3D t-SNE for one feature family and export CSVs + plots."""
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

        embed_cols = [f"tsne_{n_components}d_{idx + 1}" for idx in range(n_components)]
        embed_df = patient_df[meta_cols].copy()
        for idx, col in enumerate(embed_cols):
            embed_df[col] = embedding[:, idx]
        embedding_csv = dim_dir / f"{feature_family}_tsne_{n_components}d_embeddings.csv"
        embed_df.to_csv(embedding_csv, index=False)

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
    """Write a short machine-generated index of the exported result folders."""
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
        "## Families",
        "",
    ]
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
        "outputs/disentangleNet/v31_current_verify/window_basis_activations_all/"
        "patient_pattern_analysis/patient_profile_summary/patient_activation_profiles.csv"
    ),
    output_root: str = (
        "outputs/disentangleNet/v31_current_verify/window_basis_activations_all/"
        "patient_pattern_analysis/tsne/all/all_basis"
    ),
    exclude_basis_indices: str = "",
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """
    Main CLI entry.

    Parameters:
    - `patient_profiles_csv`: patient-level input table
    - `output_root`: root directory where all plots/embeddings will be written
    - `exclude_basis_indices`: comma-separated basis ids to ignore
    - `perplexity`: requested t-SNE perplexity
    - `random_state`: fixed seed for reproducibility
    """
    patient_profiles_path = Path(patient_profiles_csv)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    patient_df = load_patient_profiles(patient_profiles_path)
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
