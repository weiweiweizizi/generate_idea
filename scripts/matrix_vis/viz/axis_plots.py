from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _load_matplotlib():
    mpl_dir = Path("outputs/matrix_vis/.mplconfig").resolve()
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib.pyplot as plt

    return plt


def save_axis_trajectory_plot(
    *,
    output_dir: Path,
    time_grid: np.ndarray,
    trajectory: np.ndarray,
    point_ids: np.ndarray,
    axis: str,
) -> str | None:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, point_id in enumerate(point_ids.tolist()):
        ax.plot(time_grid, trajectory[idx], linewidth=1.5, alpha=0.9, label=f"p{point_id}")
    ax.set_title(f"{axis.upper()}-axis trajectories")
    ax.set_xlabel("normalized time")
    ax.set_ylabel(f"{axis} position")
    ax.grid(True, alpha=0.25)
    if trajectory.shape[0] <= 12:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "axis_trajectory.png", dpi=160)
    plt.close(fig)
    return None


def save_axis_ground_truth_comparison_plot(
    *,
    output_dir: Path,
    time_grid: np.ndarray,
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
    point_ids: np.ndarray,
    axis: str,
) -> str | None:
    if reconstructed.shape != ground_truth.shape:
        return "ground truth shape does not match reconstructed trajectory; skipped comparison plot"
    plt = _load_matplotlib()
    num_points = reconstructed.shape[0]
    cols = 4
    rows = int(np.ceil(num_points / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx, point_id in enumerate(point_ids.tolist()):
        ax = axes[idx // cols, idx % cols]
        ax.plot(time_grid, ground_truth[idx], color="tab:blue", linewidth=1.8, label="ground truth")
        ax.plot(time_grid, reconstructed[idx], color="tab:red", linewidth=1.5, linestyle="--", label="reconstructed")
        ax.set_title(f"p{point_id}", fontsize=9)
        ax.grid(True, alpha=0.25)

    for idx in range(num_points, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"{axis.upper()}-axis reconstructed vs ground truth", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_dir / "axis_ground_truth_comparison.png", dpi=160)
    plt.close(fig)
    return None
