from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np


def _load_matplotlib():
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - depends on local env
        return None
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
    if plt is None:
        return "matplotlib is unavailable; skipped axis trajectory plot"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, point_id in enumerate(point_ids.tolist()):
        ax.plot(time_grid, trajectory[idx], linewidth=1.5, label=f"p{point_id}")
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
