from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


def _load_matplotlib():
    mpl_dir = Path("outputs/matrix_vis/.mplconfig").resolve()
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib.pyplot as plt

    return plt


def save_motion_snapshot(
    *,
    output_path: Path,
    static_points: np.ndarray,
    animated_points: np.ndarray,
    title: str,
) -> None:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(static_points[:, 0], static_points[:, 1], s=28, c="#c7c7c7", label="static mesh")
    ax.plot(animated_points[:, 0], animated_points[:, 1], "-o", color="#d94841", linewidth=2, markersize=4, label="animated subset")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_motion_frames(
    *,
    output_dir: Path,
    static_points: np.ndarray,
    coordinates: np.ndarray,
    subset_mask: np.ndarray,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    frames: list[Path] = []
    x_min = float(coordinates[:, :, 0].min())
    x_max = float(coordinates[:, :, 0].max())
    y_min = float(coordinates[:, :, 1].min())
    y_max = float(coordinates[:, :, 1].max())
    x_pad = max((x_max - x_min) * 0.12, 0.2)
    y_pad = max((y_max - y_min) * 0.12, 0.2)

    for frame_idx in range(coordinates.shape[0]):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(static_points[:, 0], static_points[:, 1], s=18, c="#d9d9d9")
        current = coordinates[frame_idx]
        ax.scatter(current[~subset_mask, 0], current[~subset_mask, 1], s=18, c="#bdbdbd")
        ax.plot(current[subset_mask, 0], current[subset_mask, 1], "-o", color="#1f78b4", linewidth=2, markersize=4)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"frame {frame_idx + 1}/{coordinates.shape[0]}")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        frame_path = output_dir / f"frame_{frame_idx:03d}.png"
        fig.savefig(frame_path, dpi=140)
        plt.close(fig)
        frames.append(frame_path)
    return frames


def save_gif_from_frames(frame_paths: list[Path], output_path: Path, *, duration_ms: int = 90) -> None:
    if not frame_paths:
        raise ValueError("No frames available for GIF export")
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    first, rest = images[0], images[1:]
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
