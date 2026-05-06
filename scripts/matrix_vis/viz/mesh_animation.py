from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.matrix_vis.viz._matplotlib import load_matplotlib


STATIC_POINT_COLOR = "#d9d9d9"
SUBSET_POINT_COLOR = "#1f78b4"
ANCHOR_POINT_COLOR = "#d94841"
SNAPSHOT_STATIC_POINT_SIZE = 18
SNAPSHOT_SUBSET_POINT_SIZE = 24
SNAPSHOT_ANCHOR_POINT_SIZE = 48
FRAME_STATIC_POINT_SIZE = 18
FRAME_SUBSET_POINT_SIZE = 24
FRAME_ANCHOR_POINT_SIZE = 48


def _as_xy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected points with shape [N, >=2], got {points.shape}")
    return points[:, :2]


def _normalize_anchor_points(anchor_points: np.ndarray | None) -> np.ndarray | None:
    if anchor_points is None:
        return None
    points = _as_xy(anchor_points)
    if points.shape[0] == 0:
        return None
    return points


def _scatter_anchor_points(ax, anchor_points: np.ndarray | None, *, point_size: int) -> None:
    if anchor_points is None:
        return
    ax.scatter(anchor_points[:, 0], anchor_points[:, 1], s=point_size, c=ANCHOR_POINT_COLOR)


def save_motion_snapshot(
    *,
    output_path: Path,
    static_points: np.ndarray,
    animated_points: np.ndarray,
    title: str,
    anchor_points: np.ndarray | None = None,
) -> None:
    static_xy = _as_xy(static_points)
    animated_xy = _as_xy(animated_points)
    anchor_xy = _normalize_anchor_points(anchor_points)

    plt = load_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(static_xy[:, 0], static_xy[:, 1], s=SNAPSHOT_STATIC_POINT_SIZE, c=STATIC_POINT_COLOR)
    ax.scatter(animated_xy[:, 0], animated_xy[:, 1], s=SNAPSHOT_SUBSET_POINT_SIZE, c=SUBSET_POINT_COLOR)
    _scatter_anchor_points(ax, anchor_xy, point_size=SNAPSHOT_ANCHOR_POINT_SIZE)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_motion_frames(
    *,
    output_dir: Path,
    static_points: np.ndarray,
    subset_coordinates: np.ndarray,
    anchor_points: np.ndarray | None = None,
) -> list[Path]:
    static_xy = _as_xy(static_points)
    subset_xy = np.asarray(subset_coordinates, dtype=np.float32)
    if subset_xy.ndim != 3 or subset_xy.shape[2] < 2:
        raise ValueError(f"Expected subset_coordinates with shape [T, N, >=2], got {subset_xy.shape}")
    subset_xy = subset_xy[:, :, :2]
    anchor_xy = _normalize_anchor_points(anchor_points)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = load_matplotlib()
    frames: list[Path] = []
    x_min = min(float(static_xy[:, 0].min()), float(subset_xy[:, :, 0].min()))
    x_max = max(float(static_xy[:, 0].max()), float(subset_xy[:, :, 0].max()))
    y_min = min(float(static_xy[:, 1].min()), float(subset_xy[:, :, 1].min()))
    y_max = max(float(static_xy[:, 1].max()), float(subset_xy[:, :, 1].max()))
    x_pad = max((x_max - x_min) * 0.12, 0.05)
    y_pad = max((y_max - y_min) * 0.12, 0.05)

    for frame_idx in range(subset_xy.shape[0]):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(static_xy[:, 0], static_xy[:, 1], s=FRAME_STATIC_POINT_SIZE, c=STATIC_POINT_COLOR)
        ax.scatter(subset_xy[frame_idx, :, 0], subset_xy[frame_idx, :, 1], s=FRAME_SUBSET_POINT_SIZE, c=SUBSET_POINT_COLOR)
        _scatter_anchor_points(ax, anchor_xy, point_size=FRAME_ANCHOR_POINT_SIZE)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"frame {frame_idx + 1}/{subset_xy.shape[0]}")
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
