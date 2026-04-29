#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
from scripts.matrix_vis.io.load_mesh import _load_canonical_obj_vertices


FACE_WIDTH_POINTS = (127, 356)
FACE_HEIGHT_POINTS = (10, 152)
DEFAULT_ANCHOR_POINT_IDS = (33, 263, 10, 175)


def _load_matplotlib():
    import os

    mpl_dir = Path("outputs/matrix_vis/.mplconfig").resolve()
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib.pyplot as plt

    return plt


def _normalize_standard_facemesh(points: np.ndarray) -> np.ndarray:
    normalized = points.astype(np.float32, copy=True)
    scale_x = abs(float(points[FACE_WIDTH_POINTS[1], 0] - points[FACE_WIDTH_POINTS[0], 0]))
    scale_y = abs(float(points[FACE_HEIGHT_POINTS[1], 1] - points[FACE_HEIGHT_POINTS[0], 1]))
    if scale_x <= 0:
        scale_x = 1.0
    if scale_y <= 0:
        scale_y = 1.0
    normalized[:, 0] /= scale_x
    normalized[:, 1] /= scale_y
    return normalized


def _load_solution(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _ordered_common_ids(x_ids: np.ndarray, y_ids: np.ndarray, subset_ids: tuple[int, ...]) -> np.ndarray:
    x_set = {int(point_id) for point_id in x_ids.tolist()}
    y_set = {int(point_id) for point_id in y_ids.tolist()}
    ordered = [int(point_id) for point_id in subset_ids if int(point_id) in x_set and int(point_id) in y_set]
    if not ordered:
        raise ValueError("No overlapping ordered point ids between solutions and requested subset")
    return np.asarray(ordered, dtype=np.int64)


def _compose_subset_coordinates(
    *,
    x_solution: dict[str, np.ndarray],
    y_solution: dict[str, np.ndarray],
    subset_ids: np.ndarray,
) -> np.ndarray:
    x_lookup = {int(point_id): idx for idx, point_id in enumerate(x_solution["point_ids"].astype(np.int64).tolist())}
    y_lookup = {int(point_id): idx for idx, point_id in enumerate(y_solution["point_ids"].astype(np.int64).tolist())}
    num_frames = int(x_solution["time_grid"].shape[0])
    coordinates = np.empty((num_frames, subset_ids.shape[0], 2), dtype=np.float32)
    for subset_idx, point_id in enumerate(subset_ids.tolist()):
        coordinates[:, subset_idx, 0] = x_solution["trajectory"][x_lookup[int(point_id)]]
        coordinates[:, subset_idx, 1] = y_solution["trajectory"][y_lookup[int(point_id)]]
    return coordinates


def _save_snapshot(
    *,
    output_path: Path,
    static_points: np.ndarray,
    subset_points: np.ndarray,
    anchor_points: np.ndarray,
    title: str,
) -> None:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(static_points[:, 0], static_points[:, 1], s=12, c="#d9d9d9")
    ax.scatter(subset_points[:, 0], subset_points[:, 1], s=18, c="#1f78b4")
    ax.scatter(anchor_points[:, 0], anchor_points[:, 1], s=36, c="#d94841")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_frames(
    *,
    output_dir: Path,
    static_points: np.ndarray,
    subset_coordinates: np.ndarray,
    anchor_points: np.ndarray,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    x_min = min(float(static_points[:, 0].min()), float(subset_coordinates[:, :, 0].min()))
    x_max = max(float(static_points[:, 0].max()), float(subset_coordinates[:, :, 0].max()))
    y_min = min(float(static_points[:, 1].min()), float(subset_coordinates[:, :, 1].min()))
    y_max = max(float(static_points[:, 1].max()), float(subset_coordinates[:, :, 1].max()))
    x_pad = max((x_max - x_min) * 0.12, 0.05)
    y_pad = max((y_max - y_min) * 0.12, 0.05)

    frame_paths: list[Path] = []
    for frame_idx in range(subset_coordinates.shape[0]):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(static_points[:, 0], static_points[:, 1], s=12, c="#d9d9d9")
        ax.scatter(subset_coordinates[frame_idx, :, 0], subset_coordinates[frame_idx, :, 1], s=18, c="#1f78b4")
        ax.scatter(anchor_points[:, 0], anchor_points[:, 1], s=36, c="#d94841")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"frame {frame_idx + 1}/{subset_coordinates.shape[0]}")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        frame_path = output_dir / f"frame_{frame_idx:03d}.png"
        fig.savefig(frame_path, dpi=140)
        plt.close(fig)
        frame_paths.append(frame_path)
    return frame_paths


def _save_gif(frame_paths: list[Path], output_path: Path, *, duration_ms: int = 90) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview real 341-point reconstruction on normalized standard facemesh without polylines."
    )
    parser.add_argument(
        "--x-solution",
        type=Path,
        default=Path("outputs/matrix_vis/real/imr_00228_win005_minus_win004_axis_x_full341_anchor_facebox/solution.npz"),
    )
    parser.add_argument(
        "--y-solution",
        type=Path,
        default=Path("outputs/matrix_vis/real/imr_00228_win005_minus_win004_axis_y_full341_anchor_facebox/solution.npz"),
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("/home/weizilin/code_reproduction/canonical_face/canonical_face_model.obj"),
    )
    parser.add_argument(
        "--layout-source",
        type=Path,
        default=Path("/home/weizilin/code_reproduction/corelation-lm/project/configs/extractors.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/matrix_vis/real_preview/imr_00228_win005_minus_win004_full341_anchor_facebox"),
    )
    args = parser.parse_args()

    x_solution = _load_solution(args.x_solution.resolve())
    y_solution = _load_solution(args.y_solution.resolve())
    x_time = x_solution["time_grid"].astype(np.float32)
    y_time = y_solution["time_grid"].astype(np.float32)
    if x_time.shape != y_time.shape or not np.allclose(x_time, y_time):
        raise ValueError("x and y solutions must share the same time grid")

    standard_mesh = _load_canonical_obj_vertices(args.mesh.resolve())
    normalized_mesh = _normalize_standard_facemesh(standard_mesh)
    subset_ids = resolve_subset_layout(
        subset_layout="face_regions_grouped",
        subset_layout_source=args.layout_source.resolve(),
        subset_layout_extractor_name="mediapipe",
    )
    common_ids = _ordered_common_ids(
        x_solution["point_ids"].astype(np.int64),
        y_solution["point_ids"].astype(np.int64),
        subset_ids,
    )
    subset_coordinates = _compose_subset_coordinates(
        x_solution=x_solution,
        y_solution=y_solution,
        subset_ids=common_ids,
    )

    anchor_points = normalized_mesh[np.asarray(DEFAULT_ANCHOR_POINT_IDS, dtype=np.int64), :2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = _save_frames(
        output_dir=output_dir / "frames",
        static_points=normalized_mesh[:, :2],
        subset_coordinates=subset_coordinates,
        anchor_points=anchor_points,
    )
    _save_snapshot(
        output_path=output_dir / "snapshot_last_frame.png",
        static_points=normalized_mesh[:, :2],
        subset_points=subset_coordinates[-1],
        anchor_points=anchor_points,
        title="normalized facemesh + full341 preview",
    )
    _save_gif(frame_paths, output_dir / "preview.gif")
    np.savez(
        output_dir / "subset_motion_preview.npz",
        point_ids=common_ids.astype(np.int64),
        time_grid=x_time.astype(np.float32),
        coordinates=subset_coordinates.astype(np.float32),
        anchor_point_ids=np.asarray(DEFAULT_ANCHOR_POINT_IDS, dtype=np.int64),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
