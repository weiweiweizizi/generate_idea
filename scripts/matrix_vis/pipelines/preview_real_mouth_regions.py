from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.matrix_vis.core.composition import compose_xy_coordinates
from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
from scripts.matrix_vis.io.load_mesh import _load_canonical_obj_vertices
from scripts.matrix_vis.io.save_results import ensure_output_dir, save_json


FACE_WIDTH_POINTS = (127, 356)
FACE_HEIGHT_POINTS = (10, 152)
DEFAULT_REGION_NAMES = ("around_mouth", "mouth")
DEFAULT_ANCHOR_POINT_ID = 14
DEFAULT_MESH_SOURCE = "/home/weizilin/code_reproduction/canonical_face/canonical_face_model.obj"
DEFAULT_LANDMARK_CONFIG = "scripts/matrix_vis/configs/landmarks/mediapipe_face_regions_full.yaml"

# ？
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


def _align_subset_to_anchor(
    *,
    coordinates: np.ndarray,
    subset_ids: np.ndarray,
    normalized_mesh: np.ndarray,
    anchor_point_id: int,
) -> np.ndarray:
    if anchor_point_id not in subset_ids.tolist():
        raise ValueError(f"Anchor point {anchor_point_id} is not present in the composed subset")
    aligned = coordinates.astype(np.float32, copy=True)
    anchor_local_idx = subset_ids.tolist().index(anchor_point_id)
    anchor_target = normalized_mesh[anchor_point_id, :2].astype(np.float32, copy=False)
    for frame_idx in range(aligned.shape[0]):
        shift = anchor_target - aligned[frame_idx, anchor_local_idx]
        aligned[frame_idx, :, 0] += shift[0]
        aligned[frame_idx, :, 1] += shift[1]
    return aligned


def _save_snapshot(
    *,
    output_path: Path,
    static_points: np.ndarray,
    subset_points: np.ndarray,
    anchor_point: np.ndarray,
    title: str,
) -> None:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(static_points[:, 0], static_points[:, 1], s=18, c="#d9d9d9")
    ax.scatter(subset_points[:, 0], subset_points[:, 1], s=24, c="#1f78b4")
    ax.scatter(anchor_point[0], anchor_point[1], s=48, c="#d94841")
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
    anchor_point: np.ndarray,
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
        ax.scatter(static_points[:, 0], static_points[:, 1], s=18, c="#d9d9d9")
        ax.scatter(subset_coordinates[frame_idx, :, 0], subset_coordinates[frame_idx, :, 1], s=24, c="#1f78b4")
        ax.scatter(anchor_point[0], anchor_point[1], s=48, c="#d94841")
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


def run_preview_real_mouth_regions(
    *,
    x_solution: str,
    y_solution: str,
    output_dir: str,
    mesh_source: str = DEFAULT_MESH_SOURCE,
    landmarks_config: str = DEFAULT_LANDMARK_CONFIG,
    anchor_point_id: int = DEFAULT_ANCHOR_POINT_ID,
    title: str = "normalized facemesh + mouth regions preview",
) -> dict:
    x_solution_path = Path(x_solution).expanduser().resolve()
    y_solution_path = Path(y_solution).expanduser().resolve()
    destination = ensure_output_dir(Path(output_dir).expanduser().resolve())

    x_payload = _load_solution(x_solution_path)
    y_payload = _load_solution(y_solution_path)

    standard_mesh = _load_canonical_obj_vertices(Path(mesh_source).expanduser().resolve())
    normalized_mesh = _normalize_standard_facemesh(standard_mesh)
    subset_ids = resolve_subset_layout(
        subset_layout="face_regions_grouped",
        subset_layout_source=Path(landmarks_config).expanduser().resolve(),
        subset_layout_extractor_name="mediapipe",
        subset_layout_region_names=list(DEFAULT_REGION_NAMES),
    )
    common_ids, time_grid, subset_coordinates = compose_xy_coordinates(
        x_solution=x_payload,
        y_solution=y_payload,
        preferred_point_ids=np.asarray(subset_ids, dtype=np.int64),
    )
    subset_coordinates = _align_subset_to_anchor(
        coordinates=subset_coordinates,
        subset_ids=common_ids,
        normalized_mesh=normalized_mesh,
        anchor_point_id=anchor_point_id,
    )

    frame_paths = _save_frames(
        output_dir=destination / "frames",
        static_points=normalized_mesh[:, :2],
        subset_coordinates=subset_coordinates,
        anchor_point=normalized_mesh[anchor_point_id, :2],
    )
    _save_snapshot(
        output_path=destination / "snapshot_last_frame.png",
        static_points=normalized_mesh[:, :2],
        subset_points=subset_coordinates[-1],
        anchor_point=normalized_mesh[anchor_point_id, :2],
        title=title,
    )
    _save_gif(frame_paths, destination / "preview.gif")
    np.savez(
        destination / "aligned_subset_motion.npz",
        point_ids=common_ids.astype(np.int64),
        time_grid=time_grid.astype(np.float32),
        coordinates=subset_coordinates.astype(np.float32),
        anchor_point_id=np.asarray(anchor_point_id, dtype=np.int64),
    )

    summary = {
        "x_solution": str(x_solution_path),
        "y_solution": str(y_solution_path),
        "output_dir": str(destination),
        "num_frames": int(subset_coordinates.shape[0]),
        "num_subset_points": int(common_ids.shape[0]),
        "anchor_point_id": int(anchor_point_id),
    }
    save_json(destination / "preview_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
