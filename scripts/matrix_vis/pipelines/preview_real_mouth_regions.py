from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.matrix_vis.core.composition import compose_xy_coordinates
from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
from scripts.matrix_vis.core.projection import project_mesh_to_axis
from scripts.matrix_vis.core.types import MeshConfig, ProjectionConfig
from scripts.matrix_vis.io.load_patient_reference import load_patient_landmark_points
from scripts.matrix_vis.io.load_mesh import _load_canonical_obj_vertices, load_mesh
from scripts.matrix_vis.io.save_results import ensure_output_dir, save_json


FACE_WIDTH_POINTS = (127, 356)
FACE_HEIGHT_POINTS = (10, 152)
DEFAULT_REGION_NAMES = ("around_mouth", "mouth")
DEFAULT_ANCHOR_POINT_ID = 205
DEFAULT_ANCHOR_POINT_IDS = (205, 425, 200)
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


def _build_static_y_payload(
    *,
    mesh_source: Path,
    landmarks_config: Path,
    anchor_point_ids: tuple[int, ...],
    subset_layout_region_names: tuple[str, ...] | None,
    num_time_steps: int,
    time_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """从 mesh 投影生成静止 y 轴 solution 负载。

    当 static_y=True 时替代从文件加载 y_solution：
    y 轨迹在所有时间步上保持初始位置不变。
    """
    mesh = load_mesh(
        MeshConfig(
            source=mesh_source.resolve(),
            format="mediapipe_canonical_obj",
            dimension="3d",
            point_ids="auto",
            normalization_scope="face_regions",
        )
    )
    subset_ids = resolve_subset_layout(
        subset_layout="face_regions_grouped",
        subset_layout_source=landmarks_config.resolve(),
        subset_layout_extractor_name="mediapipe",
        subset_layout_region_names=(
            list(subset_layout_region_names) if subset_layout_region_names is not None else None
        ),
    )
    proj = project_mesh_to_axis(
        mesh,
        ProjectionConfig(
            axis="y",
            source_axis_index=1,
            subset_point_ids=subset_ids,
            anchor_point_ids=anchor_point_ids,
        ),
    )
    trajectory = np.tile(proj.subset_positions[:, None], (1, num_time_steps))
    time_grid = (
        np.asarray(time_grid, dtype=np.float32)
        if time_grid is not None
        else np.linspace(0.0, 1.0, num_time_steps, dtype=np.float32)
    )
    return {
        "point_ids": np.asarray(proj.subset_point_ids, dtype=np.int64),
        "time_grid": time_grid,
        "initial_positions": np.asarray(proj.subset_positions, dtype=np.float32),
        "trajectory": trajectory.astype(np.float32),
        "anchor_point_ids": np.asarray(proj.anchor_point_ids, dtype=np.int64),
        "anchor_point_id": np.asarray(proj.anchor_point_id, dtype=np.int64),
    }


def _align_subset_to_anchor(
    *,
    coordinates: np.ndarray,
    subset_ids: np.ndarray,
    reference_points: np.ndarray,
    anchor_point_ids: tuple[int, ...],
) -> np.ndarray:
    if not anchor_point_ids:
        raise ValueError("anchor_point_ids must be non-empty")
    subset_id_list = subset_ids.tolist()
    missing_anchor_ids = [anchor_point_id for anchor_point_id in anchor_point_ids if anchor_point_id not in subset_id_list]
    if missing_anchor_ids:
        raise ValueError(f"Anchor points {missing_anchor_ids} are not present in the composed subset")
    if np.max(np.asarray(anchor_point_ids, dtype=np.int64)) >= reference_points.shape[0]:
        raise ValueError(
            f"Anchor point ids exceed reference point count: max id {max(anchor_point_ids)}, point count {reference_points.shape[0]}"
        )
    aligned = coordinates.astype(np.float32, copy=True)
    anchor_local_indices = [subset_id_list.index(anchor_point_id) for anchor_point_id in anchor_point_ids]
    anchor_target = reference_points[np.asarray(anchor_point_ids, dtype=np.int64), :2].astype(np.float32, copy=False).mean(axis=0)
    for frame_idx in range(aligned.shape[0]):
        anchor_current = aligned[frame_idx, anchor_local_indices].mean(axis=0)
        shift = anchor_target - anchor_current
        aligned[frame_idx, :, 0] += shift[0]
        aligned[frame_idx, :, 1] += shift[1]
    return aligned


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
    ax.scatter(static_points[:, 0], static_points[:, 1], s=18, c="#d9d9d9")
    ax.scatter(subset_points[:, 0], subset_points[:, 1], s=24, c="#1f78b4")
    ax.scatter(anchor_points[:, 0], anchor_points[:, 1], s=48, c="#d94841")
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
        ax.scatter(static_points[:, 0], static_points[:, 1], s=18, c="#d9d9d9")
        ax.scatter(subset_coordinates[frame_idx, :, 0], subset_coordinates[frame_idx, :, 1], s=24, c="#1f78b4")
        ax.scatter(anchor_points[:, 0], anchor_points[:, 1], s=48, c="#d94841")
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
    y_solution: str | None = None,
    output_dir: str,
    mesh_source: str = DEFAULT_MESH_SOURCE,
    landmarks_config: str = DEFAULT_LANDMARK_CONFIG,
    anchor_point_id: int = DEFAULT_ANCHOR_POINT_ID,
    anchor_point_ids: list[int] | tuple[int, ...] | None = None,
    subset_layout_region_names: list[str] | tuple[str, ...] | None = DEFAULT_REGION_NAMES,
    title: str = "normalized facemesh + mouth regions preview",
    static_y: bool = False,
    align_to_anchor: bool = True,
    background_points_source: str | None = None,
) -> dict:
    x_solution_path = Path(x_solution).expanduser().resolve()
    destination = ensure_output_dir(Path(output_dir).expanduser().resolve())

    resolved_anchor_point_ids = tuple(int(point_id) for point_id in (anchor_point_ids or (anchor_point_id,)))

    x_payload = _load_solution(x_solution_path)
    if static_y or y_solution is None:
        x_time_grid = x_payload["time_grid"]
        num_time_steps = int(x_time_grid.shape[0])
        y_payload = _build_static_y_payload(
            mesh_source=Path(mesh_source).expanduser().resolve(),
            landmarks_config=Path(landmarks_config).expanduser().resolve(),
            anchor_point_ids=resolved_anchor_point_ids,
            subset_layout_region_names=(
                tuple(subset_layout_region_names) if subset_layout_region_names is not None else None
            ),
            num_time_steps=num_time_steps,
            time_grid=x_time_grid,
        )
        y_solution_path = None
    else:
        y_solution_path = Path(y_solution).expanduser().resolve()
        y_payload = _load_solution(y_solution_path)

    standard_mesh = _load_canonical_obj_vertices(Path(mesh_source).expanduser().resolve())
    normalized_mesh = _normalize_standard_facemesh(standard_mesh)
    if background_points_source is None:
        background_points = normalized_mesh[:, :2].astype(np.float32, copy=False)
        background_mode = "standard_template"
    else:
        background_points = load_patient_landmark_points(background_points_source).astype(np.float32, copy=False)
        if background_points.shape[1] > 2:
            background_points = background_points[:, :2]
        background_mode = "patient_initial_landmarks"
    subset_ids = resolve_subset_layout(
        subset_layout="face_regions_grouped",
        subset_layout_source=Path(landmarks_config).expanduser().resolve(),
        subset_layout_extractor_name="mediapipe",
        subset_layout_region_names=(
            list(subset_layout_region_names) if subset_layout_region_names is not None else None
        ),
    )
    common_ids, time_grid, subset_coordinates = compose_xy_coordinates(
        x_solution=x_payload,
        y_solution=y_payload,
        preferred_point_ids=np.asarray(subset_ids, dtype=np.int64),
    )
    if align_to_anchor:
        subset_coordinates = _align_subset_to_anchor(
            coordinates=subset_coordinates,
            subset_ids=common_ids,
            reference_points=background_points,
            anchor_point_ids=resolved_anchor_point_ids,
        )
    anchor_points = background_points[np.asarray(resolved_anchor_point_ids, dtype=np.int64), :2]

    frame_paths = _save_frames(
        output_dir=destination / "frames",
        static_points=background_points,
        subset_coordinates=subset_coordinates,
        anchor_points=anchor_points,
    )
    _save_snapshot(
        output_path=destination / "snapshot_last_frame.png",
        static_points=background_points,
        subset_points=subset_coordinates[-1],
        anchor_points=anchor_points,
        title=title,
    )
    _save_gif(frame_paths, destination / "preview.gif")
    np.savez(
        destination / "aligned_subset_motion.npz",
        point_ids=common_ids.astype(np.int64),
        time_grid=time_grid.astype(np.float32),
        coordinates=subset_coordinates.astype(np.float32),
        anchor_point_ids=np.asarray(resolved_anchor_point_ids, dtype=np.int64),
        anchor_point_id=np.asarray(resolved_anchor_point_ids[0], dtype=np.int64),
    )

    summary = {
        "x_solution": str(x_solution_path),
        "y_solution": str(y_solution_path) if y_solution_path is not None else None,
        "output_dir": str(destination),
        "num_frames": int(subset_coordinates.shape[0]),
        "num_subset_points": int(common_ids.shape[0]),
        "anchor_point_ids": list(resolved_anchor_point_ids),
        "anchor_point_id": int(resolved_anchor_point_ids[0]),
        "subset_layout_region_names": list(subset_layout_region_names) if subset_layout_region_names is not None else None,
        "y_mode": "static" if static_y else "file",
        "align_to_anchor": bool(align_to_anchor),
        "background_mode": background_mode,
        "background_points_source": str(Path(background_points_source).expanduser().resolve()) if background_points_source is not None else None,
    }
    save_json(destination / "preview_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
