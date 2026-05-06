from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
from scripts.matrix_vis.core.projection import project_mesh_to_axis
from scripts.matrix_vis.core.types import MeshConfig, ProjectionConfig
from scripts.matrix_vis.io.load_mesh import load_mesh
from scripts.matrix_vis.io.save_results import ensure_output_dir, load_solution_npz, save_json
from scripts.matrix_vis.pipelines.preview_real_mouth_regions import run_preview_real_mouth_regions


DEFAULT_MESH_SOURCE = "/home/weizilin/code_reproduction/canonical_face/canonical_face_model.obj"
DEFAULT_LANDMARK_CONFIG = "scripts/matrix_vis/configs/landmarks/mediapipe_face_regions_full.yaml"


def _build_y_projection() -> ProjectionConfig:
    subset_point_ids = resolve_subset_layout(
        subset_layout="face_regions_grouped",
        subset_layout_source=DEFAULT_LANDMARK_CONFIG,
        subset_layout_extractor_name="mediapipe",
        subset_layout_region_names=["around_mouth", "mouth"],
    )
    return ProjectionConfig(
        axis="y",
        source_axis_index=1,
        subset_point_ids=subset_point_ids,
        anchor_point_ids=(14,),
        subset_layout="face_regions_grouped",
        subset_layout_source=Path(DEFAULT_LANDMARK_CONFIG).resolve(),
        subset_layout_extractor_name="mediapipe",
        subset_layout_region_names=("around_mouth", "mouth"),
    )


def _save_axis_solution(
    *,
    output_path: Path,
    point_ids: np.ndarray,
    time_grid: np.ndarray,
    trajectory: np.ndarray,
    initial_positions: np.ndarray,
    anchor_point_ids: np.ndarray,
) -> None:
    np.savez(
        output_path,
        point_ids=np.asarray(point_ids, dtype=np.int64),
        time_grid=np.asarray(time_grid, dtype=np.float32),
        initial_positions=np.asarray(initial_positions, dtype=np.float32),
        trajectory=np.asarray(trajectory, dtype=np.float32),
        anchor_point_ids=np.asarray(anchor_point_ids, dtype=np.int64),
        anchor_point_id=np.asarray(anchor_point_ids, dtype=np.int64)[0],
    )


def compose_patient_static_y(
    sequence_dir: str,
    output_dir: str | None = None,
    mesh_source: str = DEFAULT_MESH_SOURCE,
) -> dict:
    sequence_path = Path(sequence_dir).expanduser().resolve()
    manifest = json.loads((sequence_path / "sequence_manifest.json").read_text(encoding="utf-8"))
    window_rows = manifest.get("windows", [])
    if not window_rows:
        raise ValueError(f"No window entries found in {sequence_path / 'sequence_manifest.json'}")

    mesh = load_mesh(
        MeshConfig(
            source=Path(mesh_source).expanduser().resolve(),
            format="mediapipe_canonical_obj",
            dimension="3d",
            point_ids="auto",
            normalization_scope="face_regions",
        )
    )
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else sequence_path / "real_preview_mouth_regions_anchor14"
    )
    ensure_output_dir(destination)

    y_projection = project_mesh_to_axis(
        mesh,
        _build_y_projection(),
    )
    stitched_x = []
    subset_ids = None
    anchor_point_ids = None
    initial_positions = None
    time_offset = 0.0
    stitched_time = []
    for row in window_rows:
        solution = load_solution_npz(Path(row["output_dir"]) / "solution.npz")
        point_ids = solution["point_ids"].astype(np.int64)
        if subset_ids is None:
            subset_ids = point_ids
            anchor_point_ids = solution["anchor_point_ids"].astype(np.int64)
            initial_positions = solution["initial_positions"].astype(np.float32)
        elif not np.array_equal(subset_ids, point_ids):
            raise ValueError("All window solutions must share the same point_ids for stitched preview")
        stitched_x.append(solution["trajectory"].astype(np.float32))
        window_time = solution["time_grid"].astype(np.float32)
        stitched_time.append(window_time + time_offset)
        time_offset = float(window_time[-1] + time_offset + 1.0)

    stitched_x_trajectory = np.concatenate(stitched_x, axis=1)
    stitched_time_grid = np.concatenate(stitched_time, axis=0)
    static_y_axis = y_projection.subset_positions.astype(np.float32)
    static_y_trajectory = np.repeat(static_y_axis[:, None], stitched_time_grid.shape[0], axis=1)

    x_solution_path = destination / "stitched_x_solution.npz"
    y_solution_path = destination / "static_y_solution.npz"
    _save_axis_solution(
        output_path=x_solution_path,
        point_ids=subset_ids,
        time_grid=stitched_time_grid,
        trajectory=stitched_x_trajectory,
        initial_positions=initial_positions,
        anchor_point_ids=anchor_point_ids,
    )
    _save_axis_solution(
        output_path=y_solution_path,
        point_ids=y_projection.subset_point_ids,
        time_grid=stitched_time_grid,
        trajectory=static_y_trajectory,
        initial_positions=static_y_axis,
        anchor_point_ids=y_projection.anchor_point_ids,
    )
    preview_summary = run_preview_real_mouth_regions(
        x_solution=str(x_solution_path),
        y_solution=str(y_solution_path),
        output_dir=str(destination),
        mesh_source=mesh_source,
        landmarks_config=DEFAULT_LANDMARK_CONFIG,
        anchor_point_id=14,
        title=sequence_path.name,
    )

    summary = {
        "sequence_dir": str(sequence_path),
        "output_dir": str(destination),
        "num_windows": len(window_rows),
        "num_frames": int(stitched_time_grid.shape[0]),
        "subset_point_count": int(subset_ids.shape[0]),
        "y_mode": "static_mesh_template",
        "x_solution": str(x_solution_path),
        "y_solution": str(y_solution_path),
        "preview_output_dir": preview_summary["output_dir"],
    }
    save_json(destination / "composed_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
