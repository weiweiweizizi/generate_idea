#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys

import fire
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet_trainprobe.analysis.export_matrix_vis_patient import export as export_patient_bundle
from scripts.matrix_vis.core.projection import project_mesh_to_axis
from scripts.matrix_vis.core.types import MeshConfig
from scripts.matrix_vis.io.load_mesh import load_mesh
from scripts.matrix_vis.io.save_results import ensure_output_dir, save_json
from scripts.matrix_vis.pipelines.patient_sequence import (
    DEFAULT_MESH_SOURCE,
    build_default_projection,
    run_patient_sequence,
)
from scripts.matrix_vis.pipelines.preview_real_mouth_regions import (
    DEFAULT_LANDMARK_CONFIG,
    run_preview_real_mouth_regions,
)


def _load_solution(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _save_axis_solution(
    *,
    output_path: Path,
    point_ids: np.ndarray,
    time_grid: np.ndarray,
    trajectory: np.ndarray,
    initial_positions: np.ndarray,
    anchor_point_ids: np.ndarray,
    basis_matrix: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_point_ids = np.asarray(anchor_point_ids, dtype=np.int64)
    np.savez(
        output_path,
        point_ids=np.asarray(point_ids, dtype=np.int64),
        time_grid=np.asarray(time_grid, dtype=np.float32),
        initial_positions=np.asarray(initial_positions, dtype=np.float32),
        trajectory=np.asarray(trajectory, dtype=np.float32),
        anchor_point_ids=anchor_point_ids,
        anchor_point_id=anchor_point_ids[0],
        basis_matrix=np.asarray(basis_matrix, dtype=np.float32),
    )


def _write_static_axis_solution(
    *,
    reference_solution_path: Path,
    axis: str,
    matrix_size: int,
    output_path: Path,
    mesh_source: str,
) -> Path:
    reference = _load_solution(reference_solution_path)
    mesh = load_mesh(
        MeshConfig(
            source=Path(mesh_source).expanduser().resolve(),
            format="mediapipe_canonical_obj",
            dimension="3d",
            point_ids="auto",
            normalization_scope="face_regions",
        )
    )
    projection = project_mesh_to_axis(mesh, build_default_projection(axis=axis, matrix_size=matrix_size))
    point_ids = np.asarray(reference["point_ids"], dtype=np.int64)
    if projection.subset_point_ids.shape != point_ids.shape or not np.array_equal(
        projection.subset_point_ids, point_ids
    ):
        raise ValueError(f"Static {axis}-axis template point_ids do not match reconstructed solution")
    time_grid = np.asarray(reference["time_grid"], dtype=np.float32)
    initial_positions = np.asarray(projection.subset_positions, dtype=np.float32)
    trajectory = np.repeat(initial_positions[:, None], time_grid.shape[0], axis=1)
    basis_matrix = np.asarray(reference["basis_matrix"], dtype=np.float32)
    _save_axis_solution(
        output_path=output_path,
        point_ids=point_ids,
        time_grid=time_grid,
        trajectory=trajectory,
        initial_positions=initial_positions,
        anchor_point_ids=np.asarray(projection.anchor_point_ids, dtype=np.int64),
        basis_matrix=basis_matrix,
    )
    return output_path


def _load_sequence_solution(sequence_dir: Path) -> tuple[Path, dict[str, np.ndarray], dict]:
    solution_path = sequence_dir / "solution.npz"
    summary_path = sequence_dir / "sequence_summary.json"
    if not solution_path.exists():
        raise FileNotFoundError(solution_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return (
        solution_path,
        _load_solution(solution_path),
        json.loads(summary_path.read_text(encoding="utf-8")),
    )


def _load_window_solution(window_dir: str | Path) -> dict[str, np.ndarray]:
    return _load_solution(Path(window_dir).expanduser().resolve() / "solution.npz")


def _align_x_solution_to_y_sequence(
    *,
    x_sequence_summary: dict,
    y_sequence_summary: dict,
    output_path: Path,
) -> tuple[Path, dict]:
    x_window_rows = list(
        zip(
            [int(value) for value in x_sequence_summary["window_indices"]],
            [str(path) for path in x_sequence_summary["window_dirs"]],
        )
    )
    y_window_rows = list(
        zip(
            [int(value) for value in y_sequence_summary["window_indices"]],
            [str(path) for path in y_sequence_summary["window_dirs"]],
        )
    )
    x_window_map = {window_idx: window_dir for window_idx, window_dir in x_window_rows}
    if not x_window_rows:
        raise ValueError("x sequence must contain at least one valid window")

    first_x_solution = _load_window_solution(x_window_rows[0][1])
    point_ids = np.asarray(first_x_solution["point_ids"], dtype=np.int64)
    anchor_point_ids = np.asarray(first_x_solution["anchor_point_ids"], dtype=np.int64)
    initial_positions = np.asarray(first_x_solution["initial_positions"], dtype=np.float32)
    basis_matrix = np.asarray(first_x_solution["basis_matrix"], dtype=np.float32)

    stitched_trajectories: list[np.ndarray] = []
    stitched_time_grids: list[np.ndarray] = []
    held_from_window_indices: list[int | None] = []
    source_kinds: list[str] = []
    time_offset = 0.0
    last_real_solution: dict[str, np.ndarray] | None = None

    for window_idx, y_window_dir in y_window_rows:
        y_solution = _load_window_solution(y_window_dir)
        y_window_time = np.asarray(y_solution["time_grid"], dtype=np.float32)
        if window_idx in x_window_map:
            x_solution = _load_window_solution(x_window_map[window_idx])
            stitched_trajectory = np.asarray(x_solution["trajectory"], dtype=np.float32)
            last_real_solution = x_solution
            source_kinds.append("real_x_window")
            held_from_window_indices.append(window_idx)
        else:
            if last_real_solution is None:
                last_real_solution = first_x_solution
            last_frame = np.asarray(last_real_solution["trajectory"], dtype=np.float32)[:, -1]
            stitched_trajectory = np.repeat(last_frame[:, None], y_window_time.shape[0], axis=1)
            source_kinds.append("held_previous_x_window")
            held_from_window_indices.append(
                None if last_real_solution is first_x_solution and x_window_rows[0][0] > window_idx else None
            )
        stitched_trajectories.append(stitched_trajectory)
        stitched_time_grids.append(y_window_time + time_offset)
        time_offset = float(y_window_time[-1] + time_offset + 1.0)

    _save_axis_solution(
        output_path=output_path,
        point_ids=point_ids,
        time_grid=np.concatenate(stitched_time_grids, axis=0),
        trajectory=np.concatenate(stitched_trajectories, axis=1),
        initial_positions=initial_positions,
        anchor_point_ids=anchor_point_ids,
        basis_matrix=basis_matrix,
    )
    return output_path, {
        "y_window_indices": [window_idx for window_idx, _ in y_window_rows],
        "x_window_indices": [window_idx for window_idx, _ in x_window_rows],
        "source_kinds": source_kinds,
    }


def _run_preview_pair(
    *,
    x_solution_path: Path,
    y_solution_path: Path,
    output_dir: Path,
    anchor_point_ids: list[int],
    title: str,
    subset_layout_region_names: list[str] | None,
    mesh_source: str,
) -> dict:
    return run_preview_real_mouth_regions(
        x_solution=str(x_solution_path),
        y_solution=str(y_solution_path),
        output_dir=str(output_dir),
        mesh_source=mesh_source,
        landmarks_config=DEFAULT_LANDMARK_CONFIG,
        anchor_point_ids=anchor_point_ids,
        subset_layout_region_names=subset_layout_region_names,
        title=title,
    )


def run(
    *,
    subject: str,
    x_checkpoint_path: str,
    y_checkpoint_path: str,
    output_dir: str | None = None,
    mesh_source: str = DEFAULT_MESH_SOURCE,
    anchor_point_ids: str | list[int] | tuple[int, ...] = "33,263,10,175",
    batch_size: int = 8,
    max_observations: int | None = None,
    renormalize_observations: bool = True,
    enforce_nonnegative_targets: bool = True,
) -> dict:
    raw_anchor_ids = anchor_point_ids
    if isinstance(raw_anchor_ids, str):
        resolved_anchor_point_ids = [int(part.strip()) for part in raw_anchor_ids.split(",") if part.strip()]
    else:
        resolved_anchor_point_ids = [int(value) for value in raw_anchor_ids]
    if not resolved_anchor_point_ids:
        raise ValueError("anchor_point_ids must not be empty")

    x_checkpoint = Path(x_checkpoint_path).expanduser().resolve()
    y_checkpoint = Path(y_checkpoint_path).expanduser().resolve()

    export_y_summary = export_patient_bundle(
        checkpoint_path=str(y_checkpoint),
        subject=str(subject),
        batch_size=int(batch_size),
    )
    export_x_summary = export_patient_bundle(
        checkpoint_path=str(x_checkpoint),
        subject=str(subject),
        batch_size=int(batch_size),
    )

    dataset_name = str(export_y_summary.get("dataset_name") or export_x_summary.get("dataset_name") or "unknown")
    resolved_subject = str(export_y_summary.get("subject") or export_x_summary.get("subject") or subject)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path("outputs/matrix_vis/patient_dual_axis/disentanglenet_trainprobe") / f"{dataset_name}_{resolved_subject}"
    )
    ensure_output_dir(destination)

    y_bundle_path = Path(export_y_summary["bundle_path"]).resolve()
    x_bundle_path = Path(export_x_summary["bundle_path"]).resolve()

    y_sequence_dir = destination / "reconstructed_y"
    x_sequence_dir = destination / "reconstructed_x"
    y_sequence_summary = run_patient_sequence(
        patient_bundle_path=str(y_bundle_path),
        output_dir=str(y_sequence_dir),
        mesh_source=mesh_source,
        max_observations=max_observations,
        renormalize_observations=bool(renormalize_observations),
        enforce_nonnegative_targets=bool(enforce_nonnegative_targets),
    )
    x_sequence_summary = run_patient_sequence(
        patient_bundle_path=str(x_bundle_path),
        output_dir=str(x_sequence_dir),
        mesh_source=mesh_source,
        max_observations=max_observations,
        renormalize_observations=bool(renormalize_observations),
        enforce_nonnegative_targets=bool(enforce_nonnegative_targets),
    )

    y_solution_path, y_solution, _ = _load_sequence_solution(y_sequence_dir)
    x_solution_path, x_solution, _ = _load_sequence_solution(x_sequence_dir)

    fixed_x_dir = ensure_output_dir(destination / "fixed_x_reconstruct_y")
    static_x_solution_path = _write_static_axis_solution(
        reference_solution_path=y_solution_path,
        axis="x",
        matrix_size=int(y_sequence_summary["matrix_size"]),
        output_path=fixed_x_dir / "static_x_solution.npz",
        mesh_source=mesh_source,
    )
    fixed_x_preview = _run_preview_pair(
        x_solution_path=static_x_solution_path,
        y_solution_path=y_solution_path,
        output_dir=fixed_x_dir,
        anchor_point_ids=resolved_anchor_point_ids,
        title=f"{dataset_name}_{resolved_subject}: fixed x + reconstructed y",
        subset_layout_region_names=None,
        mesh_source=mesh_source,
    )

    fixed_y_dir = ensure_output_dir(destination / "fixed_y_reconstruct_x")
    static_y_solution_path = _write_static_axis_solution(
        reference_solution_path=x_solution_path,
        axis="y",
        matrix_size=int(x_sequence_summary["matrix_size"]),
        output_path=fixed_y_dir / "static_y_solution.npz",
        mesh_source=mesh_source,
    )
    fixed_y_preview = _run_preview_pair(
        x_solution_path=x_solution_path,
        y_solution_path=static_y_solution_path,
        output_dir=fixed_y_dir,
        anchor_point_ids=resolved_anchor_point_ids,
        title=f"{dataset_name}_{resolved_subject}: reconstructed x + fixed y",
        subset_layout_region_names=None,
        mesh_source=mesh_source,
    )

    reconstructed_xy_dir = ensure_output_dir(destination / "reconstructed_xy")
    aligned_x_solution_path = x_solution_path
    xy_alignment_summary = None
    x_window_indices = [int(value) for value in x_sequence_summary["window_indices"]]
    y_window_indices = [int(value) for value in y_sequence_summary["window_indices"]]
    if x_window_indices != y_window_indices:
        aligned_x_solution_path, xy_alignment_summary = _align_x_solution_to_y_sequence(
            x_sequence_summary=x_sequence_summary,
            y_sequence_summary=y_sequence_summary,
            output_path=reconstructed_xy_dir / "aligned_x_solution_on_y_timeline.npz",
        )
    reconstructed_xy_preview = _run_preview_pair(
        x_solution_path=aligned_x_solution_path,
        y_solution_path=y_solution_path,
        output_dir=reconstructed_xy_dir,
        anchor_point_ids=resolved_anchor_point_ids,
        title=f"{dataset_name}_{resolved_subject}: reconstructed x + reconstructed y",
        subset_layout_region_names=None,
        mesh_source=mesh_source,
    )

    summary = {
        "subject_input": str(subject),
        "subject": resolved_subject,
        "dataset_name": dataset_name,
        "anchor_point_ids": resolved_anchor_point_ids,
        "mesh_source": str(Path(mesh_source).expanduser().resolve()),
        "output_dir": str(destination),
        "max_observations": None if max_observations is None else int(max_observations),
        "renormalize_observations": bool(renormalize_observations),
        "enforce_nonnegative_targets": bool(enforce_nonnegative_targets),
        "x_checkpoint_path": str(x_checkpoint),
        "y_checkpoint_path": str(y_checkpoint),
        "x_bundle_path": str(x_bundle_path),
        "y_bundle_path": str(y_bundle_path),
        "x_sequence_dir": str(x_sequence_dir),
        "y_sequence_dir": str(y_sequence_dir),
        "fixed_x_reconstruct_y": fixed_x_preview,
        "fixed_y_reconstruct_x": fixed_y_preview,
        "reconstructed_xy": reconstructed_xy_preview,
        "reconstructed_xy_alignment": xy_alignment_summary,
    }
    save_json(destination / "dual_axis_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"run": run})
