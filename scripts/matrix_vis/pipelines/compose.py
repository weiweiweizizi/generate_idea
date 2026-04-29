from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.matrix_vis.io.compose_config import load_compose_config
from scripts.matrix_vis.io.load_mesh import load_mesh
from scripts.matrix_vis.io.save_results import (
    ensure_output_dir,
    load_solution_npz,
    save_composed_motion_npz,
    save_json,
)
from scripts.matrix_vis.viz.mesh_animation import (
    save_gif_from_frames,
    save_motion_frames,
    save_motion_snapshot,
)


def run_motion_composition(config: str, output_dir: str | None = None) -> dict:
    cfg = load_compose_config(config)
    out_dir = ensure_output_dir(Path(output_dir).resolve()) if output_dir else ensure_output_dir(
        cfg.experiment.output_dir
    )

    mesh = load_mesh(cfg.mesh)
    x_solution = load_solution_npz(cfg.inputs.x_solution)
    y_solution = load_solution_npz(cfg.inputs.y_solution)

    x_ids = x_solution["point_ids"].astype(np.int64)
    y_ids = y_solution["point_ids"].astype(np.int64)
    common_ids = np.intersect1d(x_ids, y_ids)
    if common_ids.size == 0:
        raise ValueError("No overlapping point ids between x and y solutions")

    x_lookup = {int(point_id): idx for idx, point_id in enumerate(x_ids.tolist())}
    y_lookup = {int(point_id): idx for idx, point_id in enumerate(y_ids.tolist())}
    mesh_lookup = {int(point_id): idx for idx, point_id in enumerate(mesh.point_ids.tolist())}

    x_time = x_solution["time_grid"].astype(np.float32)
    y_time = y_solution["time_grid"].astype(np.float32)
    if x_time.shape != y_time.shape or not np.allclose(x_time, y_time):
        raise ValueError("x and y solutions must share the same time grid")

    subset_indices = np.asarray([mesh_lookup[int(point_id)] for point_id in common_ids.tolist()], dtype=np.int64)
    subset_mask = np.zeros(mesh.points.shape[0], dtype=bool)
    subset_mask[subset_indices] = True

    coordinates = np.repeat(mesh.points[None, :, :], x_time.shape[0], axis=0).astype(np.float32)
    for point_id in common_ids.tolist():
        mesh_idx = mesh_lookup[int(point_id)]
        x_idx = x_lookup[int(point_id)]
        y_idx = y_lookup[int(point_id)]
        coordinates[:, mesh_idx, 0] = x_solution["trajectory"][x_idx]
        coordinates[:, mesh_idx, 1] = y_solution["trajectory"][y_idx]

    save_motion_snapshot(
        output_path=out_dir / "motion_snapshot.png",
        static_points=mesh.points,
        animated_points=coordinates[-1, subset_indices],
        title=cfg.experiment.name,
    )

    frame_paths = []
    if cfg.export.save_animation_preview:
        frame_paths = save_motion_frames(
            output_dir=out_dir / "frames",
            static_points=mesh.points,
            coordinates=coordinates,
            subset_mask=subset_mask,
        )
        save_gif_from_frames(frame_paths, out_dir / "motion_preview.gif")

    if cfg.export.save_npz:
        save_composed_motion_npz(
            output_dir=out_dir,
            point_ids=mesh.point_ids,
            time_grid=x_time,
            coordinates=coordinates,
            subset_point_ids=common_ids,
        )

    summary = {
        "experiment_name": cfg.experiment.name,
        "output_dir": str(out_dir),
        "num_mesh_points": int(mesh.points.shape[0]),
        "num_subset_points": int(common_ids.shape[0]),
        "num_frames": int(coordinates.shape[0]),
        "saved_frame_count": int(len(frame_paths)),
        "subset_policy": cfg.subset_policy,
    }
    if cfg.export.save_json_summary:
        save_json(out_dir / "composed_summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
