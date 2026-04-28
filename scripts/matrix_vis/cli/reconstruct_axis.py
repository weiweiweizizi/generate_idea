#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
import sys

import fire
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.core.observations import basis_to_observation_table
from scripts.matrix_vis.core.projection import project_mesh_to_axis
from scripts.matrix_vis.io.config import load_config
from scripts.matrix_vis.io.load_basis import load_basis_observation
from scripts.matrix_vis.io.load_mesh import load_mesh
from scripts.matrix_vis.io.save_results import (
    ensure_output_dir,
    save_json,
    save_observations,
    save_projected_mesh,
    save_resolved_config,
    save_solution_npz,
)
from scripts.matrix_vis.qp.builder import build_axis_qp
from scripts.matrix_vis.qp.solve import solve_axis_qp
from scripts.matrix_vis.viz.axis_plots import (
    save_axis_ground_truth_comparison_plot,
    save_axis_trajectory_plot,
)


def _load_ground_truth_axis_trajectory(
    *,
    mesh,
    projection,
    basis_source: Path,
    target_time_grid: np.ndarray,
) -> np.ndarray | None:
    trajectory_path = basis_source.parent / "trajectory_2d.npy"
    if not trajectory_path.exists():
        return None

    trajectory_2d = np.load(trajectory_path).astype(np.float32, copy=False)
    if trajectory_2d.ndim != 3:
        raise ValueError(f"Expected ground truth trajectory with shape [T, N, D], got {trajectory_2d.shape}")
    if trajectory_2d.shape[1] != mesh.points.shape[0]:
        raise ValueError(
            "Ground truth trajectory point count does not match mesh: "
            f"{trajectory_2d.shape[1]} vs {mesh.points.shape[0]}"
        )

    axis_positions = trajectory_2d[:, :, projection.source_axis_index]
    point_id_to_index = {int(point_id): idx for idx, point_id in enumerate(mesh.point_ids.tolist())}
    subset_indices = [point_id_to_index[int(point_id)] for point_id in projection.subset_point_ids.tolist()]
    subset_axis = axis_positions[:, subset_indices].T

    source_time = np.linspace(0.0, 1.0, trajectory_2d.shape[0], dtype=np.float32)
    if trajectory_2d.shape[0] == target_time_grid.shape[0] and np.allclose(source_time, target_time_grid):
        return subset_axis.astype(np.float32, copy=False)

    resampled = np.empty((subset_axis.shape[0], target_time_grid.shape[0]), dtype=np.float32)
    for idx in range(subset_axis.shape[0]):
        resampled[idx] = np.interp(target_time_grid, source_time, subset_axis[idx]).astype(np.float32)
    return resampled

def reconstruct(
    config: str,
    axis: str | None = None,
    output_dir: str | None = None,
) -> dict:
    cfg = load_config(config)
    if axis is not None and axis != cfg.projection.axis:
        raise ValueError(
            f"Axis override {axis!r} does not match config projection.axis {cfg.projection.axis!r}"
        )

    out_dir = ensure_output_dir(Path(output_dir).resolve()) if output_dir else ensure_output_dir(
        cfg.experiment.output_dir
    )
    save_resolved_config(cfg, out_dir)

    mesh = load_mesh(cfg.mesh)
    projection = project_mesh_to_axis(mesh, cfg.projection)
    basis_observation = load_basis_observation(
        cfg.basis,
        subset_point_ids=projection.subset_point_ids,
    )
    observation_table = basis_to_observation_table(basis_observation).frame

    bundle = build_axis_qp(
        subset_point_ids=projection.subset_point_ids,
        initial_positions=projection.subset_positions,
        anchor_point_ids=projection.anchor_point_ids,
        observations=observation_table,
        solver_config=cfg.solver,
    )
    solve_result = solve_axis_qp(bundle)

    if cfg.export.save_projected_mesh:
        save_projected_mesh(projection, out_dir)
    save_observations(observation_table, out_dir)
    if cfg.export.save_npz:
        save_solution_npz(
            output_dir=out_dir,
            point_ids=projection.subset_point_ids,
            time_grid=bundle.time_grid,
            initial_positions=projection.subset_positions,
            trajectory=solve_result.trajectory,
            anchor_point_ids=projection.anchor_point_ids,
            basis_observation=basis_observation,
        )

    plot_warning = None
    comparison_plot_warning = None
    if cfg.export.save_axis_plot:
        plot_warning = save_axis_trajectory_plot(
            output_dir=out_dir,
            time_grid=bundle.time_grid,
            trajectory=solve_result.trajectory,
            point_ids=projection.subset_point_ids,
            axis=cfg.projection.axis,
        )
        ground_truth = _load_ground_truth_axis_trajectory(
            mesh=mesh,
            projection=projection,
            basis_source=cfg.basis.source,
            target_time_grid=bundle.time_grid,
        )
        if ground_truth is not None:
            comparison_plot_warning = save_axis_ground_truth_comparison_plot(
                output_dir=out_dir,
                time_grid=bundle.time_grid,
                reconstructed=solve_result.trajectory,
                ground_truth=ground_truth,
                point_ids=projection.subset_point_ids,
                axis=cfg.projection.axis,
            )
            comparison_metrics = {
                "ground_truth_rmse": float(np.sqrt(np.mean((solve_result.trajectory - ground_truth) ** 2))),
                "ground_truth_mae": float(np.mean(np.abs(solve_result.trajectory - ground_truth))),
                "ground_truth_max_abs_error": float(
                    np.max(np.abs(solve_result.trajectory - ground_truth))
                ),
            }
        else:
            comparison_metrics = None
    else:
        comparison_metrics = None

    summary = {
        "experiment_name": cfg.experiment.name,
        "output_dir": str(out_dir),
        "axis": cfg.projection.axis,
        "num_subset_points": int(projection.subset_point_ids.shape[0]),
        "num_pairwise_observations": int(observation_table.shape[0]),
        "anchor_point_ids": projection.anchor_point_ids.astype(int).tolist(),
        "anchor_point_id": int(projection.anchor_point_id),
        "plot_warning": plot_warning,
        "comparison_plot_warning": comparison_plot_warning,
        "comparison_metrics": comparison_metrics,
        "diagnostics": solve_result.diagnostics,
    }
    if cfg.export.save_json_summary:
        save_json(out_dir / "summary.json", summary)
    if cfg.export.save_qp_diagnostics:
        save_json(out_dir / "qp_diagnostics.json", solve_result.diagnostics)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"reconstruct": reconstruct})
