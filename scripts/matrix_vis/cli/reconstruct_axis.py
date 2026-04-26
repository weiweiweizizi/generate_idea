#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
import sys

import fire

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
from scripts.matrix_vis.viz.axis_plots import save_axis_trajectory_plot


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
        anchor_point_id=projection.anchor_point_id,
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
            anchor_point_id=projection.anchor_point_id,
            basis_observation=basis_observation,
        )

    plot_warning = None
    if cfg.export.save_axis_plot:
        plot_warning = save_axis_trajectory_plot(
            output_dir=out_dir,
            time_grid=bundle.time_grid,
            trajectory=solve_result.trajectory,
            point_ids=projection.subset_point_ids,
            axis=cfg.projection.axis,
        )

    summary = {
        "experiment_name": cfg.experiment.name,
        "output_dir": str(out_dir),
        "axis": cfg.projection.axis,
        "num_subset_points": int(projection.subset_point_ids.shape[0]),
        "num_pairwise_observations": int(observation_table.shape[0]),
        "anchor_point_id": int(projection.anchor_point_id),
        "plot_warning": plot_warning,
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
