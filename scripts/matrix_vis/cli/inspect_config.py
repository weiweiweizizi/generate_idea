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


def inspect(config: str) -> dict:
    cfg = load_config(config)
    mesh = load_mesh(cfg.mesh)
    projection = project_mesh_to_axis(mesh, cfg.projection)
    basis_observation = load_basis_observation(
        cfg.basis,
        subset_point_ids=projection.subset_point_ids,
    )
    observation_table = basis_to_observation_table(basis_observation).frame

    summary = {
        "config_path": str(cfg.config_path),
        "experiment_name": cfg.experiment.name,
        "mesh_path": str(cfg.mesh.source),
        "mesh_dimension": cfg.mesh.dimension,
        "num_mesh_points": int(mesh.points.shape[0]),
        "projection_axis": cfg.projection.axis,
        "source_axis_index": int(cfg.projection.source_axis_index),
        "subset_point_ids": projection.subset_point_ids.tolist(),
        "num_subset_points": int(projection.subset_point_ids.shape[0]),
        "anchor_point_ids": projection.anchor_point_ids.astype(int).tolist(),
        "anchor_point_id": int(projection.anchor_point_id),
        "basis_path": str(cfg.basis.source),
        "basis_shape": list(basis_observation.basis_matrix.shape),
        "num_pairwise_observations": int(observation_table.shape[0]),
        "qp_backend": cfg.solver.qp_backend,
        "num_time_steps": int(cfg.solver.num_time_steps),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"inspect": inspect})
