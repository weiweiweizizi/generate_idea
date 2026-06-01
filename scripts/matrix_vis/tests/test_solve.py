from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.matrix_vis.core.types import QPConfig
from scripts.matrix_vis.qp.builder import build_axis_qp
from scripts.matrix_vis.qp.solve import _build_linear_trajectory_reference
from scripts.matrix_vis.qp.solve import _compute_distance_signs
from scripts.matrix_vis.qp.solve import solve_axis_qp


def _build_bundle(*, max_displacement: float | None, geometry_topology_source: Path | None = None) -> object:
    observations = pd.DataFrame(
        [
            {"i": 0, "j": 1, "point_id_i": 10, "point_id_j": 11, "value": 0.15},
            {"i": 1, "j": 2, "point_id_i": 11, "point_id_j": 12, "value": 0.10},
            {"i": 0, "j": 2, "point_id_i": 10, "point_id_j": 12, "value": 0.20},
        ]
    )
    solver = QPConfig(
        num_time_steps=6,
        lambda_data=1.0,
        lambda_acc=5.0,
        lambda_vel=1.0,
        enforce_order=False,
        max_displacement=max_displacement,
        qp_backend="torch",
        lambda_laplacian=0.0,
        lambda_area_sign=0.0,
        area_barrier_margin=0.05,
        geometry_topology_source=geometry_topology_source,
    )
    return build_axis_qp(
        subset_point_ids=np.asarray([10, 11, 12], dtype=np.int64),
        initial_positions=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        orthogonal_static_positions=np.asarray([0.0, 0.4, 1.0], dtype=np.float32),
        dynamic_axis="x",
        anchor_point_ids=np.asarray([10], dtype=np.int64),
        observations=observations,
        solver_config=solver,
    )


def test_solve_axis_qp_returns_expected_shapes() -> None:
    bundle = _build_bundle(max_displacement=None)

    result = solve_axis_qp(bundle)

    assert result.trajectory.shape == (3, 6)
    assert result.diagnostics["status"] == "solved"
    assert result.diagnostics["solver_stats"]["solver_name"] == "torch"


def test_solve_axis_qp_respects_max_displacement() -> None:
    bundle = _build_bundle(max_displacement=0.05)

    result = solve_axis_qp(bundle)
    displacement = np.abs(result.trajectory - bundle.initial_positions[:, None])

    assert np.max(displacement) <= 0.051


def test_solve_axis_qp_accepts_geometry_priors(tmp_path: Path) -> None:
    topology = tmp_path / "topology.py"
    topology.write_text("FACEMESH_TESSELATION = frozenset([(10, 11, 12)])\n", encoding="utf-8")
    bundle = _build_bundle(max_displacement=0.1, geometry_topology_source=topology)
    solver_config = QPConfig(
        num_time_steps=bundle.solver_config.num_time_steps,
        lambda_data=bundle.solver_config.lambda_data,
        lambda_acc=bundle.solver_config.lambda_acc,
        lambda_vel=bundle.solver_config.lambda_vel,
        enforce_order=bundle.solver_config.enforce_order,
        max_displacement=bundle.solver_config.max_displacement,
        qp_backend=bundle.solver_config.qp_backend,
        lambda_laplacian=1.0,
        lambda_area_sign=1.0,
        area_barrier_margin=0.05,
        geometry_topology_source=topology,
    )
    bundle = build_axis_qp(
        subset_point_ids=bundle.subset_point_ids,
        initial_positions=bundle.initial_positions,
        orthogonal_static_positions=bundle.orthogonal_static_positions,
        dynamic_axis="x",
        anchor_point_ids=bundle.anchor_point_ids,
        observations=bundle.observations,
        solver_config=solver_config,
    )

    result = solve_axis_qp(bundle)

    assert result.diagnostics["solver_stats"]["num_laplacian_edges"] == 3
    assert result.diagnostics["solver_stats"]["num_area_triangles"] == 1


def test_compute_distance_signs_prefers_initial_prior_for_small_flips() -> None:
    bundle = _build_bundle(max_displacement=None)
    bundle = build_axis_qp(
        subset_point_ids=bundle.subset_point_ids,
        initial_positions=np.asarray([2.0, 1.0, 0.5], dtype=np.float32),
        orthogonal_static_positions=bundle.orthogonal_static_positions,
        dynamic_axis="x",
        anchor_point_ids=bundle.anchor_point_ids,
        observations=bundle.observations,
        solver_config=bundle.solver_config,
    )

    reference = np.asarray(
        [
            [0.0, 0.01],
            [0.0, 0.01],
            [0.0, 0.01],
        ],
        dtype=np.float32,
    )
    signs = _compute_distance_signs(bundle=bundle, reference_trajectory=reference)

    assert np.allclose(signs, np.asarray([[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32))


def test_build_linear_trajectory_reference_interpolates_from_start_to_end() -> None:
    reference = _build_linear_trajectory_reference(
        initial_positions=np.asarray([0.0, 1.0], dtype=np.float32),
        endpoint_positions=np.asarray([1.0, 3.0], dtype=np.float32),
        num_time_steps=5,
    )

    assert np.allclose(
        reference,
        np.asarray(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [1.0, 1.5, 2.0, 2.5, 3.0],
            ],
            dtype=np.float32,
        ),
    )


def test_solve_axis_qp_records_trajectory_tether_diagnostics() -> None:
    bundle = _build_bundle(max_displacement=0.05)
    solver_config = QPConfig(
        num_time_steps=bundle.solver_config.num_time_steps,
        lambda_data=bundle.solver_config.lambda_data,
        lambda_acc=bundle.solver_config.lambda_acc,
        lambda_vel=bundle.solver_config.lambda_vel,
        enforce_order=bundle.solver_config.enforce_order,
        max_displacement=bundle.solver_config.max_displacement,
        qp_backend=bundle.solver_config.qp_backend,
        lambda_laplacian=bundle.solver_config.lambda_laplacian,
        lambda_area_sign=bundle.solver_config.lambda_area_sign,
        area_barrier_margin=bundle.solver_config.area_barrier_margin,
        lambda_trajectory_tether=0.05,
        geometry_topology_source=bundle.solver_config.geometry_topology_source,
    )
    bundle = build_axis_qp(
        subset_point_ids=bundle.subset_point_ids,
        initial_positions=bundle.initial_positions,
        orthogonal_static_positions=bundle.orthogonal_static_positions,
        dynamic_axis="x",
        anchor_point_ids=bundle.anchor_point_ids,
        observations=bundle.observations,
        solver_config=solver_config,
    )

    result = solve_axis_qp(bundle)

    assert "trajectory_tether" in result.diagnostics
    assert result.diagnostics["trajectory_tether"]["enabled"] is True
