from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from scripts.matrix_vis.core.types import QPConfig
from scripts.matrix_vis.qp.builder import build_axis_qp


def test_build_axis_qp_returns_expected_layout() -> None:
    observations = pd.DataFrame(
        [
            {"i": 0, "j": 1, "point_id_i": 10, "point_id_j": 11, "value": 0.1},
            {"i": 1, "j": 2, "point_id_i": 11, "point_id_j": 12, "value": 0.2},
        ]
    )
    solver = QPConfig(
        num_time_steps=5,
        lambda_data=1.0,
        lambda_acc=10.0,
        lambda_vel=1.0,
        enforce_order=True,
        max_displacement=1.0,
        qp_backend="torch",
        lambda_laplacian=0.0,
        lambda_area_sign=0.0,
        area_barrier_margin=0.05,
    )

    bundle = build_axis_qp(
        subset_point_ids=np.asarray([10, 11, 12], dtype=np.int64),
        initial_positions=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        orthogonal_static_positions=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        dynamic_axis="x",
        anchor_point_ids=np.asarray([10, 12], dtype=np.int64),
        observations=observations,
        solver_config=solver,
    )

    assert bundle.layout.shape == (3, 5)
    assert bundle.anchor_local_indices.tolist() == [0, 2]
    assert bundle.time_grid.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_build_axis_qp_accepts_explicit_target_distance_matrix() -> None:
    observations = pd.DataFrame(
        [
            {"i": 0, "j": 1, "point_id_i": 10, "point_id_j": 11, "value": 0.1},
            {"i": 1, "j": 2, "point_id_i": 11, "point_id_j": 12, "value": 0.2},
        ]
    )
    solver = QPConfig(
        num_time_steps=5,
        lambda_data=1.0,
        lambda_acc=10.0,
        lambda_vel=1.0,
        enforce_order=False,
        max_displacement=None,
        qp_backend="torch",
        lambda_laplacian=0.0,
        lambda_area_sign=0.0,
        area_barrier_margin=0.05,
    )

    target_distance_matrix = np.asarray(
        [
            [0.0, 1.4, 2.3],
            [1.4, 0.0, 0.8],
            [2.3, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    bundle = build_axis_qp(
        subset_point_ids=np.asarray([10, 11, 12], dtype=np.int64),
        initial_positions=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        orthogonal_static_positions=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        dynamic_axis="x",
        anchor_point_ids=np.asarray([10], dtype=np.int64),
        observations=observations,
        solver_config=solver,
        target_distance_matrix=target_distance_matrix,
    )

    assert np.allclose(bundle.observation_targets, np.asarray([1.4, 0.8], dtype=np.float32))
    assert bundle.target_distance_matrix is not None


def test_build_axis_qp_sets_distance_sign_prior_from_initial_positions() -> None:
    observations = pd.DataFrame(
        [
            {"i": 0, "j": 1, "point_id_i": 10, "point_id_j": 11, "value": 0.1},
            {"i": 1, "j": 2, "point_id_i": 11, "point_id_j": 12, "value": 0.2},
            {"i": 0, "j": 2, "point_id_i": 10, "point_id_j": 12, "value": 0.3},
        ]
    )
    solver = QPConfig(
        num_time_steps=5,
        lambda_data=1.0,
        lambda_acc=10.0,
        lambda_vel=1.0,
        enforce_order=False,
        max_displacement=None,
        qp_backend="torch",
        lambda_laplacian=0.0,
        lambda_area_sign=0.0,
        area_barrier_margin=0.05,
    )

    bundle = build_axis_qp(
        subset_point_ids=np.asarray([10, 11, 12], dtype=np.int64),
        initial_positions=np.asarray([2.0, 1.0, 1.0], dtype=np.float32),
        orthogonal_static_positions=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        dynamic_axis="x",
        anchor_point_ids=np.asarray([10], dtype=np.int64),
        observations=observations,
        solver_config=solver,
    )

    assert bundle.distance_sign_prior is not None
    assert np.allclose(bundle.distance_sign_prior, np.asarray([-1.0, 1.0, -1.0], dtype=np.float32))


def test_build_axis_qp_builds_geometry_priors_from_subset_topology(tmp_path: Path) -> None:
    topology = tmp_path / "topology.py"
    topology.write_text("FACEMESH_TESSELATION = frozenset([(10, 11, 12)])\n", encoding="utf-8")
    observations = pd.DataFrame(
        [{"i": 0, "j": 1, "point_id_i": 10, "point_id_j": 11, "value": 0.1}]
    )
    solver = QPConfig(
        num_time_steps=5,
        lambda_data=1.0,
        lambda_acc=0.0,
        lambda_vel=0.0,
        enforce_order=False,
        max_displacement=None,
        qp_backend="torch",
        lambda_laplacian=2.0,
        lambda_area_sign=1.0,
        area_barrier_margin=0.05,
        geometry_topology_source=topology,
    )

    bundle = build_axis_qp(
        subset_point_ids=np.asarray([10, 11, 12], dtype=np.int64),
        initial_positions=np.asarray([0.0, 1.0, 0.2], dtype=np.float32),
        orthogonal_static_positions=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        dynamic_axis="x",
        anchor_point_ids=np.asarray([10], dtype=np.int64),
        observations=observations,
        solver_config=solver,
    )

    assert bundle.laplacian_edge_indices.shape == (3, 2)
    assert bundle.area_triangle_indices.shape == (1, 3)
    assert bundle.area_reference_signs.tolist() == [1.0]
    assert bundle.reg_p.nnz > 0
