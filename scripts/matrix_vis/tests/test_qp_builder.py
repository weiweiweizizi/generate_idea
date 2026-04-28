from __future__ import annotations

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
        qp_backend="osqp",
    )

    bundle = build_axis_qp(
        subset_point_ids=np.asarray([10, 11, 12], dtype=np.int64),
        initial_positions=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        anchor_point_ids=np.asarray([10, 12], dtype=np.int64),
        observations=observations,
        solver_config=solver,
    )

    assert bundle.layout.shape == (3, 5)
    assert bundle.anchor_local_indices.tolist() == [0, 2]
    assert bundle.time_grid.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
