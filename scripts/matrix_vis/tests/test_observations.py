from __future__ import annotations

import numpy as np

from scripts.matrix_vis.core.observations import basis_to_observation_table
from scripts.matrix_vis.core.types import BasisObservation


def test_basis_to_observation_table_uses_upper_triangle_and_global_point_ids() -> None:
    observation = BasisObservation(
        subset_point_ids=np.asarray([100, 101, 103], dtype=np.int64),
        basis_matrix=np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        ),
        value_semantics="mean_distance_delta",
    )

    table = basis_to_observation_table(observation).frame

    assert table[["i", "j"]].values.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert table[["point_id_i", "point_id_j"]].values.tolist() == [
        [100, 101],
        [100, 103],
        [101, 103],
    ]
    assert table["value"].tolist() == [1.0, 2.0, 3.0]
