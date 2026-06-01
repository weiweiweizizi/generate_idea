from __future__ import annotations

import numpy as np

from scripts.matrix_vis.core.observations import basis_to_observation_table
from scripts.matrix_vis.core.types import BasisConfig, BasisObservation
from scripts.matrix_vis.io.load_basis import load_basis_observation


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


def test_load_basis_observation_supports_window_diff_sources(tmp_path) -> None:
    prev = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    next_ = np.asarray([[0.0, 1.5], [1.5, 0.0]], dtype=np.float32)
    prev_path = tmp_path / "prev.npy"
    next_path = tmp_path / "next.npy"
    np.save(prev_path, prev)
    np.save(next_path, next_)

    observation = load_basis_observation(
        BasisConfig(
            source=None,
            prev_source=prev_path,
            next_source=next_path,
            basis_index=0,
            matrix_shape="square",
            value_semantics="mean_distance_delta",
        ),
        subset_point_ids=np.asarray([10, 11], dtype=np.int64),
    )

    assert observation.subset_point_ids.tolist() == [10, 11]
    assert observation.basis_matrix.tolist() == [[0.0, 0.5], [0.5, 0.0]]


def test_load_basis_observation_crops_full_matrix_by_layout(tmp_path) -> None:
    full = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 0.0, 5.0, 6.0, 7.0],
            [2.0, 5.0, 0.0, 8.0, 9.0],
            [3.0, 6.0, 8.0, 0.0, 10.0],
            [4.0, 7.0, 9.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )
    prev_path = tmp_path / "prev.npy"
    next_path = tmp_path / "next.npy"
    np.save(prev_path, np.zeros_like(full))
    np.save(next_path, full)
    layout_path = tmp_path / "layout.yaml"
    layout_path.write_text(
        """
mediapipe:
  face_regions:
    forehead: [1, 2]
    mouth: [3, 4, 5]
  symmetric_pairs: []
""".strip(),
        encoding="utf-8",
    )

    observation = load_basis_observation(
        BasisConfig(
            source=None,
            prev_source=prev_path,
            next_source=next_path,
            basis_index=0,
            matrix_shape="square",
            value_semantics="mean_distance_delta",
            matrix_layout="face_regions_grouped",
            matrix_layout_source=layout_path,
            matrix_layout_extractor_name="mediapipe",
            matrix_layout_region_names=("mouth",),
        ),
        subset_point_ids=np.asarray([3, 4, 5], dtype=np.int64),
    )

    assert observation.subset_point_ids.tolist() == [3, 4, 5]
    assert observation.basis_matrix.tolist() == [
        [0.0, 8.0, 9.0],
        [8.0, 0.0, 10.0],
        [9.0, 10.0, 0.0],
    ]
