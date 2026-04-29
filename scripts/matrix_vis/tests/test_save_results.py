from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.matrix_vis.core.types import BasisObservation
from scripts.matrix_vis.io.save_results import (
    load_solution_npz,
    save_composed_motion_npz,
    save_solution_npz,
)


def test_solution_npz_round_trip(tmp_path: Path) -> None:
    save_solution_npz(
        output_dir=tmp_path,
        point_ids=np.asarray([10, 11], dtype=np.int64),
        time_grid=np.asarray([0.0, 1.0], dtype=np.float32),
        initial_positions=np.asarray([1.0, 2.0], dtype=np.float32),
        trajectory=np.asarray([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32),
        anchor_point_ids=np.asarray([10], dtype=np.int64),
        basis_observation=BasisObservation(
            subset_point_ids=np.asarray([10, 11], dtype=np.int64),
            basis_matrix=np.asarray([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32),
            value_semantics="mean_distance_delta",
        ),
    )

    loaded = load_solution_npz(tmp_path / "solution.npz")

    assert loaded["point_ids"].tolist() == [10, 11]
    assert loaded["anchor_point_ids"].tolist() == [10]
    assert loaded["trajectory"].tolist() == [[1.0, 1.5], [2.0, 2.5]]


def test_save_composed_motion_npz_writes_expected_keys(tmp_path: Path) -> None:
    save_composed_motion_npz(
        output_dir=tmp_path,
        point_ids=np.asarray([10, 11, 12], dtype=np.int64),
        time_grid=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        coordinates=np.zeros((3, 3, 2), dtype=np.float32),
        subset_point_ids=np.asarray([10, 12], dtype=np.int64),
    )

    loaded = dict(np.load(tmp_path / "composed_motion.npz"))

    assert sorted(loaded.keys()) == ["coordinates", "point_ids", "subset_point_ids", "time_grid"]
    assert loaded["point_ids"].tolist() == [10, 11, 12]
    assert loaded["subset_point_ids"].tolist() == [10, 12]
