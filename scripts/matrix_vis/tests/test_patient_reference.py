from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.matrix_vis.io.load_patient_reference import (
    load_distance_matrix,
    load_label_first_line_normalized_points,
    load_subset_axis_positions,
)
from scripts.matrix_vis.pipelines.patient_sequence import build_target_distance_matrix, clamp_distance_matrix_nonnegative


def test_load_subset_axis_positions_from_label_like_table(tmp_path: Path) -> None:
    table_path = tmp_path / "landmarks.txt"
    table_path.write_text(
        "0 0.0 1.0\n"
        "1 2.0 3.0\n"
        "2 4.0 5.0\n",
        encoding="utf-8",
    )

    positions = load_subset_axis_positions(
        landmark_source=table_path,
        subset_point_ids=np.asarray([2, 0], dtype=np.int64),
        axis="x",
    )

    assert np.allclose(positions, np.asarray([4.0, 0.0], dtype=np.float32))


def test_load_label_first_line_normalized_points_preserves_y_orientation(tmp_path: Path) -> None:
    label_path = tmp_path / "lmks_crop.label"
    label_path.write_text(
        "0 0.0 1.0 0 0 0.0,1.0 0.0,3.0 0.0,5.0\n",
        encoding="utf-8",
    )

    points = load_label_first_line_normalized_points(label_path)

    assert np.allclose(points[:, 1], np.asarray([-0.5, 0.0, 0.5], dtype=np.float32))


def test_build_target_distance_matrix_accumulates_delta() -> None:
    base = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    delta = np.asarray(
        [
            [0.0, -0.25],
            [-0.25, 0.0],
        ],
        dtype=np.float32,
    )

    target = build_target_distance_matrix(reference_distance_matrix=base, delta_matrix=delta)

    assert np.allclose(target, np.asarray([[0.0, 0.75], [0.75, 0.0]], dtype=np.float32))


def test_clamp_distance_matrix_nonnegative_enforces_positive_off_diagonal() -> None:
    raw = np.asarray(
        [
            [0.2, -0.5],
            [-0.5, 0.3],
        ],
        dtype=np.float32,
    )

    clipped, clip_count = clamp_distance_matrix_nonnegative(raw, epsilon=1e-4)

    assert clip_count == 4
    assert np.allclose(clipped, np.asarray([[0.0, 1e-4], [1e-4, 0.0]], dtype=np.float32))


def test_load_distance_matrix_from_text(tmp_path: Path) -> None:
    matrix_path = tmp_path / "d1.txt"
    matrix_path.write_text("0 1\n1 0\n", encoding="utf-8")

    matrix = load_distance_matrix(matrix_path)

    assert np.allclose(matrix, np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
