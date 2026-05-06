from __future__ import annotations

import numpy as np
import pytest

from scripts.matrix_vis.core.composition import compose_xy_coordinates


def _build_solution(point_ids: list[int], trajectory: list[list[float]]) -> dict[str, np.ndarray]:
    return {
        "point_ids": np.asarray(point_ids, dtype=np.int64),
        "time_grid": np.asarray([0.0, 1.0], dtype=np.float32),
        "trajectory": np.asarray(trajectory, dtype=np.float32),
    }


def test_compose_xy_coordinates_uses_x_solution_order_by_default() -> None:
    x_solution = _build_solution(
        [10, 11, 12],
        [
            [1.0, 1.5],
            [2.0, 2.5],
            [3.0, 3.5],
        ],
    )
    y_solution = _build_solution(
        [12, 99, 10],
        [
            [30.0, 30.5],
            [40.0, 40.5],
            [50.0, 50.5],
        ],
    )

    point_ids, time_grid, coordinates = compose_xy_coordinates(
        x_solution=x_solution,
        y_solution=y_solution,
    )

    assert point_ids.tolist() == [10, 12]
    assert time_grid.tolist() == [0.0, 1.0]
    assert coordinates.tolist() == [
        [[1.0, 50.0], [3.0, 30.0]],
        [[1.5, 50.5], [3.5, 30.5]],
    ]


def test_compose_xy_coordinates_respects_preferred_order() -> None:
    x_solution = _build_solution(
        [10, 11, 12],
        [
            [1.0, 1.5],
            [2.0, 2.5],
            [3.0, 3.5],
        ],
    )
    y_solution = _build_solution(
        [12, 99, 10],
        [
            [30.0, 30.5],
            [40.0, 40.5],
            [50.0, 50.5],
        ],
    )

    point_ids, _, coordinates = compose_xy_coordinates(
        x_solution=x_solution,
        y_solution=y_solution,
        preferred_point_ids=np.asarray([12, 88, 10], dtype=np.int64),
    )

    assert point_ids.tolist() == [12, 10]
    assert coordinates.tolist() == [
        [[3.0, 30.0], [1.0, 50.0]],
        [[3.5, 30.5], [1.5, 50.5]],
    ]


def test_compose_xy_coordinates_rejects_mismatched_time_grids() -> None:
    x_solution = _build_solution(
        [10],
        [[1.0, 1.5]],
    )
    y_solution = _build_solution(
        [10],
        [[2.0, 2.5]],
    )
    y_solution["time_grid"] = np.asarray([0.0, 2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="time grid"):
        compose_xy_coordinates(
            x_solution=x_solution,
            y_solution=y_solution,
        )
