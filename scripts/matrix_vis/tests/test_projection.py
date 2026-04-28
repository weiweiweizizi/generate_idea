from __future__ import annotations

import numpy as np

from scripts.matrix_vis.core.mesh import build_mesh_template
from scripts.matrix_vis.core.projection import project_mesh_to_axis
from scripts.matrix_vis.core.types import ProjectionConfig


def test_project_mesh_to_axis_extracts_subset_and_anchor() -> None:
    points = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.5, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [4.5, 8.0, 13.0],
        ],
        dtype=np.float32,
    )
    mesh = build_mesh_template(points, dimension="3d", point_ids=[10, 11, 12, 13])
    config = ProjectionConfig(
        axis="y",
        source_axis_index=1,
        subset_point_ids=(10, 12, 13),
        anchor_point_ids=(10, 12),
    )

    projection = project_mesh_to_axis(mesh, config)

    assert projection.axis == "y"
    assert projection.anchor_point_ids.tolist() == [10, 12]
    assert projection.anchor_point_id == 10
    assert projection.subset_point_ids.tolist() == [10, 12, 13]
    assert projection.full_axis_positions.tolist() == [1.0, 2.0, 4.0, 8.0]
    assert projection.subset_positions.tolist() == [1.0, 4.0, 8.0]
