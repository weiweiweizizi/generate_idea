from __future__ import annotations

import numpy as np

from scripts.matrix_vis.core.mesh import build_mesh_template
from scripts.matrix_vis.core.projection import project_mesh_to_axis
from scripts.matrix_vis.core.types import ProjectionConfig
from scripts.matrix_vis.io.load_mesh import load_mesh
from scripts.matrix_vis.core.types import MeshConfig


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


def test_load_mesh_supports_canonical_obj_normalization(tmp_path) -> None:
    obj_path = tmp_path / "canonical.obj"
    lines = []
    for idx in range(468):
        x = float(idx)
        y = float(idx * 2)
        z = float(idx * 3)
        lines.append(f"v {x} {y} {z}")
    obj_path.write_text("\n".join(lines), encoding="utf-8")

    mesh = load_mesh(
        MeshConfig(
            source=obj_path,
            format="mediapipe_canonical_obj",
            dimension="3d",
            point_ids="auto",
            normalization_scope="face_regions",
        )
    )

    expected_scale_x = 356.0 - 127.0
    expected_scale_y = (152.0 * 2.0) - (10.0 * 2.0)
    assert mesh.points.shape == (478, 3)
    assert mesh.point_ids.tolist()[-5:] == [473, 474, 475, 476, 477]
    assert np.isclose(mesh.points[356, 0] - mesh.points[127, 0], 1.0)
    assert np.isclose(mesh.points[152, 1] - mesh.points[10, 1], 1.0)
    assert np.isclose(mesh.points[100, 0], 100.0 / expected_scale_x)
    assert np.isclose(mesh.points[100, 1], 200.0 / expected_scale_y)
    assert np.isclose(mesh.points[100, 2], 300.0)
