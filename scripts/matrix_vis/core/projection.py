from __future__ import annotations

import numpy as np

from scripts.matrix_vis.core.mesh import extract_subset_indices
from scripts.matrix_vis.core.types import AxisProjection, MeshTemplate, ProjectionConfig


def project_mesh_to_axis(
    mesh: MeshTemplate,
    projection_config: ProjectionConfig,
) -> AxisProjection:
    if projection_config.source_axis_index >= mesh.points.shape[1]:
        raise ValueError(
            f"source_axis_index={projection_config.source_axis_index} exceeds mesh width {mesh.points.shape[1]}"
        )

    full_axis_positions = mesh.points[:, projection_config.source_axis_index].astype(
        np.float32, copy=False
    )
    subset_point_ids = np.asarray(projection_config.subset_point_ids, dtype=np.int64)
    subset_indices = extract_subset_indices(mesh, subset_point_ids)
    subset_positions = full_axis_positions[subset_indices]

    return AxisProjection(
        axis=projection_config.axis,
        source_axis_index=projection_config.source_axis_index,
        full_axis_positions=full_axis_positions,
        subset_point_ids=subset_point_ids,
        subset_positions=subset_positions,
        anchor_point_ids=np.asarray(projection_config.anchor_point_ids, dtype=np.int64),
    )
