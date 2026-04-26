from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.matrix_vis.core.mesh import build_mesh_template
from scripts.matrix_vis.core.types import MeshConfig, MeshTemplate


def load_mesh(mesh_config: MeshConfig) -> MeshTemplate:
    if mesh_config.format != "numpy":
        raise ValueError(f"Unsupported mesh format: {mesh_config.format!r}")
    if not mesh_config.source.exists():
        raise FileNotFoundError(mesh_config.source)

    points = np.load(mesh_config.source)
    return build_mesh_template(
        points,
        dimension=mesh_config.dimension,
        point_ids=mesh_config.point_ids,
    )
