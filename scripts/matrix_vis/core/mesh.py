from __future__ import annotations

import numpy as np

from scripts.matrix_vis.core.types import MeshTemplate, SUPPORTED_MESH_DIMENSIONS


def validate_mesh_points(points: np.ndarray, dimension: str) -> np.ndarray:
    if dimension not in SUPPORTED_MESH_DIMENSIONS:
        raise ValueError(f"Unsupported mesh dimension: {dimension!r}")
    if points.ndim != 2:
        raise ValueError(f"Mesh points must be 2D, got shape {tuple(points.shape)}")

    expected_dims = 2 if dimension == "2d" else 3
    if points.shape[1] != expected_dims:
        raise ValueError(
            f"Mesh dimension {dimension} expects points with width {expected_dims}, "
            f"got shape {tuple(points.shape)}"
        )
    return points.astype(np.float32, copy=False)


def build_mesh_template(
    points: np.ndarray,
    *,
    dimension: str,
    point_ids: str | list[int] = "auto",
) -> MeshTemplate:
    points = validate_mesh_points(points, dimension)

    if point_ids == "auto":
        ids = np.arange(points.shape[0], dtype=np.int64)
    else:
        ids = np.asarray(point_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.shape[0] != points.shape[0]:
            raise ValueError(
                "Explicit point_ids must be a 1D array with the same length as the number of mesh points"
            )
        if np.unique(ids).shape[0] != ids.shape[0]:
            raise ValueError("Mesh point_ids must be unique")

    return MeshTemplate(points=points, point_ids=ids, dimension=dimension)


def extract_subset_indices(mesh: MeshTemplate, subset_point_ids: np.ndarray) -> np.ndarray:
    subset_point_ids = np.asarray(subset_point_ids, dtype=np.int64)
    id_to_index = {int(point_id): idx for idx, point_id in enumerate(mesh.point_ids.tolist())}
    try:
        indices = [id_to_index[int(point_id)] for point_id in subset_point_ids.tolist()]
    except KeyError as exc:
        raise ValueError(f"Subset point id {int(exc.args[0])} is not present in mesh.point_ids") from exc
    return np.asarray(indices, dtype=np.int64)
