from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.matrix_vis.core.mesh import build_mesh_template
from scripts.matrix_vis.core.types import MeshConfig, MeshTemplate


def _build_synthetic_iris_points(base_points: np.ndarray) -> np.ndarray:
    if base_points.shape != (468, 3):
        raise ValueError(f"Expected 468 base points before iris synthesis, got {tuple(base_points.shape)}")

    def iris_cluster(horizontal_pair: tuple[int, int], vertical_pair: tuple[int, int]) -> np.ndarray:
        left_corner = base_points[horizontal_pair[0]]
        right_corner = base_points[horizontal_pair[1]]
        upper_lid = base_points[vertical_pair[0]]
        lower_lid = base_points[vertical_pair[1]]
        center = (left_corner + right_corner + upper_lid + lower_lid) / 4.0
        horizontal = (right_corner - left_corner) * 0.18
        vertical = (lower_lid - upper_lid) * 0.32
        return np.asarray(
            [
                center,
                center - vertical,
                center - horizontal,
                center + vertical,
                center + horizontal,
            ],
            dtype=np.float32,
        )

    left_iris = iris_cluster((33, 133), (159, 145))
    right_iris = iris_cluster((362, 263), (386, 374))
    return np.concatenate([base_points, left_iris, right_iris], axis=0)


def _load_canonical_obj_vertices(obj_path: Path, *, num_vertices: int = 478) -> np.ndarray:
    vertices: list[list[float]] = []
    with obj_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(vertices) >= 468:
                break
    points = np.asarray(vertices, dtype=np.float32)
    if points.shape != (468, 3):
        raise ValueError(
            f"Expected 468 vertices from canonical OBJ, got shape {tuple(points.shape)}"
        )
    if num_vertices == 468:
        return points
    if num_vertices == 478:
        return _build_synthetic_iris_points(points)
    raise ValueError(f"Unsupported canonical vertex count request: {num_vertices}")


def _apply_normalization(points: np.ndarray, normalization_scope: str | None) -> np.ndarray:
    if normalization_scope is None:
        return points.astype(np.float32, copy=False)

    normalized = points.astype(np.float32, copy=True)
    if normalization_scope == "face_regions":
        scale_x = abs(float(points[356, 0] - points[127, 0]))
        scale_y = abs(float(points[152, 1] - points[10, 1]))
    elif normalization_scope == "mouth_only":
        scale_x = abs(float(points[291, 0] - points[61, 0]))
        scale_y = abs(float(points[17, 1] - points[0, 1]))
    elif normalization_scope == "eye_only":
        scale_x = abs(float(points[145, 0] - points[159, 0]))
        scale_y = abs(float(points[472, 1] - points[470, 1]))
    else:
        raise ValueError(f"Unsupported normalization scope: {normalization_scope!r}")

    if scale_x <= 0:
        scale_x = 1.0
    if scale_y <= 0:
        scale_y = 1.0
    normalized[:, 0] /= scale_x
    normalized[:, 1] /= scale_y
    return normalized


def load_mesh(mesh_config: MeshConfig) -> MeshTemplate:
    if not mesh_config.source.exists():
        raise FileNotFoundError(mesh_config.source)

    if mesh_config.format == "numpy":
        points = np.load(mesh_config.source)
    elif mesh_config.format == "mediapipe_canonical_obj":
        points = _load_canonical_obj_vertices(mesh_config.source)
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_config.format!r}")

    points = _apply_normalization(points, mesh_config.normalization_scope)
    return build_mesh_template(
        points,
        dimension=mesh_config.dimension,
        point_ids=mesh_config.point_ids,
    )
