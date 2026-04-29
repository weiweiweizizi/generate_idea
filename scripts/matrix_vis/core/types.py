from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_AXES = ("x", "y")
SUPPORTED_MESH_FORMATS = ("numpy", "mediapipe_canonical_obj")
SUPPORTED_MESH_DIMENSIONS = ("2d", "3d")
SUPPORTED_MATRIX_SHAPES = ("square",)
SUPPORTED_QP_BACKENDS = ("osqp", "matrix_free_cg")
SUPPORTED_VALUE_SEMANTICS = ("mean_distance_delta",)
SUPPORTED_NORMALIZATION_SCOPES = ("mouth_only", "eye_only", "face_regions")
SUPPORTED_SUBSET_LAYOUTS = ("face_regions_grouped", "mouth")
SUPPORTED_SUBSET_POLICIES = ("intersection",)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path


@dataclass(frozen=True)
class MeshConfig:
    source: Path
    format: str
    dimension: str
    point_ids: str | list[int]
    normalization_scope: str | None = None


@dataclass(frozen=True)
class ProjectionConfig:
    axis: str
    source_axis_index: int
    subset_point_ids: tuple[int, ...]
    anchor_point_ids: tuple[int, ...]
    subset_layout: str | None = None
    subset_layout_source: Path | None = None
    subset_layout_extractor_name: str | None = None
    subset_layout_region_names: tuple[str, ...] | None = None

    @property
    def anchor_point_id(self) -> int:
        return self.anchor_point_ids[0]


@dataclass(frozen=True)
class BasisConfig:
    source: Path | None
    basis_index: int
    matrix_shape: str
    value_semantics: str
    prev_source: Path | None = None
    next_source: Path | None = None
    matrix_layout: str | None = None
    matrix_layout_source: Path | None = None
    matrix_layout_extractor_name: str | None = None
    matrix_layout_region_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class QPConfig:
    num_time_steps: int
    lambda_data: float
    lambda_acc: float
    lambda_vel: float
    enforce_order: bool
    max_displacement: float | None
    qp_backend: str
    max_observations: int | None = None


@dataclass(frozen=True)
class ExportConfig:
    save_projected_mesh: bool = True
    save_qp_diagnostics: bool = True
    save_axis_plot: bool = True
    save_npz: bool = True
    save_json_summary: bool = True


@dataclass(frozen=True)
class ComposeInputConfig:
    x_solution: Path
    y_solution: Path


@dataclass(frozen=True)
class ComposeExportConfig:
    save_animation_preview: bool = True
    save_npz: bool = True
    save_json_summary: bool = True


@dataclass(frozen=True)
class MatrixVisConfig:
    config_path: Path
    experiment: ExperimentConfig
    mesh: MeshConfig
    projection: ProjectionConfig
    basis: BasisConfig
    solver: QPConfig
    export: ExportConfig
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class ComposeConfig:
    config_path: Path
    experiment: ExperimentConfig
    mesh: MeshConfig
    inputs: ComposeInputConfig
    subset_policy: str
    export: ComposeExportConfig
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class MeshTemplate:
    points: np.ndarray
    point_ids: np.ndarray
    dimension: str


@dataclass(frozen=True)
class AxisProjection:
    axis: str
    source_axis_index: int
    full_axis_positions: np.ndarray
    subset_point_ids: np.ndarray
    subset_positions: np.ndarray
    anchor_point_ids: np.ndarray

    @property
    def anchor_point_id(self) -> int:
        return int(self.anchor_point_ids[0])


@dataclass(frozen=True)
class BasisObservation:
    subset_point_ids: np.ndarray
    basis_matrix: np.ndarray
    value_semantics: str


@dataclass(frozen=True)
class TrajectorySolution:
    point_ids: np.ndarray
    time_grid: np.ndarray
    initial_positions: np.ndarray
    trajectory: np.ndarray
    anchor_point_ids: np.ndarray
    basis_matrix: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def anchor_point_id(self) -> int:
        return int(self.anchor_point_ids[0])


@dataclass(frozen=True)
class ComposedMotion:
    point_ids: np.ndarray
    time_grid: np.ndarray
    coordinates: np.ndarray
    static_points: np.ndarray
    metadata: dict[str, Any]
