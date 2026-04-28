from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.matrix_vis.core.types import (
    BasisConfig,
    ExportConfig,
    ExperimentConfig,
    MatrixVisConfig,
    MeshConfig,
    ProjectionConfig,
    QPConfig,
    SUPPORTED_AXES,
    SUPPORTED_MATRIX_SHAPES,
    SUPPORTED_MESH_DIMENSIONS,
    SUPPORTED_MESH_FORMATS,
    SUPPORTED_QP_BACKENDS,
    SUPPORTED_VALUE_SEMANTICS,
)

try:
    import yaml
except ImportError:  # pragma: no cover - depends on local env
    yaml = None


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping")
    return value


def _require_bool(section: dict[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Expected '{key}' to be a boolean")
    return value


def _require_int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected '{key}' to be an integer")
    return value


def _require_float(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected '{key}' to be numeric")
    return float(value)


def _require_str(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected '{key}' to be a non-empty string")
    return value.strip()


def _parse_point_ids(value: Any) -> str | list[int]:
    if value == "auto":
        return "auto"
    if not isinstance(value, list) or not value:
        raise ValueError("mesh.point_ids must be 'auto' or a non-empty list of integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("mesh.point_ids must contain only integers")
    return value


def _parse_subset_point_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("projection.subset_point_ids must be a non-empty list of integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("projection.subset_point_ids must contain only integers")
    point_ids = tuple(value)
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("projection.subset_point_ids must not contain duplicates")
    return point_ids


def _parse_anchor_point_ids(section: dict[str, Any], subset_point_ids: tuple[int, ...]) -> tuple[int, ...]:
    if "anchor_point_ids" in section:
        value = section.get("anchor_point_ids")
        if not isinstance(value, list) or not value:
            raise ValueError(
                "projection.anchor_point_ids must be a non-empty list of integers"
            )
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("projection.anchor_point_ids must contain only integers")
        anchor_point_ids = tuple(value)
        if len(set(anchor_point_ids)) != len(anchor_point_ids):
            raise ValueError("projection.anchor_point_ids must not contain duplicates")
    else:
        anchor_point_ids = (_require_int(section, "anchor_point_id"),)

    subset_point_id_set = set(subset_point_ids)
    missing_anchor_ids = [point_id for point_id in anchor_point_ids if point_id not in subset_point_id_set]
    if missing_anchor_ids:
        raise ValueError(
            "projection.anchor_point_ids must be included in projection.subset_point_ids: "
            f"{missing_anchor_ids}"
        )
    return anchor_point_ids


def _validate_projection_axis(axis: str, source_axis_index: int, dimension: str) -> None:
    if axis not in SUPPORTED_AXES:
        raise ValueError(f"Unsupported projection.axis: {axis!r}")
    max_index = 1 if dimension == "2d" else 2
    if source_axis_index < 0 or source_axis_index > max_index:
        raise ValueError(
            f"projection.source_axis_index={source_axis_index} is invalid for mesh dimension {dimension}"
        )


def _resolve_input_path(config_path: Path, candidate: Path) -> Path:
    if candidate.is_absolute():
        return candidate

    config_relative = (config_path.parent / candidate).resolve()
    if config_relative.exists():
        return config_relative
    return candidate.resolve()


def _resolve_output_path(candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else candidate.resolve()


def load_config(config_path: str | Path) -> MatrixVisConfig:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if yaml is None:
        raise ImportError("PyYAML is unavailable; install PyYAML to load matrix_vis configs")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    experiment_section = _require_mapping(raw, "experiment")
    mesh_section = _require_mapping(raw, "mesh")
    projection_section = _require_mapping(raw, "projection")
    basis_section = _require_mapping(raw, "basis")
    solver_section = _require_mapping(raw, "solver")
    export_section = _require_mapping(raw, "export")

    experiment = ExperimentConfig(
        name=_require_str(experiment_section, "name"),
        output_dir=_resolve_output_path(Path(_require_str(experiment_section, "output_dir"))),
    )

    mesh_format = _require_str(mesh_section, "format")
    if mesh_format not in SUPPORTED_MESH_FORMATS:
        raise ValueError(f"Unsupported mesh.format: {mesh_format!r}")
    mesh_dimension = _require_str(mesh_section, "dimension")
    if mesh_dimension not in SUPPORTED_MESH_DIMENSIONS:
        raise ValueError(f"Unsupported mesh.dimension: {mesh_dimension!r}")
    mesh = MeshConfig(
        source=_resolve_input_path(config_path, Path(_require_str(mesh_section, "source"))),
        format=mesh_format,
        dimension=mesh_dimension,
        point_ids=_parse_point_ids(mesh_section.get("point_ids")),
    )

    subset_point_ids = _parse_subset_point_ids(projection_section.get("subset_point_ids"))
    anchor_point_ids = _parse_anchor_point_ids(projection_section, subset_point_ids)
    source_axis_index = _require_int(projection_section, "source_axis_index")
    axis = _require_str(projection_section, "axis")
    _validate_projection_axis(axis, source_axis_index, mesh.dimension)
    projection = ProjectionConfig(
        axis=axis,
        source_axis_index=source_axis_index,
        subset_point_ids=subset_point_ids,
        anchor_point_ids=anchor_point_ids,
    )

    matrix_shape = _require_str(basis_section, "matrix_shape")
    if matrix_shape not in SUPPORTED_MATRIX_SHAPES:
        raise ValueError(f"Unsupported basis.matrix_shape: {matrix_shape!r}")
    value_semantics = _require_str(basis_section, "value_semantics")
    if value_semantics not in SUPPORTED_VALUE_SEMANTICS:
        raise ValueError(f"Unsupported basis.value_semantics: {value_semantics!r}")
    basis = BasisConfig(
        source=_resolve_input_path(config_path, Path(_require_str(basis_section, "source"))),
        basis_index=_require_int(basis_section, "basis_index"),
        matrix_shape=matrix_shape,
        value_semantics=value_semantics,
    )

    qp_backend = _require_str(solver_section, "qp_backend")
    if qp_backend not in SUPPORTED_QP_BACKENDS:
        raise ValueError(f"Unsupported solver.qp_backend: {qp_backend!r}")
    max_displacement = solver_section.get("max_displacement")
    if max_displacement is not None and (
        isinstance(max_displacement, bool) or not isinstance(max_displacement, (int, float))
    ):
        raise ValueError("solver.max_displacement must be null or numeric")
    solver = QPConfig(
        num_time_steps=_require_int(solver_section, "num_time_steps"),
        lambda_data=_require_float(solver_section, "lambda_data"),
        lambda_acc=_require_float(solver_section, "lambda_acc"),
        lambda_vel=_require_float(solver_section, "lambda_vel"),
        enforce_order=_require_bool(solver_section, "enforce_order", True),
        max_displacement=float(max_displacement) if max_displacement is not None else None,
        qp_backend=qp_backend,
    )
    if solver.num_time_steps < 2:
        raise ValueError("solver.num_time_steps must be >= 2")
    if solver.lambda_data < 0 or solver.lambda_acc < 0 or solver.lambda_vel < 0:
        raise ValueError("solver weights must be >= 0")

    export = ExportConfig(
        save_projected_mesh=_require_bool(export_section, "save_projected_mesh", True),
        save_qp_diagnostics=_require_bool(export_section, "save_qp_diagnostics", True),
        save_axis_plot=_require_bool(export_section, "save_axis_plot", True),
        save_npz=_require_bool(export_section, "save_npz", True),
        save_json_summary=_require_bool(export_section, "save_json_summary", True),
    )

    return MatrixVisConfig(
        config_path=config_path,
        experiment=experiment,
        mesh=mesh,
        projection=projection,
        basis=basis,
        solver=solver,
        export=export,
        raw=raw,
    )
