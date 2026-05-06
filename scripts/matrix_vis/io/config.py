from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
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
    SUPPORTED_NORMALIZATION_SCOPES,
    SUPPORTED_QP_BACKENDS,
    SUPPORTED_SUBSET_LAYOUTS,
    SUPPORTED_VALUE_SEMANTICS,
)
from scripts.matrix_vis.io.paths import resolve_input_path, resolve_output_path

# 解析主配置YAML，验证所有字段，解析 subset_layout（从配置文件路径resolve相对路径），构建完整的 MatrixVisConfig

try:
    import yaml
except ImportError:  # pragma: no cover - depends on local env
    yaml = None

# 验证字段
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


# 解析mesh.point_ids字段，支持"auto"或一个非空整数列表
def _parse_point_ids(value: Any) -> str | list[int]:
    if value == "auto":
        return "auto"
    if not isinstance(value, list) or not value:
        raise ValueError("mesh.point_ids must be 'auto' or a non-empty list of integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("mesh.point_ids must contain only integers")
    return value

# 解析projection.subset_point_ids字段，必须是一个非空整数列表，且不包含重复项
def _parse_subset_point_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("projection.subset_point_ids must be a non-empty list of integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("projection.subset_point_ids must contain only integers")
    point_ids = tuple(value)
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("projection.subset_point_ids must not contain duplicates")
    return point_ids

# 解析projection部分的subset配置，支持直接指定subset_point_ids或通过subset_layout指定一个布局文件来解析点ID
def _parse_projection_subset(
    *,
    projection_section: dict[str, Any],
    config_path: Path,
) -> tuple[tuple[int, ...], str | None, Path | None, str | None, tuple[str, ...] | None]:
    subset_layout_raw = projection_section.get("subset_layout")
    if subset_layout_raw is None:
        subset_point_ids = _parse_subset_point_ids(projection_section.get("subset_point_ids"))
        return subset_point_ids, None, None, None, None

    layout_name, layout_source, extractor_name, region_names = _parse_layout_spec(
        layout_raw=subset_layout_raw,
        config_path=config_path,
        section_name="projection.subset_layout",
    )
    subset_point_ids = resolve_subset_layout(
        subset_layout=layout_name,
        subset_layout_source=layout_source,
        subset_layout_extractor_name=extractor_name,
        subset_layout_region_names=list(region_names) if region_names is not None else None,
    )
    return subset_point_ids, layout_name, layout_source, extractor_name, region_names


def _parse_layout_spec(
    *,
    layout_raw: Any,
    config_path: Path,
    section_name: str,
) -> tuple[str, Path, str, tuple[str, ...] | None]:
    if not isinstance(layout_raw, dict):
        raise ValueError(f"{section_name} must be a mapping when provided")

    layout_name = _require_str(layout_raw, "name")
    if layout_name not in SUPPORTED_SUBSET_LAYOUTS:
        raise ValueError(f"Unsupported {section_name}.name: {layout_name!r}")
    layout_source = resolve_input_path(config_path, Path(_require_str(layout_raw, "source")))
    extractor_name = layout_raw.get("extractor_name", "mediapipe")
    if not isinstance(extractor_name, str) or not extractor_name.strip():
        raise ValueError(f"{section_name}.extractor_name must be a non-empty string")
    region_names_raw = layout_raw.get("region_names")
    region_names: tuple[str, ...] | None = None
    if region_names_raw is not None:
        if not isinstance(region_names_raw, list) or not region_names_raw:
            raise ValueError(f"{section_name}.region_names must be a non-empty list of strings")
        parsed_region_names: list[str] = []
        for item in region_names_raw:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{section_name}.region_names must contain only non-empty strings")
            parsed_region_names.append(item.strip())
        region_names = tuple(parsed_region_names)
    return layout_name, layout_source, extractor_name.strip(), region_names


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
        output_dir=resolve_output_path(Path(_require_str(experiment_section, "output_dir"))),
    )

    mesh_format = _require_str(mesh_section, "format")
    if mesh_format not in SUPPORTED_MESH_FORMATS:
        raise ValueError(f"Unsupported mesh.format: {mesh_format!r}")
    mesh_dimension = _require_str(mesh_section, "dimension")
    if mesh_dimension not in SUPPORTED_MESH_DIMENSIONS:
        raise ValueError(f"Unsupported mesh.dimension: {mesh_dimension!r}")
    normalization_scope = mesh_section.get("normalization_scope")
    if normalization_scope is not None:
        if not isinstance(normalization_scope, str) or normalization_scope not in SUPPORTED_NORMALIZATION_SCOPES:
            raise ValueError(
                "mesh.normalization_scope must be one of "
                f"{SUPPORTED_NORMALIZATION_SCOPES}, got {normalization_scope!r}"
            )
    mesh = MeshConfig(
        source=resolve_input_path(config_path, Path(_require_str(mesh_section, "source"))),
        format=mesh_format,
        dimension=mesh_dimension,
        point_ids=_parse_point_ids(mesh_section.get("point_ids")),
        normalization_scope=normalization_scope,
    )

    subset_point_ids, subset_layout_name, subset_layout_source, subset_layout_extractor_name, subset_layout_region_names = _parse_projection_subset(
        projection_section=projection_section,
        config_path=config_path,
    )
    anchor_point_ids = _parse_anchor_point_ids(projection_section, subset_point_ids)
    source_axis_index = _require_int(projection_section, "source_axis_index")
    axis = _require_str(projection_section, "axis")
    _validate_projection_axis(axis, source_axis_index, mesh.dimension)
    projection = ProjectionConfig(
        axis=axis,
        source_axis_index=source_axis_index,
        subset_point_ids=subset_point_ids,
        anchor_point_ids=anchor_point_ids,
        subset_layout=subset_layout_name,
        subset_layout_source=subset_layout_source,
        subset_layout_extractor_name=subset_layout_extractor_name,
        subset_layout_region_names=subset_layout_region_names,
    )

    matrix_shape = _require_str(basis_section, "matrix_shape")
    if matrix_shape not in SUPPORTED_MATRIX_SHAPES:
        raise ValueError(f"Unsupported basis.matrix_shape: {matrix_shape!r}")
    value_semantics = _require_str(basis_section, "value_semantics")
    if value_semantics not in SUPPORTED_VALUE_SEMANTICS:
        raise ValueError(f"Unsupported basis.value_semantics: {value_semantics!r}")
    source_raw = basis_section.get("source")
    prev_source_raw = basis_section.get("prev_source")
    next_source_raw = basis_section.get("next_source")
    if source_raw is not None and (prev_source_raw is not None or next_source_raw is not None):
        raise ValueError("basis.source is mutually exclusive with basis.prev_source / basis.next_source")
    if source_raw is None:
        if prev_source_raw is None or next_source_raw is None:
            raise ValueError("basis must define either source or both prev_source and next_source")
        basis_source = None
        prev_source = resolve_input_path(config_path, Path(_require_str(basis_section, "prev_source")))
        next_source = resolve_input_path(config_path, Path(_require_str(basis_section, "next_source")))
    else:
        basis_source = resolve_input_path(config_path, Path(_require_str(basis_section, "source")))
        prev_source = None
        next_source = None
    matrix_layout_raw = basis_section.get("matrix_layout")
    if matrix_layout_raw is None:
        matrix_layout = None
        matrix_layout_source = None
        matrix_layout_extractor_name = None
        matrix_layout_region_names = None
    else:
        matrix_layout, matrix_layout_source, matrix_layout_extractor_name, matrix_layout_region_names = _parse_layout_spec(
            layout_raw=matrix_layout_raw,
            config_path=config_path,
            section_name="basis.matrix_layout",
        )
    basis = BasisConfig(
        source=basis_source,
        basis_index=_require_int(basis_section, "basis_index"),
        matrix_shape=matrix_shape,
        value_semantics=value_semantics,
        prev_source=prev_source,
        next_source=next_source,
        matrix_layout=matrix_layout,
        matrix_layout_source=matrix_layout_source,
        matrix_layout_extractor_name=matrix_layout_extractor_name,
        matrix_layout_region_names=matrix_layout_region_names,
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
        max_observations=solver_section.get("max_observations"),
    )
    if solver.num_time_steps < 2:
        raise ValueError("solver.num_time_steps must be >= 2")
    if solver.lambda_data < 0 or solver.lambda_acc < 0 or solver.lambda_vel < 0:
        raise ValueError("solver weights must be >= 0")
    if solver.max_observations is not None:
        if isinstance(solver.max_observations, bool) or not isinstance(solver.max_observations, int):
            raise ValueError("solver.max_observations must be null or integer")
        if solver.max_observations <= 0:
            raise ValueError("solver.max_observations must be > 0 when provided")

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
