from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.matrix_vis.core.types import (
    ComposeConfig,
    ComposeExportConfig,
    ComposeInputConfig,
    ExperimentConfig,
    MeshConfig,
    SUPPORTED_MESH_DIMENSIONS,
    SUPPORTED_MESH_FORMATS,
    SUPPORTED_NORMALIZATION_SCOPES,
    SUPPORTED_SUBSET_POLICIES,
)
from scripts.matrix_vis.io.paths import resolve_input_path, resolve_output_path

try:
    import yaml
except ImportError:  # pragma: no cover - depends on local env
    yaml = None


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping")
    return value


def _require_str(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected '{key}' to be a non-empty string")
    return value.strip()


def _require_bool(section: dict[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Expected '{key}' to be a boolean")
    return value


def load_compose_config(config_path: str | Path) -> ComposeConfig:
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
    inputs_section = _require_mapping(raw, "inputs")
    compose_section = raw.get("compose", {})
    export_section = raw.get("export", {})
    if not isinstance(compose_section, dict):
        raise ValueError("Expected 'compose' to be a mapping")
    if not isinstance(export_section, dict):
        raise ValueError("Expected 'export' to be a mapping")

    mesh_format = _require_str(mesh_section, "format")
    if mesh_format not in SUPPORTED_MESH_FORMATS:
        raise ValueError(f"Unsupported mesh.format: {mesh_format!r}")
    mesh_dimension = _require_str(mesh_section, "dimension")
    if mesh_dimension not in SUPPORTED_MESH_DIMENSIONS:
        raise ValueError(f"Unsupported mesh.dimension: {mesh_dimension!r}")
    normalization_scope = mesh_section.get("normalization_scope")
    if normalization_scope is not None and normalization_scope not in SUPPORTED_NORMALIZATION_SCOPES:
        raise ValueError(f"Unsupported mesh.normalization_scope: {normalization_scope!r}")

    subset_policy = compose_section.get("subset_policy", "intersection")
    if subset_policy not in SUPPORTED_SUBSET_POLICIES:
        raise ValueError(f"Unsupported compose.subset_policy: {subset_policy!r}")

    return ComposeConfig(
        config_path=config_path,
        experiment=ExperimentConfig(
            name=_require_str(experiment_section, "name"),
            output_dir=resolve_output_path(Path(_require_str(experiment_section, "output_dir"))),
        ),
        mesh=MeshConfig(
            source=resolve_input_path(config_path, Path(_require_str(mesh_section, "source"))),
            format=mesh_format,
            dimension=mesh_dimension,
            point_ids=mesh_section.get("point_ids", "auto"),
            normalization_scope=normalization_scope,
        ),
        inputs=ComposeInputConfig(
            x_solution=resolve_input_path(config_path, Path(_require_str(inputs_section, "x_solution"))),
            y_solution=resolve_input_path(config_path, Path(_require_str(inputs_section, "y_solution"))),
        ),
        subset_policy=subset_policy,
        export=ComposeExportConfig(
            save_animation_preview=_require_bool(export_section, "save_animation_preview", True),
            save_npz=_require_bool(export_section, "save_npz", True),
            save_json_summary=_require_bool(export_section, "save_json_summary", True),
        ),
        raw=raw,
    )
