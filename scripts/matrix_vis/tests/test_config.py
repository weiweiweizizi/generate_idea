from __future__ import annotations

from pathlib import Path

import pytest

from scripts.matrix_vis.io.config import load_config


VALID_CONFIG = """
experiment:
  name: toy_open_mouth_x
  output_dir: outputs/matrix_vis/toy_open_mouth_x

mesh:
  source: toy/double_crescent_mesh.npy
  format: numpy
  dimension: 2d
  point_ids: auto

projection:
  axis: x
  source_axis_index: 0
  subset_point_ids: [10, 11, 12, 13]
  anchor_point_ids: [10, 13]

basis:
  source: toy/basis_x.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta

solver:
  num_time_steps: 25
  lambda_data: 1.0
  lambda_acc: 10.0
  lambda_vel: 1.0
  lambda_laplacian: 0.0
  lambda_area_sign: 0.0
  area_barrier_margin: 0.05
  enforce_order: true
  max_displacement: null
  qp_backend: torch

export:
  save_projected_mesh: true
  save_qp_diagnostics: true
  save_axis_plot: true
  save_npz: true
  save_json_summary: true
"""


def write_config(tmp_path: Path, content: str) -> Path:
    toy_dir = tmp_path / "toy"
    toy_dir.mkdir(parents=True, exist_ok=True)
    (toy_dir / "double_crescent_mesh.npy").write_bytes(b"")
    (toy_dir / "basis_x.npy").write_bytes(b"")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_load_config_parses_valid_minimal_config(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG))

    assert config.experiment.name == "toy_open_mouth_x"
    assert config.mesh.dimension == "2d"
    assert config.projection.axis == "x"
    assert config.projection.subset_point_ids == (10, 11, 12, 13)
    assert config.projection.anchor_point_ids == (10, 13)
    assert config.projection.anchor_point_id == 10
    assert config.solver.num_time_steps == 25
    assert config.solver.lambda_laplacian == 0.0
    assert config.solver.lambda_area_sign == 0.0
    assert config.basis.value_semantics == "mean_distance_delta"
    assert config.mesh.source == (tmp_path / "toy/double_crescent_mesh.npy").resolve()


def test_load_config_rejects_anchor_outside_subset(tmp_path: Path) -> None:
    invalid = VALID_CONFIG.replace("anchor_point_ids: [10, 13]", "anchor_point_ids: [10, 99]")

    with pytest.raises(ValueError, match="anchor_point_ids"):
        load_config(write_config(tmp_path, invalid))


def test_load_config_rejects_invalid_axis_index_for_2d_mesh(tmp_path: Path) -> None:
    invalid = VALID_CONFIG.replace("source_axis_index: 0", "source_axis_index: 2")

    with pytest.raises(ValueError, match="source_axis_index=2"):
        load_config(write_config(tmp_path, invalid))


def test_load_config_supports_legacy_single_anchor_key(tmp_path: Path) -> None:
    legacy = VALID_CONFIG.replace("anchor_point_ids: [10, 13]", "anchor_point_id: 10")

    config = load_config(write_config(tmp_path, legacy))

    assert config.projection.anchor_point_ids == (10,)


def test_load_config_normalizes_legacy_qp_backend(tmp_path: Path) -> None:
    legacy = VALID_CONFIG.replace("qp_backend: torch", "qp_backend: matrix_free_cg")

    config = load_config(write_config(tmp_path, legacy))

    assert config.solver.qp_backend == "torch"


def test_load_config_resolves_grouped_face_region_subset_and_diff_sources(tmp_path: Path) -> None:
    toy_dir = tmp_path / "toy"
    toy_dir.mkdir(parents=True, exist_ok=True)
    (toy_dir / "double_crescent_mesh.npy").write_bytes(b"")
    (toy_dir / "win_prev.npy").write_bytes(b"")
    (toy_dir / "win_next.npy").write_bytes(b"")
    extractor = tmp_path / "extractors.yaml"
    extractor.write_text(
        """
mediapipe:
  symmetric_pairs: [(10, 110), (11, 111)]
  face_regions:
    forehead: [10, 12]
    mouth: [61, 0, 291]
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment:
  name: real_like
  output_dir: outputs/matrix_vis/real_like

mesh:
  source: toy/double_crescent_mesh.npy
  format: numpy
  dimension: 2d
  point_ids: auto
  normalization_scope: face_regions

projection:
  axis: x
  source_axis_index: 0
  subset_layout:
    name: face_regions_grouped
    source: {extractor.name}
  anchor_point_ids: [12]

basis:
  prev_source: toy/win_prev.npy
  next_source: toy/win_next.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta

solver:
  num_time_steps: 20
  lambda_data: 1.0
  lambda_acc: 10.0
  lambda_vel: 1.0
  enforce_order: false
  max_displacement: null
  qp_backend: torch

export:
  save_projected_mesh: true
  save_qp_diagnostics: true
  save_axis_plot: true
  save_npz: true
  save_json_summary: true
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.mesh.normalization_scope == "face_regions"
    assert config.projection.subset_layout == "face_regions_grouped"
    assert config.projection.subset_point_ids == (10, 12, 110, 61, 0, 291)
    assert config.basis.source is None
    assert config.basis.prev_source == (toy_dir / "win_prev.npy").resolve()
    assert config.basis.next_source == (toy_dir / "win_next.npy").resolve()


def test_load_config_filters_grouped_subset_by_region_names(tmp_path: Path) -> None:
    toy_dir = tmp_path / "toy"
    toy_dir.mkdir(parents=True, exist_ok=True)
    (toy_dir / "double_crescent_mesh.npy").write_bytes(b"")
    (toy_dir / "win_prev.npy").write_bytes(b"")
    (toy_dir / "win_next.npy").write_bytes(b"")
    extractor = tmp_path / "extractors.yaml"
    extractor.write_text(
        """
mediapipe:
  symmetric_pairs: [(10, 110), (61, 291)]
  face_regions:
    forehead: [10, 12]
    around_mouth: [164, 18]
    mouth: [61, 0]
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment:
  name: local_regions
  output_dir: outputs/matrix_vis/local_regions

mesh:
  source: toy/double_crescent_mesh.npy
  format: numpy
  dimension: 2d
  point_ids: auto

projection:
  axis: x
  source_axis_index: 0
  subset_layout:
    name: face_regions_grouped
    source: {extractor.name}
    region_names: [around_mouth, mouth]
  anchor_point_ids: [18]

basis:
  prev_source: toy/win_prev.npy
  next_source: toy/win_next.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta

solver:
  num_time_steps: 20
  lambda_data: 1.0
  lambda_acc: 10.0
  lambda_vel: 1.0
  enforce_order: false
  max_displacement: null
  qp_backend: torch

export:
  save_projected_mesh: true
  save_qp_diagnostics: true
  save_axis_plot: true
  save_npz: true
  save_json_summary: true
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.projection.subset_layout_region_names == ("around_mouth", "mouth")
    assert config.projection.subset_point_ids == (164, 18, 61, 0, 291)
