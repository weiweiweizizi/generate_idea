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
  enforce_order: true
  max_displacement: null
  qp_backend: osqp

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
