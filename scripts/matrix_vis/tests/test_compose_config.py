from __future__ import annotations

from pathlib import Path

import pytest

from scripts.matrix_vis.io.compose_config import load_compose_config


def write_compose_config(tmp_path: Path, content: str) -> Path:
    toy_dir = tmp_path / "toy"
    toy_dir.mkdir(parents=True, exist_ok=True)
    (toy_dir / "mesh.npy").write_bytes(b"")
    (toy_dir / "x_solution.npz").write_bytes(b"")
    (toy_dir / "y_solution.npz").write_bytes(b"")
    config_path = tmp_path / "compose.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


VALID_COMPOSE_CONFIG = """
experiment:
  name: compose_demo
  output_dir: outputs/matrix_vis/compose_demo

mesh:
  source: toy/mesh.npy
  format: numpy
  dimension: 2d
  point_ids: auto

inputs:
  x_solution: toy/x_solution.npz
  y_solution: toy/y_solution.npz

compose:
  subset_policy: intersection

export:
  save_animation_preview: false
  save_npz: true
  save_json_summary: false
"""


def test_load_compose_config_parses_valid_config(tmp_path: Path) -> None:
    config = load_compose_config(write_compose_config(tmp_path, VALID_COMPOSE_CONFIG))

    assert config.experiment.name == "compose_demo"
    assert config.mesh.source == (tmp_path / "toy/mesh.npy").resolve()
    assert config.inputs.x_solution == (tmp_path / "toy/x_solution.npz").resolve()
    assert config.inputs.y_solution == (tmp_path / "toy/y_solution.npz").resolve()
    assert config.subset_policy == "intersection"
    assert config.export.save_animation_preview is False
    assert config.export.save_npz is True
    assert config.export.save_json_summary is False


def test_load_compose_config_rejects_unknown_subset_policy(tmp_path: Path) -> None:
    invalid = VALID_COMPOSE_CONFIG.replace("subset_policy: intersection", "subset_policy: union")

    with pytest.raises(ValueError, match="subset_policy"):
        load_compose_config(write_compose_config(tmp_path, invalid))
