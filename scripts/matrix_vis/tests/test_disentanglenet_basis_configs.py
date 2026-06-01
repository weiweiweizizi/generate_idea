from __future__ import annotations

import json
from pathlib import Path

from scripts.matrix_vis.scripts.generate_disentanglenet_basis_configs import generate_configs


def _write_axis_template(path: Path, *, axis: str) -> None:
    path.write_text(
        f"""
experiment:
  name: demo_axis_{axis}
  output_dir: outputs/matrix_vis/demo_axis_{axis}

mesh:
  source: toy_mesh.npy
  format: numpy
  dimension: 2d
  point_ids: auto

projection:
  subset_layout:
    name: face_regions_grouped
    source: scripts/matrix_vis/configs/landmarks/mediapipe_face_regions_full.yaml
    extractor_name: mediapipe
  anchor_point_ids: [14]
  axis: {axis}
  source_axis_index: {0 if axis == "x" else 1}

basis:
  source: toy_basis.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta

solver:
  method: osqp

export:
  save_npz: true
""".strip(),
        encoding="utf-8",
    )


def test_generate_configs_switches_to_axis_y_for_y_mode(tmp_path: Path) -> None:
    manifest_path = tmp_path / "basis_manifest.json"
    manifest = {
        "mode": "y",
        "num_basis": 2,
        "side_basis_count": 1,
        "point_layout": "face_regions_grouped",
        "point_layout_region_names": None,
        "exported_basis_path": str((tmp_path / "basis_bank_y.npy").resolve()),
        "exported_side_basis_path": str((tmp_path / "side_basis_bank_y.npy").resolve()),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    axis_x_template = tmp_path / "axis_x.yaml"
    axis_y_template = tmp_path / "axis_y.yaml"
    _write_axis_template(axis_x_template, axis="x")
    _write_axis_template(axis_y_template, axis="y")

    summary = generate_configs(
        manifest_path=str(manifest_path),
        output_dir=str(tmp_path / "generated"),
        x_template_config=str(axis_x_template),
        fixed_y_config=str(axis_y_template),
        anchor_point_ids="33,263",
    )

    assert summary["mode"] == "y"
    assert summary["reconstruction_axis"] == "y"
    assert summary["static_axis"] == "x"
    assert summary["fixed_other_config"] is None
    assert summary["fixed_other_solution"] is None
    assert summary["anchor_point_ids"] == [33, 263]
    assert len(summary["axis_configs"]) == 3
    assert all(path.endswith("_axis_y.yaml") for path in summary["axis_configs"])
    assert all("/no_motion_x/" in path for path in summary["no_motion_other_preview_output_dirs"])

    first_axis_config = Path(summary["axis_configs"][0])
    payload = first_axis_config.read_text(encoding="utf-8")
    assert "axis_y" in first_axis_config.name
    assert "axis_y" in payload
    assert "source_axis_index: 1" in payload
