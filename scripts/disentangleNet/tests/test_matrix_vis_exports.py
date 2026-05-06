from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet.analysis.export_matrix_vis_basis import (
    build_basis_manifest,
    compute_level_boundaries,
    export_basis_bundle,
)


def test_compute_level_boundaries() -> None:
    assert compute_level_boundaries((2, 6)) == [0, 2, 8]


def test_export_basis_bundle_writes_bridge_artifacts(tmp_path: Path) -> None:
    basis = np.arange(8 * 4 * 4, dtype=np.float32).reshape(8, 4, 4)
    side_basis = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    config = {
        "mode": "x",
        "region": "mouth",
        "levels": "2,6",
        "basis_orthogonalization": "joint_global_qr",
        "quantizer_type": "residual_fsq",
    }
    summary = export_basis_bundle(
        output_dir=tmp_path,
        checkpoint_path=tmp_path / "best.pt",
        config=config,
        basis=basis,
        side_basis=side_basis,
        save_heatmaps=False,
    )

    basis_path = tmp_path / "basis_bank_x.npy"
    side_basis_path = tmp_path / "side_basis_bank_x.npy"
    manifest_path = tmp_path / "basis_manifest.json"
    summary_path = tmp_path / "export_summary.json"

    assert basis_path.exists()
    assert side_basis_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()

    np.testing.assert_array_equal(np.load(basis_path), basis)
    np.testing.assert_array_equal(np.load(side_basis_path), side_basis)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["mode"] == "x"
    assert manifest["region"] == "mouth"
    assert manifest["num_basis"] == 8
    assert manifest["side_basis_count"] == 3
    assert manifest["point_layout"] == "face_regions_grouped"
    assert manifest["point_layout_region_names"] == ["around_mouth", "mouth"]
    assert manifest["level_boundaries"] == [0, 2, 8]
    assert manifest["value_semantics"] == "mean_distance_delta"

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["basis_shape"] == [8, 4, 4]
    assert summary["point_layout"] == "face_regions_grouped"


def test_build_basis_manifest_records_export_paths(tmp_path: Path) -> None:
    manifest = build_basis_manifest(
        checkpoint_path=tmp_path / "best.pt",
        config={"mode": "x", "region": "mouth", "levels": "2,6"},
        basis=np.zeros((8, 4, 4), dtype=np.float32),
        side_basis=np.zeros((3, 4, 4), dtype=np.float32),
        point_layout="face_regions_grouped",
        point_layout_region_names=["around_mouth", "mouth"],
        exported_basis_path=tmp_path / "basis_bank_x.npy",
        exported_side_basis_path=tmp_path / "side_basis_bank_x.npy",
    )

    assert manifest["exported_basis_path"].endswith("basis_bank_x.npy")
    assert manifest["exported_side_basis_path"].endswith("side_basis_bank_x.npy")
    assert manifest["bridge_scope"]["step1"].startswith("basis_wise")
    assert manifest["point_layout_region_names"] == ["around_mouth", "mouth"]
