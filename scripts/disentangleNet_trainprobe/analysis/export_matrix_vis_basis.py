#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys

import fire
import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet_trainprobe.analysis.common import (
    load_model_from_checkpoint,
    parse_levels,
    plot_basis_grid,
)


def export(
    checkpoint_path: str,
    output_dir: str | None = None,
    save_heatmaps: bool = True,
):
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    ckpt = torch.load(checkpoint, map_location="cpu")
    config = dict(ckpt.get("config", {}))
    model, loaded_config = load_model_from_checkpoint(checkpoint, num_dataset_classes=1)
    basis = model.get_structured_basis().detach().cpu().numpy().astype(np.float32)
    side_basis = model.get_side_basis().detach().cpu().numpy().astype(np.float32)
    levels = parse_levels(loaded_config.get("levels", "2,6"))

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint.parent / "matrix_vis_exports" / "basis"
    )
    destination.mkdir(parents=True, exist_ok=True)

    basis_path = destination / "basis_bank_x.npy"
    side_basis_path = destination / "side_basis_bank_x.npy"
    np.save(basis_path, basis)
    np.save(side_basis_path, side_basis)

    basis_heatmap = None
    side_basis_heatmap = None
    if save_heatmaps:
        basis_heatmap = destination / "basis_bank_x_heatmap.png"
        plot_basis_grid(basis, levels, basis_heatmap)
        side_basis_heatmap = destination / "side_basis_bank_x_heatmap.png"
        plot_basis_grid(side_basis, (side_basis.shape[0],), side_basis_heatmap)

    manifest = {
        "checkpoint_path": str(checkpoint),
        "mode": str(config.get("mode", "x")),
        "region": str(config.get("region", "full")),
        "matrix_size": int(basis.shape[-1]),
        "num_basis": int(basis.shape[0]),
        "levels": list(levels),
        "side_basis_count": int(side_basis.shape[0]),
        "basis_orthogonalization": str(config.get("basis_orthogonalization", "joint_global_qr")),
        "quantizer_type": str(config.get("quantizer_type", "residual_fsq")),
        "point_layout": "face_regions_grouped",
        "point_layout_region_names": None,
        "value_semantics": "mean_distance_delta",
        "exported_basis_path": str(basis_path),
        "exported_side_basis_path": str(side_basis_path),
        "artifacts": {
            "basis_bank_heatmap": str(basis_heatmap) if basis_heatmap is not None else None,
            "side_basis_bank_heatmap": str(side_basis_heatmap)
            if side_basis_heatmap is not None
            else None,
        },
    }
    manifest_path = destination / "basis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    fire.Fire({"export": export})
