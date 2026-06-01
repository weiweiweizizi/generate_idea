from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from disentangleNet.analysis.loaders import infer_checkpoint_contract, load_model_for_analysis
from disentangleNet.analysis.utils import (
    compute_level_boundaries,
    get_shared_basis_bank,
    get_side_basis_bank,
    parse_levels,
    plot_basis_grid,
    resolve_bridge_point_layout,
    resolve_bridge_point_layout_region_names,
    save_json,
)
from disentangleNet.training.data import build_specs


def build_basis_manifest(
    *,
    checkpoint_path: Path,
    contract_framework: str,
    config: dict[str, Any],
    basis: np.ndarray,
    side_basis: np.ndarray,
    point_layout: str,
    point_layout_region_names: list[str] | None,
    exported_basis_path: Path,
    exported_side_basis_path: Path,
) -> dict[str, Any]:
    levels = parse_levels(config.get("levels"), default=(2, 6))
    manifest: dict[str, Any] = {
        "framework": contract_framework,
        "checkpoint_path": str(checkpoint_path),
        "mode": str(config.get("mode", "x")),
        "region": str(config.get("region", "mouth")),
        "matrix_size": int(basis.shape[-1]),
        "num_basis": int(basis.shape[0]),
        "levels": list(levels),
        "level_boundaries": compute_level_boundaries(levels),
        "basis_orthogonalization": str(config.get("basis_orthogonalization", "none")),
        "quantizer_type": str(config.get("quantizer_type", "residual_fsq")),
        "point_layout": point_layout,
        "point_layout_region_names": point_layout_region_names,
        "value_semantics": "mean_distance_delta",
        "exported_basis_path": str(exported_basis_path),
        "side_basis_count": int(side_basis.shape[0]),
        "exported_side_basis_path": str(exported_side_basis_path),
        "bridge_scope": {
            "step1": "basis_wise_x_reconstruct_then_compose_with_fixed_y",
            "step2": "patient_coeff_compose_then_x_reconstruct",
        },
    }
    if contract_framework == "disentangleNet_lowrank":
        manifest["lowrank"] = {
            "level_ranks": list(config.get("lowrank_level_ranks", [3, 5])),
            "private_branch_enabled": bool(config.get("private_branch_enabled", True)),
            "side_residual_enabled": bool(config.get("side_residual_enabled", False)),
            "action_side_input": str(config.get("action_side_input", "free_path_coeff")),
        }
    return manifest


def export_basis(
    *,
    checkpoint_path: str,
    output_dir: str | None = None,
    save_heatmaps: bool = True,
):
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    contract = infer_checkpoint_contract(checkpoint)
    data_roots = contract.config.get("data_roots")
    num_dataset_classes = len(build_specs(str(data_roots))) if data_roots else int(contract.config.get("num_dataset_classes", 1))
    model, config, contract = load_model_for_analysis(
        checkpoint,
        num_dataset_classes=max(int(num_dataset_classes), 1),
    )

    basis = get_shared_basis_bank(model)
    side_basis = get_side_basis_bank(model)
    levels = parse_levels(config.get("levels"), default=contract.levels)
    mode = str(config.get("mode", "x"))

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint.parent / "matrix_vis_exports" / "basis"
    )
    destination.mkdir(parents=True, exist_ok=True)

    basis_path = destination / f"basis_bank_{mode}.npy"
    side_basis_path = destination / f"side_basis_bank_{mode}.npy"
    np.save(basis_path, basis.astype(np.float32, copy=False))
    np.save(side_basis_path, side_basis.astype(np.float32, copy=False))

    basis_plot_path = None
    side_basis_plot_path = None
    if save_heatmaps:
        basis_plot_path = destination / f"basis_bank_{mode}_heatmap.png"
        plot_basis_grid(basis, levels, basis_plot_path)
        if side_basis.shape[0] > 0:
            side_basis_plot_path = destination / f"side_basis_bank_{mode}_heatmap.png"
            plot_basis_grid(side_basis, (side_basis.shape[0],), side_basis_plot_path)

    point_layout = resolve_bridge_point_layout(region=str(config.get("region", "mouth")))
    point_layout_region_names = resolve_bridge_point_layout_region_names(region=str(config.get("region", "mouth")))
    manifest = build_basis_manifest(
        checkpoint_path=checkpoint,
        contract_framework=contract.framework,
        config=config,
        basis=basis,
        side_basis=side_basis,
        point_layout=point_layout,
        point_layout_region_names=point_layout_region_names,
        exported_basis_path=basis_path,
        exported_side_basis_path=side_basis_path,
    )
    manifest["artifacts"] = {
        "basis_bank_heatmap": str(basis_plot_path) if basis_plot_path is not None else None,
        "side_basis_bank_heatmap": str(side_basis_plot_path) if side_basis_plot_path is not None else None,
    }

    manifest_path = destination / "basis_manifest.json"
    save_json(manifest_path, manifest)
    summary = {
        "output_dir": str(destination),
        "basis_shape": list(basis.shape),
        "side_basis_shape": list(side_basis.shape),
        "manifest_path": str(manifest_path),
        "basis_path": str(basis_path),
        "side_basis_path": str(side_basis_path),
    }
    save_json(destination / "export_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
