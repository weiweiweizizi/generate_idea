#!/usr/bin/env python
"""
Export matrix-visualization basis bundles from one checkpoint.

What this script does:
- Load structured free and side basis matrices from a checkpoint.
- Export them as `.npy` arrays for downstream matrix-visualization tools.
- Optionally render compact heatmap grids.
- Save a JSON manifest describing level boundaries, point layout, and file semantics.

Typical usage:
1. Default export:
   `python scripts/disentangleNet/analysis/export_matrix_vis_basis.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt`
2. Disable heatmaps:
   `python scripts/disentangleNet/analysis/export_matrix_vis_basis.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \\
      --save_heatmaps False`
3. Custom destination:
   `python scripts/disentangleNet/analysis/export_matrix_vis_basis.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \\
      --output_dir outputs/disentangleNet/v31_current_verify/matrix_vis_exports/basis_custom`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import fire
import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet.analysis.analyze_checkpoint import plot_basis_grid
from scripts.disentangleNet.model.basis import get_joint_structured_basis


def parse_levels(levels) -> tuple[int, ...]:
    if isinstance(levels, str):
        return tuple(int(v) for v in levels.split(",") if str(v).strip())
    if isinstance(levels, (tuple, list)):
        return tuple(int(v) for v in levels)
    raise TypeError(f"Unsupported levels value: {levels!r}")


def compute_level_boundaries(levels: tuple[int, ...]) -> list[int]:
    boundaries = [0]
    running = 0
    for level in levels:
        running += int(level)
        boundaries.append(running)
    return boundaries


def resolve_bridge_point_layout(*, region: str) -> str:
    return "face_regions_grouped"


def resolve_bridge_point_layout_region_names(*, region: str) -> list[str] | None:
    if region == "mouth":
        return ["around_mouth", "mouth"]
    return None


def extract_structured_basis_from_checkpoint(
    *,
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    config = dict(checkpoint.get("config", {}))
    state_dict = checkpoint.get("model", {})
    if "action_basis_bank" not in state_dict:
        raise KeyError("Checkpoint model state is missing 'action_basis_bank'")

    action_basis = state_dict["action_basis_bank"].detach().cpu().float()
    side_basis = state_dict.get("side_basis_bank")
    if side_basis is None:
        side_basis = torch.zeros((0, action_basis.shape[-1], action_basis.shape[-1]), dtype=torch.float32)
    else:
        side_basis = side_basis.detach().cpu().float()

    levels = parse_levels(config.get("levels", "2,3,6"))
    side_basis_count = int(config.get("side_basis_count", int(side_basis.shape[0])))
    basis_size = int(config.get("basis_size", action_basis.shape[-1]))
    shared_structured, side_structured = get_joint_structured_basis(
        action_basis,
        side_basis,
        levels=levels,
        total_basis_num=int(sum(levels)),
        side_basis_count=side_basis_count,
        basis_size=basis_size,
        basis_orthogonalization=str(config.get("basis_orthogonalization", "normalize")),
    )
    return (
        config,
        shared_structured.detach().cpu().numpy().astype(np.float32, copy=False),
        side_structured.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def build_basis_manifest(
    *,
    checkpoint_path: Path,
    config: dict[str, Any],
    basis: np.ndarray,
    side_basis: np.ndarray,
    point_layout: str,
    point_layout_region_names: list[str] | None,
    exported_basis_path: Path,
    exported_side_basis_path: Path,
) -> dict[str, Any]:
    levels = parse_levels(config.get("levels", "2,3,6"))
    return {
        "checkpoint_path": str(checkpoint_path),
        "mode": str(config.get("mode", "x")),
        "region": str(config.get("region", "mouth")),
        "matrix_size": int(basis.shape[-1]),
        "num_basis": int(basis.shape[0]),
        "levels": list(levels),
        "level_boundaries": compute_level_boundaries(levels),
        "basis_orthogonalization": str(config.get("basis_orthogonalization", "normalize")),
        "quantizer_type": str(config.get("quantizer_type", "latent_quantize")),
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


def export_basis_bundle(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    config: dict[str, Any],
    basis: np.ndarray,
    side_basis: np.ndarray,
    save_heatmaps: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    basis_path = output_dir / "basis_bank_x.npy"
    side_basis_path = output_dir / "side_basis_bank_x.npy"
    np.save(basis_path, basis.astype(np.float32, copy=False))
    np.save(side_basis_path, side_basis.astype(np.float32, copy=False))

    basis_plot_path = None
    side_basis_plot_path = None
    levels = parse_levels(config.get("levels", "2,3,6"))
    if save_heatmaps:
        basis_plot_path = output_dir / "basis_bank_x_heatmap.png"
        plot_basis_grid(basis, levels, basis_plot_path)
        if side_basis.shape[0] > 0:
            side_basis_plot_path = output_dir / "side_basis_bank_x_heatmap.png"
            plot_basis_grid(side_basis, (side_basis.shape[0],), side_basis_plot_path)

    point_layout = resolve_bridge_point_layout(region=str(config.get("region", "mouth")))
    point_layout_region_names = resolve_bridge_point_layout_region_names(
        region=str(config.get("region", "mouth"))
    )
    manifest = build_basis_manifest(
        checkpoint_path=checkpoint_path,
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
        "side_basis_bank_heatmap": str(side_basis_plot_path)
        if side_basis_plot_path is not None
        else None,
    }

    manifest_path = output_dir / "basis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "output_dir": str(output_dir),
        "basis_shape": list(basis.shape),
        "side_basis_shape": list(side_basis.shape),
        "manifest_path": str(manifest_path),
        "basis_path": str(basis_path),
        "side_basis_path": str(side_basis_path),
        "point_layout": point_layout,
        "point_layout_region_names": point_layout_region_names,
    }
    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def export(
    checkpoint_path: str,
    output_dir: str | None = None,
    save_heatmaps: bool = True,
) -> dict[str, Any]:
    """
    Main CLI entry for exporting basis bundles for matrix visualization.

    Parameters:
    - `checkpoint_path`: trained checkpoint
    - `output_dir`: optional custom destination
    - `save_heatmaps`: whether to render compact PNG heatmaps in addition to `.npy` exports
    """
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    ckpt = torch.load(checkpoint, map_location="cpu")
    loaded_config, basis, side_basis = extract_structured_basis_from_checkpoint(checkpoint=ckpt)

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint.parent / "matrix_vis_exports" / "basis"
    )
    summary = export_basis_bundle(
        output_dir=destination,
        checkpoint_path=checkpoint,
        config=loaded_config,
        basis=basis,
        side_basis=side_basis,
        save_heatmaps=save_heatmaps,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"export": export})
