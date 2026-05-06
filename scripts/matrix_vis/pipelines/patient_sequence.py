from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
from scripts.matrix_vis.core.observations import basis_to_observation_table
from scripts.matrix_vis.core.projection import project_mesh_to_axis
from scripts.matrix_vis.core.types import BasisObservation, MeshConfig, ProjectionConfig, QPConfig
from scripts.matrix_vis.io.load_mesh import load_mesh
from scripts.matrix_vis.io.save_results import (
    ensure_output_dir,
    save_json,
    save_observations,
    save_projected_mesh,
    save_solution_npz,
)
from scripts.matrix_vis.qp.builder import build_axis_qp
from scripts.matrix_vis.qp.solve import solve_axis_qp
from scripts.matrix_vis.viz.axis_plots import save_axis_trajectory_plot

# 对患者的面部运动序列进行轴重建的主函数，包含以下步骤：
# 1. 加载患者数据bundle，包含多个时间窗口的基函数观测
DEFAULT_MESH_SOURCE = "/home/weizilin/code_reproduction/canonical_face/canonical_face_model.obj"
DEFAULT_LANDMARK_CONFIG = "scripts/matrix_vis/configs/landmarks/mediapipe_face_regions_full.yaml"

def build_default_projection() -> ProjectionConfig:
    subset_point_ids = resolve_subset_layout(
        subset_layout="face_regions_grouped",
        subset_layout_source=DEFAULT_LANDMARK_CONFIG,
        subset_layout_extractor_name="mediapipe",
        subset_layout_region_names=["around_mouth", "mouth"],
    )
    return ProjectionConfig(
        axis="x",
        source_axis_index=0,
        subset_point_ids=subset_point_ids,
        anchor_point_ids=(14,),
        subset_layout="face_regions_grouped",
        subset_layout_source=Path(DEFAULT_LANDMARK_CONFIG).resolve(),
        subset_layout_extractor_name="mediapipe",
        subset_layout_region_names=("around_mouth", "mouth"),
    )


def build_default_solver(num_time_steps: int = 20) -> QPConfig:
    return QPConfig(
        num_time_steps=int(num_time_steps),
        lambda_data=1.0,
        lambda_acc=10.0,
        lambda_vel=1.0,
        enforce_order=False,
        max_displacement=None,
        qp_backend="matrix_free_cg",
        max_observations=None,
    )


def run_patient_sequence(
    patient_bundle_path: str,
    output_dir: str | None = None,
    mesh_source: str = DEFAULT_MESH_SOURCE,
) -> dict:
    bundle_path = Path(patient_bundle_path).expanduser().resolve()
    data = np.load(bundle_path, allow_pickle=True)

    mesh = load_mesh(
        MeshConfig(
            source=Path(mesh_source).expanduser().resolve(),
            format="mediapipe_canonical_obj",
            dimension="3d",
            point_ids="auto",
            normalization_scope="face_regions",
        )
    )
    projection = build_default_projection()
    axis_projection = project_mesh_to_axis(mesh, projection)
    solver_config = build_default_solver()

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else bundle_path.parent / "matrix_vis_sequence_x"
    )
    ensure_output_dir(destination)

    initial_positions = axis_projection.subset_positions.copy()
    per_window_rows = []

    for item_idx, window_idx in enumerate(data["window_indices"].astype(np.int64).tolist()):
        window_dir = ensure_output_dir(destination / f"window_{int(window_idx):03d}")
        basis_observation = BasisObservation(
            subset_point_ids=axis_projection.subset_point_ids,
            basis_matrix=data["composed_basis_matrices"][item_idx].astype(np.float32),
            value_semantics="mean_distance_delta",
        )
        observations = basis_to_observation_table(basis_observation).frame
        qp_bundle = build_axis_qp(
            subset_point_ids=axis_projection.subset_point_ids,
            initial_positions=initial_positions,
            anchor_point_ids=axis_projection.anchor_point_ids,
            observations=observations,
            solver_config=solver_config,
        )
        solve_result = solve_axis_qp(qp_bundle)

        save_projected_mesh(axis_projection, window_dir)
        save_observations(observations, window_dir)
        save_solution_npz(
            output_dir=window_dir,
            point_ids=axis_projection.subset_point_ids,
            time_grid=qp_bundle.time_grid,
            initial_positions=initial_positions,
            trajectory=solve_result.trajectory,
            anchor_point_ids=axis_projection.anchor_point_ids,
            basis_observation=basis_observation,
        )
        save_axis_trajectory_plot(
            output_dir=window_dir,
            time_grid=qp_bundle.time_grid,
            trajectory=solve_result.trajectory,
            point_ids=axis_projection.subset_point_ids,
            axis="x",
        )
        window_summary = {
            "window_idx": int(window_idx),
            "prev_window_idx": int(data["prev_window_indices"][item_idx]),
            "group_id": str(data["group_id"][item_idx]),
            "side_pred": int(data["side_pred"][item_idx]),
            "side_true": int(data["side_true"][item_idx]),
            "output_dir": str(window_dir),
            "diagnostics": solve_result.diagnostics,
        }
        save_json(window_dir / "window_summary.json", window_summary)
        per_window_rows.append(window_summary)
        initial_positions = solve_result.trajectory[:, -1].astype(np.float32, copy=False)

    sequence_summary = {
        "patient_bundle_path": str(bundle_path),
        "output_dir": str(destination),
        "num_windows": int(len(per_window_rows)),
        "window_indices": [row["window_idx"] for row in per_window_rows],
        "window_dirs": [row["output_dir"] for row in per_window_rows],
    }
    save_json(destination / "sequence_summary.json", sequence_summary)
    save_json(destination / "sequence_manifest.json", {"windows": per_window_rows})
    print(json.dumps(sequence_summary, indent=2, ensure_ascii=False))
    return sequence_summary
