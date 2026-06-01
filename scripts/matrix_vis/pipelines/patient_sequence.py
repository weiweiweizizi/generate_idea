from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from disentangleNet.bridge.matrix_vis import (
    load_patient_bundle_bridge,
    restore_physical_observation_scale as bridge_restore_physical_observation_scale,
)
from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
from scripts.matrix_vis.core.mesh import extract_subset_indices
from scripts.matrix_vis.core.observations import basis_to_observation_table
from scripts.matrix_vis.core.projection import project_mesh_to_axis
from scripts.matrix_vis.core.types import BasisObservation, MeshConfig, ProjectionConfig, QPConfig
from scripts.matrix_vis.io.load_patient_reference import load_distance_matrix, load_subset_axis_positions
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
from scripts.matrix_vis.pipelines.reconstruct import truncate_observations
from scripts.matrix_vis.viz.axis_plots import save_axis_trajectory_plot

# 对患者的面部运动序列进行轴重建的主函数，包含以下步骤：
# 1. 加载患者数据bundle，包含多个时间窗口的基函数观测
DEFAULT_MESH_SOURCE = "/home/weizilin/code_reproduction/canonical_face/canonical_face_model.obj"
DEFAULT_LANDMARK_CONFIG = "scripts/matrix_vis/configs/landmarks/mediapipe_face_regions_full.yaml"


def _resolve_subset_point_ids(matrix_size: int) -> tuple[int, ...]:
    if int(matrix_size) == 341:
        return resolve_subset_layout(
            subset_layout="face_regions_grouped",
            subset_layout_source=DEFAULT_LANDMARK_CONFIG,
            subset_layout_extractor_name="mediapipe",
            subset_layout_region_names=None,
        )
    return resolve_subset_layout(
        subset_layout="face_regions_grouped",
        subset_layout_source=DEFAULT_LANDMARK_CONFIG,
        subset_layout_extractor_name="mediapipe",
        subset_layout_region_names=["around_mouth", "mouth"],
    )


def _default_anchor_point_ids(matrix_size: int) -> tuple[int, ...]:
    if int(matrix_size) == 341:
        return (33, 263, 10, 175)
    return (205, 425, 200)


def build_default_projection(*, axis: str, matrix_size: int) -> ProjectionConfig:
    subset_point_ids = _resolve_subset_point_ids(matrix_size)
    anchor_point_ids = _default_anchor_point_ids(matrix_size)
    return ProjectionConfig(
        axis=axis,
        source_axis_index=0 if axis == "x" else 1,
        subset_point_ids=subset_point_ids,
        anchor_point_ids=anchor_point_ids,
        subset_layout="face_regions_grouped",
        subset_layout_source=Path(DEFAULT_LANDMARK_CONFIG).resolve(),
        subset_layout_extractor_name="mediapipe",
        subset_layout_region_names=None if int(matrix_size) == 341 else ("around_mouth", "mouth"),
    )


def build_default_solver(
    num_time_steps: int = 20,
    *,
    lambda_acc: float = 10.0,
    max_displacement: float | None = 0.2,
    max_observations: int | None = None,
    lambda_laplacian: float = 0.0,
    lambda_area_sign: float = 0.0,
    area_barrier_margin: float = 0.05,
    lambda_trajectory_tether: float = 0.0,
) -> QPConfig:
    return QPConfig(
        num_time_steps=int(num_time_steps),
        lambda_data=1.0,
        lambda_acc=float(lambda_acc),
        lambda_vel=1.0,
        enforce_order=False,
        max_displacement=max_displacement,
        qp_backend="torch",
        max_observations=max_observations,
        lambda_laplacian=float(lambda_laplacian),
        lambda_area_sign=float(lambda_area_sign),
        area_barrier_margin=float(area_barrier_margin),
        lambda_trajectory_tether=float(lambda_trajectory_tether),
    )


def restore_physical_observation_scale(
    basis_matrix: np.ndarray,
    observation_scale: float | None,
    *,
    observation_matrix_space: str,
) -> np.ndarray:
    return bridge_restore_physical_observation_scale(
        basis_matrix,
        observation_scale,
        observation_matrix_space=observation_matrix_space,
    )


def clamp_distance_matrix_nonnegative(
    matrix: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, int]:
    clipped = np.asarray(matrix, dtype=np.float32).copy()
    if clipped.ndim != 2 or clipped.shape[0] != clipped.shape[1]:
        raise ValueError(f"Expected square distance matrix, got shape {tuple(clipped.shape)}")
    off_diagonal_mask = ~np.eye(clipped.shape[0], dtype=bool)
    original = clipped.copy()
    clipped[off_diagonal_mask] = np.maximum(clipped[off_diagonal_mask], float(epsilon))
    clipped[np.diag_indices(clipped.shape[0])] = 0.0
    clip_count = int(np.count_nonzero(np.abs(clipped - original) > 1e-8))
    return clipped, clip_count


def clamp_observation_deltas(
    observations,
    *,
    initial_positions: np.ndarray,
    epsilon: float = 1e-6,
):
    clipped = observations.copy()
    i_idx = clipped["i"].to_numpy(dtype=np.int64, copy=False)
    j_idx = clipped["j"].to_numpy(dtype=np.int64, copy=False)
    lower_bounds = -np.abs(initial_positions[j_idx] - initial_positions[i_idx]).astype(np.float32, copy=False) + float(epsilon)
    values = clipped["value"].to_numpy(dtype=np.float32, copy=False)
    clipped_values = np.maximum(values, lower_bounds)
    clipped["value"] = clipped_values
    clip_count = int(np.count_nonzero(clipped_values > values))
    return clipped, clip_count


def build_target_distance_matrix(
    *,
    reference_distance_matrix: np.ndarray,
    delta_matrix: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_distance_matrix, dtype=np.float32)
    delta = np.asarray(delta_matrix, dtype=np.float32)
    if reference.shape != delta.shape:
        raise ValueError(
            "reference_distance_matrix and delta_matrix shape mismatch: "
            f"{tuple(reference.shape)} vs {tuple(delta.shape)}"
        )
    return (reference + delta).astype(np.float32, copy=False)


def run_patient_sequence(
    patient_bundle_path: str,
    output_dir: str | None = None,
    mesh_source: str = DEFAULT_MESH_SOURCE,
    initial_landmark_source: str | None = None,
    initial_distance_matrix_source: str | None = None,
    include_initial_reference_window: bool = False,
    lambda_acc: float = 10.0,
    max_displacement: float | None = 0.2,
    max_observations: int | None = None,
    renormalize_observations: bool = True,
    enforce_nonnegative_targets: bool = True,
    carry_forward_initial_positions: bool = True,
    lambda_laplacian: float = 0.0,
    lambda_area_sign: float = 0.0,
    area_barrier_margin: float = 0.05,
    lambda_trajectory_tether: float = 0.0,
) -> dict:
    bridge = load_patient_bundle_bridge(patient_bundle_path)
    bundle_path = bridge.bundle_path
    data = bridge.data
    bundle_contract = bridge.contract
    mode = bridge.mode
    matrix_size = bridge.matrix_size
    dataset_name = bridge.dataset_name
    subject = bridge.subject
    group_ids = bridge.group_ids

    mesh = load_mesh(
        MeshConfig(
            source=Path(mesh_source).expanduser().resolve(),
            format="mediapipe_canonical_obj",
            dimension="3d",
            point_ids="auto",
            normalization_scope="face_regions",
        )
    )
    projection = build_default_projection(axis=mode, matrix_size=matrix_size)
    axis_projection = project_mesh_to_axis(mesh, projection)
    orthogonal_axis_index = 1 if mode == "x" else 0
    subset_indices = extract_subset_indices(mesh, axis_projection.subset_point_ids)
    orthogonal_static_positions = mesh.points[subset_indices, orthogonal_axis_index].astype(np.float32, copy=False)
    solver_config = build_default_solver(
        lambda_acc=lambda_acc,
        max_displacement=max_displacement,
        max_observations=max_observations,
        lambda_laplacian=lambda_laplacian,
        lambda_area_sign=lambda_area_sign,
        area_barrier_margin=area_barrier_margin,
        lambda_trajectory_tether=lambda_trajectory_tether,
    )

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else bundle_path.parent / f"matrix_vis_sequence_{mode}"
    )
    ensure_output_dir(destination)

    template_initial_positions = axis_projection.subset_positions.copy()
    if initial_landmark_source is None:
        initial_positions = template_initial_positions.copy()
        initial_positions_source = "canonical_template"
    else:
        initial_positions = load_subset_axis_positions(
            landmark_source=initial_landmark_source,
            subset_point_ids=axis_projection.subset_point_ids,
            axis=mode,
        )
        initial_positions_source = str(Path(initial_landmark_source).expanduser().resolve())
    warm_start_trajectory: np.ndarray | None = None
    use_distance_recursion = initial_distance_matrix_source is not None
    if use_distance_recursion:
        reference_distance_matrix = load_distance_matrix(initial_distance_matrix_source)
        if reference_distance_matrix.shape != (axis_projection.subset_point_ids.shape[0], axis_projection.subset_point_ids.shape[0]):
            raise ValueError(
                "initial_distance_matrix_source shape does not match subset size: "
                f"got {tuple(reference_distance_matrix.shape)}, expected "
                f"{(axis_projection.subset_point_ids.shape[0], axis_projection.subset_point_ids.shape[0])}"
            )
        reference_distance_matrix, initial_distance_clip_count = clamp_distance_matrix_nonnegative(
            reference_distance_matrix,
        ) if enforce_nonnegative_targets else (reference_distance_matrix.astype(np.float32, copy=False), 0)
        initial_reference_distance_matrix = reference_distance_matrix.copy()
    else:
        reference_distance_matrix = None
        initial_reference_distance_matrix = None
        initial_distance_clip_count = 0
    per_window_rows = []
    observation_scales = bridge.observation_scales
    value_semantics = str(bundle_contract.get("value_semantics", "mean_distance_delta"))
    observation_matrix_space = str(
        bundle_contract.get("observation_matrix_space", "normalized_input_space")
    )

    def _solve_single_window(
        *,
        window_idx: int,
        prev_window_idx: int,
        group_id: str,
        side_pred: int,
        side_true: int,
        observation_scale: float | None,
        observation_matrix: np.ndarray,
        target_distance_matrix: np.ndarray | None,
        distance_reference_source: str,
        warm_start_trajectory: np.ndarray | None,
    ) -> tuple[dict[str, object], np.ndarray | None, np.ndarray | None]:
        window_dir = ensure_output_dir(destination / f"window_{int(window_idx):03d}")
        working_observation_matrix = np.asarray(observation_matrix, dtype=np.float32)
        if renormalize_observations and observation_scale is not None:
            working_observation_matrix = restore_physical_observation_scale(
                working_observation_matrix,
                observation_scale,
                observation_matrix_space=observation_matrix_space,
            )
        basis_observation = BasisObservation(
            subset_point_ids=axis_projection.subset_point_ids,
            basis_matrix=working_observation_matrix,
            value_semantics=value_semantics,
        )
        full_delta_observations = basis_to_observation_table(basis_observation).frame
        observations, truncation_info = truncate_observations(
            full_delta_observations,
            max_observations=solver_config.max_observations,
        )
        clipped_target_count = 0
        if target_distance_matrix is not None and enforce_nonnegative_targets:
            target_distance_matrix, clipped_target_count = clamp_distance_matrix_nonnegative(
                target_distance_matrix,
            )
        elif target_distance_matrix is None and enforce_nonnegative_targets:
            observations, clipped_target_count = clamp_observation_deltas(
                observations,
                initial_positions=initial_positions,
            )
        qp_bundle = build_axis_qp(
            subset_point_ids=axis_projection.subset_point_ids,
            initial_positions=initial_positions,
            orthogonal_static_positions=orthogonal_static_positions,
            dynamic_axis=mode,
            anchor_point_ids=axis_projection.anchor_point_ids,
            observations=observations,
            solver_config=solver_config,
            target_distance_matrix=target_distance_matrix,
            warm_start_trajectory=warm_start_trajectory,
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
            axis=mode,
        )
        window_summary = {
            "window_idx": int(window_idx),
            "prev_window_idx": int(prev_window_idx),
            "group_id": str(group_id),
            "side_pred": int(side_pred),
            "side_true": int(side_true),
            "observation_scale": None if observation_scale is None else float(observation_scale),
            "renormalize_observations": bool(renormalize_observations),
            "enforce_nonnegative_targets": bool(enforce_nonnegative_targets),
            "observation_truncation": truncation_info,
            "clipped_target_count": int(clipped_target_count),
            "distance_reference_source": distance_reference_source,
            "output_dir": str(window_dir),
            "diagnostics": solve_result.diagnostics,
        }
        save_json(window_dir / "window_summary.json", window_summary)
        return (
            window_summary,
            solve_result.trajectory.astype(np.float32, copy=False),
            target_distance_matrix,
        )

    if use_distance_recursion and include_initial_reference_window:
        assert reference_distance_matrix is not None
        initial_window_summary, initial_trajectory, initial_target_distance_matrix = _solve_single_window(
            window_idx=0,
            prev_window_idx=-1,
            group_id="initial_reference",
            side_pred=-1,
            side_true=-1,
            observation_scale=None,
            observation_matrix=reference_distance_matrix,
            target_distance_matrix=reference_distance_matrix.copy(),
            distance_reference_source=str(Path(initial_distance_matrix_source).expanduser().resolve()),
            warm_start_trajectory=warm_start_trajectory,
        )
        per_window_rows.append(initial_window_summary)
        if initial_target_distance_matrix is not None:
            reference_distance_matrix = initial_target_distance_matrix
        if carry_forward_initial_positions:
            warm_start_trajectory = initial_trajectory.astype(np.float32, copy=False)
            initial_positions = initial_trajectory[:, -1].astype(np.float32, copy=False)

    for item_idx, window_idx in enumerate(data["window_indices"].astype(np.int64).tolist()):
        observation_scale = None if observation_scales is None else float(observation_scales[item_idx])
        observation_matrix = data["composed_basis_matrices"][item_idx].astype(np.float32)
        target_distance_matrix = None
        distance_reference_source = "legacy_initial_position_gaps"
        if use_distance_recursion:
            assert reference_distance_matrix is not None
            delta_matrix = observation_matrix.astype(np.float32, copy=False)
            if renormalize_observations and observation_scale is not None:
                delta_matrix = restore_physical_observation_scale(
                    delta_matrix,
                    observation_scale,
                    observation_matrix_space=observation_matrix_space,
                )
            target_distance_matrix = build_target_distance_matrix(
                reference_distance_matrix=reference_distance_matrix,
                delta_matrix=delta_matrix,
            )
            distance_reference_source = (
                str(Path(initial_distance_matrix_source).expanduser().resolve())
                if item_idx == 0
                else "previous_window_target_distance_matrix"
            )
        window_summary, solve_trajectory, effective_target_distance_matrix = _solve_single_window(
            window_idx=int(window_idx),
            prev_window_idx=int(data["prev_window_indices"][item_idx]),
            group_id=str(group_ids[item_idx]),
            side_pred=int(data["side_pred"][item_idx]),
            side_true=int(data["side_true"][item_idx]),
            observation_scale=observation_scale,
            observation_matrix=observation_matrix,
            target_distance_matrix=target_distance_matrix,
            distance_reference_source=distance_reference_source,
            warm_start_trajectory=warm_start_trajectory,
        )
        per_window_rows.append(window_summary)
        if use_distance_recursion and effective_target_distance_matrix is not None:
            reference_distance_matrix = effective_target_distance_matrix
        if carry_forward_initial_positions:
            warm_start_trajectory = solve_trajectory.astype(np.float32, copy=False)
            initial_positions = solve_trajectory[:, -1].astype(np.float32, copy=False)
        else:
            warm_start_trajectory = None

    stitched_trajectories = []
    stitched_time_grids = []
    time_offset = 0.0
    stitched_initial_positions = None
    for row in per_window_rows:
        solution = np.load(Path(row["output_dir"]) / "solution.npz")
        stitched_trajectories.append(solution["trajectory"].astype(np.float32))
        window_time = solution["time_grid"].astype(np.float32)
        stitched_time_grids.append(window_time + time_offset)
        time_offset = float(window_time[-1] + time_offset + 1.0)
        if stitched_initial_positions is None:
            stitched_initial_positions = solution["initial_positions"].astype(np.float32)

    save_solution_npz(
        output_dir=destination,
        point_ids=axis_projection.subset_point_ids,
        time_grid=np.concatenate(stitched_time_grids, axis=0),
        initial_positions=(
            stitched_initial_positions
            if stitched_initial_positions is not None
            else axis_projection.subset_positions
        ),
        trajectory=np.concatenate(stitched_trajectories, axis=1),
        anchor_point_ids=axis_projection.anchor_point_ids,
        basis_observation=BasisObservation(
            subset_point_ids=axis_projection.subset_point_ids,
            basis_matrix=(
                initial_reference_distance_matrix
                if use_distance_recursion and initial_reference_distance_matrix is not None
                else (
                    restore_physical_observation_scale(
                        data["composed_basis_matrices"][0].astype(np.float32),
                        None if observation_scales is None else float(observation_scales[0]),
                        observation_matrix_space=observation_matrix_space,
                    )
                    if renormalize_observations
                    else data["composed_basis_matrices"][0].astype(np.float32)
                )
            ),
            value_semantics=value_semantics,
        ),
    )
    stitched_solution_path = destination / "solution.npz"

    sequence_summary = {
        "patient_bundle_path": str(bundle_path),
        "output_dir": str(destination),
        "bundle_contract": bundle_contract,
        "mode": mode,
        "matrix_size": matrix_size,
        "anchor_point_ids": axis_projection.anchor_point_ids.astype(int).tolist(),
        "num_windows": int(len(per_window_rows)),
        "lambda_acc": float(lambda_acc),
        "max_displacement": None if max_displacement is None else float(max_displacement),
        "max_observations": None if max_observations is None else int(max_observations),
        "lambda_laplacian": float(solver_config.lambda_laplacian),
        "lambda_area_sign": float(solver_config.lambda_area_sign),
        "area_barrier_margin": float(solver_config.area_barrier_margin),
        "lambda_trajectory_tether": float(solver_config.lambda_trajectory_tether),
        "renormalize_observations": bool(renormalize_observations),
        "enforce_nonnegative_targets": bool(enforce_nonnegative_targets),
        "initial_positions_source": initial_positions_source,
        "initial_distance_matrix_source": (
            str(Path(initial_distance_matrix_source).expanduser().resolve())
            if initial_distance_matrix_source is not None
            else None
        ),
        "initial_distance_clip_count": int(initial_distance_clip_count),
        "uses_distance_recursion": bool(use_distance_recursion),
        "carry_forward_initial_positions": bool(carry_forward_initial_positions),
        "window_indices": [row["window_idx"] for row in per_window_rows],
        "window_dirs": [row["output_dir"] for row in per_window_rows],
        "sequence_solution": str(stitched_solution_path),
    }
    save_json(destination / "sequence_summary.json", sequence_summary)
    save_json(destination / "sequence_manifest.json", {"windows": per_window_rows})
    print(json.dumps(sequence_summary, indent=2, ensure_ascii=False))
    return sequence_summary
