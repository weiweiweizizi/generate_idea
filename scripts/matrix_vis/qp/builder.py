from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.matrix_vis.core.types import QPConfig
from scripts.matrix_vis.qp.variables import VariableLayout, build_time_grid, find_anchor_local_indices

# 构建二次规划
@dataclass(frozen=True)
class AxisQPBundle:
    pair_weights: np.ndarray
    layout: VariableLayout
    time_grid: np.ndarray
    anchor_local_indices: np.ndarray
    observations: pd.DataFrame
    initial_positions: np.ndarray
    subset_point_ids: np.ndarray
    anchor_point_ids: np.ndarray
    solver_config: QPConfig
    observation_i: np.ndarray
    observation_j: np.ndarray
    observation_values: np.ndarray
    observation_targets: np.ndarray
    data_row_indices: np.ndarray
    data_col_indices: np.ndarray
    flat_j_indices: np.ndarray
    flat_i_indices: np.ndarray
    observation_weight_per_time: np.ndarray
    data_diag: np.ndarray
    fixed_var_indices: np.ndarray
    fixed_var_values: np.ndarray
    reg_p: sparse.csc_matrix
    constraint_a: sparse.csc_matrix
    constraint_l: np.ndarray
    constraint_u: np.ndarray


def _flat_index(point_idx: int, time_idx: int, num_time_steps: int) -> int:
    return point_idx * num_time_steps + time_idx


def _build_acceleration_matrix(layout: VariableLayout) -> sparse.csr_matrix:
    if layout.num_time_steps < 3:
        return sparse.csr_matrix((0, layout.num_points * layout.num_time_steps), dtype=np.float32)
    base = sparse.diags(
        diagonals=[np.ones(layout.num_time_steps - 2), -2.0 * np.ones(layout.num_time_steps - 2), np.ones(layout.num_time_steps - 2)],
        offsets=[0, 1, 2],
        shape=(layout.num_time_steps - 2, layout.num_time_steps),
        dtype=np.float32,
    )
    return sparse.kron(sparse.eye(layout.num_points, format="csr", dtype=np.float32), base, format="csr")


def _build_velocity_center_matrix(layout: VariableLayout) -> sparse.csr_matrix:
    if layout.num_time_steps < 2:
        return sparse.csr_matrix((0, layout.num_points * layout.num_time_steps), dtype=np.float32)
    diff = sparse.diags(
        diagonals=[-np.ones(layout.num_time_steps - 1), np.ones(layout.num_time_steps - 1)],
        offsets=[0, 1],
        shape=(layout.num_time_steps - 1, layout.num_time_steps),
        dtype=np.float32,
    ).toarray()
    mean_center = np.eye(layout.num_time_steps - 1, dtype=np.float32) - (
        np.ones((layout.num_time_steps - 1, layout.num_time_steps - 1), dtype=np.float32) / float(layout.num_time_steps - 1)
    )
    centered = mean_center @ diff
    base = sparse.csr_matrix(centered, dtype=np.float32)
    return sparse.kron(sparse.eye(layout.num_points, format="csr", dtype=np.float32), base, format="csr")


def _build_constraint_system(
    *,
    layout: VariableLayout,
    initial_positions: np.ndarray,
    anchor_local_indices: np.ndarray,
    max_displacement: float | None,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row_idx = 0

    for point_idx, initial_value in enumerate(initial_positions.tolist()):
        rows.append(row_idx)
        cols.append(_flat_index(point_idx, 0, layout.num_time_steps))
        data.append(1.0)
        lower.append(float(initial_value))
        upper.append(float(initial_value))
        row_idx += 1

    for anchor_idx in anchor_local_indices.tolist():
        anchor_value = float(initial_positions[anchor_idx])
        for time_idx in range(layout.num_time_steps):
            rows.append(row_idx)
            cols.append(_flat_index(int(anchor_idx), time_idx, layout.num_time_steps))
            data.append(1.0)
            lower.append(anchor_value)
            upper.append(anchor_value)
            row_idx += 1

    if max_displacement is not None:
        displacement_bound = float(max_displacement)
        for point_idx, initial_value in enumerate(initial_positions.tolist()):
            for time_idx in range(layout.num_time_steps):
                rows.append(row_idx)
                cols.append(_flat_index(point_idx, time_idx, layout.num_time_steps))
                data.append(1.0)
                lower.append(float(initial_value) - displacement_bound)
                upper.append(float(initial_value) + displacement_bound)
                row_idx += 1

    a_matrix = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(row_idx, layout.num_points * layout.num_time_steps),
        dtype=np.float32,
    ).tocsc()
    return a_matrix, np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def _build_fixed_variable_spec(
    *,
    layout: VariableLayout,
    initial_positions: np.ndarray,
    anchor_local_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixed: dict[int, float] = {}
    for point_idx, initial_value in enumerate(initial_positions.tolist()):
        fixed[_flat_index(point_idx, 0, layout.num_time_steps)] = float(initial_value)
    for anchor_idx in anchor_local_indices.tolist():
        anchor_value = float(initial_positions[int(anchor_idx)])
        for time_idx in range(layout.num_time_steps):
            fixed[_flat_index(int(anchor_idx), time_idx, layout.num_time_steps)] = anchor_value
    fixed_indices = np.asarray(sorted(fixed.keys()), dtype=np.int64)
    fixed_values = np.asarray([fixed[int(index)] for index in fixed_indices.tolist()], dtype=np.float32)
    return fixed_indices, fixed_values


def _build_regularization_p(
    *,
    layout: VariableLayout,
    solver_config: QPConfig,
) -> sparse.csc_matrix:
    num_variables = layout.num_points * layout.num_time_steps
    p_matrix = sparse.csc_matrix((num_variables, num_variables), dtype=np.float32)
    acc_matrix = _build_acceleration_matrix(layout)
    if acc_matrix.shape[0] > 0 and solver_config.lambda_acc > 0.0:
        p_matrix = p_matrix + (2.0 * solver_config.lambda_acc) * (acc_matrix.T @ acc_matrix).tocsc()
    vel_matrix = _build_velocity_center_matrix(layout)
    if vel_matrix.shape[0] > 0 and solver_config.lambda_vel > 0.0:
        p_matrix = p_matrix + (2.0 * solver_config.lambda_vel) * (vel_matrix.T @ vel_matrix).tocsc()
    return p_matrix


def _compute_pair_weights(
    *,
    initial_positions: np.ndarray,
    observations: pd.DataFrame,
) -> np.ndarray:
    if observations.empty:
        return np.zeros((0,), dtype=np.float32)

    i_idx = observations["i"].to_numpy(dtype=np.int64, copy=False)
    j_idx = observations["j"].to_numpy(dtype=np.int64, copy=False)
    raw_gaps = np.abs(initial_positions[j_idx] - initial_positions[i_idx]).astype(np.float32, copy=False)
    positive_gaps = raw_gaps[raw_gaps > 1e-6]
    if positive_gaps.size == 0:
        return np.ones_like(raw_gaps, dtype=np.float32)

    scale = float(np.median(positive_gaps))
    normalized = raw_gaps / max(scale, 1e-6)
    weights = np.clip(normalized, 0.1, 3.0)
    weights /= float(np.mean(weights))
    return weights.astype(np.float32, copy=False)


def build_axis_qp(
    *,
    subset_point_ids: np.ndarray,
    initial_positions: np.ndarray,
    anchor_point_ids: np.ndarray,
    observations: pd.DataFrame,
    solver_config: QPConfig,
) -> AxisQPBundle:
    subset_point_ids = np.asarray(subset_point_ids, dtype=np.int64)
    initial_positions = np.asarray(initial_positions, dtype=np.float32)
    if subset_point_ids.ndim != 1 or initial_positions.ndim != 1:
        raise ValueError("subset_point_ids and initial_positions must be 1D")
    if subset_point_ids.shape[0] != initial_positions.shape[0]:
        raise ValueError("subset_point_ids and initial_positions length mismatch")

    layout = VariableLayout(
        num_points=int(subset_point_ids.shape[0]),
        num_time_steps=int(solver_config.num_time_steps),
    )
    time_grid = build_time_grid(layout.num_time_steps)
    anchor_local_indices = find_anchor_local_indices(subset_point_ids, anchor_point_ids)

    pair_weights = _compute_pair_weights(
        initial_positions=initial_positions,
        observations=observations,
    )
    observation_i = observations["i"].to_numpy(dtype=np.int64, copy=False)
    observation_j = observations["j"].to_numpy(dtype=np.int64, copy=False)
    observation_values = observations["value"].to_numpy(dtype=np.float32, copy=False)
    observation_targets = (
        observation_values + np.abs(initial_positions[observation_j] - initial_positions[observation_i]).astype(np.float32, copy=False)
    )
    time_offsets = np.arange(layout.num_time_steps, dtype=np.int32)
    cols_j = observation_j[:, None] * layout.num_time_steps + time_offsets[None, :]
    cols_i = observation_i[:, None] * layout.num_time_steps + time_offsets[None, :]
    data_row_indices = np.repeat(
        np.arange(observations.shape[0], dtype=np.int32),
        layout.num_time_steps * 2,
    )
    flat_j_indices = cols_j.reshape(-1).astype(np.int32, copy=False)
    flat_i_indices = cols_i.reshape(-1).astype(np.int32, copy=False)
    data_col_indices = np.stack([cols_j, cols_i], axis=2).reshape(-1).astype(np.int32, copy=False)
    observation_weight_per_time = (
        np.sqrt(pair_weights).astype(np.float32, copy=False) / float(layout.num_time_steps)
    )
    point_weight_sum = np.bincount(
        np.concatenate([observation_i, observation_j]),
        weights=np.concatenate([pair_weights, pair_weights]).astype(np.float64, copy=False),
        minlength=layout.num_points,
    )
    data_diag = np.repeat(
        point_weight_sum.astype(np.float32, copy=False) / float(layout.num_time_steps ** 2),
        layout.num_time_steps,
    )

    reg_p = _build_regularization_p(
        layout=layout,
        solver_config=solver_config,
    )
    constraint_a, constraint_l, constraint_u = _build_constraint_system(
        layout=layout,
        initial_positions=initial_positions,
        anchor_local_indices=anchor_local_indices,
        max_displacement=solver_config.max_displacement,
    )
    fixed_var_indices, fixed_var_values = _build_fixed_variable_spec(
        layout=layout,
        initial_positions=initial_positions,
        anchor_local_indices=anchor_local_indices,
    )

    return AxisQPBundle(
        pair_weights=pair_weights,
        layout=layout,
        time_grid=time_grid,
        anchor_local_indices=anchor_local_indices,
        observations=observations,
        initial_positions=initial_positions,
        subset_point_ids=subset_point_ids,
        anchor_point_ids=np.asarray(anchor_point_ids, dtype=np.int64),
        solver_config=solver_config,
        observation_i=observation_i,
        observation_j=observation_j,
        observation_values=observation_values,
        observation_targets=observation_targets,
        data_row_indices=data_row_indices,
        data_col_indices=data_col_indices,
        flat_j_indices=flat_j_indices,
        flat_i_indices=flat_i_indices,
        observation_weight_per_time=observation_weight_per_time,
        data_diag=data_diag,
        fixed_var_indices=fixed_var_indices,
        fixed_var_values=fixed_var_values,
        reg_p=reg_p,
        constraint_a=constraint_a,
        constraint_l=constraint_l,
        constraint_u=constraint_u,
    )
