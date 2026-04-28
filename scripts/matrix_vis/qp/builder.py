from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from scripts.matrix_vis.core.types import QPConfig
from scripts.matrix_vis.qp.constraints import build_constraints
from scripts.matrix_vis.qp.objective import (
    build_acceleration_term,
    build_data_term,
    build_velocity_change_term,
)
from scripts.matrix_vis.qp.variables import VariableLayout, build_time_grid, find_anchor_local_indices


@dataclass(frozen=True)
class AxisQPBundle:
    problem: cp.Problem
    x_var: cp.Variable
    distance_signs: cp.Parameter
    pair_weights: np.ndarray
    layout: VariableLayout
    time_grid: np.ndarray
    anchor_local_indices: np.ndarray
    observations: pd.DataFrame
    initial_positions: np.ndarray
    subset_point_ids: np.ndarray
    anchor_point_ids: np.ndarray
    solver_config: QPConfig


def _compute_pair_weights(
    *,
    initial_positions: np.ndarray,
    observations: pd.DataFrame,
) -> np.ndarray:
    if observations.empty:
        return np.zeros((0,), dtype=np.float32)

    raw_gaps = []
    for row in observations.itertuples(index=False):
        raw_gaps.append(abs(float(initial_positions[row.j] - initial_positions[row.i])))

    raw_gaps = np.asarray(raw_gaps, dtype=np.float32)
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

    x_var = cp.Variable(layout.shape)
    distance_signs = cp.Parameter((int(observations.shape[0]), layout.num_time_steps))
    pair_weights = _compute_pair_weights(
        initial_positions=initial_positions,
        observations=observations,
    )
    data_term = build_data_term(
        x_var,
        initial_positions=initial_positions,
        observations=observations,
        distance_signs=distance_signs,
        pair_weights=pair_weights,
    )
    acc_term = build_acceleration_term(x_var)
    vel_term = build_velocity_change_term(x_var)

    objective = cp.Minimize(
        solver_config.lambda_data * data_term
        + solver_config.lambda_acc * acc_term
        + solver_config.lambda_vel * vel_term
    )
    constraints = build_constraints(
        x_var,
        initial_positions=initial_positions,
        anchor_local_indices=anchor_local_indices,
        enforce_order=solver_config.enforce_order,
        max_displacement=solver_config.max_displacement,
    )
    problem = cp.Problem(objective, constraints)

    return AxisQPBundle(
        problem=problem,
        x_var=x_var,
        distance_signs=distance_signs,
        pair_weights=pair_weights,
        layout=layout,
        time_grid=time_grid,
        anchor_local_indices=anchor_local_indices,
        observations=observations,
        initial_positions=initial_positions,
        subset_point_ids=subset_point_ids,
        anchor_point_ids=np.asarray(anchor_point_ids, dtype=np.int64),
        solver_config=solver_config,
    )
