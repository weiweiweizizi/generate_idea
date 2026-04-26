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
from scripts.matrix_vis.qp.variables import VariableLayout, build_time_grid, find_anchor_local_index


@dataclass(frozen=True)
class AxisQPBundle:
    problem: cp.Problem
    x_var: cp.Variable
    layout: VariableLayout
    time_grid: np.ndarray
    anchor_local_index: int
    observations: pd.DataFrame
    initial_positions: np.ndarray


def build_axis_qp(
    *,
    subset_point_ids: np.ndarray,
    initial_positions: np.ndarray,
    anchor_point_id: int,
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
    anchor_local_index = find_anchor_local_index(subset_point_ids, anchor_point_id)

    x_var = cp.Variable(layout.shape)
    data_term = build_data_term(
        x_var,
        initial_positions=initial_positions,
        observations=observations,
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
        anchor_local_index=anchor_local_index,
        enforce_order=solver_config.enforce_order,
        max_displacement=solver_config.max_displacement,
    )
    problem = cp.Problem(objective, constraints)

    return AxisQPBundle(
        problem=problem,
        x_var=x_var,
        layout=layout,
        time_grid=time_grid,
        anchor_local_index=anchor_local_index,
        observations=observations,
        initial_positions=initial_positions,
    )
