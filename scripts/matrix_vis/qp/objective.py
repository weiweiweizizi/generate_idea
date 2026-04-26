from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def build_data_term(
    x_var: cp.Variable,
    *,
    initial_positions: np.ndarray,
    observations: pd.DataFrame,
) -> cp.Expression:
    if observations.empty:
        return cp.Constant(0.0)

    residuals = []
    num_time_steps = x_var.shape[1]
    for row in observations.itertuples(index=False):
        initial_distance = float(initial_positions[row.j] - initial_positions[row.i])
        mean_distance = cp.sum(x_var[row.j, :] - x_var[row.i, :]) / num_time_steps
        predicted_value = mean_distance - initial_distance
        residuals.append(predicted_value - float(row.value))
    return cp.sum_squares(cp.hstack(residuals))


def build_acceleration_term(x_var: cp.Variable) -> cp.Expression:
    if x_var.shape[1] < 3:
        return cp.Constant(0.0)
    second_diff = x_var[:, 2:] - 2.0 * x_var[:, 1:-1] + x_var[:, :-2]
    return cp.sum_squares(second_diff)


def build_velocity_change_term(x_var: cp.Variable) -> cp.Expression:
    if x_var.shape[1] < 3:
        return cp.Constant(0.0)
    velocity = x_var[:, 1:] - x_var[:, :-1]
    velocity_delta = velocity[:, 1:] - velocity[:, :-1]
    return cp.sum_squares(velocity_delta)
