from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def build_data_term(
    x_var: cp.Variable,
    *,
    initial_positions: np.ndarray,
    observations: pd.DataFrame,
    distance_signs: cp.Parameter,
    pair_weights: np.ndarray,
) -> cp.Expression:
    if observations.empty:
        return cp.Constant(0.0)

    pair_weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    if pair_weights.shape[0] != observations.shape[0]:
        raise ValueError("pair_weights length must match number of observations")

    residuals = []
    num_time_steps = x_var.shape[1]
    for obs_idx, row in enumerate(observations.itertuples(index=False)):
        initial_distance = float(abs(initial_positions[row.j] - initial_positions[row.i]))
        signed_gap = cp.multiply(distance_signs[obs_idx, :], x_var[row.j, :] - x_var[row.i, :])
        mean_distance = cp.sum(signed_gap) / num_time_steps
        predicted_value = mean_distance - initial_distance
        residuals.append(np.sqrt(float(pair_weights[obs_idx])) * (predicted_value - float(row.value)))
    return cp.sum_squares(cp.hstack(residuals))


def build_acceleration_term(x_var: cp.Variable) -> cp.Expression:
    if x_var.shape[1] < 3:
        return cp.Constant(0.0)
    second_diff = x_var[:, 2:] - 2.0 * x_var[:, 1:-1] + x_var[:, :-2]
    return cp.sum_squares(second_diff)


def build_velocity_change_term(x_var: cp.Variable) -> cp.Expression:
    if x_var.shape[1] < 2:
        return cp.Constant(0.0)
    velocity = x_var[:, 1:] - x_var[:, :-1]
    mean_velocity = cp.sum(velocity, axis=1, keepdims=True) / velocity.shape[1]
    return cp.sum_squares(velocity - mean_velocity)
