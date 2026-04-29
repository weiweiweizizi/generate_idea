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

    initial_positions = np.asarray(initial_positions, dtype=np.float32).reshape(-1)
    pair_weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    if pair_weights.shape[0] != observations.shape[0]:
        raise ValueError("pair_weights length must match number of observations")

    num_time_steps = int(x_var.shape[1])
    i_idx = observations["i"].to_numpy(dtype=np.int64, copy=False)
    j_idx = observations["j"].to_numpy(dtype=np.int64, copy=False)
    observed_values = observations["value"].to_numpy(dtype=np.float32, copy=False)
    initial_distances = np.abs(initial_positions[j_idx] - initial_positions[i_idx]).astype(np.float32, copy=False)

    # Build one batched residual expression instead of thousands of scalar cvxpy nodes.
    signed_gaps = cp.multiply(distance_signs, x_var[j_idx, :] - x_var[i_idx, :])
    mean_distances = cp.sum(signed_gaps, axis=1) / float(num_time_steps)
    residuals = mean_distances - initial_distances - observed_values
    sqrt_weights = np.sqrt(pair_weights).astype(np.float32, copy=False)
    weighted_residuals = cp.multiply(sqrt_weights, residuals)
    return cp.sum_squares(weighted_residuals)


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
