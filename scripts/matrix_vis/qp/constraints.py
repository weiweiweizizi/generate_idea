from __future__ import annotations

import cvxpy as cp
import numpy as np


def build_order_indices(initial_positions: np.ndarray) -> np.ndarray:
    initial_positions = np.asarray(initial_positions, dtype=np.float32)
    if initial_positions.ndim != 1:
        raise ValueError("initial_positions must be 1D")
    # Stable sort keeps deterministic behavior for ties.
    return np.argsort(initial_positions, kind="mergesort").astype(np.int64)


def build_constraints(
    x_var: cp.Variable,
    *,
    initial_positions: np.ndarray,
    anchor_local_index: int,
    enforce_order: bool,
    max_displacement: float | None,
) -> list[cp.Constraint]:
    initial_positions = np.asarray(initial_positions, dtype=np.float32)
    constraints: list[cp.Constraint] = [
        x_var[:, 0] == initial_positions,
        x_var[anchor_local_index, :] == initial_positions[anchor_local_index],
    ]

    if enforce_order:
        order_indices = build_order_indices(initial_positions)
        for left_idx, right_idx in zip(order_indices[:-1], order_indices[1:]):
            constraints.append(x_var[int(left_idx), :] <= x_var[int(right_idx), :])

    if max_displacement is not None:
        centered = x_var - initial_positions[:, None]
        constraints.extend(
            [
                centered <= float(max_displacement),
                centered >= -float(max_displacement),
            ]
        )

    return constraints
