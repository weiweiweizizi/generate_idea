from __future__ import annotations

import cvxpy as cp
import numpy as np


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
        for idx in range(x_var.shape[0] - 1):
            constraints.append(x_var[idx, :] <= x_var[idx + 1, :])

    if max_displacement is not None:
        centered = x_var - initial_positions[:, None]
        constraints.extend(
            [
                centered <= float(max_displacement),
                centered >= -float(max_displacement),
            ]
        )

    return constraints
