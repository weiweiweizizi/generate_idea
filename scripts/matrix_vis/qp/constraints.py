from __future__ import annotations

import numpy as np
import cvxpy as cp


def build_constraints(
    x_var: cp.Variable,
    *,
    initial_positions: np.ndarray,
    anchor_local_indices: np.ndarray,
    enforce_order: bool,
    max_displacement: float | None,
) -> list[cp.Constraint]:
    initial_positions = np.asarray(initial_positions, dtype=np.float32)
    anchor_local_indices = np.asarray(anchor_local_indices, dtype=np.int64)
    anchor_targets = np.repeat(
        initial_positions[anchor_local_indices, None],
        x_var.shape[1],
        axis=1,
    )
    constraints: list[cp.Constraint] = [
        x_var[:, 0] == initial_positions,
        x_var[anchor_local_indices, :] == anchor_targets,
    ]

    if max_displacement is not None:
        centered = x_var - initial_positions[:, None]
        constraints.extend(
            [
                centered <= float(max_displacement),
                centered >= -float(max_displacement),
            ]
        )

    return constraints
