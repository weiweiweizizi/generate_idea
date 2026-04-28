from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scripts.matrix_vis.qp.builder import AxisQPBundle


@dataclass(frozen=True)
class AxisQPSolveResult:
    trajectory: np.ndarray
    diagnostics: dict[str, Any]


def _compute_distance_signs(
    *,
    bundle: AxisQPBundle,
    reference_trajectory: np.ndarray | None,
) -> np.ndarray:
    num_observations = int(bundle.observations.shape[0])
    num_time_steps = int(bundle.layout.num_time_steps)
    signs = np.ones((num_observations, num_time_steps), dtype=np.float32)

    initial_positions = np.asarray(bundle.initial_positions, dtype=np.float32)
    if reference_trajectory is None:
        reference_trajectory = np.repeat(initial_positions[:, None], num_time_steps, axis=1)
    else:
        reference_trajectory = np.asarray(reference_trajectory, dtype=np.float32)

    for obs_idx, row in enumerate(bundle.observations.itertuples(index=False)):
        ref_gap = reference_trajectory[row.j, :] - reference_trajectory[row.i, :]
        sign_row = np.sign(ref_gap).astype(np.float32, copy=False)
        if np.any(sign_row == 0.0):
            fallback = float(np.sign(initial_positions[row.j] - initial_positions[row.i]))
            if fallback == 0.0:
                fallback = 1.0
            sign_row[sign_row == 0.0] = fallback
        signs[obs_idx, :] = sign_row
    return signs


def solve_axis_qp(bundle: AxisQPBundle) -> AxisQPSolveResult:
    trajectory_reference: np.ndarray | None = None
    last_signs: np.ndarray | None = None
    num_outer_iterations = 4

    for outer_idx in range(num_outer_iterations):
        current_signs = _compute_distance_signs(
            bundle=bundle,
            reference_trajectory=trajectory_reference,
        )
        bundle.distance_signs.value = current_signs
        bundle.problem.solve(
            solver="OSQP",
            verbose=False,
            eps_abs=1e-8,
            eps_rel=1e-8,
            polish=True,
        )
        if bundle.x_var.value is None:
            raise RuntimeError(f"QP solve failed with status {bundle.problem.status!r}")

        solved_trajectory = np.asarray(bundle.x_var.value, dtype=np.float32)
        if last_signs is not None and np.array_equal(current_signs, last_signs):
            trajectory_reference = solved_trajectory
            break

        trajectory_reference = solved_trajectory
        last_signs = current_signs
    else:
        outer_idx = num_outer_iterations - 1

    diagnostics = {
        "status": bundle.problem.status,
        "objective_value": float(bundle.problem.value) if bundle.problem.value is not None else None,
        "num_points": int(bundle.layout.num_points),
        "num_time_steps": int(bundle.layout.num_time_steps),
        "num_observations": int(bundle.observations.shape[0]),
        "num_outer_iterations": int(outer_idx + 1),
        "solver_stats": {
            "solver_name": getattr(bundle.problem.solver_stats, "solver_name", None),
            "num_iters": getattr(bundle.problem.solver_stats, "num_iters", None),
            "solve_time": getattr(bundle.problem.solver_stats, "solve_time", None),
        },
    }
    return AxisQPSolveResult(
        trajectory=np.asarray(bundle.x_var.value, dtype=np.float32),
        diagnostics=diagnostics,
    )
