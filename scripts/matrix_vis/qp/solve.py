from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.matrix_vis.qp.builder import AxisQPBundle


@dataclass(frozen=True)
class AxisQPSolveResult:
    trajectory: np.ndarray
    diagnostics: dict[str, Any]


def solve_axis_qp(bundle: AxisQPBundle) -> AxisQPSolveResult:
    bundle.problem.solve(
        solver="OSQP",
        verbose=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
        polish=True,
    )
    if bundle.x_var.value is None:
        raise RuntimeError(f"QP solve failed with status {bundle.problem.status!r}")

    diagnostics = {
        "status": bundle.problem.status,
        "objective_value": float(bundle.problem.value) if bundle.problem.value is not None else None,
        "num_points": int(bundle.layout.num_points),
        "num_time_steps": int(bundle.layout.num_time_steps),
        "num_observations": int(bundle.observations.shape[0]),
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
