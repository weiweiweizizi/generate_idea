from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import osqp
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
import time
from scripts.matrix_vis.qp.builder import AxisQPBundle

# 两种求解器的后端
@dataclass(frozen=True)
class AxisQPSolveResult:
    trajectory: np.ndarray
    diagnostics: dict[str, Any]


def _compute_distance_signs(
    *,
    bundle: AxisQPBundle,
    reference_trajectory: np.ndarray | None,
) -> np.ndarray:
    num_time_steps = int(bundle.layout.num_time_steps)
    initial_positions = np.asarray(bundle.initial_positions, dtype=np.float32)
    if reference_trajectory is None:
        reference_trajectory = np.repeat(initial_positions[:, None], num_time_steps, axis=1)
    else:
        reference_trajectory = np.asarray(reference_trajectory, dtype=np.float32)
    ref_gap = reference_trajectory[bundle.observation_j, :] - reference_trajectory[bundle.observation_i, :]
    signs = np.sign(ref_gap).astype(np.float32, copy=False)
    fallback = np.sign(initial_positions[bundle.observation_j] - initial_positions[bundle.observation_i]).astype(
        np.float32,
        copy=False,
    )
    fallback[fallback == 0.0] = 1.0
    zero_mask = signs == 0.0
    if np.any(zero_mask):
        signs[zero_mask] = np.broadcast_to(fallback[:, None], signs.shape)[zero_mask]
    return signs


def _build_data_matrix(
    *,
    bundle: AxisQPBundle,
    distance_signs: np.ndarray,
) -> sparse.csc_matrix:
    num_observations = int(bundle.observations.shape[0])
    signed_weight = bundle.observation_weight_per_time[:, None] * distance_signs
    values = np.stack([signed_weight, -signed_weight], axis=2).reshape(-1).astype(np.float32, copy=False)
    return sparse.coo_matrix(
        (values, (bundle.data_row_indices, bundle.data_col_indices)),
        shape=(num_observations, bundle.layout.num_points * bundle.layout.num_time_steps),
        dtype=np.float32,
    ).tocsc()


def _build_q_vector(
    *,
    data_matrix: sparse.csc_matrix,
    target: np.ndarray,
    lambda_data: float,
) -> np.ndarray:
    q_vector = (-2.0 * lambda_data) * (data_matrix.T @ target)
    return np.asarray(q_vector, dtype=np.float64)


def _csc_keys(matrix: sparse.csc_matrix) -> np.ndarray:
    num_rows = matrix.shape[0]
    cols = np.repeat(np.arange(matrix.shape[1], dtype=np.int64), np.diff(matrix.indptr))
    rows = matrix.indices.astype(np.int64, copy=False)
    return cols * num_rows + rows


def _align_csc_data_to_pattern(
    *,
    pattern: sparse.csc_matrix,
    pattern_keys: np.ndarray,
    current: sparse.csc_matrix,
) -> np.ndarray:
    aligned = np.zeros_like(pattern.data, dtype=np.float64)
    current_keys = _csc_keys(current)
    positions = np.searchsorted(pattern_keys, current_keys)
    if positions.size and (
        np.any(positions >= pattern_keys.shape[0]) or not np.array_equal(pattern_keys[positions], current_keys)
    ):
        raise ValueError("Updated P matrix sparsity pattern does not match the initialized OSQP pattern")
    aligned[positions] = current.data.astype(np.float64, copy=False)
    return aligned


def _build_osqp_terms(
    *,
    bundle: AxisQPBundle,
    distance_signs: np.ndarray,
) -> tuple[sparse.csc_matrix, np.ndarray]:
    data_matrix = _build_data_matrix(
        bundle=bundle,
        distance_signs=distance_signs,
    )
    p_matrix = bundle.reg_p + (2.0 * bundle.solver_config.lambda_data) * (data_matrix.T @ data_matrix).tocsc()
    p_triu = sparse.triu(p_matrix, format="csc")
    target = np.sqrt(bundle.pair_weights).astype(np.float32, copy=False) * bundle.observation_targets
    q_vector = _build_q_vector(
        data_matrix=data_matrix,
        target=target,
        lambda_data=bundle.solver_config.lambda_data,
    )
    return p_triu, q_vector


def _solve_with_solver(
    *,
    solver: osqp.OSQP,
    q_vector: np.ndarray,
    p_values: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    update_kwargs: dict[str, Any] = {"q": q_vector}
    if p_values is not None:
        update_kwargs["Px"] = p_values
    solver.update(**update_kwargs)
    result = solver.solve()
    if result.x is None:
        raise RuntimeError(f"Sparse OSQP solve failed with status {result.info.status!r}")
    diagnostics = {
        "status": result.info.status,
        "objective_value": float(result.info.obj_val) if result.info.obj_val is not None else None,
        "solver_stats": {
            "solver_name": "OSQP",
            "num_iters": int(result.info.iter) if result.info.iter is not None else None,
            "solve_time": float(result.info.solve_time) if result.info.solve_time is not None else None,
            "setup_time": float(result.info.setup_time) if result.info.setup_time is not None else None,
            "run_time": float(result.info.run_time) if result.info.run_time is not None else None,
        },
    }
    return np.asarray(result.x, dtype=np.float32), diagnostics


def _apply_data_matrix(
    *,
    bundle: AxisQPBundle,
    distance_signs: np.ndarray,
    x_vector: np.ndarray,
) -> np.ndarray:
    x_matrix = np.asarray(x_vector, dtype=np.float64).reshape(bundle.layout.shape)
    signed_gaps = distance_signs * (x_matrix[bundle.observation_j, :] - x_matrix[bundle.observation_i, :])
    return np.sum(
        bundle.observation_weight_per_time[:, None].astype(np.float64, copy=False) * signed_gaps,
        axis=1,
        dtype=np.float64,
    )


def _apply_data_transpose(
    *,
    bundle: AxisQPBundle,
    distance_signs: np.ndarray,
    obs_vector: np.ndarray,
) -> np.ndarray:
    obs_vector = np.asarray(obs_vector, dtype=np.float64).reshape(-1)
    scaled = (
        bundle.observation_weight_per_time[:, None].astype(np.float64, copy=False)
        * obs_vector[:, None]
        * distance_signs.astype(np.float64, copy=False)
    )
    flat = np.zeros(bundle.layout.num_points * bundle.layout.num_time_steps, dtype=np.float64)
    scaled_flat = scaled.reshape(-1)
    np.add.at(flat, bundle.flat_j_indices, scaled_flat)
    np.add.at(flat, bundle.flat_i_indices, -scaled_flat)
    return flat


def _apply_full_p(
    *,
    bundle: AxisQPBundle,
    distance_signs: np.ndarray,
    x_vector: np.ndarray,
) -> np.ndarray:
    x_vector = np.asarray(x_vector, dtype=np.float64).reshape(-1)
    reg_term = bundle.reg_p @ x_vector
    data_forward = _apply_data_matrix(
        bundle=bundle,
        distance_signs=distance_signs,
        x_vector=x_vector,
    )
    data_term = _apply_data_transpose(
        bundle=bundle,
        distance_signs=distance_signs,
        obs_vector=data_forward,
    )
    return np.asarray(reg_term, dtype=np.float64).reshape(-1) + (2.0 * bundle.solver_config.lambda_data) * data_term


def _build_matrix_free_q(
    *,
    bundle: AxisQPBundle,
    distance_signs: np.ndarray,
) -> np.ndarray:
    target = np.sqrt(bundle.pair_weights).astype(np.float64, copy=False) * bundle.observation_targets.astype(
        np.float64,
        copy=False,
    )
    return (-2.0 * bundle.solver_config.lambda_data) * _apply_data_transpose(
        bundle=bundle,
        distance_signs=distance_signs,
        obs_vector=target,
    )


def _solve_matrix_free_cg(bundle: AxisQPBundle) -> AxisQPSolveResult:
    if bundle.solver_config.max_displacement is not None:
        raise NotImplementedError("matrix_free_cg does not support max_displacement constraints yet")

    num_variables = bundle.layout.num_points * bundle.layout.num_time_steps
    fixed_mask = np.zeros(num_variables, dtype=bool)
    fixed_mask[bundle.fixed_var_indices] = True
    free_mask = ~fixed_mask
    free_indices = np.flatnonzero(free_mask)

    fixed_vector = np.zeros(num_variables, dtype=np.float64)
    fixed_vector[bundle.fixed_var_indices] = bundle.fixed_var_values.astype(np.float64, copy=False)
    reg_diag = np.asarray(bundle.reg_p.diagonal(), dtype=np.float64).reshape(-1)
    data_diag = (2.0 * bundle.solver_config.lambda_data) * bundle.data_diag.astype(np.float64, copy=False)
    full_diag = reg_diag + data_diag
    free_diag = np.maximum(full_diag[free_indices], 1e-6)

    def precondition(v: np.ndarray) -> np.ndarray:
        return np.asarray(v, dtype=np.float64) / free_diag

    preconditioner = LinearOperator(
        shape=(free_indices.shape[0], free_indices.shape[0]),
        matvec=precondition,
        dtype=np.float64,
    )

    trajectory_reference: np.ndarray | None = None
    last_signs: np.ndarray | None = None
    num_outer_iterations = 4
    outer_wall_times: list[float] = []
    cg_iterations: list[int] = []

    for outer_idx in range(num_outer_iterations):
        current_signs = _compute_distance_signs(
            bundle=bundle,
            reference_trajectory=trajectory_reference,
        )
        q_full = _build_matrix_free_q(
            bundle=bundle,
            distance_signs=current_signs,
        )
        p_fixed = _apply_full_p(
            bundle=bundle,
            distance_signs=current_signs,
            x_vector=fixed_vector,
        )
        rhs = -(q_full + p_fixed)[free_indices]

        def free_matvec(free_vector: np.ndarray) -> np.ndarray:
            full_vector = np.zeros(num_variables, dtype=np.float64)
            full_vector[free_indices] = np.asarray(free_vector, dtype=np.float64)
            product = _apply_full_p(
                bundle=bundle,
                distance_signs=current_signs,
                x_vector=full_vector,
            )
            return product[free_indices]

        operator = LinearOperator(
            shape=(free_indices.shape[0], free_indices.shape[0]),
            matvec=free_matvec,
            dtype=np.float64,
        )

        if trajectory_reference is None:
            initial_guess = np.repeat(
                bundle.initial_positions[:, None].astype(np.float64, copy=False),
                bundle.layout.num_time_steps,
                axis=1,
            ).reshape(-1)[free_indices]
        else:
            initial_guess = trajectory_reference.astype(np.float64, copy=False).reshape(-1)[free_indices]

        cg_state = {"iters": 0}

        def _count_iterations(_: np.ndarray) -> None:
            cg_state["iters"] += 1

        solve_start = time.perf_counter()
        free_solution, info = cg(
            operator,
            rhs,
            x0=initial_guess,
            tol=1e-6,
            atol=1e-8,
            maxiter=max(200, min(4000, free_indices.shape[0] * 2)),
            M=preconditioner,
            callback=_count_iterations,
        )
        outer_wall_times.append(time.perf_counter() - solve_start)
        cg_iterations.append(int(cg_state["iters"]))
        if info != 0:
            raise RuntimeError(f"matrix_free_cg failed to converge, info={info}")

        solved_full = fixed_vector.copy()
        solved_full[free_indices] = free_solution
        solved_trajectory = solved_full.reshape(bundle.layout.shape).astype(np.float32, copy=False)
        if last_signs is not None and np.array_equal(current_signs, last_signs):
            trajectory_reference = solved_trajectory
            break

        trajectory_reference = solved_trajectory
        last_signs = current_signs
    else:
        outer_idx = num_outer_iterations - 1

    final_vector = trajectory_reference.astype(np.float64, copy=False).reshape(-1)
    final_q = _build_matrix_free_q(
        bundle=bundle,
        distance_signs=last_signs if last_signs is not None else current_signs,
    )
    final_objective = 0.5 * float(final_vector @ _apply_full_p(
        bundle=bundle,
        distance_signs=last_signs if last_signs is not None else current_signs,
        x_vector=final_vector,
    )) + float(final_q @ final_vector)

    diagnostics = {
        "status": "solved",
        "objective_value": final_objective,
        "num_points": int(bundle.layout.num_points),
        "num_time_steps": int(bundle.layout.num_time_steps),
        "num_observations": int(bundle.observations.shape[0]),
        "num_outer_iterations": int(outer_idx + 1),
        "wall_time": {
            "setup_time": 0.0,
            "solve_times": [float(value) for value in outer_wall_times],
            "total_time": float(sum(outer_wall_times)),
        },
        "solver_stats": {
            "solver_name": "matrix_free_cg",
            "num_iters": int(cg_iterations[-1]) if cg_iterations else 0,
            "solve_time": float(outer_wall_times[-1]) if outer_wall_times else 0.0,
            "setup_time": 0.0,
            "run_time": float(sum(outer_wall_times)),
            "outer_cg_iterations": [int(value) for value in cg_iterations],
        },
    }
    return AxisQPSolveResult(
        trajectory=np.asarray(trajectory_reference, dtype=np.float32),
        diagnostics=diagnostics,
    )


def solve_axis_qp(bundle: AxisQPBundle) -> AxisQPSolveResult:
    if bundle.solver_config.qp_backend == "matrix_free_cg":
        return _solve_matrix_free_cg(bundle)

    trajectory_reference: np.ndarray | None = None
    last_signs: np.ndarray | None = None
    num_outer_iterations = 4
    setup_wall_time = 0.0
    solve_wall_times: list[float] = []
    solver: osqp.OSQP | None = None
    p_pattern: sparse.csc_matrix | None = None
    p_pattern_keys: np.ndarray | None = None

    for outer_idx in range(num_outer_iterations):
        current_signs = _compute_distance_signs(
            bundle=bundle,
            reference_trajectory=trajectory_reference,
        )
        p_triu, q_vector = _build_osqp_terms(
            bundle=bundle,
            distance_signs=current_signs,
        )
        solve_start = time.perf_counter()
        if solver is None:
            solver = osqp.OSQP()
            setup_start = time.perf_counter()
            solver.setup(
                P=p_triu,
                q=q_vector,
                A=bundle.constraint_a,
                l=np.asarray(bundle.constraint_l, dtype=np.float64),
                u=np.asarray(bundle.constraint_u, dtype=np.float64),
                verbose=False,
                eps_abs=1e-8,
                eps_rel=1e-8,
                polish=True,
                scaling=10,
                adaptive_rho=True,
                warm_start=True,
            )
            setup_wall_time += time.perf_counter() - setup_start
            p_pattern = p_triu
            p_pattern_keys = _csc_keys(p_pattern)
            solved_vector, iteration_diag = _solve_with_solver(
                solver=solver,
                q_vector=q_vector,
            )
        else:
            if p_pattern is None or p_pattern_keys is None:
                raise RuntimeError("OSQP update state is incomplete")
            p_values = _align_csc_data_to_pattern(
                pattern=p_pattern,
                pattern_keys=p_pattern_keys,
                current=p_triu,
            )
            solved_vector, iteration_diag = _solve_with_solver(
                solver=solver,
                q_vector=q_vector,
                p_values=p_values,
            )
        solve_wall_times.append(time.perf_counter() - solve_start)
        solved_trajectory = solved_vector.reshape(bundle.layout.shape)
        if last_signs is not None and np.array_equal(current_signs, last_signs):
            trajectory_reference = solved_trajectory
            break

        trajectory_reference = solved_trajectory
        last_signs = current_signs
    else:
        outer_idx = num_outer_iterations - 1

    diagnostics = {
        "status": iteration_diag["status"],
        "objective_value": iteration_diag["objective_value"],
        "num_points": int(bundle.layout.num_points),
        "num_time_steps": int(bundle.layout.num_time_steps),
        "num_observations": int(bundle.observations.shape[0]),
        "num_outer_iterations": int(outer_idx + 1),
        "wall_time": {
            "setup_time": float(setup_wall_time),
            "solve_times": [float(value) for value in solve_wall_times],
            "total_time": float(setup_wall_time + sum(solve_wall_times)),
        },
        "solver_stats": iteration_diag["solver_stats"],
    }
    return AxisQPSolveResult(
        trajectory=np.asarray(trajectory_reference, dtype=np.float32),
        diagnostics=diagnostics,
    )
