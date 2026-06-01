from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch
from scipy import sparse

from scripts.matrix_vis.qp.builder import AxisQPBundle


@dataclass(frozen=True)
class AxisQPSolveResult:
    trajectory: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _TorchBundle:
    reg_p: torch.Tensor
    fixed_full: torch.Tensor
    free_indices: torch.Tensor
    observation_i: torch.Tensor
    observation_j: torch.Tensor
    observation_targets: torch.Tensor
    observation_weight_per_time: torch.Tensor
    flat_i_indices: torch.Tensor
    flat_j_indices: torch.Tensor
    free_center: torch.Tensor
    free_lower: torch.Tensor | None
    free_upper: torch.Tensor | None
    free_diag: torch.Tensor
    num_variables: int
    shape: tuple[int, int]
    orthogonal_static_positions: torch.Tensor
    area_triangle_indices: torch.Tensor
    area_reference_signs: torch.Tensor
    area_reference_scales: torch.Tensor
    trajectory_tether_reference_full: torch.Tensor | None
    dynamic_axis: str


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
    prior = bundle.distance_sign_prior
    if prior is not None:
        prior = np.asarray(prior, dtype=np.float32)
        if prior.shape != (bundle.observations.shape[0],):
            raise ValueError(
                "distance_sign_prior shape does not match observations: "
                f"got {tuple(prior.shape)}, expected {(bundle.observations.shape[0],)}"
            )
        prior_matrix = np.broadcast_to(prior[:, None], signs.shape)
        initial_gap = initial_positions[bundle.observation_j] - initial_positions[bundle.observation_i]
        flip_threshold = np.maximum(0.05, 0.25 * np.abs(initial_gap).astype(np.float32, copy=False))
        allow_flip = np.abs(ref_gap) > flip_threshold[:, None]
        signs = np.where(allow_flip, signs, prior_matrix)
    zero_mask = signs == 0.0
    if np.any(zero_mask):
        signs[zero_mask] = np.broadcast_to(fallback[:, None], signs.shape)[zero_mask]
    return signs


def _build_linear_trajectory_reference(
    *,
    initial_positions: np.ndarray,
    endpoint_positions: np.ndarray,
    num_time_steps: int,
) -> np.ndarray:
    initial_positions = np.asarray(initial_positions, dtype=np.float32).reshape(-1)
    endpoint_positions = np.asarray(endpoint_positions, dtype=np.float32).reshape(-1)
    if initial_positions.shape != endpoint_positions.shape:
        raise ValueError(
            "initial_positions and endpoint_positions shape mismatch: "
            f"{tuple(initial_positions.shape)} vs {tuple(endpoint_positions.shape)}"
        )
    if num_time_steps < 1:
        raise ValueError(f"num_time_steps must be positive, got {num_time_steps}")

    if num_time_steps == 1:
        return initial_positions[:, None].astype(np.float32, copy=False)

    alpha = np.linspace(0.0, 1.0, num_time_steps, dtype=np.float32)[None, :]
    delta = (endpoint_positions - initial_positions).astype(np.float32, copy=False)[:, None]
    return (initial_positions[:, None] + delta * alpha).astype(np.float32, copy=False)


def _scipy_to_torch_sparse(matrix: sparse.csc_matrix, *, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
    values = coo.data.astype(np.float32, copy=False)
    return torch.sparse_coo_tensor(
        indices=torch.as_tensor(indices, dtype=torch.int64, device=device),
        values=torch.as_tensor(values, dtype=torch.float32, device=device),
        size=coo.shape,
        device=device,
    ).coalesce()


def _select_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu_fallback"


def _prepare_torch_bundle(
    bundle: AxisQPBundle,
    *,
    device: torch.device,
    trajectory_tether_reference: np.ndarray | None = None,
) -> _TorchBundle:
    num_variables = bundle.layout.num_points * bundle.layout.num_time_steps
    fixed_mask = np.zeros(num_variables, dtype=bool)
    fixed_mask[bundle.fixed_var_indices] = True
    free_mask = ~fixed_mask
    free_indices = np.flatnonzero(free_mask).astype(np.int64, copy=False)

    fixed_full = np.zeros(num_variables, dtype=np.float32)
    fixed_full[bundle.fixed_var_indices] = bundle.fixed_var_values.astype(np.float32, copy=False)

    centers = np.repeat(
        bundle.initial_positions[:, None].astype(np.float32, copy=False),
        bundle.layout.num_time_steps,
        axis=1,
    ).reshape(-1)
    free_center = centers[free_indices]

    if bundle.solver_config.max_displacement is None:
        free_lower = None
        free_upper = None
    else:
        displacement_bound = float(bundle.solver_config.max_displacement)
        free_lower = free_center - displacement_bound
        free_upper = free_center + displacement_bound

    reg_diag = np.asarray(bundle.reg_p.diagonal(), dtype=np.float32).reshape(-1)
    data_diag = (2.0 * bundle.solver_config.lambda_data) * bundle.data_diag.astype(np.float32, copy=False)
    tether_diag = 0.0
    if bundle.solver_config.lambda_trajectory_tether > 0.0 and trajectory_tether_reference is not None:
        tether_diag = 2.0 * float(bundle.solver_config.lambda_trajectory_tether)
    full_diag = reg_diag + data_diag + tether_diag
    free_diag = np.maximum(full_diag[free_indices], 1e-6)

    trajectory_tether_reference_full = None
    if trajectory_tether_reference is not None:
        tether_full = np.asarray(trajectory_tether_reference, dtype=np.float32)
        if tether_full.shape != bundle.layout.shape:
            raise ValueError(
                "trajectory_tether_reference shape mismatch: "
                f"got {tuple(tether_full.shape)}, expected {bundle.layout.shape}"
            )
        trajectory_tether_reference_full = torch.as_tensor(
            tether_full.reshape(-1),
            dtype=torch.float32,
            device=device,
        )

    return _TorchBundle(
        reg_p=_scipy_to_torch_sparse(bundle.reg_p, device=device),
        fixed_full=torch.as_tensor(fixed_full, dtype=torch.float32, device=device),
        free_indices=torch.as_tensor(free_indices, dtype=torch.int64, device=device),
        observation_i=torch.as_tensor(bundle.observation_i.astype(np.int64, copy=False), dtype=torch.int64, device=device),
        observation_j=torch.as_tensor(bundle.observation_j.astype(np.int64, copy=False), dtype=torch.int64, device=device),
        observation_targets=torch.as_tensor(
            (np.sqrt(bundle.pair_weights).astype(np.float32, copy=False) * bundle.observation_targets).astype(
                np.float32,
                copy=False,
            ),
            dtype=torch.float32,
            device=device,
        ),
        observation_weight_per_time=torch.as_tensor(
            bundle.observation_weight_per_time.astype(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
        ),
        flat_i_indices=torch.as_tensor(bundle.flat_i_indices.astype(np.int64, copy=False), dtype=torch.int64, device=device),
        flat_j_indices=torch.as_tensor(bundle.flat_j_indices.astype(np.int64, copy=False), dtype=torch.int64, device=device),
        free_center=torch.as_tensor(free_center, dtype=torch.float32, device=device),
        free_lower=None if free_lower is None else torch.as_tensor(free_lower, dtype=torch.float32, device=device),
        free_upper=None if free_upper is None else torch.as_tensor(free_upper, dtype=torch.float32, device=device),
        free_diag=torch.as_tensor(free_diag, dtype=torch.float32, device=device),
        num_variables=num_variables,
        shape=bundle.layout.shape,
        orthogonal_static_positions=torch.as_tensor(
            bundle.orthogonal_static_positions.astype(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
        ),
        area_triangle_indices=torch.as_tensor(
            bundle.area_triangle_indices.astype(np.int64, copy=False),
            dtype=torch.int64,
            device=device,
        ),
        area_reference_signs=torch.as_tensor(
            bundle.area_reference_signs.astype(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
        ),
        area_reference_scales=torch.as_tensor(
            bundle.area_reference_scales.astype(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
        ),
        trajectory_tether_reference_full=trajectory_tether_reference_full,
        dynamic_axis=bundle.dynamic_axis,
    )


def _apply_data_matrix_torch(
    *,
    state: _TorchBundle,
    distance_signs: torch.Tensor,
    x_vector: torch.Tensor,
) -> torch.Tensor:
    x_matrix = x_vector.reshape(state.shape)
    signed_gaps = distance_signs * (x_matrix[state.observation_j, :] - x_matrix[state.observation_i, :])
    return torch.sum(state.observation_weight_per_time[:, None] * signed_gaps, dim=1)


def _apply_data_transpose_torch(
    *,
    state: _TorchBundle,
    distance_signs: torch.Tensor,
    obs_vector: torch.Tensor,
) -> torch.Tensor:
    scaled = state.observation_weight_per_time[:, None] * obs_vector[:, None] * distance_signs
    flat = torch.zeros(state.num_variables, dtype=torch.float32, device=obs_vector.device)
    scaled_flat = scaled.reshape(-1)
    flat.index_add_(0, state.flat_j_indices, scaled_flat)
    flat.index_add_(0, state.flat_i_indices, -scaled_flat)
    return flat


def _apply_full_p_torch(
    *,
    bundle: AxisQPBundle,
    state: _TorchBundle,
    distance_signs: torch.Tensor,
    x_vector: torch.Tensor,
) -> torch.Tensor:
    reg_term = torch.sparse.mm(state.reg_p, x_vector[:, None]).reshape(-1)
    data_forward = _apply_data_matrix_torch(
        state=state,
        distance_signs=distance_signs,
        x_vector=x_vector,
    )
    data_term = _apply_data_transpose_torch(
        state=state,
        distance_signs=distance_signs,
        obs_vector=data_forward,
    )
    tether_term = torch.zeros_like(reg_term)
    if bundle.solver_config.lambda_trajectory_tether > 0.0:
        tether_term = (2.0 * bundle.solver_config.lambda_trajectory_tether) * x_vector
    return reg_term + (2.0 * bundle.solver_config.lambda_data) * data_term + tether_term


def _build_matrix_free_q_torch(
    *,
    bundle: AxisQPBundle,
    state: _TorchBundle,
    distance_signs: torch.Tensor,
) -> torch.Tensor:
    q = (-2.0 * bundle.solver_config.lambda_data) * _apply_data_transpose_torch(
        state=state,
        distance_signs=distance_signs,
        obs_vector=state.observation_targets,
    )
    if bundle.solver_config.lambda_trajectory_tether > 0.0 and state.trajectory_tether_reference_full is not None:
        q = q - (2.0 * bundle.solver_config.lambda_trajectory_tether) * state.trajectory_tether_reference_full
    return q


def _assemble_full_vector(state: _TorchBundle, free_values: torch.Tensor) -> torch.Tensor:
    full = state.fixed_full.clone()
    full[state.free_indices] = free_values
    return full


def _clamp_free_values(state: _TorchBundle, free_values: torch.Tensor) -> torch.Tensor:
    if state.free_lower is None or state.free_upper is None:
        return free_values
    return torch.minimum(torch.maximum(free_values, state.free_lower), state.free_upper)


def _initial_free_guess(
    *,
    bundle: AxisQPBundle,
    state: _TorchBundle,
    trajectory_reference: np.ndarray | None,
) -> torch.Tensor:
    if trajectory_reference is None:
        full = np.repeat(
            bundle.initial_positions[:, None].astype(np.float32, copy=False),
            bundle.layout.num_time_steps,
            axis=1,
        ).reshape(-1)
    else:
        full = np.asarray(trajectory_reference, dtype=np.float32).reshape(-1)
    guess = torch.as_tensor(full, dtype=torch.float32, device=state.fixed_full.device)[state.free_indices]
    return _clamp_free_values(state, guess)


def _solve_free_cg_torch(
    *,
    bundle: AxisQPBundle,
    state: _TorchBundle,
    distance_signs: torch.Tensor,
    rhs: torch.Tensor,
    initial_guess: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if rhs.numel() == 0:
        return initial_guess.clone(), 0

    free_diag = torch.clamp(state.free_diag, min=1e-6)

    def apply_free_matrix(free_vector: torch.Tensor) -> torch.Tensor:
        full_vector = torch.zeros(state.num_variables, dtype=torch.float32, device=rhs.device)
        full_vector[state.free_indices] = free_vector
        product = _apply_full_p_torch(
            bundle=bundle,
            state=state,
            distance_signs=distance_signs,
            x_vector=full_vector,
        )
        return product[state.free_indices]

    x = initial_guess.clone()
    r = rhs - apply_free_matrix(x)
    z = r / free_diag
    p = z.clone()
    rz_old = torch.dot(r, z)
    rhs_norm = float(torch.linalg.norm(rhs).item())
    tolerance = max(1e-6, rhs_norm * 1e-5)
    max_iter = max(200, min(4000, int(state.free_indices.shape[0]) * 2))

    for iteration in range(max_iter):
        ap = apply_free_matrix(p)
        denom = torch.dot(p, ap)
        if torch.abs(denom) < 1e-12:
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        residual_norm = float(torch.linalg.norm(r).item())
        if residual_norm <= tolerance:
            return x, iteration + 1
        z = r / free_diag
        rz_new = torch.dot(r, z)
        beta = rz_new / torch.clamp(rz_old, min=1e-12)
        p = z + beta * p
        rz_old = rz_new

    raise RuntimeError(
        f"torch CG failed to converge within {max_iter} iterations; residual={float(torch.linalg.norm(r).item()):.6e}"
    )


def _quadratic_objective_torch(
    *,
    bundle: AxisQPBundle,
    state: _TorchBundle,
    distance_signs: torch.Tensor,
    full_vector: torch.Tensor,
    q_full: torch.Tensor,
) -> torch.Tensor:
    quadratic = _apply_full_p_torch(
        bundle=bundle,
        state=state,
        distance_signs=distance_signs,
        x_vector=full_vector,
    )
    base = 0.5 * torch.dot(full_vector, quadratic) + torch.dot(q_full, full_vector)
    return base + _area_sign_barrier_torch(
        bundle=bundle,
        state=state,
        full_vector=full_vector,
    )


def _area_sign_barrier_torch(
    *,
    bundle: AxisQPBundle,
    state: _TorchBundle,
    full_vector: torch.Tensor,
) -> torch.Tensor:
    if bundle.solver_config.lambda_area_sign <= 0.0 or state.area_triangle_indices.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=full_vector.device)

    x_matrix = full_vector.reshape(state.shape)
    tri = state.area_triangle_indices
    if state.dynamic_axis == "x":
        x_coords = x_matrix
        y_coords = state.orthogonal_static_positions[:, None].expand(-1, state.shape[1])
    else:
        x_coords = state.orthogonal_static_positions[:, None].expand(-1, state.shape[1])
        y_coords = x_matrix

    ax = x_coords[tri[:, 0], :]
    ay = y_coords[tri[:, 0], :]
    bx = x_coords[tri[:, 1], :]
    by = y_coords[tri[:, 1], :]
    cx = x_coords[tri[:, 2], :]
    cy = y_coords[tri[:, 2], :]
    oriented_double_area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    normalized_signed_area = (
        state.area_reference_signs[:, None]
        * oriented_double_area
        / torch.clamp(state.area_reference_scales[:, None], min=1e-6)
    )
    barrier_input = bundle.solver_config.area_barrier_margin - normalized_signed_area
    return bundle.solver_config.lambda_area_sign * torch.mean(
        torch.nn.functional.softplus(barrier_input) ** 2
    )


def _solve_lbfgs_torch(
    *,
    bundle: AxisQPBundle,
    state: _TorchBundle,
    distance_signs: torch.Tensor,
    q_full: torch.Tensor,
    initial_guess: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if initial_guess.numel() == 0:
        return initial_guess.clone(), 0

    if state.free_lower is None or state.free_upper is None:
        optimize_var = initial_guess.detach().clone().requires_grad_(True)

        def materialize_free_values() -> torch.Tensor:
            return optimize_var

    else:
        scale = torch.clamp((state.free_upper - state.free_lower) * 0.5, min=1e-4)
        center = (state.free_upper + state.free_lower) * 0.5
        normalized = torch.clamp((initial_guess - center) / scale, min=-0.999, max=0.999)
        optimize_var = torch.atanh(normalized).detach().clone().requires_grad_(True)

        def materialize_free_values() -> torch.Tensor:
            return center + scale * torch.tanh(optimize_var)

    optimizer = torch.optim.LBFGS(
        [optimize_var],
        lr=0.8,
        max_iter=80,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    closure_calls = {"count": 0}

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        closure_calls["count"] += 1
        free_values = materialize_free_values()
        full_vector = _assemble_full_vector(state, free_values)
        objective = _quadratic_objective_torch(
            bundle=bundle,
            state=state,
            distance_signs=distance_signs,
            full_vector=full_vector,
            q_full=q_full,
        )
        objective.backward()
        return objective

    optimizer.step(closure)
    with torch.no_grad():
        free_solution = materialize_free_values()
    return free_solution.detach(), int(closure_calls["count"])


def _solve_axis_qp_once(
    bundle: AxisQPBundle,
    *,
    trajectory_tether_reference: np.ndarray | None = None,
) -> AxisQPSolveResult:
    device, device_mode = _select_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    state = _prepare_torch_bundle(
        bundle,
        device=device,
        trajectory_tether_reference=trajectory_tether_reference,
    )
    trajectory_reference: np.ndarray | None = None
    warm_start_reference = bundle.warm_start_trajectory
    last_signs: np.ndarray | None = None
    num_outer_iterations = 4
    outer_wall_times: list[float] = []
    inner_iterations: list[int] = []
    inner_solver = "cg"

    for outer_idx in range(num_outer_iterations):
        current_signs_np = _compute_distance_signs(
            bundle=bundle,
            reference_trajectory=trajectory_reference,
        )
        current_signs = torch.as_tensor(current_signs_np, dtype=torch.float32, device=device)
        q_full = _build_matrix_free_q_torch(
            bundle=bundle,
            state=state,
            distance_signs=current_signs,
        )
        fixed_product = _apply_full_p_torch(
            bundle=bundle,
            state=state,
            distance_signs=current_signs,
            x_vector=state.fixed_full,
        )
        rhs = -(q_full + fixed_product)[state.free_indices]
        initial_guess = _initial_free_guess(
            bundle=bundle,
            state=state,
            trajectory_reference=trajectory_reference if trajectory_reference is not None else warm_start_reference,
        )

        solve_start = time.perf_counter()
        use_lbfgs = state.free_lower is not None or bundle.solver_config.lambda_area_sign > 0.0
        if not use_lbfgs:
            free_solution, step_count = _solve_free_cg_torch(
                bundle=bundle,
                state=state,
                distance_signs=current_signs,
                rhs=rhs,
                initial_guess=initial_guess,
            )
        else:
            inner_solver = "lbfgs_box" if state.free_lower is not None else "lbfgs"
            free_solution, step_count = _solve_lbfgs_torch(
                bundle=bundle,
                state=state,
                distance_signs=current_signs,
                q_full=q_full,
                initial_guess=initial_guess,
            )
        outer_wall_times.append(time.perf_counter() - solve_start)
        inner_iterations.append(int(step_count))

        solved_full = _assemble_full_vector(state, free_solution)
        solved_trajectory = solved_full.reshape(bundle.layout.shape).detach().cpu().numpy().astype(np.float32, copy=False)
        if last_signs is not None and np.array_equal(current_signs_np, last_signs):
            trajectory_reference = solved_trajectory
            break

        trajectory_reference = solved_trajectory
        last_signs = current_signs_np
    else:
        outer_idx = num_outer_iterations - 1

    final_signs_np = last_signs if last_signs is not None else current_signs_np
    final_signs = torch.as_tensor(final_signs_np, dtype=torch.float32, device=device)
    final_q = _build_matrix_free_q_torch(
        bundle=bundle,
        state=state,
        distance_signs=final_signs,
    )
    final_vector = torch.as_tensor(
        np.asarray(trajectory_reference, dtype=np.float32).reshape(-1),
        dtype=torch.float32,
        device=device,
    )
    final_objective = float(
        _quadratic_objective_torch(
            bundle=bundle,
            state=state,
            distance_signs=final_signs,
            full_vector=final_vector,
            q_full=final_q,
        ).item()
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mib = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    else:
        peak_memory_mib = None

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
            "solver_name": "torch",
            "device": device.type,
            "device_mode": device_mode,
            "inner_solver": inner_solver,
            "num_iters": int(inner_iterations[-1]) if inner_iterations else 0,
            "solve_time": float(outer_wall_times[-1]) if outer_wall_times else 0.0,
            "setup_time": 0.0,
            "run_time": float(sum(outer_wall_times)),
            "outer_iterations": [int(value) for value in inner_iterations],
            "peak_memory_mib": peak_memory_mib,
            "num_laplacian_edges": int(bundle.laplacian_edge_indices.shape[0]),
            "num_area_triangles": int(bundle.area_triangle_indices.shape[0]),
        },
    }
    return AxisQPSolveResult(
        trajectory=np.asarray(trajectory_reference, dtype=np.float32),
        diagnostics=diagnostics,
    )


def solve_axis_qp(bundle: AxisQPBundle) -> AxisQPSolveResult:
    if bundle.solver_config.lambda_trajectory_tether <= 0.0:
        return _solve_axis_qp_once(bundle)

    coarse_result = _solve_axis_qp_once(bundle)
    reference_trajectory = _build_linear_trajectory_reference(
        initial_positions=bundle.initial_positions,
        endpoint_positions=coarse_result.trajectory[:, -1],
        num_time_steps=bundle.layout.num_time_steps,
    )
    refined_result = _solve_axis_qp_once(
        bundle,
        trajectory_tether_reference=reference_trajectory,
    )
    return AxisQPSolveResult(
        trajectory=refined_result.trajectory,
        diagnostics={
            **refined_result.diagnostics,
            "trajectory_tether": {
                "enabled": True,
                "weight": float(bundle.solver_config.lambda_trajectory_tether),
                "reference_source": "linear_interpolation_from_coarse_endpoint",
                "coarse_objective_value": coarse_result.diagnostics.get("objective_value"),
                "coarse_num_outer_iterations": coarse_result.diagnostics.get("num_outer_iterations"),
            },
        },
    )
