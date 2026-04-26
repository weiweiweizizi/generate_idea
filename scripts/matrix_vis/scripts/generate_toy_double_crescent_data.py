from __future__ import annotations

from pathlib import Path

import numpy as np


OUTPUT_DIR = Path("data/toy/matrix_vis/double_crescent_mouth_opening")


def make_arc(
    *,
    radius_x: float,
    radius_y: float,
    theta_start: float,
    theta_end: float,
    num_points: int,
    y_offset: float = 0.0,
) -> np.ndarray:
    theta = np.linspace(theta_start, theta_end, num_points, dtype=np.float32)
    x = radius_x * np.cos(theta)
    y = radius_y * np.sin(theta) + y_offset
    return np.stack([x, y], axis=1)


def build_double_crescent_mesh() -> np.ndarray:
    upper_outer = make_arc(
        radius_x=2.8,
        radius_y=1.0,
        theta_start=np.deg2rad(200.0),
        theta_end=np.deg2rad(340.0),
        num_points=8,
        y_offset=0.55,
    )
    lower_outer = make_arc(
        radius_x=2.8,
        radius_y=1.1,
        theta_start=np.deg2rad(20.0),
        theta_end=np.deg2rad(160.0),
        num_points=8,
        y_offset=-0.55,
    )
    return np.concatenate([upper_outer, lower_outer], axis=0).astype(np.float32)


def build_open_mouth_trajectory(mesh_2d: np.ndarray, num_steps: int = 25) -> np.ndarray:
    coordinates = np.repeat(mesh_2d[None, :, :], num_steps, axis=0)
    midpoint = mesh_2d.shape[0] // 2
    amplitudes = np.linspace(0.0, 0.75, num_steps, dtype=np.float32)
    stretch_profile = np.abs(mesh_2d[:, 0]) / max(float(np.abs(mesh_2d[:, 0]).max()), 1e-6)

    for step_idx, amplitude in enumerate(amplitudes):
        coordinates[step_idx, :midpoint, 1] += amplitude * 0.45
        coordinates[step_idx, midpoint:, 1] -= amplitude
        coordinates[step_idx, :, 0] += np.sign(mesh_2d[:, 0]) * stretch_profile * amplitude * 0.16
        coordinates[step_idx, 0, :] = mesh_2d[0]
    return coordinates.astype(np.float32)


def pairwise_axis_delta_average(trajectory: np.ndarray, axis_index: int) -> np.ndarray:
    axis_positions = trajectory[:, :, axis_index]
    initial = axis_positions[0]
    time_mean = axis_positions.mean(axis=0)
    delta = time_mean - initial
    pairwise = np.zeros((delta.shape[0], delta.shape[0]), dtype=np.float32)
    for i in range(delta.shape[0]):
        for j in range(i + 1, delta.shape[0]):
            value = float(delta[j] - delta[i])
            pairwise[i, j] = value
            pairwise[j, i] = value
    return pairwise


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_2d = build_double_crescent_mesh()
    trajectory_2d = build_open_mouth_trajectory(mesh_2d)
    basis_x = pairwise_axis_delta_average(trajectory_2d, axis_index=0)
    basis_y = pairwise_axis_delta_average(trajectory_2d, axis_index=1)

    np.save(OUTPUT_DIR / "mesh_2d.npy", mesh_2d)
    np.save(OUTPUT_DIR / "trajectory_2d.npy", trajectory_2d)
    np.save(OUTPUT_DIR / "basis_open_mouth_x.npy", basis_x)
    np.save(OUTPUT_DIR / "basis_open_mouth_y.npy", basis_y)


if __name__ == "__main__":
    main()
