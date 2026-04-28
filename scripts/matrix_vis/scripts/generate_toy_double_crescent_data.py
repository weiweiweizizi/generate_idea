from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


OUTPUT_DIR = Path("data/toy/matrix_vis/leaf_to_rectangle_mouth_opening")

def make_superellipse_contour(
    *,
    radius_x: float,
    radius_y: float,
    exponent: float,
    num_points: int,
) -> np.ndarray:
    theta = np.linspace(np.pi, -np.pi, num_points, endpoint=False, dtype=np.float32)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x = radius_x * np.sign(cos_theta) * (np.abs(cos_theta) ** (2.0 / exponent))
    y = radius_y * np.sign(sin_theta) * (np.abs(sin_theta) ** (2.0 / exponent))
    return np.stack([x, y], axis=1).astype(np.float32)


def build_flat_leaf_mesh(num_points: int = 16) -> np.ndarray:
    return make_superellipse_contour(
        radius_x=2.8,
        radius_y=0.52,
        exponent=1.35,
        num_points=num_points,
    )


def build_near_rectangle_target(num_points: int = 16) -> np.ndarray:
    return make_superellipse_contour(
        radius_x=2.25,
        radius_y=1.18,
        exponent=8.0,
        num_points=num_points,
    )


def build_open_mouth_trajectory(mesh_2d: np.ndarray, num_steps: int = 25) -> np.ndarray:
    coordinates = np.repeat(mesh_2d[None, :, :], num_steps, axis=0).astype(np.float32)
    rectangle_target = build_near_rectangle_target(mesh_2d.shape[0])
    full_displacement = rectangle_target - mesh_2d
    full_displacement -= full_displacement[0]
    time = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    amplitudes = 0.5 * (1.0 - np.cos(np.pi * time)) ** 0.85

    for step_idx, amplitude in enumerate(amplitudes):
        coordinates[step_idx] = mesh_2d + amplitude * full_displacement
        coordinates[step_idx, 0, :] = mesh_2d[0]
    return coordinates.astype(np.float32)


def pairwise_axis_distance_matrix_diff(trajectory: np.ndarray, axis_index: int) -> np.ndarray:
    axis_positions = trajectory[:, :, axis_index]
    initial = axis_positions[0]
    next_distance = np.abs(axis_positions[:, :, None] - axis_positions[:, None, :]).mean(axis=0)
    prev_distance = np.abs(initial[:, None] - initial[None, :])
    return (next_distance - prev_distance).astype(np.float32)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_2d = build_flat_leaf_mesh()
    trajectory_2d = build_open_mouth_trajectory(mesh_2d)
    basis_x = pairwise_axis_distance_matrix_diff(trajectory_2d, axis_index=0)
    basis_y = pairwise_axis_distance_matrix_diff(trajectory_2d, axis_index=1)

    np.save(OUTPUT_DIR / "mesh_2d.npy", mesh_2d)
    np.save(OUTPUT_DIR / "trajectory_2d.npy", trajectory_2d)
    np.save(OUTPUT_DIR / "basis_open_mouth_x.npy", basis_x)
    np.save(OUTPUT_DIR / "basis_open_mouth_y.npy", basis_y)


if __name__ == "__main__":
    main()
