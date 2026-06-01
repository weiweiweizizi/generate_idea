#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import sys

import fire

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.pipelines.patient_sequence import run_patient_sequence


def reconstruct(
    patient_bundle_path: str,
    output_dir: str | None = None,
    mesh_source: str | None = None,
    initial_landmark_source: str | None = None,
    initial_distance_matrix_source: str | None = None,
    lambda_acc: float = 10.0,
    max_displacement: float | None = 0.2,
    carry_forward_initial_positions: bool = True,
    lambda_laplacian: float = 0.0,
    lambda_area_sign: float = 0.0,
    area_barrier_margin: float = 0.05,
    lambda_trajectory_tether: float = 0.0,
) -> dict:
    kwargs = {
        "patient_bundle_path": patient_bundle_path,
        "output_dir": output_dir,
        "lambda_acc": lambda_acc,
        "max_displacement": max_displacement,
        "carry_forward_initial_positions": carry_forward_initial_positions,
        "lambda_laplacian": lambda_laplacian,
        "lambda_area_sign": lambda_area_sign,
        "area_barrier_margin": area_barrier_margin,
        "lambda_trajectory_tether": lambda_trajectory_tether,
    }
    if mesh_source is not None:
        kwargs["mesh_source"] = mesh_source
    if initial_landmark_source is not None:
        kwargs["initial_landmark_source"] = initial_landmark_source
    if initial_distance_matrix_source is not None:
        kwargs["initial_distance_matrix_source"] = initial_distance_matrix_source
    return run_patient_sequence(**kwargs)


if __name__ == "__main__":
    fire.Fire({"reconstruct": reconstruct})
