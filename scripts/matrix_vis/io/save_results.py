from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.matrix_vis.core.types import AxisProjection, BasisObservation, MatrixVisConfig


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_resolved_config(config: MatrixVisConfig, output_dir: Path) -> None:
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.raw, handle, allow_unicode=True, sort_keys=False)


def save_projected_mesh(projection: AxisProjection, output_dir: Path) -> None:
    frame = pd.DataFrame(
        {
            "point_id": projection.subset_point_ids.tolist(),
            "axis_position": projection.subset_positions.tolist(),
        }
    )
    frame.to_csv(output_dir / "projected_mesh.csv", index=False)


def save_observations(observations: pd.DataFrame, output_dir: Path) -> None:
    observations.to_csv(output_dir / "observations.csv", index=False)


def save_solution_npz(
    *,
    output_dir: Path,
    point_ids: np.ndarray,
    time_grid: np.ndarray,
    initial_positions: np.ndarray,
    trajectory: np.ndarray,
    anchor_point_id: int,
    basis_observation: BasisObservation,
) -> None:
    np.savez(
        output_dir / "solution.npz",
        point_ids=np.asarray(point_ids, dtype=np.int64),
        time_grid=np.asarray(time_grid, dtype=np.float32),
        initial_positions=np.asarray(initial_positions, dtype=np.float32),
        trajectory=np.asarray(trajectory, dtype=np.float32),
        anchor_point_id=np.asarray(anchor_point_id, dtype=np.int64),
        basis_matrix=np.asarray(basis_observation.basis_matrix, dtype=np.float32),
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
