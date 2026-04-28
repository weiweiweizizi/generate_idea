from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VariableLayout:
    num_points: int
    num_time_steps: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_points, self.num_time_steps)


def build_time_grid(num_time_steps: int) -> np.ndarray:
    if num_time_steps < 2:
        raise ValueError("num_time_steps must be >= 2")
    return np.linspace(0.0, 1.0, num_time_steps, dtype=np.float32)


def find_anchor_local_indices(
    subset_point_ids: np.ndarray,
    anchor_point_ids: np.ndarray | list[int] | tuple[int, ...],
) -> np.ndarray:
    subset_point_ids = np.asarray(subset_point_ids, dtype=np.int64)
    anchor_point_ids = np.asarray(anchor_point_ids, dtype=np.int64)
    local_indices: list[int] = []
    for anchor_point_id in anchor_point_ids.tolist():
        matches = np.flatnonzero(subset_point_ids == int(anchor_point_id))
        if matches.size != 1:
            raise ValueError(f"Could not locate anchor_point_id={anchor_point_id} in subset_point_ids")
        local_indices.append(int(matches[0]))
    return np.asarray(local_indices, dtype=np.int64)
