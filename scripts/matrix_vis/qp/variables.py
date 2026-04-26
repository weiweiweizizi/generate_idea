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


def find_anchor_local_index(subset_point_ids: np.ndarray, anchor_point_id: int) -> int:
    matches = np.flatnonzero(np.asarray(subset_point_ids, dtype=np.int64) == int(anchor_point_id))
    if matches.size != 1:
        raise ValueError(f"Could not locate anchor_point_id={anchor_point_id} in subset_point_ids")
    return int(matches[0])
