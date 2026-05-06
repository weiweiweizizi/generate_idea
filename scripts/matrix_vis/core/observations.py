from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.matrix_vis.core.types import BasisObservation


@dataclass(frozen=True)
class ObservationTable:
    frame: pd.DataFrame

# 将矩阵转换为观察表（一个DataFrame）
# 因为矩阵的表示是冗余的（对称性），只保留上三角部分（不包括对角线）来构建观察表
def basis_to_observation_table(observation: BasisObservation) -> ObservationTable:
    basis = observation.basis_matrix
    point_ids = observation.subset_point_ids
    rows: list[dict[str, float | int]] = []

    for i in range(basis.shape[0]):
        for j in range(i + 1, basis.shape[1]):
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "point_id_i": int(point_ids[i]),
                    "point_id_j": int(point_ids[j]),
                    "value": float(basis[i, j]),
                }
            )

    return ObservationTable(frame=pd.DataFrame(rows))
