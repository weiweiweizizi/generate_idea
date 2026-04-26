from __future__ import annotations

import numpy as np

from scripts.matrix_vis.core.types import BasisConfig, BasisObservation


def load_basis_observation(
    basis_config: BasisConfig,
    *,
    subset_point_ids: np.ndarray,
) -> BasisObservation:
    if not basis_config.source.exists():
        raise FileNotFoundError(basis_config.source)

    raw = np.load(basis_config.source)
    if raw.ndim == 2:
        basis_matrix = raw
    elif raw.ndim == 3:
        if basis_config.basis_index < 0 or basis_config.basis_index >= raw.shape[0]:
            raise IndexError(
                f"basis_index={basis_config.basis_index} is out of range for basis stack with shape {tuple(raw.shape)}"
            )
        basis_matrix = raw[basis_config.basis_index]
    else:
        raise ValueError(
            f"Basis source must be 2D or 3D square matrix data, got shape {tuple(raw.shape)}"
        )

    expected_size = int(np.asarray(subset_point_ids).shape[0])
    if basis_matrix.shape != (expected_size, expected_size):
        raise ValueError(
            "Basis shape does not match subset point count: "
            f"got {tuple(basis_matrix.shape)}, expected {(expected_size, expected_size)}"
        )

    return BasisObservation(
        subset_point_ids=np.asarray(subset_point_ids, dtype=np.int64),
        basis_matrix=basis_matrix.astype(np.float32, copy=False),
        value_semantics=basis_config.value_semantics,
    )
