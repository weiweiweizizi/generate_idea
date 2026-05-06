from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet.analysis.export_matrix_vis_patient import compose_window_matrix


def test_compose_window_matrix_combines_shared_and_side_weights() -> None:
    shared_basis = np.stack(
        [
            np.full((2, 2), 1.0, dtype=np.float32),
            np.full((2, 2), 2.0, dtype=np.float32),
        ],
        axis=0,
    )
    side_basis = np.stack(
        [
            np.full((2, 2), 10.0, dtype=np.float32),
        ],
        axis=0,
    )
    result = compose_window_matrix(
        shared_weights=np.asarray([0.5, 1.5], dtype=np.float32),
        shared_basis_bank=shared_basis,
        side_weights=np.asarray([0.25], dtype=np.float32),
        side_basis_bank=side_basis,
    )
    expected = 0.5 * shared_basis[0] + 1.5 * shared_basis[1] + 0.25 * side_basis[0]
    np.testing.assert_allclose(result, expected)
