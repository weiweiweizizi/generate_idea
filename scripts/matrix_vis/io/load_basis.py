from __future__ import annotations

import numpy as np

from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
from scripts.matrix_vis.core.types import BasisConfig, BasisObservation

# 支持直接加载或者差分加载；
# 支持整理矩阵行列顺序以适配subset point id的任意排列（前提是matrix_layout正确指定了行列顺序）；
def load_basis_observation(
    basis_config: BasisConfig,
    *,
    subset_point_ids: np.ndarray,
) -> BasisObservation:
    if basis_config.source is not None:
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
    else:
        if basis_config.prev_source is None or basis_config.next_source is None:
            raise ValueError("Basis diff mode requires both prev_source and next_source")
        if not basis_config.prev_source.exists():
            raise FileNotFoundError(basis_config.prev_source)
        if not basis_config.next_source.exists():
            raise FileNotFoundError(basis_config.next_source)
        prev_matrix = np.load(basis_config.prev_source)
        next_matrix = np.load(basis_config.next_source)
        if prev_matrix.shape != next_matrix.shape:
            raise ValueError(
                "prev_source and next_source shapes must match: "
                f"{tuple(prev_matrix.shape)} vs {tuple(next_matrix.shape)}"
            )
        basis_matrix = next_matrix - prev_matrix

    expected_size = int(np.asarray(subset_point_ids).shape[0])
    if basis_matrix.shape != (expected_size, expected_size):
        if (
            basis_config.matrix_layout is None
            or basis_config.matrix_layout_source is None
            or basis_matrix.shape[0] != basis_matrix.shape[1]
        ):
            raise ValueError(
                "Basis shape does not match subset point count: "
                f"got {tuple(basis_matrix.shape)}, expected {(expected_size, expected_size)}"
            )

        def _crop_from_layout(layout_region_names: list[str] | None) -> np.ndarray | None:
            layout_point_ids = resolve_subset_layout(
                subset_layout=basis_config.matrix_layout,
                subset_layout_source=basis_config.matrix_layout_source,
                subset_layout_extractor_name=basis_config.matrix_layout_extractor_name or "mediapipe",
                subset_layout_region_names=layout_region_names,
            )
            if basis_matrix.shape != (len(layout_point_ids), len(layout_point_ids)):
                return None
            index_by_point_id = {int(point_id): idx for idx, point_id in enumerate(layout_point_ids)}
            try:
                subset_indices = [index_by_point_id[int(point_id)] for point_id in np.asarray(subset_point_ids).tolist()]
            except KeyError as exc:
                raise ValueError(
                    f"Subset point id {int(exc.args[0])} is not present in basis matrix_layout ordering"
                ) from exc
            return basis_matrix[np.ix_(subset_indices, subset_indices)]

        cropped_matrix = _crop_from_layout(
            list(basis_config.matrix_layout_region_names)
            if basis_config.matrix_layout_region_names is not None
            else None
        )
        if cropped_matrix is None and basis_config.matrix_layout_region_names is not None:
            cropped_matrix = _crop_from_layout(None)
        if cropped_matrix is None:
            candidate_layout = resolve_subset_layout(
                subset_layout=basis_config.matrix_layout,
                subset_layout_source=basis_config.matrix_layout_source,
                subset_layout_extractor_name=basis_config.matrix_layout_extractor_name or "mediapipe",
                subset_layout_region_names=list(basis_config.matrix_layout_region_names)
                if basis_config.matrix_layout_region_names is not None
                else None,
            )
            full_layout = resolve_subset_layout(
                subset_layout=basis_config.matrix_layout,
                subset_layout_source=basis_config.matrix_layout_source,
                subset_layout_extractor_name=basis_config.matrix_layout_extractor_name or "mediapipe",
                subset_layout_region_names=None,
            )
            raise ValueError(
                "Basis matrix_layout size does not match loaded basis matrix: "
                f"subset layout has {len(candidate_layout)} points, full layout has {len(full_layout)} points, "
                f"matrix shape is {tuple(basis_matrix.shape)}"
            )
        basis_matrix = cropped_matrix

    return BasisObservation(
        subset_point_ids=np.asarray(subset_point_ids, dtype=np.int64),
        basis_matrix=basis_matrix.astype(np.float32, copy=False),
        value_semantics=basis_config.value_semantics,
    )
