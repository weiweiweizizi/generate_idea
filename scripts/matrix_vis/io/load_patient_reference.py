from __future__ import annotations

from pathlib import Path

import numpy as np


def _parse_numeric_text_table(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(token) for token in line.replace(",", " ").split()])
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Inconsistent column counts in {path}")
    return np.asarray(rows, dtype=np.float32)


def _looks_like_point_id_column(values: np.ndarray) -> bool:
    if values.ndim != 1:
        return False
    rounded = np.round(values)
    if not np.allclose(values, rounded, atol=1e-4):
        return False
    unique_count = np.unique(rounded.astype(np.int64)).shape[0]
    return unique_count == values.shape[0]


def load_patient_landmark_points(path: str | Path) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix == ".npy":
        points = np.load(source).astype(np.float32, copy=False)
    elif suffix == ".npz":
        payload = np.load(source)
        if "points" in payload:
            points = payload["points"].astype(np.float32, copy=False)
        else:
            first_key = payload.files[0]
            points = payload[first_key].astype(np.float32, copy=False)
    elif source.name.endswith(".label"):
        points = load_label_first_line_normalized_points(source)
    else:
        points = _parse_numeric_text_table(source)

    if points.ndim != 2:
        raise ValueError(f"Expected 2D landmark table in {source}, got shape {tuple(points.shape)}")
    if points.shape[1] < 2:
        raise ValueError(f"Expected at least 2 numeric columns in {source}, got shape {tuple(points.shape)}")

    if points.shape[1] >= 3 and _looks_like_point_id_column(points[:, 0]):
        coords = points[:, 1:]
    else:
        coords = points

    if coords.shape[1] < 2:
        raise ValueError(f"Expected at least x/y coordinate columns in {source}, got shape {tuple(coords.shape)}")
    return coords[:, :2].astype(np.float32, copy=False)


def load_subset_axis_positions(
    *,
    landmark_source: str | Path,
    subset_point_ids: np.ndarray,
    axis: str,
) -> np.ndarray:
    points = load_patient_landmark_points(landmark_source)
    axis_index = 0 if axis == "x" else 1
    point_ids = np.asarray(subset_point_ids, dtype=np.int64)
    if np.any(point_ids < 0) or np.any(point_ids >= points.shape[0]):
        raise ValueError(
            f"Subset point ids exceed landmark table bounds: max id {int(point_ids.max())}, point count {points.shape[0]}"
        )
    return points[point_ids, axis_index].astype(np.float32, copy=False)


def load_distance_matrix(path: str | Path) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(source).astype(np.float32, copy=False)
    elif suffix == ".npz":
        payload = np.load(source)
        if "distance_matrix" in payload:
            matrix = payload["distance_matrix"].astype(np.float32, copy=False)
        else:
            first_key = payload.files[0]
            matrix = payload[first_key].astype(np.float32, copy=False)
    else:
        matrix = _parse_numeric_text_table(source)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square distance matrix in {source}, got shape {tuple(matrix.shape)}")
    return matrix.astype(np.float32, copy=False)


def load_label_first_line_normalized_points(path: str | Path) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    first_line = source.read_text(encoding="utf-8").splitlines()[0].strip().split()
    if len(first_line) < 6:
        raise ValueError(f"Expected metadata plus x,y tokens in {source}, got {len(first_line)} tokens")
    pair_tokens = first_line[5:]
    coords = np.asarray([[float(value) for value in token.split(",")] for token in pair_tokens], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected x,y coordinate pairs in {source}, got shape {tuple(coords.shape)}")

    x = coords[:, 0]
    y = coords[:, 1]
    x_norm = (x - x.min()) / max(float(x.max() - x.min()), 1e-6) - 0.5
    y_norm = (y - y.min()) / max(float(y.max() - y.min()), 1e-6) - 0.5
    return np.stack([x_norm, y_norm], axis=1).astype(np.float32, copy=False)
