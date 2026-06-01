from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class GeometryTopology:
    edges_local: np.ndarray
    triangles_local: np.ndarray


def default_topology_source() -> Path:
    return (Path(__file__).resolve().parents[1] / "face_mesh_connections.py").resolve()


def _recover_triangles_from_edges(edges: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    adjacency: dict[int, set[int]] = {}
    for left, right in edges:
        adjacency.setdefault(int(left), set()).add(int(right))
        adjacency.setdefault(int(right), set()).add(int(left))

    triangles: set[tuple[int, int, int]] = set()
    for node_i, neighbors_i in adjacency.items():
        for node_j in neighbors_i:
            if node_j <= node_i:
                continue
            shared = neighbors_i.intersection(adjacency.get(node_j, set()))
            for node_k in shared:
                if node_k <= node_j:
                    continue
                triangles.add((int(node_i), int(node_j), int(node_k)))
    return sorted(triangles)


def load_facemesh_tesselation(
    topology_source: str | Path | None,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    source = default_topology_source() if topology_source is None else Path(topology_source).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("matrix_vis_face_mesh_connections", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load topology module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = getattr(module, "FACEMESH_TESSELATION", None)
    if raw is None:
        raise ValueError(f"FACEMESH_TESSELATION is missing in {source}")

    edges: set[tuple[int, int]] = set()
    triangles: list[tuple[int, int, int]] = []
    for item in raw:
        if len(item) == 2:
            left = int(item[0])
            right = int(item[1])
            if left != right:
                edges.add((min(left, right), max(left, right)))
            continue
        if len(item) == 3:
            tri = (int(item[0]), int(item[1]), int(item[2]))
            triangles.append(tri)
            for left, right in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                if left != right:
                    edges.add((min(left, right), max(left, right)))
    if edges and not triangles:
        triangles = _recover_triangles_from_edges(edges)
    if not edges:
        raise ValueError(f"No valid edge tuples found in {source}")
    if not triangles:
        raise ValueError(f"No valid triangle tuples found in {source}")
    return tuple(sorted(edges)), tuple(triangles)


def build_subset_topology(
    *,
    subset_point_ids: np.ndarray,
    topology_source: str | Path | None,
) -> GeometryTopology:
    point_to_local = {
        int(point_id): idx
        for idx, point_id in enumerate(np.asarray(subset_point_ids, dtype=np.int64).tolist())
    }
    edges_full, triangles_full = load_facemesh_tesselation(topology_source)

    triangles_local: list[tuple[int, int, int]] = []
    edge_set: set[tuple[int, int]] = set()
    for left, right in edges_full:
        if left not in point_to_local or right not in point_to_local:
            continue
        local_left = point_to_local[left]
        local_right = point_to_local[right]
        if local_left == local_right:
            continue
        edge_set.add((min(local_left, local_right), max(local_left, local_right)))
    for tri in triangles_full:
        if tri[0] not in point_to_local or tri[1] not in point_to_local or tri[2] not in point_to_local:
            continue
        local_tri = (
            point_to_local[tri[0]],
            point_to_local[tri[1]],
            point_to_local[tri[2]],
        )
        if len({local_tri[0], local_tri[1], local_tri[2]}) < 3:
            continue
        triangles_local.append(local_tri)

    edges_local = (
        np.asarray(sorted(edge_set), dtype=np.int64)
        if edge_set
        else np.zeros((0, 2), dtype=np.int64)
    )
    triangles_array = (
        np.asarray(triangles_local, dtype=np.int64)
        if triangles_local
        else np.zeros((0, 3), dtype=np.int64)
    )
    return GeometryTopology(
        edges_local=edges_local,
        triangles_local=triangles_array,
    )


def build_graph_laplacian(*, num_points: int, edges_local: np.ndarray) -> sparse.csr_matrix:
    if int(num_points) <= 0 or edges_local.size == 0:
        return sparse.csr_matrix((num_points, num_points), dtype=np.float32)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    degree = np.zeros(int(num_points), dtype=np.float32)
    for left, right in np.asarray(edges_local, dtype=np.int64).tolist():
        if int(left) == int(right):
            continue
        degree[int(left)] += 1.0
        degree[int(right)] += 1.0
        rows.extend([int(left), int(right)])
        cols.extend([int(right), int(left)])
        data.extend([-1.0, -1.0])

    rows.extend(range(int(num_points)))
    cols.extend(range(int(num_points)))
    data.extend(degree.tolist())
    return sparse.coo_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(int(num_points), int(num_points)),
        dtype=np.float32,
    ).tocsr()
