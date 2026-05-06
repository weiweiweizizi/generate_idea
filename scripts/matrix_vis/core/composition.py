from __future__ import annotations

import numpy as np

# 合并x轴和y轴的数据

# 验证x_solution和y_solution是否共享相同的时间网格，如果不共享则抛出异常
def require_shared_time_grid(
    x_solution: dict[str, np.ndarray],
    y_solution: dict[str, np.ndarray],
) -> np.ndarray:
    x_time = x_solution["time_grid"].astype(np.float32)
    y_time = y_solution["time_grid"].astype(np.float32)
    if x_time.shape != y_time.shape or not np.allclose(x_time, y_time):
        raise ValueError("x and y solutions must share the same time grid")
    return x_time


# 根据x_solution和y_solution中的point_ids
# 按照preferred_point_ids的顺序返回一个有序的数组
def ordered_common_point_ids(
    x_ids: np.ndarray,
    y_ids: np.ndarray,
    *,
    preferred_point_ids: np.ndarray | None = None,
) -> np.ndarray:
    x_ids = x_ids.astype(np.int64)
    y_ids = y_ids.astype(np.int64)
    x_set = {int(point_id) for point_id in x_ids.tolist()}
    y_set = {int(point_id) for point_id in y_ids.tolist()}

    if preferred_point_ids is None:
        ordered = [int(point_id) for point_id in x_ids.tolist() if int(point_id) in y_set]
    else:
        preferred = np.asarray(preferred_point_ids, dtype=np.int64)
        ordered = [
            int(point_id)
            for point_id in preferred.tolist()
            if int(point_id) in x_set and int(point_id) in y_set
        ]

    if not ordered:
        raise ValueError("No overlapping ordered point ids between x and y solutions")
    return np.asarray(ordered, dtype=np.int64)


# 合并x轴和y轴的数据
# dim=(Time,Number,2)
def compose_xy_coordinates(
    *,
    x_solution: dict[str, np.ndarray],
    y_solution: dict[str, np.ndarray],
    preferred_point_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_grid = require_shared_time_grid(x_solution, y_solution)
    ordered_ids = ordered_common_point_ids(
        x_solution["point_ids"],
        y_solution["point_ids"],
        preferred_point_ids=preferred_point_ids,
    )

    x_lookup = {
        int(point_id): idx
        for idx, point_id in enumerate(x_solution["point_ids"].astype(np.int64).tolist())
    }
    y_lookup = {
        int(point_id): idx
        for idx, point_id in enumerate(y_solution["point_ids"].astype(np.int64).tolist())
    }
    coordinates = np.empty((time_grid.shape[0], ordered_ids.shape[0], 2), dtype=np.float32)
    for local_idx, point_id in enumerate(ordered_ids.tolist()):
        coordinates[:, local_idx, 0] = x_solution["trajectory"][x_lookup[int(point_id)]]
        coordinates[:, local_idx, 1] = y_solution["trajectory"][y_lookup[int(point_id)]]

    return ordered_ids, time_grid, coordinates
