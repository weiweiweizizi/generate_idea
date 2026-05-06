from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.matrix_vis.core.composition import compose_xy_coordinates
from scripts.matrix_vis.io.compose_config import load_compose_config
from scripts.matrix_vis.io.load_mesh import load_mesh
from scripts.matrix_vis.io.save_results import (
    ensure_output_dir,
    load_solution_npz,
    save_composed_motion_npz,
    save_json,
)
from scripts.matrix_vis.viz.mesh_animation import (
    save_gif_from_frames,
    save_motion_frames,
    save_motion_snapshot,
)

# 运行运动合成流程的主函数，包含以下步骤：
# 1. load_compose_config() 加载配置文件，解析实验设置。
# 2. compose_xy_coordinates() 根据 x_solution 和 y_solution 计算合成坐标。
# 3. 合成gif和帧图像预览。


def _resolve_anchor_points(
    *,
    mesh_points,
    mesh_point_ids,
    x_solution: dict,
    y_solution: dict,
):
    x_anchor_ids = np.asarray(x_solution.get("anchor_point_ids", []), dtype=np.int64)
    y_anchor_ids = np.asarray(y_solution.get("anchor_point_ids", []), dtype=np.int64)
    if x_anchor_ids.size == 0 and y_anchor_ids.size == 0:
        return None
    if x_anchor_ids.size == 0:
        anchor_ids = y_anchor_ids
    elif y_anchor_ids.size == 0:
        anchor_ids = x_anchor_ids
    else:
        anchor_ids = np.intersect1d(x_anchor_ids, y_anchor_ids)
        if anchor_ids.size == 0:
            anchor_ids = x_anchor_ids

    mesh_lookup = {int(point_id): idx for idx, point_id in enumerate(mesh_point_ids.tolist())}
    anchor_indices = [mesh_lookup[int(point_id)] for point_id in anchor_ids.tolist() if int(point_id) in mesh_lookup]
    if not anchor_indices:
        return None
    return mesh_points[anchor_indices, :2]


def run_motion_composition(config: str, output_dir: str | None = None) -> dict:
    cfg = load_compose_config(config)
    out_dir = ensure_output_dir(Path(output_dir).resolve()) if output_dir else ensure_output_dir(
        cfg.experiment.output_dir
    )

    mesh = load_mesh(cfg.mesh)
    x_solution = load_solution_npz(cfg.inputs.x_solution)
    y_solution = load_solution_npz(cfg.inputs.y_solution)

    common_ids, time_grid, coordinates = compose_xy_coordinates(
        x_solution=x_solution,
        y_solution=y_solution,
    )
    anchor_points = _resolve_anchor_points(
        mesh_points=mesh.points,
        mesh_point_ids=mesh.point_ids,
        x_solution=x_solution,
        y_solution=y_solution,
    )
    mesh_lookup = {int(point_id): idx for idx, point_id in enumerate(mesh.point_ids.tolist())}
    missing_from_mesh = [int(point_id) for point_id in common_ids.tolist() if int(point_id) not in mesh_lookup]
    if missing_from_mesh:
        raise ValueError(f"Composed point ids are missing from mesh: {missing_from_mesh[:10]}")

    save_motion_snapshot(
        output_path=out_dir / "motion_snapshot.png",
        static_points=mesh.points,
        animated_points=coordinates[-1],
        title=cfg.experiment.name,
        anchor_points=anchor_points,
    )

    frame_paths = []
    if cfg.export.save_animation_preview:
        frame_paths = save_motion_frames(
            output_dir=out_dir / "frames",
            static_points=mesh.points,
            subset_coordinates=coordinates,
            anchor_points=anchor_points,
        )
        save_gif_from_frames(frame_paths, out_dir / "motion_preview.gif")

    if cfg.export.save_npz:
        save_composed_motion_npz(
            output_dir=out_dir,
            point_ids=common_ids,
            time_grid=time_grid,
            coordinates=coordinates,
            subset_point_ids=common_ids,
        )

    summary = {
        "experiment_name": cfg.experiment.name,
        "output_dir": str(out_dir),
        "num_static_mesh_points": int(mesh.points.shape[0]),
        "num_composed_points": int(common_ids.shape[0]),
        "num_frames": int(coordinates.shape[0]),
        "saved_frame_count": int(len(frame_paths)),
        "subset_policy": cfg.subset_policy,
    }
    if cfg.export.save_json_summary:
        save_json(out_dir / "composed_summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
