#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
import sys

import fire
import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.io.load_mesh import load_mesh
from scripts.matrix_vis.io.save_results import ensure_output_dir, save_json
from scripts.matrix_vis.core.types import MeshConfig
from scripts.matrix_vis.viz.mesh_animation import (
    save_gif_from_frames,
    save_motion_frames,
    save_motion_snapshot,
)


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = (config_path.parent / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def _load_compose_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Compose config root must be a mapping")
    raw["_config_path"] = str(config_path)
    return raw


def _load_solution(path: Path) -> dict:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def compose(config: str, output_dir: str | None = None) -> dict:
    raw = _load_compose_config(config)
    config_path = Path(raw["_config_path"])
    experiment = raw["experiment"]
    mesh_section = raw["mesh"]
    inputs = raw["inputs"]
    compose_section = raw.get("compose", {})
    export = raw.get("export", {})

    out_dir = ensure_output_dir(Path(output_dir).resolve()) if output_dir else ensure_output_dir(
        _resolve_path(config_path, experiment["output_dir"])
    )

    mesh = load_mesh(
        MeshConfig(
            source=_resolve_path(config_path, mesh_section["source"]),
            format=mesh_section["format"],
            dimension=mesh_section["dimension"],
            point_ids=mesh_section.get("point_ids", "auto"),
        )
    )

    x_solution = _load_solution(_resolve_path(config_path, inputs["x_solution"]))
    y_solution = _load_solution(_resolve_path(config_path, inputs["y_solution"]))

    x_ids = x_solution["point_ids"].astype(np.int64)
    y_ids = y_solution["point_ids"].astype(np.int64)
    if compose_section.get("subset_policy", "intersection") != "intersection":
        raise ValueError("Only subset_policy=intersection is supported right now")
    common_ids = np.intersect1d(x_ids, y_ids)
    if common_ids.size == 0:
        raise ValueError("No overlapping point ids between x and y solutions")

    x_lookup = {int(point_id): idx for idx, point_id in enumerate(x_ids.tolist())}
    y_lookup = {int(point_id): idx for idx, point_id in enumerate(y_ids.tolist())}
    mesh_lookup = {int(point_id): idx for idx, point_id in enumerate(mesh.point_ids.tolist())}

    x_time = x_solution["time_grid"].astype(np.float32)
    y_time = y_solution["time_grid"].astype(np.float32)
    if x_time.shape != y_time.shape or not np.allclose(x_time, y_time):
        raise ValueError("x and y solutions must share the same time grid")

    subset_indices = np.asarray([mesh_lookup[int(point_id)] for point_id in common_ids.tolist()], dtype=np.int64)
    subset_mask = np.zeros(mesh.points.shape[0], dtype=bool)
    subset_mask[subset_indices] = True

    coordinates = np.repeat(mesh.points[None, :, :], x_time.shape[0], axis=0).astype(np.float32)
    for point_id in common_ids.tolist():
        mesh_idx = mesh_lookup[int(point_id)]
        x_idx = x_lookup[int(point_id)]
        y_idx = y_lookup[int(point_id)]
        coordinates[:, mesh_idx, 0] = x_solution["trajectory"][x_idx]
        coordinates[:, mesh_idx, 1] = y_solution["trajectory"][y_idx]

    save_motion_snapshot(
        output_path=out_dir / "motion_snapshot.png",
        static_points=mesh.points,
        animated_points=coordinates[-1, subset_indices],
        title=experiment["name"],
    )

    frame_paths = []
    if export.get("save_animation_preview", True):
        frame_paths = save_motion_frames(
            output_dir=out_dir / "frames",
            static_points=mesh.points,
            coordinates=coordinates,
            subset_mask=subset_mask,
        )
        save_gif_from_frames(frame_paths, out_dir / "motion_preview.gif")

    if export.get("save_npz", True):
        np.savez(
            out_dir / "composed_motion.npz",
            point_ids=mesh.point_ids.astype(np.int64),
            time_grid=x_time.astype(np.float32),
            coordinates=coordinates.astype(np.float32),
            subset_point_ids=common_ids.astype(np.int64),
        )

    summary = {
        "experiment_name": experiment["name"],
        "output_dir": str(out_dir),
        "num_mesh_points": int(mesh.points.shape[0]),
        "num_subset_points": int(common_ids.shape[0]),
        "num_frames": int(coordinates.shape[0]),
        "saved_frame_count": int(len(frame_paths)),
        "subset_policy": compose_section.get("subset_policy", "intersection"),
    }
    if export.get("save_json_summary", True):
        save_json(out_dir / "composed_summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"compose": compose})
