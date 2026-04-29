#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
from scripts.matrix_vis.io.load_mesh import _load_canonical_obj_vertices


FACE_WIDTH_POINTS = (127, 356)
FACE_HEIGHT_POINTS = (10, 152)
DEFAULT_LAYOUT_SOURCE = Path("/home/weizilin/code_reproduction/corelation-lm/project/configs/extractors.yaml")
DEFAULT_MESH = Path("/home/weizilin/code_reproduction/canonical_face/canonical_face_model.obj")


def _load_matplotlib():
    import os

    mpl_dir = Path("outputs/matrix_vis/.mplconfig").resolve()
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib.pyplot as plt

    return plt


def _normalize_standard_facemesh(points: np.ndarray) -> np.ndarray:
    normalized = points.astype(np.float32, copy=True)
    scale_x = abs(float(points[FACE_WIDTH_POINTS[1], 0] - points[FACE_WIDTH_POINTS[0], 0]))
    scale_y = abs(float(points[FACE_HEIGHT_POINTS[1], 1] - points[FACE_HEIGHT_POINTS[0], 1]))
    if scale_x <= 0:
        scale_x = 1.0
    if scale_y <= 0:
        scale_y = 1.0
    normalized[:, 0] /= scale_x
    normalized[:, 1] /= scale_y
    return normalized


def _load_solution(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _ordered_point_ids(layout_source: Path) -> np.ndarray:
    subset_ids = resolve_subset_layout(
        subset_layout="face_regions_grouped",
        subset_layout_source=layout_source,
        subset_layout_extractor_name="mediapipe",
    )
    return np.asarray(subset_ids, dtype=np.int64)


def _compose_coordinates(
    *,
    x_solution: dict[str, np.ndarray],
    y_solution: dict[str, np.ndarray],
    ordered_ids: np.ndarray,
) -> np.ndarray:
    x_ids = x_solution["point_ids"].astype(np.int64)
    y_ids = y_solution["point_ids"].astype(np.int64)
    if not np.array_equal(x_ids, y_ids):
        raise ValueError("x and y point_ids differ; comparison script expects identical subset membership")
    x_lookup = {int(point_id): idx for idx, point_id in enumerate(x_ids.tolist())}
    y_lookup = {int(point_id): idx for idx, point_id in enumerate(y_ids.tolist())}
    coords = np.empty((x_solution["time_grid"].shape[0], ordered_ids.shape[0], 2), dtype=np.float32)
    for local_idx, point_id in enumerate(ordered_ids.tolist()):
        coords[:, local_idx, 0] = x_solution["trajectory"][x_lookup[int(point_id)]]
        coords[:, local_idx, 1] = y_solution["trajectory"][y_lookup[int(point_id)]]
    return coords


def _build_summary(
    *,
    osqp_xy: np.ndarray,
    cg_xy: np.ndarray,
    ordered_ids: np.ndarray,
) -> dict[str, object]:
    diff_xy = cg_xy - osqp_xy
    pointwise_l2 = np.linalg.norm(diff_xy, axis=2)
    frame_mean = pointwise_l2.mean(axis=1)
    frame_max = pointwise_l2.max(axis=1)
    final_diff = pointwise_l2[-1]
    top_indices = np.argsort(final_diff)[-10:][::-1]
    return {
        "num_points": int(ordered_ids.shape[0]),
        "num_frames": int(osqp_xy.shape[0]),
        "x": {
            "mae": float(np.mean(np.abs(diff_xy[:, :, 0]))),
            "rmse": float(np.sqrt(np.mean(diff_xy[:, :, 0] ** 2))),
            "max_abs": float(np.max(np.abs(diff_xy[:, :, 0]))),
        },
        "y": {
            "mae": float(np.mean(np.abs(diff_xy[:, :, 1]))),
            "rmse": float(np.sqrt(np.mean(diff_xy[:, :, 1] ** 2))),
            "max_abs": float(np.max(np.abs(diff_xy[:, :, 1]))),
        },
        "xy_l2": {
            "mae": float(np.mean(pointwise_l2)),
            "rmse": float(np.sqrt(np.mean(pointwise_l2 ** 2))),
            "max_abs": float(np.max(pointwise_l2)),
            "frame_mean": frame_mean.astype(float).tolist(),
            "frame_max": frame_max.astype(float).tolist(),
        },
        "top_final_frame_point_diffs": [
            {
                "point_id": int(ordered_ids[idx]),
                "l2_diff": float(final_diff[idx]),
                "dx": float(diff_xy[-1, idx, 0]),
                "dy": float(diff_xy[-1, idx, 1]),
            }
            for idx in top_indices.tolist()
        ],
    }


def _save_figure(
    *,
    output_path: Path,
    normalized_mesh: np.ndarray,
    osqp_xy: np.ndarray,
    cg_xy: np.ndarray,
    ordered_ids: np.ndarray,
) -> None:
    plt = _load_matplotlib()
    final_osqp = osqp_xy[-1]
    final_cg = cg_xy[-1]
    final_diff = np.linalg.norm(final_cg - final_osqp, axis=1)
    frame_mean = np.linalg.norm(cg_xy - osqp_xy, axis=2).mean(axis=1)
    frame_max = np.linalg.norm(cg_xy - osqp_xy, axis=2).max(axis=1)
    time_grid = np.linspace(0.0, 1.0, osqp_xy.shape[0], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    ax.scatter(normalized_mesh[:, 0], normalized_mesh[:, 1], s=10, c="#dddddd")
    ax.scatter(final_osqp[:, 0], final_osqp[:, 1], s=16, c="#1f78b4", label="OSQP")
    ax.scatter(final_cg[:, 0], final_cg[:, 1], s=12, c="#d94841", alpha=0.7, label="matrix_free_cg")
    ax.set_title("Final Frame Overlay")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.scatter(normalized_mesh[:, 0], normalized_mesh[:, 1], s=8, c="#e6e6e6")
    scatter = ax.scatter(
        final_osqp[:, 0],
        final_osqp[:, 1],
        c=final_diff,
        s=20,
        cmap="magma",
    )
    ax.set_title("Final Frame Pointwise L2 Difference")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, shrink=0.8, label="L2 diff")

    ax = axes[1, 0]
    ax.plot(time_grid, frame_mean, color="#1f78b4", linewidth=2, label="frame mean")
    ax.plot(time_grid, frame_max, color="#d94841", linewidth=2, label="frame max")
    ax.set_title("Per-frame Difference Summary")
    ax.set_xlabel("normalized time")
    ax.set_ylabel("L2 diff")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.hist(final_diff, bins=30, color="#6a3d9a", alpha=0.85)
    ax.set_title("Final Frame L2 Difference Histogram")
    ax.set_xlabel("pointwise L2 diff")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Full341 Solver Comparison: OSQP vs matrix_free_cg\nmax final diff={final_diff.max():.6f}, mean final diff={final_diff.mean():.6f}",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full341 OSQP and matrix_free_cg reconstructions.")
    parser.add_argument(
        "--osqp-x",
        type=Path,
        default=Path("outputs/matrix_vis/real/imr_00228_win005_minus_win004_axis_x_full341_anchor_facebox/solution.npz"),
    )
    parser.add_argument(
        "--osqp-y",
        type=Path,
        default=Path("outputs/matrix_vis/real/imr_00228_win005_minus_win004_axis_y_full341_anchor_facebox/solution.npz"),
    )
    parser.add_argument(
        "--cg-x",
        type=Path,
        default=Path("outputs/matrix_vis/real/imr_00228_win005_minus_win004_axis_x_full341_anchor_facebox_matrixfree/solution.npz"),
    )
    parser.add_argument(
        "--cg-y",
        type=Path,
        default=Path("outputs/matrix_vis/real/imr_00228_win005_minus_win004_axis_y_full341_anchor_facebox_matrixfree/solution.npz"),
    )
    parser.add_argument("--layout-source", type=Path, default=DEFAULT_LAYOUT_SOURCE)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/matrix_vis/real_compare/imr_00228_win005_minus_win004_full341_osqp_vs_matrixfree"),
    )
    args = parser.parse_args()

    osqp_x = _load_solution(args.osqp_x.resolve())
    osqp_y = _load_solution(args.osqp_y.resolve())
    cg_x = _load_solution(args.cg_x.resolve())
    cg_y = _load_solution(args.cg_y.resolve())

    ordered_ids = _ordered_point_ids(args.layout_source.resolve())
    osqp_xy = _compose_coordinates(x_solution=osqp_x, y_solution=osqp_y, ordered_ids=ordered_ids)
    cg_xy = _compose_coordinates(x_solution=cg_x, y_solution=cg_y, ordered_ids=ordered_ids)
    normalized_mesh = _normalize_standard_facemesh(_load_canonical_obj_vertices(args.mesh.resolve()))[:, :2]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(osqp_xy=osqp_xy, cg_xy=cg_xy, ordered_ids=ordered_ids)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _save_figure(
        output_path=output_dir / "comparison.png",
        normalized_mesh=normalized_mesh,
        osqp_xy=osqp_xy,
        cg_xy=cg_xy,
        ordered_ids=ordered_ids,
    )
    np.savez(
        output_dir / "difference_data.npz",
        point_ids=ordered_ids.astype(np.int64),
        osqp_coordinates=osqp_xy.astype(np.float32),
        matrixfree_coordinates=cg_xy.astype(np.float32),
        difference=(cg_xy - osqp_xy).astype(np.float32),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
