#!/usr/bin/env python
"""预览 full341 重建结果在标准 facemesh 上的动态轨迹。

这个脚本面向 full341 子集的单次可视化检查：
- 读取一对 `axis_x` / `axis_y` solution.npz
- 合成为二维轨迹
- 在标准化 canonical facemesh 上导出：
  - 逐帧 PNG
  - 最后一帧 snapshot
  - 动态 GIF
  - 便于下游复用的 `subset_motion_preview.npz`

常用命令：
```bash
python scripts/matrix_vis/scripts/preview_real_full341.py

python scripts/matrix_vis/scripts/preview_real_full341.py \
  --x-solution outputs/matrix_vis/.../axis_x/solution.npz \
  --y-solution outputs/matrix_vis/.../axis_y/solution.npz \
  --output-dir outputs/matrix_vis/real_preview/.../preview
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.core.composition import compose_xy_coordinates
from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
from scripts.matrix_vis.io.load_mesh import _load_canonical_obj_vertices


FACE_WIDTH_POINTS = (127, 356)
FACE_HEIGHT_POINTS = (10, 152)
DEFAULT_ANCHOR_POINT_IDS = (33, 263, 10, 175)


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


def _save_snapshot(
    *,
    output_path: Path,
    static_points: np.ndarray,
    subset_points: np.ndarray,
    anchor_points: np.ndarray,
    title: str,
) -> None:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(static_points[:, 0], static_points[:, 1], s=12, c="#d9d9d9")
    ax.scatter(subset_points[:, 0], subset_points[:, 1], s=18, c="#1f78b4")
    ax.scatter(anchor_points[:, 0], anchor_points[:, 1], s=36, c="#d94841")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_frames(
    *,
    output_dir: Path,
    static_points: np.ndarray,
    subset_coordinates: np.ndarray,
    anchor_points: np.ndarray,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    x_min = min(float(static_points[:, 0].min()), float(subset_coordinates[:, :, 0].min()))
    x_max = max(float(static_points[:, 0].max()), float(subset_coordinates[:, :, 0].max()))
    y_min = min(float(static_points[:, 1].min()), float(subset_coordinates[:, :, 1].min()))
    y_max = max(float(static_points[:, 1].max()), float(subset_coordinates[:, :, 1].max()))
    x_pad = max((x_max - x_min) * 0.12, 0.05)
    y_pad = max((y_max - y_min) * 0.12, 0.05)

    frame_paths: list[Path] = []
    for frame_idx in range(subset_coordinates.shape[0]):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(static_points[:, 0], static_points[:, 1], s=12, c="#d9d9d9")
        ax.scatter(subset_coordinates[frame_idx, :, 0], subset_coordinates[frame_idx, :, 1], s=18, c="#1f78b4")
        ax.scatter(anchor_points[:, 0], anchor_points[:, 1], s=36, c="#d94841")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"frame {frame_idx + 1}/{subset_coordinates.shape[0]}")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        frame_path = output_dir / f"frame_{frame_idx:03d}.png"
        fig.savefig(frame_path, dpi=140)
        plt.close(fig)
        frame_paths.append(frame_path)
    return frame_paths


def _save_gif(frame_paths: list[Path], output_path: Path, *, duration_ms: int = 90) -> None:
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    first, rest = images[0], images[1:]
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main() -> None:
    """CLI 入口：导出 full341 预览图、GIF 与坐标缓存。"""
    parser = argparse.ArgumentParser(
        description="Preview real 341-point reconstruction on normalized standard facemesh without polylines."
    )
    parser.add_argument(
        "--x-solution",
        type=Path,
        default=Path(
            "outputs/matrix_vis/real/imr_00228_win005_minus_win004/full341/anchor_033_263_010_175/matrix_free_cg/axis_x/solution.npz"
        ),
    )
    parser.add_argument(
        "--y-solution",
        type=Path,
        default=Path(
            "outputs/matrix_vis/real/imr_00228_win005_minus_win004/full341/anchor_033_263_010_175/matrix_free_cg/axis_y/solution.npz"
        ),
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("/home/weizilin/code_reproduction/canonical_face/canonical_face_model.obj"),
    )
    parser.add_argument(
        "--layout-source",
        type=Path,
        default=Path("/home/weizilin/code_reproduction/corelation-lm/project/configs/extractors.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/matrix_vis/real_preview/imr_00228_win005_minus_win004/full341/anchor_033_263_010_175/default/preview"
        ),
    )
    args = parser.parse_args()

    x_solution = _load_solution(args.x_solution.resolve())
    y_solution = _load_solution(args.y_solution.resolve())

    standard_mesh = _load_canonical_obj_vertices(args.mesh.resolve())
    normalized_mesh = _normalize_standard_facemesh(standard_mesh)
    subset_ids = resolve_subset_layout(
        subset_layout="face_regions_grouped",
        subset_layout_source=args.layout_source.resolve(),
        subset_layout_extractor_name="mediapipe",
    )
    common_ids, time_grid, subset_coordinates = compose_xy_coordinates(
        x_solution=x_solution,
        y_solution=y_solution,
        preferred_point_ids=np.asarray(subset_ids, dtype=np.int64),
    )

    anchor_points = normalized_mesh[np.asarray(DEFAULT_ANCHOR_POINT_IDS, dtype=np.int64), :2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = _save_frames(
        output_dir=output_dir / "frames",
        static_points=normalized_mesh[:, :2],
        subset_coordinates=subset_coordinates,
        anchor_points=anchor_points,
    )
    _save_snapshot(
        output_path=output_dir / "snapshot_last_frame.png",
        static_points=normalized_mesh[:, :2],
        subset_points=subset_coordinates[-1],
        anchor_points=anchor_points,
        title="normalized facemesh + full341 preview",
    )
    _save_gif(frame_paths, output_dir / "preview.gif")
    np.savez(
        output_dir / "subset_motion_preview.npz",
        point_ids=common_ids.astype(np.int64),
        time_grid=time_grid.astype(np.float32),
        coordinates=subset_coordinates.astype(np.float32),
        anchor_point_ids=np.asarray(DEFAULT_ANCHOR_POINT_IDS, dtype=np.int64),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
