#!/usr/bin/env python
"""预览 mouth_regions 重建结果在标准 facemesh 上的动态轨迹。

这是 `scripts.matrix_vis.pipelines.preview_real_mouth_regions` 的薄 CLI 包装：
- 读取一对 `axis_x` / `axis_y` solution.npz
- 调用 pipeline 做 mouth + around_mouth 可视化
- 导出 preview 图像、GIF 和中间缓存

常用命令：
```bash
# 标准双轴预览：
python scripts/matrix_vis/scripts/preview_real_mouth_regions.py \
  --x-solution outputs/.../axis_x/solution.npz \
  --y-solution outputs/.../axis_y/solution.npz \
  --anchor-point-ids 205,425,200 \
  --output-dir outputs/disentangleNet/.../matrix_vis_exports/.../preview_anchor205_425_200

# y 轴静止模式（无需 --y-solution，从 mesh 投影生成静态 y）：
python scripts/matrix_vis/scripts/preview_real_mouth_regions.py \
  --x-solution outputs/.../axis_x/solution.npz \
  --static-y \
  --anchor-point-ids 205,425,200 \
  --output-dir outputs/disentangleNet/.../matrix_vis_exports/.../preview_x_static_y
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.pipelines.preview_real_mouth_regions import (
    DEFAULT_ANCHOR_POINT_ID,
    DEFAULT_LANDMARK_CONFIG,
    DEFAULT_MESH_SOURCE,
    run_preview_real_mouth_regions,
)


def main() -> None:
    """CLI 入口：调用 mouth-region preview pipeline。"""
    parser = argparse.ArgumentParser(
        description="Preview real mouth + around_mouth results with normalized standard facemesh and anchor 205,425,200 alignment."
    )
    parser.add_argument(
        "--x-solution",
        type=Path,
        default=Path(
            "outputs/disentangleNet/.../matrix_vis_exports/.../axis_x/solution.npz"
        ),
    )
    parser.add_argument(
        "--y-solution",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--static-y",
        action="store_true",
        help="y 轴保持静止（从 mesh 投影生成静态 y，无需 --y-solution）",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(DEFAULT_MESH_SOURCE),
    )
    parser.add_argument(
        "--landmarks-config",
        type=Path,
        default=Path(DEFAULT_LANDMARK_CONFIG),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/disentangleNet/.../matrix_vis_exports/.../preview_x_static_y"
        ),
    )
    parser.add_argument("--anchor-point-id", type=int, default=DEFAULT_ANCHOR_POINT_ID)
    parser.add_argument(
        "--anchor-point-ids",
        type=str,
        default=None,
        help="逗号分隔的锚点 ID 列表（如 '205,425,200'），优先级高于 --anchor-point-id",
    )
    args = parser.parse_args()
    resolved_anchor_point_ids = (
        [int(p.strip()) for p in args.anchor_point_ids.split(",") if p.strip()]
        if args.anchor_point_ids is not None
        else None
    )
    run_preview_real_mouth_regions(
        x_solution=str(args.x_solution),
        y_solution=str(args.y_solution) if args.y_solution is not None else None,
        output_dir=str(args.output_dir),
        mesh_source=str(args.mesh),
        landmarks_config=str(args.landmarks_config),
        anchor_point_id=args.anchor_point_id,
        anchor_point_ids=resolved_anchor_point_ids,
        static_y=bool(args.static_y),
    )


if __name__ == "__main__":
    main()
