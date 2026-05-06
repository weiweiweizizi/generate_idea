#!/usr/bin/env python
"""预览 mouth_regions 重建结果在标准 facemesh 上的动态轨迹。

这是 `scripts.matrix_vis.pipelines.preview_real_mouth_regions` 的薄 CLI 包装：
- 读取一对 `axis_x` / `axis_y` solution.npz
- 调用 pipeline 做 mouth + around_mouth 可视化
- 导出 preview 图像、GIF 和中间缓存

常用命令：
```bash
python scripts/matrix_vis/scripts/preview_real_mouth_regions.py

python scripts/matrix_vis/scripts/preview_real_mouth_regions.py \
  --x-solution outputs/matrix_vis/.../axis_x/solution.npz \
  --y-solution outputs/matrix_vis/.../axis_y/solution.npz \
  --anchor-point-id 14 \
  --output-dir outputs/matrix_vis/real_preview/.../preview
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
        description="Preview real mouth + around_mouth results with normalized standard facemesh and anchor-14 alignment."
    )
    parser.add_argument(
        "--x-solution",
        type=Path,
        default=Path(
            "outputs/matrix_vis/real/imr_00228_win005_minus_win004/mouth_regions/anchor_014/matrix_free_cg/axis_x/solution.npz"
        ),
    )
    parser.add_argument(
        "--y-solution",
        type=Path,
        default=Path(
            "outputs/matrix_vis/real/imr_00228_win005_minus_win004/mouth_regions/anchor_014/matrix_free_cg/axis_y/solution.npz"
        ),
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
            "outputs/matrix_vis/real_preview/imr_00228_win005_minus_win004/mouth_regions/anchor_014/default/preview"
        ),
    )
    parser.add_argument("--anchor-point-id", type=int, default=DEFAULT_ANCHOR_POINT_ID)
    args = parser.parse_args()
    run_preview_real_mouth_regions(
        x_solution=str(args.x_solution),
        y_solution=str(args.y_solution),
        output_dir=str(args.output_dir),
        mesh_source=str(args.mesh),
        landmarks_config=str(args.landmarks_config),
        anchor_point_id=args.anchor_point_id,
    )


if __name__ == "__main__":
    main()
