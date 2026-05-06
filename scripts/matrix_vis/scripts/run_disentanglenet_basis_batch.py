#!/usr/bin/env python
"""批量运行 disentangleNet basis 的 matrix_vis 重建与预览。

这个脚本是 `scripts/matrix_vis/scripts/generate_disentanglenet_basis_configs.py`
的下游执行器。它读取一个 disentangleNet basis 导出 manifest，先生成一批
`axis_x` / `fixed_y` 配置，然后按 basis 逐个运行：

1. `axis_x` 重建
2. 使用固定 `axis_y` 解做 mouth region 预览
3. 可选地额外生成一份 “y 不动” 的 synthetic 解，再做 no-motion-y 预览

适用场景：
- 已经有 `exported_basis_manifest.json`
- 想批量把 free basis / side basis 逐个投影回 mouth-region 轨迹
- 想同时拿到 reconstruction 输出和 preview GIF/PNG

主要输入：
- `manifest_path`:
  `export_matrix_vis_basis.py` 导出的 manifest JSON
- `fixed_y_config`:
  用于生成固定 y 参考解的模板配置
- `fixed_y_solution`:
  仅作为模板记录用途；真正批跑时默认会优先使用生成后的 fixed-y 输出路径

主要输出：
- 生成的 YAML 配置：
  `scripts/matrix_vis/configs/real/disentanglenet/generated/<run_name>/<anchor_tag>/`
- x 轴重建结果：
  `outputs/matrix_vis/real/disentanglenet/<run_name>/<anchor_tag>/<basis_label>_<idx>/axis_x/`
- fixed-y 预览：
  `outputs/matrix_vis/real_preview/disentanglenet/<run_name>/<anchor_tag>/fixed_y/...`
- no-motion-y 预览（可选）：
  `outputs/matrix_vis/real_preview/disentanglenet/<run_name>/<anchor_tag>/no_motion_y/...`
- 汇总文件：
  `<generated_dir>/batch_run_summary.json`

常用命令：
```bash
python scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py run \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json

python scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py run \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json \
  --anchor_point_id 14 \
  --limit_basis 3

python scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py run \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json \
  --run_no_motion_y_preview True
```
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import fire
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.pipelines.preview_real_mouth_regions import run_preview_real_mouth_regions
from scripts.matrix_vis.pipelines.reconstruct import run_axis_reconstruction
from scripts.matrix_vis.scripts.generate_disentanglenet_basis_configs import (
    DEFAULT_FIXED_Y_CONFIG,
    DEFAULT_FIXED_Y_SOLUTION,
    generate_configs,
)


def _write_no_motion_y_solution(
    *,
    x_solution_path: Path,
    reference_y_solution_path: Path,
    output_path: Path,
) -> Path:
    """基于 fixed-y 参考解构造一份 “y 完全静止” 的 synthetic y 解。"""
    x_payload = np.load(x_solution_path)
    reference_y_payload = np.load(reference_y_solution_path)
    point_ids = np.asarray(x_payload["point_ids"], dtype=np.int64)
    time_grid = np.asarray(x_payload["time_grid"], dtype=np.float32)
    reference_point_ids = np.asarray(reference_y_payload["point_ids"], dtype=np.int64)
    reference_time_grid = np.asarray(reference_y_payload["time_grid"], dtype=np.float32)
    if point_ids.shape != reference_point_ids.shape or not np.array_equal(point_ids, reference_point_ids):
        raise ValueError("reference y solution must share the same point_ids as x solution for no-motion export")
    if time_grid.shape != reference_time_grid.shape or not np.allclose(time_grid, reference_time_grid):
        raise ValueError("reference y solution must share the same time grid as x solution for no-motion export")

    initial_positions = np.asarray(reference_y_payload["initial_positions"], dtype=np.float32)
    trajectory = np.repeat(initial_positions[:, None], time_grid.shape[0], axis=1)
    anchor_point_ids = np.asarray(reference_y_payload.get("anchor_point_ids", []), dtype=np.int64)
    basis_matrix = np.asarray(reference_y_payload["basis_matrix"], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        point_ids=point_ids,
        time_grid=time_grid,
        initial_positions=initial_positions,
        trajectory=trajectory,
        anchor_point_ids=anchor_point_ids,
        anchor_point_id=anchor_point_ids[0] if anchor_point_ids.size > 0 else np.asarray(-1, dtype=np.int64),
        basis_matrix=basis_matrix,
    )
    return output_path


def run_batch(
    manifest_path: str,
    generated_dir: str | None = None,
    fixed_y_config: str = DEFAULT_FIXED_Y_CONFIG,
    fixed_y_solution: str = DEFAULT_FIXED_Y_SOLUTION,
    anchor_point_id: int = 14,
    run_fixed_y_if_missing: bool = True,
    run_fixed_y_preview: bool = True,
    run_no_motion_y_preview: bool = False,
    limit_basis: int | None = None,
) -> dict:
    """批量运行 basis 对应的 axis_x 重建和预览。

    参数说明：
    - `manifest_path`: basis 导出 manifest 路径
    - `generated_dir`: 生成 YAML 配置的输出目录；为空时使用默认 generated 目录
    - `fixed_y_config`: fixed-y 模板配置
    - `fixed_y_solution`: fixed-y 模板解路径，仅用于配置生成记录
    - `anchor_point_id`: mouth region 对齐锚点
    - `run_fixed_y_if_missing`: 若 fixed-y 解不存在，是否先补跑
    - `run_fixed_y_preview`: 是否生成 fixed-y 预览
    - `run_no_motion_y_preview`: 是否额外生成 y-static 预览
    - `limit_basis`: 调试用途，只跑前 N 个 basis

    返回：
    - 一个可序列化字典，同时会写入 `batch_run_summary.json`
    """
    generation = generate_configs(
        manifest_path=manifest_path,
        output_dir=generated_dir,
        fixed_y_config=fixed_y_config,
        fixed_y_solution=fixed_y_solution,
        anchor_point_id=anchor_point_id,
    )

    fixed_y_solution_path = Path(generation["fixed_y_solution"])
    if run_fixed_y_if_missing and not fixed_y_solution_path.exists():
        run_axis_reconstruction(config=generation["fixed_y_config"])

    axis_jobs = generation.get("axis_jobs")
    if axis_jobs is None:
        axis_configs = generation["axis_configs"]
        preview_output_dirs = generation["preview_output_dirs"]
        axis_jobs = [
            {
                "basis_label": "basis",
                "basis_index": idx,
                "axis_config": axis_config,
                "fixed_y_preview_output_dir": preview_output_dir,
                "no_motion_y_preview_output_dir": str(
                    Path(preview_output_dir).parent / "no_motion_y" / Path(preview_output_dir).name
                ),
            }
            for idx, (axis_config, preview_output_dir) in enumerate(zip(axis_configs, preview_output_dirs))
        ]
    if limit_basis is not None:
        axis_jobs = axis_jobs[: int(limit_basis)]

    axis_results = []
    fixed_y_preview_results = []
    no_motion_y_preview_results = []
    for job in axis_jobs:
        axis_result = run_axis_reconstruction(config=job["axis_config"])
        axis_results.append(
            {
                **axis_result,
                "basis_label": job["basis_label"],
                "basis_index": int(job["basis_index"]),
            }
        )

        x_solution_path = Path(axis_result["output_dir"]) / "solution.npz"
        if run_fixed_y_preview:
            fixed_y_preview_results.append(
                run_preview_real_mouth_regions(
                    x_solution=str(x_solution_path),
                    y_solution=str(fixed_y_solution_path),
                    output_dir=job["fixed_y_preview_output_dir"],
                    anchor_point_id=int(anchor_point_id),
                )
            )
        if run_no_motion_y_preview:
            no_motion_y_solution_path = _write_no_motion_y_solution(
                x_solution_path=x_solution_path,
                reference_y_solution_path=fixed_y_solution_path,
                output_path=Path(job["no_motion_y_preview_output_dir"]) / "synthetic_no_motion_y_solution.npz",
            )
            no_motion_y_preview_results.append(
                run_preview_real_mouth_regions(
                    x_solution=str(x_solution_path),
                    y_solution=str(no_motion_y_solution_path),
                    output_dir=job["no_motion_y_preview_output_dir"],
                    anchor_point_id=int(anchor_point_id),
                    title="normalized facemesh + mouth regions preview (y static)",
                )
            )

    summary = {
        "manifest_path": generation["manifest_path"],
        "generated_dir": generation["generated_dir"],
        "anchor_point_id": int(anchor_point_id),
        "fixed_y_solution": generation["fixed_y_solution"],
        "num_total_jobs": len(axis_results),
        "num_axis_runs": len(axis_results),
        "num_fixed_y_preview_runs": len(fixed_y_preview_results),
        "num_no_motion_y_preview_runs": len(no_motion_y_preview_results),
        "axis_output_dirs": [result["output_dir"] for result in axis_results],
        "fixed_y_preview_output_dirs": [result["output_dir"] for result in fixed_y_preview_results],
        "no_motion_y_preview_output_dirs": [result["output_dir"] for result in no_motion_y_preview_results],
        "axis_jobs": axis_jobs,
    }
    summary_path = Path(generation["generated_dir"]) / "batch_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"run": run_batch})
