#!/usr/bin/env python
"""批量运行 disentangleNet basis 的 matrix_vis 重建与预览。

这个脚本是 `scripts/matrix_vis/scripts/generate_disentanglenet_basis_configs.py`
的下游执行器。它读取一个 disentangleNet basis 导出 manifest，先生成一批
按 manifest `mode` 生成并运行单轴重建配置：

1. `mode=x`: `axis_x` 重建
2. `mode=x`: 使用固定 `axis_y` 解做 preview
3. `mode=x`: 可选生成 “y 不动” synthetic 解做 no-motion-y preview
4. `mode=y`: `axis_y` 重建
5. `mode=y`: 默认生成 “x 不动” synthetic 解做 no-motion-x preview

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
- 单轴重建结果：
  `outputs/matrix_vis/real/disentanglenet/<run_name>/<anchor_tag>/<basis_label>_<idx>/axis_{x|y}/`
- 固定另一轴预览（若启用）：
  `outputs/disentangleNet_*/matrix_vis_exports/basis/preview_anchor205_425_200/fixed_{x|y}/...`
- 另一轴静止预览（可选/默认视 mode 而定）：
  `outputs/disentangleNet_*/matrix_vis_exports/basis/preview_anchor205_425_200/no_motion_{x|y}/...`
- 汇总文件：
  `<generated_dir>/batch_run_summary.json`

常用命令：
```bash
python scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py run \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json

python scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py run \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json \
  --anchor_point_id 205 \
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

from scripts.matrix_vis.core.projection import project_mesh_to_axis
from scripts.matrix_vis.io.config import load_config
from scripts.matrix_vis.io.load_mesh import load_mesh
from scripts.matrix_vis.pipelines.preview_real_mouth_regions import run_preview_real_mouth_regions
from scripts.matrix_vis.pipelines.reconstruct import run_axis_reconstruction
from scripts.matrix_vis.scripts.generate_disentanglenet_basis_configs import (
    DEFAULT_FIXED_Y_CONFIG,
    DEFAULT_FIXED_Y_SOLUTION,
    DEFAULT_X_TEMPLATE_CONFIG,
    generate_configs,
)


def _write_no_motion_solution(
    *,
    reference_solution_path: Path,
    output_path: Path,
    validate_against_solution_path: Path | None = None,
) -> Path:
    """基于参考解构造一份“另一轴完全静止”的 synthetic 解。"""
    reference_payload = np.load(reference_solution_path)
    point_ids = np.asarray(reference_payload["point_ids"], dtype=np.int64)
    time_grid = np.asarray(reference_payload["time_grid"], dtype=np.float32)

    if validate_against_solution_path is not None:
        validate_payload = np.load(validate_against_solution_path)
        validate_point_ids = np.asarray(validate_payload["point_ids"], dtype=np.int64)
        validate_time_grid = np.asarray(validate_payload["time_grid"], dtype=np.float32)
        if point_ids.shape != validate_point_ids.shape or not np.array_equal(point_ids, validate_point_ids):
            raise ValueError("reference solution must share the same point_ids as the reconstructed axis solution")
        if time_grid.shape != validate_time_grid.shape or not np.allclose(time_grid, validate_time_grid):
            raise ValueError("reference solution must share the same time grid as the reconstructed axis solution")

    initial_positions = np.asarray(reference_payload["initial_positions"], dtype=np.float32)
    trajectory = np.repeat(initial_positions[:, None], time_grid.shape[0], axis=1)
    anchor_point_ids = np.asarray(reference_payload.get("anchor_point_ids", []), dtype=np.int64)
    basis_matrix = np.asarray(reference_payload["basis_matrix"], dtype=np.float32)

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


def _write_no_motion_solution_from_config(
    *,
    config_path: str,
    validate_against_solution_path: Path,
    output_path: Path,
) -> Path:
    """基于标准脸配置直接导出静止单轴解。"""
    cfg = load_config(config_path)
    mesh = load_mesh(cfg.mesh)
    projection = project_mesh_to_axis(mesh, cfg.projection)

    validate_payload = np.load(validate_against_solution_path)
    point_ids = np.asarray(validate_payload["point_ids"], dtype=np.int64)
    time_grid = np.asarray(validate_payload["time_grid"], dtype=np.float32)
    basis_matrix = np.asarray(validate_payload["basis_matrix"], dtype=np.float32)

    if point_ids.shape != projection.subset_point_ids.shape or not np.array_equal(
        point_ids, projection.subset_point_ids
    ):
        raise ValueError("template projection point_ids do not match the reconstructed axis solution")

    initial_positions = np.asarray(projection.subset_positions, dtype=np.float32)
    trajectory = np.repeat(initial_positions[:, None], time_grid.shape[0], axis=1)
    anchor_point_ids = np.asarray(projection.anchor_point_ids, dtype=np.int64)

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
    reconstruction_root: str | None = None,
    preview_root: str | None = None,
    x_template_config: str = DEFAULT_X_TEMPLATE_CONFIG,
    fixed_y_config: str = DEFAULT_FIXED_Y_CONFIG,
    fixed_y_solution: str = DEFAULT_FIXED_Y_SOLUTION,
    anchor_point_id: int = 205,
    anchor_point_ids: str | list[int] | tuple[int, ...] | None = None,
    run_fixed_y_if_missing: bool = True,
    run_fixed_y_preview: bool = True,
    run_no_motion_y_preview: bool = False,
    run_fixed_other_preview: bool | None = None,
    run_no_motion_other_preview: bool | None = None,
    limit_basis: int | None = None,
    lambda_laplacian: float | None = None,
    lambda_area_sign: float | None = None,
    area_barrier_margin: float | None = None,
) -> dict:
    """批量运行 basis 对应的单轴重建和预览。

    参数说明：
    - `manifest_path`: basis 导出 manifest 路径
    - `generated_dir`: 生成 YAML 配置的输出目录；为空时使用默认 generated 目录
    - `x_template_config`: x 轴模板配置；`mode=y` 时主要用于兼容保留
    - `fixed_y_config`: fixed-y 模板配置；`mode=y` 时会作为 axis-y 模板
    - `fixed_y_solution`: fixed-y 模板解路径，仅用于配置生成记录
    - `anchor_point_id`: mouth region 对齐锚点
    - `run_fixed_y_if_missing`: 兼容旧 x-mode 接口；若 fixed-y 解不存在，是否先补跑
    - `run_fixed_y_preview`: 兼容旧 x-mode 接口；是否生成 fixed-y 预览
    - `run_no_motion_y_preview`: 兼容旧 x-mode 接口；是否额外生成 y-static 预览
    - `run_fixed_other_preview`: 是否生成固定另一轴预览；默认 x-mode=True, y-mode=False
    - `run_no_motion_other_preview`: 是否生成另一轴静止预览；默认 x-mode=False, y-mode=True
    - `limit_basis`: 调试用途，只跑前 N 个 basis

    返回：
    - 一个可序列化字典，同时会写入 `batch_run_summary.json`
    """
    generation = generate_configs(
        manifest_path=manifest_path,
        output_dir=generated_dir,
        x_template_config=x_template_config,
        fixed_y_config=fixed_y_config,
        fixed_y_solution=fixed_y_solution,
        anchor_point_id=anchor_point_id,
        anchor_point_ids=anchor_point_ids,
        reconstruction_root=reconstruction_root,
        preview_root=preview_root,
        lambda_laplacian=lambda_laplacian,
        lambda_area_sign=lambda_area_sign,
        area_barrier_margin=area_barrier_margin,
    )
    resolved_anchor_point_ids = generation.get("anchor_point_ids") or [int(anchor_point_id)]
    reconstruction_axis = str(generation.get("reconstruction_axis") or generation.get("mode") or "x")
    static_axis = str(generation.get("static_axis") or ("y" if reconstruction_axis == "x" else "x"))
    manifest = json.loads(Path(generation["manifest_path"]).read_text(encoding="utf-8"))
    subset_layout_region_names = manifest.get("point_layout_region_names")
    preview_title = (
        "normalized facemesh + face regions preview"
        if subset_layout_region_names is None
        else "normalized facemesh + selected face regions preview"
    )

    if run_fixed_other_preview is None:
        run_fixed_other_preview = run_fixed_y_preview if reconstruction_axis == "x" else False
    if run_no_motion_other_preview is None:
        run_no_motion_other_preview = run_no_motion_y_preview if reconstruction_axis == "x" else True

    fixed_other_solution_raw = generation.get("fixed_other_solution")
    fixed_other_config_raw = generation.get("fixed_other_config")
    fixed_other_solution_path = Path(fixed_other_solution_raw) if fixed_other_solution_raw else None
    fixed_other_solution_is_compatible = False
    if fixed_other_solution_path is not None and fixed_other_solution_path.exists():
        try:
            fixed_other_payload = np.load(fixed_other_solution_path)
            fixed_other_basis_matrix = np.asarray(fixed_other_payload["basis_matrix"])
            fixed_other_solution_is_compatible = int(fixed_other_basis_matrix.shape[0]) == int(
                manifest.get("matrix_size", 0) or 0
            )
        except Exception:
            fixed_other_solution_is_compatible = False
    if (
        run_fixed_other_preview
        and fixed_other_config_raw is not None
        and run_fixed_y_if_missing
        and not fixed_other_solution_is_compatible
    ):
        run_axis_reconstruction(config=fixed_other_config_raw)
        fixed_other_solution_is_compatible = False
        if fixed_other_solution_path is not None and fixed_other_solution_path.exists():
            try:
                fixed_other_payload = np.load(fixed_other_solution_path)
                fixed_other_basis_matrix = np.asarray(fixed_other_payload["basis_matrix"])
                fixed_other_solution_is_compatible = int(fixed_other_basis_matrix.shape[0]) == int(
                    manifest.get("matrix_size", 0) or 0
                )
            except Exception:
                fixed_other_solution_is_compatible = False
    if run_fixed_other_preview and not fixed_other_solution_is_compatible:
        print(
            "[WARN] fixed-other preview is skipped because the cached fixed solution is missing or incompatible "
            f"with matrix_size={manifest.get('matrix_size')}"
        )
        run_fixed_other_preview = False

    axis_jobs = generation.get("axis_jobs")
    if axis_jobs is None:
        axis_configs = generation["axis_configs"]
        preview_output_dirs = generation["preview_output_dirs"]
        axis_jobs = [
            {
                "basis_label": "basis",
                "basis_index": idx,
                "axis_config": axis_config,
                "fixed_other_preview_output_dir": preview_output_dir,
                "no_motion_other_preview_output_dir": str(
                    Path(preview_output_dir).parent / f"no_motion_{static_axis}" / Path(preview_output_dir).name
                ),
            }
            for idx, (axis_config, preview_output_dir) in enumerate(zip(axis_configs, preview_output_dirs))
        ]
    if limit_basis is not None:
        axis_jobs = axis_jobs[: int(limit_basis)]

    axis_results = []
    fixed_other_preview_results = []
    no_motion_other_preview_results = []
    for job in axis_jobs:
        axis_result = run_axis_reconstruction(config=job["axis_config"])
        axis_results.append(
            {
                **axis_result,
                "basis_label": job["basis_label"],
                "basis_index": int(job["basis_index"]),
            }
        )

        reconstructed_solution_path = Path(axis_result["output_dir"]) / "solution.npz"
        if run_fixed_other_preview:
            if fixed_other_solution_path is None:
                raise ValueError(
                    f"fixed-{static_axis} preview requested, but generation did not provide a fixed solution"
                )
            x_solution_path = (
                reconstructed_solution_path if reconstruction_axis == "x" else fixed_other_solution_path
            )
            y_solution_path = (
                fixed_other_solution_path if reconstruction_axis == "x" else reconstructed_solution_path
            )
            fixed_other_preview_results.append(
                run_preview_real_mouth_regions(
                    x_solution=str(x_solution_path),
                    y_solution=str(y_solution_path),
                    output_dir=job["fixed_other_preview_output_dir"],
                    anchor_point_ids=resolved_anchor_point_ids,
                    subset_layout_region_names=subset_layout_region_names,
                    title=preview_title,
                )
            )
        if run_no_motion_other_preview:
            no_motion_solution_output_path = (
                Path(job["no_motion_other_preview_output_dir"])
                / f"synthetic_no_motion_{static_axis}_solution.npz"
            )
            if reconstruction_axis == "y":
                no_motion_solution_path = _write_no_motion_solution_from_config(
                    config_path=str(generation["x_template_config"]),
                    validate_against_solution_path=reconstructed_solution_path,
                    output_path=no_motion_solution_output_path,
                )
            else:
                fixed_other_config = generation.get("fixed_other_config")
                if fixed_other_config is None:
                    raise ValueError("no-motion-y preview requires generation['fixed_other_config']")
                no_motion_solution_path = _write_no_motion_solution_from_config(
                    config_path=str(fixed_other_config),
                    validate_against_solution_path=reconstructed_solution_path,
                    output_path=no_motion_solution_output_path,
                )
            x_solution_path = (
                reconstructed_solution_path if reconstruction_axis == "x" else no_motion_solution_path
            )
            y_solution_path = (
                no_motion_solution_path if reconstruction_axis == "x" else reconstructed_solution_path
            )
            no_motion_other_preview_results.append(
                run_preview_real_mouth_regions(
                    x_solution=str(x_solution_path),
                    y_solution=str(y_solution_path),
                    output_dir=job["no_motion_other_preview_output_dir"],
                    anchor_point_ids=resolved_anchor_point_ids,
                    subset_layout_region_names=subset_layout_region_names,
                    title=f"{preview_title} ({static_axis} static)",
                    align_to_anchor=False,
                )
            )

    summary = {
        "manifest_path": generation["manifest_path"],
        "generated_dir": generation["generated_dir"],
        "mode": reconstruction_axis,
        "reconstruction_axis": reconstruction_axis,
        "static_axis": static_axis,
        "anchor_point_ids": resolved_anchor_point_ids,
        "anchor_point_id": int(resolved_anchor_point_ids[0]),
        "fixed_other_solution": generation.get("fixed_other_solution"),
        "num_total_jobs": len(axis_results),
        "num_axis_runs": len(axis_results),
        "num_fixed_other_preview_runs": len(fixed_other_preview_results),
        "num_no_motion_other_preview_runs": len(no_motion_other_preview_results),
        "axis_output_dirs": [result["output_dir"] for result in axis_results],
        "fixed_other_preview_output_dirs": [result["output_dir"] for result in fixed_other_preview_results],
        "no_motion_other_preview_output_dirs": [
            result["output_dir"] for result in no_motion_other_preview_results
        ],
        "axis_jobs": axis_jobs,
    }
    if reconstruction_axis == "x":
        summary["fixed_y_solution"] = summary["fixed_other_solution"]
        summary["num_fixed_y_preview_runs"] = summary["num_fixed_other_preview_runs"]
        summary["num_no_motion_y_preview_runs"] = summary["num_no_motion_other_preview_runs"]
        summary["fixed_y_preview_output_dirs"] = summary["fixed_other_preview_output_dirs"]
        summary["no_motion_y_preview_output_dirs"] = summary["no_motion_other_preview_output_dirs"]
    summary_path = Path(generation["generated_dir"]) / "batch_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"run": run_batch})
