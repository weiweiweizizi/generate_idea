#!/usr/bin/env python
"""为 disentangleNet basis 导出生成 matrix_vis 批处理配置。

这个脚本负责把一个 basis manifest 转成一组可直接运行的 YAML：
- 1 份 fixed-y `axis_y` 配置
- 多份 `axis_x` basis 配置
  - `basis_*` 对应 free basis
  - `side_basis_*` 对应 side basis（如果 manifest 里存在）

它本身不执行重建，只负责“配配置 + 组织目录 + 产出 job 清单”。
通常由 `run_disentanglenet_basis_batch.py` 继续消费。

主要输入：
- `manifest_path`: `export_matrix_vis_basis.py` 导出的 manifest JSON
- `x_template_config`: x 轴模板 YAML
- `fixed_y_config`: y 轴模板 YAML

主要输出：
- `generated_dir/`
  - `fixed_y_axis_y.yaml`
  - `basis_00_axis_x.yaml` 等
  - `side_basis_00_axis_x.yaml` 等
  - `generation_summary.json`

常用命令：
```bash
python scripts/matrix_vis/scripts/generate_disentanglenet_basis_configs.py generate \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json

python scripts/matrix_vis/scripts/generate_disentanglenet_basis_configs.py generate \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json \
  --anchor_point_id 14
```
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import fire
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


DEFAULT_X_TEMPLATE_CONFIG = (
    "scripts/matrix_vis/configs/real/imr_00228_win005_minus_win004/mouth_regions/anchor_014/matrix_free_cg/axis_x.yaml"
)
DEFAULT_FIXED_Y_CONFIG = (
    "scripts/matrix_vis/configs/real/imr_00228_win005_minus_win004/mouth_regions/anchor_014/matrix_free_cg/axis_y.yaml"
)
DEFAULT_FIXED_Y_SOLUTION = (
    "outputs/matrix_vis/real/imr_00228_win005_minus_win004/mouth_regions/anchor_018/osqp/axis_y/solution.npz"
)


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 并确保顶层为 mapping。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return payload


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    """写出 YAML，自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def sanitize_run_name(manifest_path: Path) -> str:
    """从 manifest 路径中提炼稳定的 run 名称，用于 generated/output 目录。"""
    if manifest_path.parent.name == "basis" and manifest_path.parents[1].name == "matrix_vis_exports":
        candidate = manifest_path.parents[2].name
    elif manifest_path.parent.name == "basis":
        candidate = manifest_path.parents[1].name
    else:
        candidate = manifest_path.parent.name
    return candidate.replace(" ", "_")


def build_projection_section(
    *,
    x_template: dict[str, Any],
    manifest: dict[str, Any],
    anchor_point_ids: list[int],
) -> dict[str, Any]:
    """构造 matrix_vis 投影段，固定为 subset-layout + axis-x 模式。"""
    projection = dict(x_template.get("projection", {}))
    region_names = manifest.get("point_layout_region_names")
    projection["subset_layout"] = {
        "name": manifest["point_layout"],
        "source": "scripts/matrix_vis/configs/landmarks/mediapipe_face_regions_full.yaml",
        "extractor_name": "mediapipe",
        "region_names": region_names,
    }
    projection["anchor_point_ids"] = [int(point_id) for point_id in anchor_point_ids]
    projection["axis"] = "x"
    projection["source_axis_index"] = 0
    return projection


def _anchor_tag(anchor_point_ids: list[int]) -> str:
    return "anchor_" + "_".join(str(int(point_id)) for point_id in anchor_point_ids)


def build_fixed_y_config(
    *,
    fixed_y_template: dict[str, Any],
    manifest: dict[str, Any],
    run_name: str,
    anchor_point_ids: list[int],
) -> dict[str, Any]:
    """基于模板生成 fixed-y `axis_y` 配置。"""
    anchor_tag = _anchor_tag(anchor_point_ids)
    payload = json.loads(json.dumps(fixed_y_template))
    payload["experiment"]["name"] = f"disentanglenet_{run_name}_{anchor_tag}_fixed_y_axis_y"
    payload["experiment"]["output_dir"] = (
        f"outputs/matrix_vis/real/disentanglenet/{run_name}/{anchor_tag}/fixed_y_axis_y"
    )
    payload["projection"] = build_projection_section(
        x_template=fixed_y_template,
        manifest=manifest,
        anchor_point_ids=anchor_point_ids,
    )
    payload["projection"]["axis"] = "y"
    payload["projection"]["source_axis_index"] = 1
    payload["projection"].pop("anchor_point_id", None)
    basis_matrix_layout = dict(payload.get("basis", {}).get("matrix_layout", {}))
    basis_matrix_layout["name"] = manifest["point_layout"]
    basis_matrix_layout["source"] = "scripts/matrix_vis/configs/landmarks/mediapipe_face_regions_full.yaml"
    basis_matrix_layout["extractor_name"] = "mediapipe"
    region_names = manifest.get("point_layout_region_names")
    if region_names is None:
        basis_matrix_layout.pop("region_names", None)
    else:
        basis_matrix_layout["region_names"] = region_names
    payload.setdefault("basis", {})["matrix_layout"] = basis_matrix_layout
    return payload


def build_basis_config(
    *,
    basis_stack_path: str,
    basis_index: int,
) -> dict[str, Any]:
    """把某个 basis stack 路径和 basis 索引包装成 matrix_vis basis 段。"""
    return {
        "source": basis_stack_path,
        "basis_index": int(basis_index),
        "matrix_shape": "square",
        "value_semantics": "mean_distance_delta",
    }


def build_axis_config(
    *,
    x_template: dict[str, Any],
    manifest: dict[str, Any],
    run_name: str,
    basis_label: str,
    basis_index: int,
    basis_stack_path: str,
    anchor_point_ids: list[int],
) -> dict[str, Any]:
    """为单个 free/side basis 生成一份 `axis_x` YAML payload。"""
    anchor_tag = _anchor_tag(anchor_point_ids)
    experiment_name = f"disentanglenet_{run_name}_{basis_label}_{basis_index:02d}_axis_x"
    return {
        "experiment": {
            "name": f"{experiment_name}_{anchor_tag}",
            "output_dir": (
                f"outputs/matrix_vis/real/disentanglenet/{run_name}/{anchor_tag}/"
                f"{basis_label}_{basis_index:02d}/axis_x"
            ),
        },
        "mesh": dict(x_template.get("mesh", {})),
        "projection": build_projection_section(
            x_template=x_template,
            manifest=manifest,
            anchor_point_ids=anchor_point_ids,
        ),
        "basis": build_basis_config(
            basis_stack_path=basis_stack_path,
            basis_index=basis_index,
        ),
        "solver": dict(x_template.get("solver", {})),
        "export": dict(x_template.get("export", {})),
    }


def _override_anchor_point_id(config: dict[str, Any], anchor_point_ids: list[int]) -> None:
    """统一把旧格式/新格式锚点字段收敛到 `anchor_point_ids`。"""
    config["projection"]["anchor_point_ids"] = [int(point_id) for point_id in anchor_point_ids]
    config["projection"].pop("anchor_point_id", None)


def generate_configs(
    manifest_path: str,
    output_dir: str | None = None,
    x_template_config: str = DEFAULT_X_TEMPLATE_CONFIG,
    fixed_y_config: str = DEFAULT_FIXED_Y_CONFIG,
    fixed_y_solution: str = DEFAULT_FIXED_Y_SOLUTION,
    anchor_point_id: int = 14,
    anchor_point_ids: str | list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """根据 basis manifest 批量生成 fixed-y 与 axis-x 配置。

    返回的 summary 可直接交给 `run_disentanglenet_basis_batch.py` 使用。
    其中 `axis_jobs` 已经按 `basis` / `side_basis` 组织好，每项包含：
    - `basis_label`
    - `basis_index`
    - `axis_config`
    - `fixed_y_preview_output_dir`
    - `no_motion_y_preview_output_dir`
    """
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    x_template_file = Path(x_template_config).expanduser().resolve()
    fixed_y_config_file = Path(fixed_y_config).expanduser().resolve()

    x_template = load_yaml(x_template_file)
    fixed_y_template = load_yaml(fixed_y_config_file)
    run_name = sanitize_run_name(manifest_file)
    if anchor_point_ids is None:
        resolved_anchor_point_ids = [int(anchor_point_id)]
    elif isinstance(anchor_point_ids, str):
        resolved_anchor_point_ids = [int(part.strip()) for part in anchor_point_ids.split(",") if part.strip()]
    else:
        resolved_anchor_point_ids = [int(point_id) for point_id in anchor_point_ids]
    if not resolved_anchor_point_ids:
        raise ValueError("anchor_point_ids must not be empty")
    anchor_tag = _anchor_tag(resolved_anchor_point_ids)
    base_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path("scripts/matrix_vis/configs/real/disentanglenet/generated").resolve() / run_name / anchor_tag
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    fixed_y_generated_config = build_fixed_y_config(
        fixed_y_template=fixed_y_template,
        manifest=manifest,
        run_name=run_name,
        anchor_point_ids=resolved_anchor_point_ids,
    )
    fixed_y_generated_config_path = base_dir / "fixed_y_axis_y.yaml"
    dump_yaml(fixed_y_generated_config_path, fixed_y_generated_config)
    fixed_y_generated_solution_path = (
        Path(fixed_y_generated_config["experiment"]["output_dir"]).resolve() / "solution.npz"
    )

    generated_axes: list[str] = []
    fixed_y_preview_output_dirs: list[str] = []
    no_motion_y_preview_output_dirs: list[str] = []
    axis_jobs: list[dict[str, Any]] = []

    job_specs = [
        {
            "basis_label": "basis",
            "count": int(manifest["num_basis"]),
            "stack_path": str(manifest["exported_basis_path"]),
        }
    ]
    side_basis_count = int(manifest.get("side_basis_count", 0))
    side_basis_path = manifest.get("exported_side_basis_path")
    if side_basis_count > 0 and side_basis_path:
        job_specs.append(
            {
                "basis_label": "side_basis",
                "count": side_basis_count,
                "stack_path": str(side_basis_path),
            }
        )

    for job_spec in job_specs:
        basis_label = str(job_spec["basis_label"])
        for basis_index in range(int(job_spec["count"])):
            axis_config = build_axis_config(
                x_template=x_template,
                manifest=manifest,
                run_name=run_name,
                basis_label=basis_label,
                basis_index=basis_index,
                basis_stack_path=str(job_spec["stack_path"]),
                anchor_point_ids=resolved_anchor_point_ids,
            )
            _override_anchor_point_id(axis_config, resolved_anchor_point_ids)
            axis_path = base_dir / f"{basis_label}_{basis_index:02d}_axis_x.yaml"
            dump_yaml(axis_path, axis_config)
            generated_axes.append(str(axis_path))
            anchor_preview_tag = "_".join(str(int(point_id)) for point_id in resolved_anchor_point_ids)

            fixed_y_preview_dir = str(
                Path(
                    f"outputs/matrix_vis/real_preview/disentanglenet/{run_name}/{anchor_tag}/fixed_y/"
                    f"{basis_label}_{basis_index:02d}_preview_anchor{anchor_preview_tag}"
                ).resolve()
            )
            no_motion_y_preview_dir = str(
                Path(
                    f"outputs/matrix_vis/real_preview/disentanglenet/{run_name}/{anchor_tag}/no_motion_y/"
                    f"{basis_label}_{basis_index:02d}_preview_anchor{anchor_preview_tag}"
                ).resolve()
            )
            fixed_y_preview_output_dirs.append(fixed_y_preview_dir)
            no_motion_y_preview_output_dirs.append(no_motion_y_preview_dir)
            axis_jobs.append(
                {
                    "basis_label": basis_label,
                    "basis_index": basis_index,
                    "axis_config": str(axis_path),
                    "fixed_y_preview_output_dir": fixed_y_preview_dir,
                    "no_motion_y_preview_output_dir": no_motion_y_preview_dir,
                }
            )

    summary = {
        "manifest_path": str(manifest_file),
        "generated_dir": str(base_dir),
        "run_name": run_name,
        "anchor_point_ids": resolved_anchor_point_ids,
        "anchor_point_id": int(resolved_anchor_point_ids[0]),
        "anchor_tag": anchor_tag,
        "num_basis": int(manifest["num_basis"]),
        "num_side_basis": side_basis_count,
        "num_total_jobs": len(axis_jobs),
        "fixed_y_template_config": str(fixed_y_config_file),
        "fixed_y_template_solution": str(Path(fixed_y_solution).expanduser().resolve()),
        "fixed_y_config": str(fixed_y_generated_config_path),
        "fixed_y_solution": str(fixed_y_generated_solution_path),
        "axis_configs": generated_axes,
        "preview_output_dirs": fixed_y_preview_output_dirs,
        "fixed_y_preview_output_dirs": fixed_y_preview_output_dirs,
        "no_motion_y_preview_output_dirs": no_motion_y_preview_output_dirs,
        "axis_jobs": axis_jobs,
    }
    summary_path = base_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"generate": generate_configs})
