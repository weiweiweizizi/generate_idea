#!/usr/bin/env python
"""
从指定 checkpoint 导出 basis 矩阵束，供下游 matrix-visualization 工具使用。

此脚本功能：
- 从 checkpoint 加载结构化的 free 和 side basis 矩阵。
- 将其导出为 `.npy` 数组，供 matrix-visualization 使用。
- 可选地渲染紧凑的热图网格。
- 保存一份 JSON manifest，记录 level 边界、点布局和文件语义。

典型用法：
1. 默认导出：
   `python scripts/disentangleNet/analysis/export_matrix_vis_basis.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt`
2. 禁用热图：
   `python scripts/disentangleNet/analysis/export_matrix_vis_basis.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \\
      --save_heatmaps False`
3. 自定义输出目录：
   `python scripts/disentangleNet/analysis/export_matrix_vis_basis.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \\
      --output_dir outputs/disentangleNet/v31_current_verify/matrix_vis_exports/basis_custom`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import fire
import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet.analysis.analyze_checkpoint import plot_basis_grid
from scripts.disentangleNet.model.basis import get_joint_structured_basis


def parse_levels(levels) -> tuple[int, ...]:
    """
    解析 levels 配置，支持字符串（逗号分隔）、元组或列表格式。
    返回整数元组，例如 (2, 3, 6)。
    """
    if isinstance(levels, str):
        return tuple(int(v) for v in levels.split(",") if str(v).strip())
    if isinstance(levels, (tuple, list)):
        return tuple(int(v) for v in levels)
    raise TypeError(f"Unsupported levels value: {levels!r}")


def compute_level_boundaries(levels: tuple[int, ...]) -> list[int]:
    """
    根据各 level 的 basis 数量计算累积边界索引。
    例如 levels=(2,3,6) → [0, 2, 5, 11]。
    """
    boundaries = [0]
    running = 0
    for level in levels:
        running += int(level)
        boundaries.append(running)
    return boundaries


def resolve_bridge_point_layout(*, region: str) -> str:
    """根据 region 返回 bridge 模块使用的点布局方案名称"""
    return "face_regions_grouped"


def resolve_bridge_point_layout_region_names(*, region: str) -> list[str] | None:
    """根据 region 返回 bridge 模块使用的区域子名称列表"""
    if region == "mouth":
        return ["around_mouth", "mouth"]
    return None


def extract_structured_basis_from_checkpoint(
    *,
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """
    从 checkpoint 中解析结构化的 shared basis 和 side basis。

    流程：
    1. 读取 config 和 model state_dict
    2. 提取 action_basis_bank 和 side_basis_bank
    3. 调用 get_joint_structured_basis 构造层次化 basis 结构

    返回：(config字典, shared_basis矩阵, side_basis矩阵)
    """
    config = dict(checkpoint.get("config", {}))
    state_dict = checkpoint.get("model", {})
    if "action_basis_bank" not in state_dict:
        raise KeyError("Checkpoint model state is missing 'action_basis_bank'")

    action_basis = state_dict["action_basis_bank"].detach().cpu().float()
    side_basis = state_dict.get("side_basis_bank")
    if side_basis is None:
        # 若无 side basis，填充空 tensor
        side_basis = torch.zeros((0, action_basis.shape[-1], action_basis.shape[-1]), dtype=torch.float32)
    else:
        side_basis = side_basis.detach().cpu().float()

    levels = parse_levels(config.get("levels", "2,3,6"))
    side_basis_count = int(config.get("side_basis_count", int(side_basis.shape[0])))
    basis_size = int(config.get("basis_size", action_basis.shape[-1]))
    shared_structured, side_structured = get_joint_structured_basis(
        action_basis,
        side_basis,
        levels=levels,
        total_basis_num=int(sum(levels)),
        side_basis_count=side_basis_count,
        basis_size=basis_size,
        basis_orthogonalization=str(config.get("basis_orthogonalization", "normalize")),
    )
    return (
        config,
        shared_structured.detach().cpu().numpy().astype(np.float32, copy=False),
        side_structured.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def build_basis_manifest(
    *,
    checkpoint_path: Path,
    config: dict[str, Any],
    basis: np.ndarray,
    side_basis: np.ndarray,
    point_layout: str,
    point_layout_region_names: list[str] | None,
    exported_basis_path: Path,
    exported_side_basis_path: Path,
) -> dict[str, Any]:
    """
    构造 basis 导出的 manifest 元数据。

    包含 checkpoint 来源、矩阵维度、level 结构、bridge 调用方式等，
    供下游 matrix-visualization 工具在加载时使用。
    """
    levels = parse_levels(config.get("levels", "2,3,6"))
    return {
        "checkpoint_path": str(checkpoint_path),
        "mode": str(config.get("mode", "x")),
        "region": str(config.get("region", "mouth")),
        "matrix_size": int(basis.shape[-1]),
        "num_basis": int(basis.shape[0]),
        "levels": list(levels),
        "level_boundaries": compute_level_boundaries(levels),
        "basis_orthogonalization": str(config.get("basis_orthogonalization", "normalize")),
        "quantizer_type": str(config.get("quantizer_type", "latent_quantize")),
        "point_layout": point_layout,
        "point_layout_region_names": point_layout_region_names,
        "value_semantics": "mean_distance_delta",
        "exported_basis_path": str(exported_basis_path),
        "side_basis_count": int(side_basis.shape[0]),
        "exported_side_basis_path": str(exported_side_basis_path),
        "bridge_scope": {
            "step1": "basis_wise_x_reconstruct_then_compose_with_fixed_y",
            "step2": "patient_coeff_compose_then_x_reconstruct",
        },
    }


def export_basis_bundle(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    config: dict[str, Any],
    basis: np.ndarray,
    side_basis: np.ndarray,
    save_heatmaps: bool,
) -> dict[str, Any]:
    """
    执行实际的 basis 导出逻辑。

    步骤：
    1. 创建输出目录
    2. 保存 basis_bank_x.npy 和 side_basis_bank_x.npy
    3. 可选：调用 plot_basis_grid 渲染热图
    4. 生成 manifest.json 和 export_summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    basis_path = output_dir / "basis_bank_x.npy"
    side_basis_path = output_dir / "side_basis_bank_x.npy"
    np.save(basis_path, basis.astype(np.float32, copy=False))
    np.save(side_basis_path, side_basis.astype(np.float32, copy=False))

    basis_plot_path = None
    side_basis_plot_path = None
    levels = parse_levels(config.get("levels", "2,3,6"))
    if save_heatmaps:
        # 渲染 shared basis 热图（按 level 分块）
        basis_plot_path = output_dir / "basis_bank_x_heatmap.png"
        plot_basis_grid(basis, levels, basis_plot_path)
        if side_basis.shape[0] > 0:
            # side basis 不分层，统一作为一个 block 渲染
            side_basis_plot_path = output_dir / "side_basis_bank_x_heatmap.png"
            plot_basis_grid(side_basis, (side_basis.shape[0],), side_basis_plot_path)

    point_layout = resolve_bridge_point_layout(region=str(config.get("region", "mouth")))
    point_layout_region_names = resolve_bridge_point_layout_region_names(
        region=str(config.get("region", "mouth"))
    )
    manifest = build_basis_manifest(
        checkpoint_path=checkpoint_path,
        config=config,
        basis=basis,
        side_basis=side_basis,
        point_layout=point_layout,
        point_layout_region_names=point_layout_region_names,
        exported_basis_path=basis_path,
        exported_side_basis_path=side_basis_path,
    )
    manifest["artifacts"] = {
        "basis_bank_heatmap": str(basis_plot_path) if basis_plot_path is not None else None,
        "side_basis_bank_heatmap": str(side_basis_plot_path)
        if side_basis_plot_path is not None
        else None,
    }

    manifest_path = output_dir / "basis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "output_dir": str(output_dir),
        "basis_shape": list(basis.shape),
        "side_basis_shape": list(side_basis.shape),
        "manifest_path": str(manifest_path),
        "basis_path": str(basis_path),
        "side_basis_path": str(side_basis_path),
        "point_layout": point_layout,
        "point_layout_region_names": point_layout_region_names,
    }
    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def export(
    checkpoint_path: str,
    output_dir: str | None = None,
    save_heatmaps: bool = True,
) -> dict[str, Any]:
    """
    主 CLI 入口：导出 basis 矩阵束供 matrix 可视化使用。

    参数：
    - `checkpoint_path`：训练好的 checkpoint 路径
    - `output_dir`：可选的自定义输出目录
    - `save_heatmaps`：是否额外渲染 PNG 热图
    """
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    ckpt = torch.load(checkpoint, map_location="cpu")
    loaded_config, basis, side_basis = extract_structured_basis_from_checkpoint(checkpoint=ckpt)

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint.parent / "matrix_vis_exports" / "basis"
    )
    summary = export_basis_bundle(
        output_dir=destination,
        checkpoint_path=checkpoint,
        config=loaded_config,
        basis=basis,
        side_basis=side_basis,
        save_heatmaps=save_heatmaps,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"export": export})
