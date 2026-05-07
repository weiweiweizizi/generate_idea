#!/usr/bin/env python
"""
导出患者级 matrix-visualization 矩阵束。

此脚本功能：
- 加载一个 checkpoint 和一个患者序列。
- 利用 free/side basis 系数重建每个窗口的组合矩阵。
- 导出 `.npz` 矩阵束（供可视化使用）和 side 预测 CSV。
- 同时保留用于重建的 free/side basis bank，供后续可视化工具使用。

典型用法：
1. 默认患者导出：
   `python scripts/disentangleNet/analysis/export_matrix_vis_patient.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt`
2. 指定患者导出：
   `python scripts/disentangleNet/analysis/export_matrix_vis_patient.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \\
      --subject 844697`
3. 自定义数据根和输出目录：
   `python scripts/disentangleNet/analysis/export_matrix_vis_patient.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \\
      --data_roots data/win20-step20/IMR,data/win20-step20/TT \\
      --output_dir outputs/disentangleNet/v31_current_verify/matrix_vis_exports/patient_custom`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import fire
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet.analysis.analyze_checkpoint import build_specs
from scripts.disentangleNet.analysis.analyze_side_interpretability import (
    SIDE_LABEL_NAMES,
    load_model_from_checkpoint,
)
from scripts.disentangleNet.data import FacialMotionSequenceDataset


def resolve_target_spec(*, data_roots: str, subject: str):
    """
    在 data_roots 中查找目标患者所属的数据集 spec。
    返回匹配的第一个 spec（患者只需存在于其中一个 root）。
    """
    for spec in build_specs(data_roots):
        if (spec.root / subject).exists():
            return spec
    raise FileNotFoundError(f"Subject {subject!r} was not found in any data root from {data_roots!r}")


def compose_window_matrix(
    *,
    shared_weights: np.ndarray,
    shared_basis_bank: np.ndarray,
    side_weights: np.ndarray,
    side_basis_bank: np.ndarray,
) -> np.ndarray:
    """
    将 free path 和 side path 的 basis 组合为单帧窗口矩阵。

    公式：M = Σ_k(shared_weights[k] * shared_basis_bank[k])
              + Σ_k(side_weights[k] * side_basis_bank[k])

    矩阵维度：(basis_size, basis_size)，与距离矩阵 D 对应。
    """
    composed = np.einsum("k,kxy->xy", shared_weights, shared_basis_bank, optimize=True)
    if side_basis_bank.size and side_weights.size:
        composed = composed + np.einsum("k,kxy->xy", side_weights, side_basis_bank, optimize=True)
    return composed.astype(np.float32, copy=False)


def masked_group_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    计算 group 内有效帧的加权平均。

    参数：
    - values：[B, T, ...] 形状的逐帧特征
    - valid_mask：[B, T] 二值掩码，标记有效帧

    返回：[B, ...] 的组平均表征，供 side 分类使用。
    """
    if values.ndim < 3:
        raise ValueError(f"Expected at least [B, T, ...], got {tuple(values.shape)}")
    if values.ndim > 3:
        values = values.reshape(values.shape[0], values.shape[1], -1)
    weights = valid_mask.to(values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom


def export_patient_bundle(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    patient_rows: list[dict[str, Any]],
    shared_basis_bank: np.ndarray,
    side_basis_bank: np.ndarray,
) -> dict[str, Any]:
    """
    将患者的重建数据聚合为标准化的 .npz 束和 CSV 文件。

    npz 包含内容：
    - window_indices / prev_window_indices：窗口索引序列
    - side_pred / side_true：side 预测与真实标签
    - shared_basis_coeffs / side_basis_coeffs：逐窗口 basis 系数
    - free_path_usage / side_path_usage：路径 usage 向量
    - composed_basis_matrices：逐窗口重建矩阵
    - shared_basis_bank / side_basis_bank：basis bank（供可视化重建用）
    - group_id：窗口所属的 group id

    CSV 包含窗口级的 side 预测详情（可读标签名）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按 window_idx 排序，保证时序连贯
    patient_rows = sorted(patient_rows, key=lambda row: int(row["window_idx"]))
    if not patient_rows:
        raise ValueError("patient_rows must not be empty")

    window_indices = np.asarray([int(row["window_idx"]) for row in patient_rows], dtype=np.int64)
    prev_window_indices = np.asarray([int(row["prev_window_idx"]) for row in patient_rows], dtype=np.int64)
    side_pred = np.asarray([int(row["side_pred"]) for row in patient_rows], dtype=np.int64)
    side_true = np.asarray([int(row["side_true"]) for row in patient_rows], dtype=np.int64)
    shared_coeffs = np.stack([row["shared_basis_coeffs"] for row in patient_rows], axis=0).astype(np.float32)
    side_coeffs = np.stack([row["side_basis_coeffs"] for row in patient_rows], axis=0).astype(np.float32)
    free_usage = np.stack([row["free_path_usage"] for row in patient_rows], axis=0).astype(np.float32)
    side_usage = np.stack([row["side_path_usage"] for row in patient_rows], axis=0).astype(np.float32)
    composed_matrices = np.stack([row["composed_basis_matrix"] for row in patient_rows], axis=0).astype(np.float32)

    bundle_path = output_dir / f"patient_{patient_rows[0]['subject']}_x_sequence.npz"
    np.savez(
        bundle_path,
        window_indices=window_indices,
        prev_window_indices=prev_window_indices,
        side_pred=side_pred,
        side_true=side_true,
        shared_basis_coeffs=shared_coeffs,
        side_basis_coeffs=side_coeffs,
        free_path_usage=free_usage,
        side_path_usage=side_usage,
        composed_basis_matrices=composed_matrices,
        shared_basis_bank=shared_basis_bank.astype(np.float32, copy=False),
        side_basis_bank=side_basis_bank.astype(np.float32, copy=False),
        group_id=np.asarray([row["group_id"] for row in patient_rows], dtype=object),
    )

    # 构建 side 预测 CSV（带可读标签名）
    side_df = pd.DataFrame(
        [
            {
                "dataset_name": row["dataset_name"],
                "subject": row["subject"],
                "group_id": row["group_id"],
                "window_idx": int(row["window_idx"]),
                "prev_window_idx": int(row["prev_window_idx"]),
                "side_pred": int(row["side_pred"]),
                "side_pred_name": SIDE_LABEL_NAMES.get(int(row["side_pred"]), str(int(row["side_pred"]))),
                "side_true": int(row["side_true"]),
                "side_true_name": SIDE_LABEL_NAMES.get(int(row["side_true"]), str(int(row["side_true"]))),
            }
            for row in patient_rows
        ]
    )
    side_csv_path = output_dir / f"patient_{patient_rows[0]['subject']}_side_predictions.csv"
    side_df.to_csv(side_csv_path, index=False)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_name": str(patient_rows[0]["dataset_name"]),
        "subject": str(patient_rows[0]["subject"]),
        "mode": "x",
        "region": "mouth",
        "point_layout": "face_regions_grouped",
        "point_layout_region_names": ["around_mouth", "mouth"],
        "matrix_size": int(composed_matrices.shape[-1]),
        "num_valid_windows": int(len(patient_rows)),
        "composition_rule": "shared_free_representation_plus_side_path_representation_weighted_basis_sum",
        "bundle_path": str(bundle_path),
        "side_predictions_csv": str(side_csv_path),
    }
    summary_path = output_dir / f"patient_{patient_rows[0]['subject']}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def export(
    checkpoint_path: str,
    subject: str = "844697",
    data_roots: str | None = None,
    output_dir: str | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """
    主 CLI 入口：导出单个患者的矩阵重建序列。

    参数：
    - `checkpoint_path`：训练好的 checkpoint 路径
    - `subject`：要导出的患者 ID
    - `data_roots`：可选的数据集根目录列表；默认使用 checkpoint 中的配置
    - `output_dir`：可选的自定义输出目录
    - `batch_size`：DataLoader 的 batch size（按 group 分组）
    """
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    subject = str(subject)

    ckpt = torch.load(checkpoint, map_location="cpu")
    config = dict(ckpt.get("config", {}))
    resolved_data_roots = data_roots or config.get("data_roots")
    if not resolved_data_roots:
        raise ValueError("data_roots must be provided either via argument or checkpoint config")

    # 确定数据集 spec
    spec = resolve_target_spec(data_roots=str(resolved_data_roots), subject=subject)
    dataset = FacialMotionSequenceDataset(
        spec,
        [subject],
        mode=str(config.get("mode", "x")),
        region=str(config.get("region", "mouth")),
        use_difference=bool(config.get("use_difference", True)),
        signed_normalize=str(config.get("signed_normalize", "per_sample")),
        group_size=int(config.get("group_size", 4)),
        apply_deleted_filter=bool(config.get("apply_deleted_filter", True)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model, loaded_config = load_model_from_checkpoint(checkpoint, num_dataset_classes=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    patient_rows: list[dict[str, Any]] = []
    shared_basis_bank = None
    side_basis_bank = None

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["images"].to(device), return_group_pooled=True)

            # 从首个 batch 中保存 basis bank（所有 basis 共享同一 bank）
            if shared_basis_bank is None:
                shared_basis_bank = outputs["basis"].detach().cpu().numpy().astype(np.float32)
            if side_basis_bank is None:
                side_basis_bank = outputs["side_basis"].detach().cpu().numpy().astype(np.float32)

            # group 级别的 side 预测（取 group 内有效帧平均后分类）
            valid_mask_torch = batch["valid_mask"].to(device)
            group_side_rep = masked_group_mean(outputs["side_path_representation"], valid_mask_torch)
            group_side_logits = model.classify_side_group(group_side_rep)
            group_side_pred = group_side_logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

            # 提取逐帧特征
            free_rep = outputs["free_path_representation"].detach().cpu().numpy().astype(np.float32)
            side_rep = outputs["side_path_representation"].detach().cpu().numpy().astype(np.float32)
            free_usage = outputs["free_path_usage"].detach().cpu().numpy().astype(np.float32)
            side_usage = outputs["side_path_usage"].detach().cpu().numpy().astype(np.float32)
            valid_mask = batch["valid_mask"].cpu().numpy().astype(bool)
            window_indices = batch["window_indices"].cpu().numpy().astype(np.int64)
            group_ids = list(batch["group_id"])
            subjects = list(batch["subject"])
            dataset_names = list(batch["dataset_name"])
            side_true = batch["side_label"].cpu().numpy().astype(np.int64)
            # prev_window_indices：[num_groups, T] 形状，存储每个 group 中每帧对应的上一窗口索引
            prev_window_groups = [
                [
                    int(per_frame_values[batch_idx].item())
                    if hasattr(per_frame_values[batch_idx], "item")
                    else int(per_frame_values[batch_idx])
                    for per_frame_values in batch["prev_window_indices"]
                ]
                for batch_idx in range(len(subjects))
            ]

            for batch_idx, batch_subject in enumerate(subjects):
                if str(batch_subject) != str(subject):
                    continue
                prev_values = prev_window_groups[batch_idx]
                for frame_idx, keep in enumerate(valid_mask[batch_idx]):
                    if not keep:
                        continue
                    # 组合该帧的 basis 矩阵
                    shared_weights = free_rep[batch_idx, frame_idx]
                    side_weights = side_rep[batch_idx, frame_idx]
                    composed = compose_window_matrix(
                        shared_weights=shared_weights,
                        shared_basis_bank=shared_basis_bank,
                        side_weights=side_weights,
                        side_basis_bank=side_basis_bank,
                    )
                    patient_rows.append(
                        {
                            "dataset_name": dataset_names[batch_idx],
                            "subject": str(batch_subject),
                            "group_id": group_ids[batch_idx],
                            "window_idx": int(window_indices[batch_idx, frame_idx]),
                            "prev_window_idx": int(prev_values[frame_idx]),
                            "side_pred": int(group_side_pred[batch_idx]),
                            "side_true": int(side_true[batch_idx]),
                            "shared_basis_coeffs": shared_weights,
                            "side_basis_coeffs": side_weights,
                            "free_path_usage": free_usage[batch_idx, frame_idx],
                            "side_path_usage": side_usage[batch_idx, frame_idx],
                            "composed_basis_matrix": composed,
                        }
                    )

    if shared_basis_bank is None or side_basis_bank is None:
        raise RuntimeError(f"No valid batches were exported for subject {subject}")

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint.parent / "matrix_vis_exports" / "patients" / f"TT_{subject}"
    )
    summary = export_patient_bundle(
        output_dir=destination,
        checkpoint_path=checkpoint,
        patient_rows=patient_rows,
        shared_basis_bank=shared_basis_bank,
        side_basis_bank=side_basis_bank,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"export": export})
