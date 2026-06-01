#!/usr/bin/env python
"""
检查单个 checkpoint 的 side-basis 行为。

此脚本功能：
- 加载单个 checkpoint 及其分组评测 split。
- 导出原始 side basis 矩阵。
- 测量 basis 幅值、左右交换后的对称性、block 级质量集中度。
- 聚合 group 级 side usage 和 side coefficients。
- 导出 basis 统计、group 语义和紧凑 JSON 汇总。

典型用法：
1. 默认 validation split：
   `python scripts/disentangleNet_trainprobe/analysis/analyze_side_interpretability.py \\
      outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/best.pt`
2. 评估所有分组样本：
   `python scripts/disentangleNet_trainprobe/analysis/analyze_side_interpretability.py \\
      outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/best.pt --split all`
3. 覆盖 block 边界：
   `python scripts/disentangleNet_trainprobe/analysis/analyze_side_interpretability.py \\
      outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/best.pt \\
      --block_boundaries 0,22,45,82,119`

主要输出：
- `<output_dir>/side_basis_stats.csv`
- `<output_dir>/group_side_semantics.csv`
- `<output_dir>/summary.json`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import fire
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet_trainprobe.analysis.analyze_checkpoint import (
    build_eval_dataset,
    build_specs,
    fit_linear_probe,
)
from scripts.disentangleNet_trainprobe.analysis.common import (
    SIDE_LABEL_NAMES,
    load_model_from_checkpoint as load_trainprobe_model_from_checkpoint,
)
from scripts.disentangleNet_trainprobe.regions import REGION_NAMES, REGION_SPECS, get_region_spec

# 脚本功能概述：
# 1. 加载 checkpoint
# 2. side basis 基矩阵统计
#   - 每个 basis 的 fro_norm、l1_sum、mean_abs、max_abs
#   - 每个 basis 与 swap 后的 cosine 相似度和 l1 差异（分析左右对称性）
#   - 每个 basis 在 block 上的绝对值分布（分析是否集中在某些 block 上）
# 3. 每个 group 的 side semantics 统计
#   - 每个 group 的 side basis usage 和 side coeff 的均值
# 4. 汇总输出：summary.json，side_basis_stats.csv，group_side_semantics.csv

def parse_levels(levels) -> tuple[int, ...]:
    """解析 levels 配置字符串或列表"""
    if isinstance(levels, str):
        return tuple(int(v) for v in levels.split(",") if str(v).strip())
    if isinstance(levels, (tuple, list)):
        return tuple(int(v) for v in levels)
    raise TypeError(f"Unsupported levels value: {levels!r}")


def masked_group_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """用 valid_mask 对每个分组序列做加权均值池化"""
    if values.ndim < 3:
        raise ValueError(f"Expected at least [B, T, ...], got {tuple(values.shape)}")
    if values.ndim > 3:
        values = values.reshape(values.shape[0], values.shape[1], -1)
    weights = valid_mask.to(values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    num_dataset_classes: int,
):
    return load_trainprobe_model_from_checkpoint(
        checkpoint_path,
        num_dataset_classes=num_dataset_classes,
    )


def infer_half_block_structure(region: str) -> tuple[list[int], list[str]]:
    """
    根据 region 定义自动推导 half-block 边界和标签。

    约定：每个原始 region 内，前一半是 left，后一半是 right。
    返回：
    - boundaries: 局部裁剪坐标系中的 block 边界
    - block_labels: 与边界相邻区间一一对应的标签
    """
    crop_spec = get_region_spec(region)
    boundaries = [0]
    block_labels: list[str] = []

    for region_name in REGION_NAMES:
        spec = REGION_SPECS[region_name]
        if spec.end <= crop_spec.start or spec.start >= crop_spec.end:
            continue
        if spec.start < crop_spec.start or spec.end > crop_spec.end:
            raise ValueError(
                f"Region {region_name!r} is only partially covered by crop {region!r}; "
                "cannot infer left-right half boundaries safely."
            )

        local_start = spec.start - crop_spec.start
        local_end = spec.end - crop_spec.start
        local_mid = local_start + (local_end - local_start) // 2
        boundaries.extend([local_mid, local_end])
        block_labels.extend([f"{region_name}_left", f"{region_name}_right"])

    if boundaries[-1] != crop_spec.end - crop_spec.start:
        raise ValueError(
            f"Inferred boundaries do not cover crop {region!r}: "
            f"{boundaries[-1]} vs {crop_spec.end - crop_spec.start}"
        )
    return boundaries, block_labels


def build_swap_permutation(boundaries: list[int], block_labels: list[str]) -> np.ndarray:
    """构建左右 half-block 交换索引排列。"""
    if len(boundaries) != len(block_labels) + 1:
        raise ValueError("boundaries length must equal len(block_labels) + 1")

    label_to_segment = {
        label: np.arange(boundaries[idx], boundaries[idx + 1], dtype=np.int64)
        for idx, label in enumerate(block_labels)
    }
    swapped_segments = []
    for label in block_labels:
        if label.endswith("_left"):
            target = label[:-5] + "_right"
        elif label.endswith("_right"):
            target = label[:-6] + "_left"
        else:
            raise ValueError(f"Unsupported block label for left-right swap: {label}")
        if target not in label_to_segment:
            raise ValueError(f"Missing mirrored block for {label}: {target}")
        swapped_segments.append(label_to_segment[target])
    return np.concatenate(swapped_segments, axis=0)


def compute_side_basis_stats(
    side_basis: np.ndarray,
    boundaries: list[int],
    block_labels: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    计算 side basis 统计量：
    - 各 basis 的 Frobenius 范数、l1 和、mean_abs、max_abs
    - 左右交换后的 cosine 相似度和 l1 差异
    - block 级质量分布（对角块 vs 非对角块）
    """
    if side_basis.ndim != 3:
        raise ValueError(f"Expected side basis bank [K, H, W], got {side_basis.shape}")

    flat = side_basis.reshape(side_basis.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    cosine = (flat / norms) @ (flat / norms).T

    # 左右交换后的 basis
    swap_perm = build_swap_permutation(boundaries, block_labels)
    side_basis_swapped = side_basis[:, swap_perm][:, :, swap_perm]

    rows = []
    block_abs_mass = []
    for idx in range(side_basis.shape[0]):
        basis = side_basis[idx]
        swapped = side_basis_swapped[idx]
        # block 级绝对值质量矩阵
        block_matrix = np.zeros((len(boundaries) - 1, len(boundaries) - 1), dtype=np.float64)
        for row_idx in range(len(boundaries) - 1):
            rs = slice(boundaries[row_idx], boundaries[row_idx + 1])
            for col_idx in range(len(boundaries) - 1):
                cs = slice(boundaries[col_idx], boundaries[col_idx + 1])
                block_matrix[row_idx, col_idx] = float(np.abs(basis[rs, cs]).sum())
        block_abs_mass.append(block_matrix)
        total_abs = float(np.abs(basis).sum()) + 1e-8
        rows.append(
            {
                "basis_index": idx,
                "fro_norm": float(np.linalg.norm(basis)),
                "l1_sum": float(np.abs(basis).sum()),
                "mean_abs": float(np.abs(basis).mean()),
                "max_abs": float(np.abs(basis).max()),
                "swap_cosine": float(
                    np.dot(basis.reshape(-1), swapped.reshape(-1))
                    / max(np.linalg.norm(basis.reshape(-1)) * np.linalg.norm(swapped.reshape(-1)), 1e-8)
                ),
                "swap_delta_l1_mean": float(np.abs(basis - swapped).mean()),
                "diag_block_mass_ratio": float(np.trace(block_matrix) / total_abs),
                "offdiag_block_mass_ratio": float((block_matrix.sum() - np.trace(block_matrix)) / total_abs),
            }
        )
        for block_idx, label in enumerate(block_labels):
            rows[-1][f"{label}_self_mass_ratio"] = float(block_matrix[block_idx, block_idx] / total_abs)

        region_prefixes = sorted({label.rsplit("_", 1)[0] for label in block_labels})
        for region_prefix in region_prefixes:
            left_label = f"{region_prefix}_left"
            right_label = f"{region_prefix}_right"
            if left_label in block_labels and right_label in block_labels:
                left_idx = block_labels.index(left_label)
                right_idx = block_labels.index(right_label)
                rows[-1][f"{region_prefix}_lr_cross_mass_ratio"] = float(
                    (block_matrix[left_idx, right_idx] + block_matrix[right_idx, left_idx]) / total_abs
                )

    return pd.DataFrame(rows), np.stack(block_abs_mass, axis=0)


def collect_group_side_semantics(
    model: DistNet,
    loader: DataLoader,
    device: str,
) -> pd.DataFrame:
    """
    遍历整个数据集，收集每个 group 的 side semantics 信息。
    返回 DataFrame：group_id / subject / usage / rep / coeff 等。
    """
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["images"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            outputs = model(x, return_group_pooled=True)

            group_valid_mask = valid_mask.any(dim=1)
            if not group_valid_mask.any():
                continue

            # masked mean pooling
            usage = masked_group_mean(outputs["side_path_usage"], valid_mask)
            rep = masked_group_mean(outputs["side_path_representation"], valid_mask)
            coeff = masked_group_mean(outputs["side_path_coefficients"], valid_mask).squeeze(-1)
            side_recon_l1 = masked_group_mean(
                outputs["shared_side_reconstruction"].abs().mean(dim=(2, 3, 4), keepdim=True),
                valid_mask,
            ).squeeze(-1)

            usage_np = usage[group_valid_mask].cpu().numpy().astype(np.float32)
            rep_np = rep[group_valid_mask].cpu().numpy().astype(np.float32)
            coeff_np = coeff[group_valid_mask].cpu().numpy().astype(np.float32)
            side_recon_l1_np = side_recon_l1[group_valid_mask].cpu().numpy().astype(np.float32)

            keep = group_valid_mask.cpu().numpy().astype(bool)
            side_labels = batch["side_label"][group_valid_mask.cpu()].cpu().numpy().astype(np.int64)
            dataset_labels = batch["dataset_label"][group_valid_mask.cpu()].cpu().numpy().astype(np.int64)
            subjects = [subject for subject, flag in zip(batch["subject"], keep) if flag]
            group_ids = [group_id for group_id, flag in zip(batch["group_id"], keep) if flag]
            dataset_names = [name for name, flag in zip(batch["dataset_name"], keep) if flag]

            for idx, group_id in enumerate(group_ids):
                row = {
                    "group_id": group_id,
                    "subject": subjects[idx],
                    "dataset_name": dataset_names[idx],
                    "dataset_label": int(dataset_labels[idx]),
                    "side_label": int(side_labels[idx]),
                    "side_label_name": SIDE_LABEL_NAMES.get(int(side_labels[idx]), str(int(side_labels[idx]))),
                    "side_coeff_mean": float(coeff_np[idx]),
                    "side_coeff_abs_mean": float(abs(coeff_np[idx])),
                    "side_recon_l1_mean": float(side_recon_l1_np[idx]),
                }
                for basis_idx in range(usage_np.shape[1]):
                    row[f"usage_b{basis_idx}"] = float(usage_np[idx, basis_idx])
                    row[f"rep_b{basis_idx}"] = float(rep_np[idx, basis_idx])
                rows.append(row)
    return pd.DataFrame(rows)


def summarise_group_side_semantics(df: pd.DataFrame, side_basis_count: int) -> dict:
    """
    对 group side semantics 表做汇总统计：
    - 按 side_label 和 dataset_name 分组均值
    - usage/coeff 与 side_coeff 的相关性
    - 各 basis 的 top 使用 group 和 top 正/负 rep group
    - 探针准确率：usage / coeff / usage+coeff 预测 side_label 和 dataset_label
    """
    usage_cols = [f"usage_b{i}" for i in range(side_basis_count)]
    rep_cols = [f"rep_b{i}" for i in range(side_basis_count)]

    usage = df[usage_cols].to_numpy(dtype=np.float32)
    rep = df[rep_cols].to_numpy(dtype=np.float32)
    coeff = df[["side_coeff_mean"]].to_numpy(dtype=np.float32)
    usage_coeff = np.concatenate([usage, coeff], axis=1)
    side_labels = df["side_label"].to_numpy(dtype=np.int64)
    dataset_labels = df["dataset_label"].to_numpy(dtype=np.int64)

    # 多组探针
    side_probe_usage = fit_linear_probe(usage, side_labels, seed=42)
    dataset_probe_usage = fit_linear_probe(usage, dataset_labels, seed=42)
    side_probe_coeff = fit_linear_probe(coeff, side_labels, seed=42)
    dataset_probe_coeff = fit_linear_probe(coeff, dataset_labels, seed=42)
    side_probe_usage_coeff = fit_linear_probe(usage_coeff, side_labels, seed=42)
    dataset_probe_usage_coeff = fit_linear_probe(usage_coeff, dataset_labels, seed=42)

    summary = {
        "num_groups": int(df.shape[0]),
        "side_label_counts": {
            SIDE_LABEL_NAMES.get(int(k), str(int(k))): int(v)
            for k, v in df["side_label"].value_counts().sort_index().items()
        },
        "dataset_counts": {
            str(k): int(v) for k, v in df["dataset_name"].value_counts().sort_index().items()
        },
        "coeff_overall": {
            "mean": float(df["side_coeff_mean"].mean()),
            "std": float(df["side_coeff_mean"].std(ddof=0)),
            "mean_abs": float(df["side_coeff_abs_mean"].mean()),
        },
        "probe_summary": {
            "side_from_usage_acc": side_probe_usage["accuracy"],
            "dataset_from_usage_acc": dataset_probe_usage["accuracy"],
            "side_from_coeff_acc": side_probe_coeff["accuracy"],
            "dataset_from_coeff_acc": dataset_probe_coeff["accuracy"],
            "side_from_usage_coeff_acc": side_probe_usage_coeff["accuracy"],
            "dataset_from_usage_coeff_acc": dataset_probe_usage_coeff["accuracy"],
        },
        "usage_means_by_side": {},
        "usage_means_by_dataset": {},
        "coeff_means_by_side": {},
        "coeff_means_by_dataset": {},
        "rep_means_by_side": {},
        "basis_coeff_correlations": {},
        "top_groups_by_usage": {},
        "top_groups_by_positive_rep": {},
        "top_groups_by_negative_rep": {},
    }

    # 按 side_label 分组均值
    for side_label, side_name in SIDE_LABEL_NAMES.items():
        subset = df[df["side_label"] == side_label]
        if subset.empty:
            continue
        summary["usage_means_by_side"][side_name] = {
            col: float(subset[col].mean()) for col in usage_cols
        }
        summary["rep_means_by_side"][side_name] = {
            col: float(subset[col].mean()) for col in rep_cols
        }
        summary["coeff_means_by_side"][side_name] = {
            "mean": float(subset["side_coeff_mean"].mean()),
            "mean_abs": float(subset["side_coeff_abs_mean"].mean()),
            "std": float(subset["side_coeff_mean"].std(ddof=0)),
        }

    # 按 dataset_name 分组均值
    for dataset_name in sorted(df["dataset_name"].unique().tolist()):
        subset = df[df["dataset_name"] == dataset_name]
        summary["usage_means_by_dataset"][dataset_name] = {
            col: float(subset[col].mean()) for col in usage_cols
        }
        summary["coeff_means_by_dataset"][dataset_name] = {
            "mean": float(subset["side_coeff_mean"].mean()),
            "mean_abs": float(subset["side_coeff_abs_mean"].mean()),
            "std": float(subset["side_coeff_mean"].std(ddof=0)),
        }

    # basis 与 side_coeff 的相关性
    coeff_vector = df["side_coeff_mean"].to_numpy(dtype=np.float64)
    for basis_idx in range(side_basis_count):
        usage_vector = df[f"usage_b{basis_idx}"].to_numpy(dtype=np.float64)
        if np.std(usage_vector) < 1e-8 or np.std(coeff_vector) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(usage_vector, coeff_vector)[0, 1])
        summary["basis_coeff_correlations"][f"b{basis_idx}"] = corr

        # top groups by usage / positive rep / negative rep
        top_usage = df.nlargest(5, f"usage_b{basis_idx}")[
            ["group_id", "dataset_name", "subject", "side_label_name", f"usage_b{basis_idx}", "side_coeff_mean", f"rep_b{basis_idx}"]
        ]
        top_pos_rep = df.nlargest(5, f"rep_b{basis_idx}")[
            ["group_id", "dataset_name", "subject", "side_label_name", f"usage_b{basis_idx}", "side_coeff_mean", f"rep_b{basis_idx}"]
        ]
        top_neg_rep = df.nsmallest(5, f"rep_b{basis_idx}")[
            ["group_id", "dataset_name", "subject", "side_label_name", f"usage_b{basis_idx}", "side_coeff_mean", f"rep_b{basis_idx}"]
        ]
        summary["top_groups_by_usage"][f"b{basis_idx}"] = top_usage.to_dict(orient="records")
        summary["top_groups_by_positive_rep"][f"b{basis_idx}"] = top_pos_rep.to_dict(orient="records")
        summary["top_groups_by_negative_rep"][f"b{basis_idx}"] = top_neg_rep.to_dict(orient="records")

    return summary


def analyze(
    checkpoint_path: str,
    data_roots: str | None = None,
    split: str = "val",
    batch_size: int = 64,
    num_workers: int = 0,
    output_dir: str | None = None,
    block_boundaries: str | None = None,
):
    """
    主入口函数。

    参数：
    - `checkpoint_path`：已训练 checkpoint
    - `data_roots`：可选的数据集根路径，默认使用 checkpoint 配置
    - `split`：`train` 或 `val`
    - `batch_size`、`num_workers`：DataLoader 控制参数
    - `output_dir`：输出目录
    - `block_boundaries`：可选的逗号分隔 block 边界；默认按 region 定义自动推导
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})
    data_roots = data_roots or config.get("data_roots")
    if not data_roots:
        raise ValueError("data_roots must be provided or present in checkpoint config")

    mode = config.get("mode", "x")
    region = config.get("region", "mouth")
    use_difference = bool(config.get("use_difference", True))
    signed_normalize = config.get("signed_normalize", "per_sample")
    val_ratio = float(config.get("val_ratio", 0.2))
    seed = int(config.get("seed", 42))
    group_size = int(config.get("group_size", 4))
    apply_deleted_filter = bool(config.get("apply_deleted_filter", True))

    output_dir = Path(output_dir or checkpoint_path.parent / "side_interpretability")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(data_roots)
    dataset = build_eval_dataset(
        specs,
        mode=mode,
        region=region,
        use_difference=use_difference,
        signed_normalize=signed_normalize,
        val_ratio=val_ratio,
        seed=seed,
        group_size=group_size,
        apply_deleted_filter=apply_deleted_filter,
        split=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_config = load_model_from_checkpoint(
        checkpoint_path,
        num_dataset_classes=len(specs),
    )
    model = model.to(device)

    # side basis 统计
    side_basis = model.get_side_basis().detach().cpu().numpy()
    if block_boundaries:
        boundaries = [int(v) for v in block_boundaries.split(",") if v.strip()]
        block_labels = [f"block_{idx}" for idx in range(len(boundaries) - 1)]
    else:
        boundaries, block_labels = infer_half_block_structure(str(region))

    side_basis_stats_df, side_basis_block_abs_mass = compute_side_basis_stats(
        side_basis,
        boundaries,
        block_labels,
    )
    side_basis_stats_path = output_dir / "side_basis_stats.csv"
    side_basis_stats_df.to_csv(side_basis_stats_path, index=False)
    np.save(output_dir / "side_basis_block_abs_mass.npy", side_basis_block_abs_mass.astype(np.float32))
    np.save(output_dir / "side_basis_bank.npy", side_basis.astype(np.float32))

    # group 级 side semantics
    group_side_df = collect_group_side_semantics(model, loader, device)
    group_side_path = output_dir / "group_side_semantics.csv"
    group_side_df.to_csv(group_side_path, index=False)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "analysis_split": split,
        "mode": mode,
        "region": region,
        "half_block_boundaries": boundaries,
        "half_block_labels": block_labels,
        "data_roots": data_roots,
        "side_pooling": model_config.get("side_pooling"),
        "side_basis_count": int(model_config.get("side_basis_count", side_basis.shape[0])),
        "group_semantics": summarise_group_side_semantics(group_side_df, side_basis.shape[0]),
        "basis_summary": {
            "mean_swap_cosine": float(side_basis_stats_df["swap_cosine"].mean()),
            "max_pairwise_cosine_offdiag": float(
                np.max(
                    np.where(
                        np.eye(side_basis.shape[0], dtype=bool),
                        -np.inf,
                        (side_basis.reshape(side_basis.shape[0], -1) /
                         np.clip(np.linalg.norm(side_basis.reshape(side_basis.shape[0], -1), axis=1, keepdims=True), 1e-8, None))
                        @
                        (side_basis.reshape(side_basis.shape[0], -1) /
                         np.clip(np.linalg.norm(side_basis.reshape(side_basis.shape[0], -1), axis=1, keepdims=True), 1e-8, None)).T,
                    )
                )
            ),
            "basis_rows": side_basis_stats_df.to_dict(orient="records"),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved side basis stats: {side_basis_stats_path}")
    print(f"Saved group side semantics: {group_side_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    fire.Fire(analyze)
