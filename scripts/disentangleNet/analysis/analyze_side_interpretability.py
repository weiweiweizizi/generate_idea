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
   `python scripts/disentangleNet/analysis/analyze_side_interpretability.py \\
      outputs/disentangleNet/v31_current_verify/best.pt`
2. 评估所有分组样本：
   `python scripts/disentangleNet/analysis/analyze_side_interpretability.py \\
      outputs/disentangleNet/v31_current_verify/best.pt --split all`
3. 覆盖 block 边界：
   `python scripts/disentangleNet/analysis/analyze_side_interpretability.py \\
      outputs/disentangleNet/v31_current_verify/best.pt \\
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

from scripts.disentangleNet.analysis.analyze_checkpoint import (
    build_eval_dataset,
    build_specs,
    fit_linear_probe,
)
from scripts.disentangleNet.model.distnet import DistNet

# 脚本功能概述：
# 1. 加载 checkpoint
# 2. side basis 基矩阵统计
#   - 每个 basis 的 fro_norm、l1_sum、mean_abs、max_abs
#   - 每个 basis 与 swap 后的 cosine 相似度和 l1 差异（分析左右对称性）
#   - 每个 basis 在 block 上的绝对值分布（分析是否集中在某些 block 上）
# 3. 每个 group 的 side semantics 统计
#   - 每个 group 的 side basis usage 和 side coeff 的均值
# 4. 汇总输出：summary.json，side_basis_stats.csv，group_side_semantics.csv

SIDE_LABEL_NAMES = {
    0: "Left",
    1: "Normal",
    2: "Right",
}


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
) -> tuple[DistNet, dict]:
    """从 checkpoint 加载 DistNet 模型并恢复配置"""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})

    mode = config.get("mode", "x")
    region = config.get("region", "mouth")
    hidden_dim = int(config.get("hidden_dim", 32))
    basis_size = int(config.get("basis_size", 119))
    pool_size = int(config.get("pool_size", 1))
    shared_dim = config.get("shared_dim")
    if shared_dim is not None:
        shared_dim = int(shared_dim)
    private_dim = int(config.get("private_dim", 32))
    private_decoder_hidden_dim = config.get("private_decoder_hidden_dim")
    if private_decoder_hidden_dim is not None:
        private_decoder_hidden_dim = int(private_decoder_hidden_dim)
    levels = parse_levels(config.get("levels", "2,3,6"))
    use_dataset_aux = bool(config.get("use_dataset_aux", False))
    side_semantic_enabled = bool(config.get("side_semantic_enabled", False))
    side_basis_count = int(config.get("side_basis_count", 0))
    side_pooling = str(config.get("side_pooling", "masked_mean"))
    side_subspace_dim = config.get("side_subspace_dim")
    if side_subspace_dim is not None:
        side_subspace_dim = int(side_subspace_dim)
    side_free_frame_qr = bool(config.get("side_free_frame_qr", False))
    early_branch_factorization = bool(config.get("early_branch_factorization", False))
    free_pool_size = int(config.get("free_pool_size", 2))
    side_pool_size = int(config.get("side_pool_size", 2))
    private_pool_size = int(config.get("private_pool_size", 1))
    free_z_dim = config.get("free_z_dim")
    if free_z_dim is not None:
        free_z_dim = int(free_z_dim)
    side_z_dim = config.get("side_z_dim")
    if side_z_dim is not None:
        side_z_dim = int(side_z_dim)
    private_adapter_enabled = bool(config.get("private_adapter_enabled", False))
    num_side_classes = int(config.get("num_side_classes", 3))
    target_label_mode = str(config.get("target_label_mode", "side"))

    model = DistNet(
        levels=levels,
        basis_size=basis_size,
        hidden_dim=hidden_dim,
        pool_size=pool_size,
        shared_dim=shared_dim,
        private_dim=private_dim,
        private_decoder_hidden_dim=private_decoder_hidden_dim,
        num_side_classes=num_side_classes,
        num_dataset_classes=num_dataset_classes,
        target_label_mode=target_label_mode,
        private_residual_weight=float(config.get("private_residual_weight", 0.25)),
        private_residual_max_l1=config.get("private_residual_max_l1"),
        shared_basis_soft_mixing=bool(config.get("shared_basis_soft_mixing", False)),
        shared_basis_anchor_bias=float(config.get("shared_basis_anchor_bias", 1.0)),
        shared_basis_topk=config.get("shared_basis_topk"),
        grl_lambda=float(config.get("grl_lambda", 1.0)),
        use_dataset_aux=use_dataset_aux,
        side_semantic_enabled=side_semantic_enabled,
        side_basis_count=side_basis_count,
        side_pooling=side_pooling,
        side_subspace_dim=side_subspace_dim,
        side_free_frame_qr=side_free_frame_qr,
        early_branch_factorization=early_branch_factorization,
        free_pool_size=free_pool_size,
        side_pool_size=side_pool_size,
        private_pool_size=private_pool_size,
        free_z_dim=free_z_dim,
        side_z_dim=side_z_dim,
        private_adapter_enabled=private_adapter_enabled,
        action_basis_init_path=None,
        lq_commitment_loss_weight=float(config.get("lq_commitment_loss_weight", 0.1)),
        lq_quantization_loss_weight=float(config.get("lq_quantization_loss_weight", 0.1)),
        lq_optimize_values=bool(config.get("lq_optimize_values", True)),
        quantizer_type=config.get("quantizer_type", "latent_quantize"),
        fsq_preserve_symmetry=bool(config.get("fsq_preserve_symmetry", True)),
        basis_orthogonalization=config.get("basis_orthogonalization", "normalize"),
        discrete_side_loss_enabled=bool(config.get("discrete_side_loss_enabled", True)),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    config["mode"] = mode
    config["region"] = region
    return model, config


def build_swap_permutation(boundaries: list[int]) -> np.ndarray:
    """
    构建左右 block 交换的索引排列。
    交换 segments[0] <-> segments[1]（around_left <-> around_right），
    segments[2] <-> segments[3]（mouth_left <-> mouth_right）。
    """
    if boundaries != [0, 22, 45, 82, 119]:
        raise ValueError(f"Unsupported boundaries for left-right block swap: {boundaries}")
    segments = [np.arange(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    swapped = [segments[1], segments[0], segments[3], segments[2]]
    return np.concatenate(swapped, axis=0)


def compute_side_basis_stats(side_basis: np.ndarray, boundaries: list[int]) -> tuple[pd.DataFrame, np.ndarray]:
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
    swap_perm = build_swap_permutation(boundaries)
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
                "around_left_left_mass_ratio": float(block_matrix[0, 0] / total_abs),
                "around_right_right_mass_ratio": float(block_matrix[1, 1] / total_abs),
                "mouth_left_left_mass_ratio": float(block_matrix[2, 2] / total_abs),
                "mouth_right_right_mass_ratio": float(block_matrix[3, 3] / total_abs),
                "around_lr_cross_mass_ratio": float((block_matrix[0, 1] + block_matrix[1, 0]) / total_abs),
                "mouth_lr_cross_mass_ratio": float((block_matrix[2, 3] + block_matrix[3, 2]) / total_abs),
                "around_to_mouth_mass_ratio": float(
                    (
                        block_matrix[0, 2]
                        + block_matrix[0, 3]
                        + block_matrix[1, 2]
                        + block_matrix[1, 3]
                        + block_matrix[2, 0]
                        + block_matrix[2, 1]
                        + block_matrix[3, 0]
                        + block_matrix[3, 1]
                    )
                    / total_abs
                ),
            }
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
    block_boundaries: str = "0,22,45,82,119",
):
    """
    主入口函数。

    参数：
    - `checkpoint_path`：已训练 checkpoint
    - `data_roots`：可选的数据集根路径，默认使用 checkpoint 配置
    - `split`：`train` 或 `val`
    - `batch_size`、`num_workers`：DataLoader 控制参数
    - `output_dir`：输出目录
    - `block_boundaries`：逗号分隔的矩阵 block 边界，用于 block 质量统计
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
    boundaries = [int(v) for v in block_boundaries.split(",") if v.strip()]
    side_basis_stats_df, side_basis_block_abs_mass = compute_side_basis_stats(side_basis, boundaries)
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