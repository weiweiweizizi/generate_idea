"""
SVD 重构极限 MAE 基准测试。

在 data/win20-step20 数据上：
1. 对每个 region (IMR, TT)，将患者随机分为 5 份
2. 用其中 4 份训练 SVD（联合分解），用剩余 1 份测试
3. 重构时：
   - 保持与训练流程一致的 per-sample signed normalization（p98 归一化到 [-1,1]）
   - 先在归一化后数据上做 SVD fit，然后对测试样本做 transform + inverse
   - MAE 与 neural network 保持一致：|recon - x|.mean()（平均到每个元素）

可选 per-PC rank cap：
   --per_pc_rank_caps 3,5,7
   前 3 个主成分分别做 inner SVD，第一个截断到 rank=3，第二个 rank=5，第三个 rank=7

可选可视化：
   --visualize 生成患者级 MAE 分布直方图
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD


def crop_region(mat, start=188, end=307):
    """Crop square matrix to mouth region [start:end, start:end]"""
    return mat[start:end, start:end]


def load_patient_windows(dataset_path, subj_id, mode="x"):
    subj_path = dataset_path / subj_id
    if not subj_path.exists():
        return None

    win_files = sorted(subj_path.glob(f"win_*_{mode}.npy"))
    if len(win_files) == 0:
        return None

    matrices = [np.load(f) for f in win_files]
    return np.array(matrices, dtype=np.float32)


def compute_diff_matrices(windows):
    """前后帧差分，返回 shape (n-1, H, W)，裁剪到 mouth 区域"""
    diffs = []
    for i in range(1, len(windows)):
        diff = windows[i] - windows[i - 1]
        diff = crop_region(diff)
        diffs.append(diff)
    return np.array(diffs)


def per_sample_normalize(mats):
    """
    与训练流程一致：每个矩阵按自身 p98 归一化到 [-1, 1]。
    输入 shape: (N, H, W)
    输出 shape: (N, H, W)
    """
    result = np.zeros_like(mats)
    for i in range(len(mats)):
        scale = np.percentile(np.abs(mats[i]), 98)
        scale = max(scale, 1e-6)
        result[i] = np.clip(mats[i] / scale, -1.0, 1.0)
    return result


def compute_mae(recon, original):
    """|recon - original|.mean()，平均到每个元素"""
    return np.abs(recon - original).mean()


def visualize_patient_mae(all_patient_maes: dict, method_label: str, output_dir: Path) -> None:
    """
    为每个患者绘制 MAE 分布直方图。

    all_patient_maes: {dataset: {subj_id: [window_mae_0, window_mae_1, ...]}}
    每个患者可能有不等数量的窗口 MAE。
    """
    fig, axes = plt.subplots(1, len(all_patient_maes), figsize=(5 * len(all_patient_maes), 4))
    if len(all_patient_maes) == 1:
        axes = [axes]

    colors = {"IMR": "#4C78A8", "TT": "#F58518"}

    for ax, (ds_name, patient_data) in zip(axes, all_patient_maes.items()):
        # 每个患者取平均 MAE，然后收集所有患者
        patient_mean_maes = []
        patient_n_wins = []
        patient_ids = []

        for subj_id, mae_list in patient_data.items():
            patient_mean_maes.append(np.mean(mae_list))
            patient_n_wins.append(len(mae_list))
            patient_ids.append(subj_id)

        patient_mean_maes = np.array(patient_mean_maes)
        patient_n_wins = np.array(patient_n_wins)

        # 颜色按窗口数映射
        normed_nwins = patient_n_wins / patient_n_wins.max()
        bar_colors = [colors.get(ds_name, "#616E7C") for _ in patient_ids]

        # 按 MAE 排序
        order = np.argsort(patient_mean_maes)
        sorted_maes = patient_mean_maes[order]
        sorted_colors = [bar_colors[i] for i in order]
        sorted_ids = [patient_ids[i] for i in order]

        bars = ax.bar(range(len(sorted_maes)), sorted_maes, color=sorted_colors, width=0.8)

        # 叠加标准差（如果有多个窗口）
        for i, (subj_id, mae_list) in enumerate(sorted(zip(sorted_ids, sorted_maes), key=lambda x: x[1])):
            patient_maes_full = patient_data[subj_id]
            if len(patient_maes_full) > 1:
                std = np.std(patient_maes_full)
                ax.errorbar(i, mae_list, yerr=std, fmt="none", color="black", capsize=2, alpha=0.5)

        ax.axhline(np.mean(patient_mean_maes), color="red", linestyle="--",
                   linewidth=1.5, label=f"Mean={np.mean(patient_mean_maes):.3f}")

        ax.set_xlabel("Patient (sorted by MAE)")
        ax.set_ylabel("MAE")
        ax.set_title(f"{ds_name}: {len(patient_data)} patients\n{method_label}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(sorted_maes) * 1.2)
        ax.tick_params(axis="x", labelsize=5)

        # 统计注释
        ax.text(0.02, 0.98, f"n_wins: min={patient_n_wins.min()}, max={patient_n_wins.max()}, median={np.median(patient_n_wins):.0f}",
                transform=ax.transAxes, fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"Patient-level SVD Reconstruction MAE\n{method_label}", fontsize=11, y=1.02)
    plt.tight_layout()

    safe_label = method_label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    plot_path = output_dir / f"patient_mae_histogram_{safe_label}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  患者级 MAE 分布图已保存: {plot_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SVD reconstruction MAE benchmark")
    parser.add_argument("--n_components", type=int, default=10)
    parser.add_argument("--per_pc_rank_caps", type=str, default=None,
                        help="Per-PC rank caps as comma-separated ints, e.g. '3,5,7'."
                             "When set, overrides n_components and applies inner-SVD rank truncation to each PC.")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="x", choices=["x", "y"])
    parser.add_argument("--dataset", type=str, default=None, help="IMR or TT, if None do both")
    parser.add_argument("--visualize", action="store_true", help="Generate patient-level MAE distribution histograms")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    data_root = repo_root / "data" / "win20-step20"

    np.random.seed(args.seed)

    datasets = ["IMR", "TT"] if args.dataset is None else [args.dataset]

    # Parse per-PC rank caps up front
    per_pc_rank_caps = None
    n_components = args.n_components
    method_label = f"{n_components}-comp SVD"
    if args.per_pc_rank_caps:
        per_pc_rank_caps = [int(x) for x in args.per_pc_rank_caps.split(",")]
        n_components = len(per_pc_rank_caps)
        method_label = f"SVD-{'-'.join(str(r) for r in per_pc_rank_caps)}-rank-cap"

    print(f"Method: {method_label}")

    results = {}
    output_dir = repo_root / "outputs" / "pilot_feasibility" / "svd" / "svd_recon_mae_benchmark"
    output_dir.mkdir(exist_ok=True, parents=True)

    for ds_name in datasets:
        ds_path = data_root / ds_name
        if not ds_path.exists():
            continue

        patients = sorted([p.name for p in ds_path.iterdir() if p.is_dir()])
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}, {len(patients)} patients")
        print(f"{'='*60}")

        # 加载所有患者数据（diff matrices，normalized）
        patient_data = {}
        for subj_id in patients:
            wins = load_patient_windows(ds_path, subj_id, mode=args.mode)
            if wins is None or len(wins) < 3:
                continue
            diffs = compute_diff_matrices(wins)
            if len(diffs) < 2:
                continue
            patient_data[subj_id] = diffs

        valid_patients = list(patient_data.keys())
        total_wins = sum(len(v) for v in patient_data.values())
        print(f"Valid patients: {len(valid_patients)}, total diff windows: {total_wins}")

        # 分成 n_folds 份
        np.random.shuffle(valid_patients)
        n_folds = min(args.n_folds, len(valid_patients))
        fold_size = len(valid_patients) // n_folds

        fold_maes = []
        # 每个患者在所有 fold 中只记录一次：
        # - 测试患者：out-of-sample MAE（来自 hold-out fold）
        # - 训练患者：in-sample MAE（仅取第一个出现的 fold，避免重复）
        test_patient_maes = {}   # {subj_id: [mae_win0, ...]}
        train_patient_maes = {}  # {subj_id: [mae_win0, ...]}，仅记录第一次出现

        for fold_idx in range(n_folds):
            test_start = fold_idx * fold_size
            test_end = test_start + fold_size if fold_idx < n_folds - 1 else len(valid_patients)
            test_patients = valid_patients[test_start:test_end]
            train_patients = [p for p in valid_patients if p not in set(test_patients)]

            print(f"\n  Fold {fold_idx+1}/{n_folds}: train={len(train_patients)}, test={len(test_patients)}")

            # 加载训练数据（normalized）
            train_mats = []
            for subj_id in train_patients:
                normed = per_sample_normalize(patient_data[subj_id])
                train_mats.append(normed)

            train_stacked = np.vstack(train_mats)
            n_samples, H, W = train_stacked.shape
            train_flat = train_stacked.reshape(n_samples, -1)

            print(f"    Train stacked: {train_stacked.shape}, flattened: {train_flat.shape}")

            # SVD fit
            svd = TruncatedSVD(n_components=n_components, random_state=args.seed)
            U_train = svd.fit_transform(train_flat)
            Vt = svd.components_  # (n_pc, H*W)

            # Per-PC rank truncation via inner SVD
            if per_pc_rank_caps:
                Vt_parts = []
                for pc_idx, rank_cap in enumerate(per_pc_rank_caps):
                    v_i = Vt[pc_idx].reshape(H, W)
                    inner_u, inner_s, inner_vt = np.linalg.svd(v_i, full_matrices=False)
                    for j in range(rank_cap):
                        vt_row = (inner_s[j] * inner_u[:, j])[:, None] @ inner_vt[j:j+1]
                        Vt_parts.append(vt_row.flatten())
                U_aug = np.column_stack([
                    U_train[:, pc_idx] for pc_idx in range(len(per_pc_rank_caps))
                    for _ in range(per_pc_rank_caps[pc_idx])
                ])
                Vt_aug = np.stack(Vt_parts, axis=0)
            else:
                U_aug = U_train
                Vt_aug = Vt

            # In-sample reconstruction
            recon_train = U_aug @ Vt_aug
            recon_train_3d = recon_train.reshape(n_samples, H, W)
            mae_train = compute_mae(recon_train_3d, train_stacked)
            print(f"    In-sample MAE (train): {mae_train:.4f}")

            # 逐患者 in-sample MAE（训练集患者）
            # 仅记录第一次出现（避免同一患者在多个 fold 中重复记录）
            offset = 0
            for subj_id in train_patients:
                n_wins = patient_data[subj_id].shape[0] - 1  # diff 后少一帧
                if n_wins <= 0:
                    continue
                if subj_id not in train_patient_maes:  # 只记录第一次
                    win_recon = recon_train_3d[offset:offset + n_wins]
                    win_orig = train_stacked[offset:offset + n_wins]
                    win_maes = np.abs(win_recon - win_orig).mean(axis=(1, 2))
                    train_patient_maes[subj_id] = win_maes.tolist()
                offset += n_wins

            # Out-of-sample reconstruction（逐患者处理，保留患者 ID）
            if per_pc_rank_caps:
                U_test_raw = svd.transform(train_flat)  # placeholder
                for subj_id in test_patients:
                    mats = per_sample_normalize(patient_data[subj_id])
                    flat = mats.reshape(len(mats), -1)
                    u_raw = svd.transform(flat)
                    u_parts = []
                    for pc_idx, rank_cap in enumerate(per_pc_rank_caps):
                        u_i = u_raw[:, pc_idx]
                        v_i = Vt[pc_idx].reshape(H, W)
                        _, inner_s, inner_vt = np.linalg.svd(v_i, full_matrices=False)
                        for _ in range(rank_cap):
                            u_parts.append(u_i)
                    U_test_aug = np.column_stack(u_parts)
                    recon = U_test_aug @ Vt_aug
                    recon_3d = recon.reshape(len(mats), H, W)
                    win_maes = np.abs(recon_3d - mats).mean(axis=(1, 2))
                    test_patient_maes[subj_id] = win_maes.tolist()
            else:
                for subj_id in test_patients:
                    mats = per_sample_normalize(patient_data[subj_id])
                    flat = mats.reshape(len(mats), -1)
                    u_test = svd.transform(flat)  # (n_win, n_pc)
                    recon = u_test @ Vt_aug
                    recon_3d = recon.reshape(len(mats), H, W)
                    win_maes = np.abs(recon_3d - mats).mean(axis=(1, 2))
                    test_patient_maes[subj_id] = win_maes.tolist()

            # 同时做批量测试集重构（用于 fold 级 aggregate MAE）
            test_mats_all = []
            for subj_id in test_patients:
                normed = per_sample_normalize(patient_data[subj_id])
                test_mats_all.append(normed)
            test_stacked = np.vstack(test_mats_all)
            n_test = test_stacked.shape[0]
            test_flat = test_stacked.reshape(n_test, -1)
            U_test_raw = svd.transform(test_flat)  # (n_test, n_pc)

            if per_pc_rank_caps:
                # Build augmented U for batch test
                U_test_parts = []
                for pc_idx, rank_cap in enumerate(per_pc_rank_caps):
                    u_i = U_test_raw[:, pc_idx]  # (n_test,)
                    for _ in range(rank_cap):
                        U_test_parts.append(u_i)
                U_test_aug_batch = np.column_stack(U_test_parts)  # (n_test, total_rank)
                recon_test = U_test_aug_batch @ Vt_aug
            else:
                recon_test = U_test_raw @ Vt_aug
            recon_test_3d = recon_test.reshape(n_test, H, W)
            mae_test = compute_mae(recon_test_3d, test_stacked)
            print(f"    Out-of-sample MAE (test): {mae_test:.4f}")

            fold_maes.append(mae_test)

            sample_maes = np.abs(recon_test_3d - test_stacked).mean(axis=(1, 2))
            print(f"    Test MAE per sample: mean={sample_maes.mean():.4f}, "
                  f"std={sample_maes.std():.4f}, min={sample_maes.min():.4f}, max={sample_maes.max():.4f}")

        mean_mae = np.mean(fold_maes)
        std_mae = np.std(fold_maes)
        print(f"\n  [{ds_name}] {method_label} MAE: {mean_mae:.4f} ± {std_mae:.4f}")

        results[ds_name] = {
            "method": method_label,
            "n_folds": n_folds,
            "fold_maes": [float(v) for v in fold_maes],
            "mean_mae": float(mean_mae),
            "std_mae": float(std_mae),
            "n_test_patients": len(test_patient_maes),
            "n_test_windows": sum(len(v) for v in test_patient_maes.values()),
            "n_train_patients": len(train_patient_maes),
            "n_train_windows": sum(len(v) for v in train_patient_maes.values()),
        }

        # 可视化：每个患者的 MAE 分布
        if args.visualize:
            visualize_patient_mae({ds_name: test_patient_maes}, method_label + " (test)", output_dir)
            visualize_patient_mae({ds_name: train_patient_maes}, method_label + " (train)", output_dir)

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    for ds_name, res in results.items():
        print(f"  {ds_name}: {res['method']} MAE = {res['mean_mae']:.4f} ± {res['std_mae']:.4f}")

    # 保存结果
    safe_name = "_".join(str(r) for r in per_pc_rank_caps) if per_pc_rank_caps else str(args.n_components)
    out_file = output_dir / f"svd_mae_{safe_name}comp_{args.mode}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存到: {out_file}")


if __name__ == "__main__":
    main()