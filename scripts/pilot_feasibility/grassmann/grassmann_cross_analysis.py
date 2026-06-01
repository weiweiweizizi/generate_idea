"""
Grassmann流形跨数据集分析
比较多个数据集之间的主模态相似性。

对每个参考数据集 A：
1. 计算 A 的联合 SVD 子空间
2. 计算每个来源数据集 B 的单患者 SVD 子空间
3. 统计 B_single vs A_joint 的 principal angles
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win20-step20"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "pilot_feasibility" / "grassmann" / "win20-step20"
N_COMPONENTS = 4
RANDOM_STATE = 42


def list_available_datasets():
    """返回 data/win20-step20 下可用的原始数据集目录名。"""
    datasets = []
    for path in sorted(DATA_ROOT.iterdir()):
        if not path.is_dir() or path.name.endswith("-SVD"):
            continue
        if (path / "config.json").exists():
            datasets.append(path.name)
    return datasets


def load_patient_windows(dataset_path, subj_id):
    """加载单个患者所有窗口的 x 和 y 矩阵。"""
    subj_path = dataset_path / subj_id
    if not subj_path.exists():
        return None, None

    win_x_files = sorted(subj_path.glob("win_*_x.npy"))
    win_y_files = sorted(subj_path.glob("win_*_y.npy"))

    if len(win_x_files) == 0:
        return None, None

    windows_x = [np.load(f) for f in win_x_files]
    windows_y = [np.load(f) for f in win_y_files]
    return np.array(windows_x), np.array(windows_y)


def compute_diff_matrix(windows):
    """计算前后差分矩阵。"""
    diffs = []
    for i in range(1, len(windows)):
        diffs.append(windows[i] - windows[i - 1])
    return np.array(diffs)


def run_single_patient_svd(diff_data, n_components):
    """对单个患者的数据做 SVD。"""
    n_samples = diff_data.shape[0]
    X = diff_data.reshape(n_samples, -1)

    n_components = min(n_components, n_samples, X.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    _ = svd.fit_transform(X)
    return svd.components_


def run_joint_svd(all_diffs, n_components):
    """对所有患者的差分数据堆叠后做联合 SVD。"""
    X_list = [d.reshape(d.shape[0], -1) for d in all_diffs]
    X_stacked = np.vstack(X_list)

    n_components = min(n_components, X_stacked.shape[0], X_stacked.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    _ = svd.fit_transform(X_stacked)
    return svd.components_, svd.singular_values_


def vt_to_subspace(Vt):
    """将 Vt 矩阵转换为正交基。"""
    Q = Vt.T
    Q, _ = np.linalg.qr(Q)
    return Q


def compute_principal_angles_efficient(Q1, Q2):
    """高效计算两个子空间之间的 principal angles。"""
    Q1, _ = np.linalg.qr(Q1)
    Q2, _ = np.linalg.qr(Q2)

    M = Q1.T @ Q2
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, 0, 1)
    return np.degrees(np.arccos(s))


def pad_angles(angles, target_len):
    """将角度数组填充到目标长度，不足的用 NaN 填充。"""
    padded = np.full(target_len, np.nan)
    n = min(len(angles), target_len)
    padded[:n] = angles[:n]
    return padded


def collect_dataset_diffs(dataset):
    """收集一个数据集内所有患者的差分矩阵。"""
    dataset_path = DATA_ROOT / dataset
    diff_x_list = []
    diff_y_list = []
    patient_ids = []

    for subj_dir in sorted(dataset_path.iterdir()):
        if not subj_dir.is_dir():
            continue
        windows_x, windows_y = load_patient_windows(dataset_path, subj_dir.name)
        if windows_x is None or windows_y is None:
            continue

        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)
        if diff_x.shape[0] < 3 or diff_y.shape[0] < 3:
            continue

        diff_x_list.append(diff_x)
        diff_y_list.append(diff_y)
        patient_ids.append(subj_dir.name)

    return {
        "patient_ids": patient_ids,
        "diff_x": diff_x_list,
        "diff_y": diff_y_list,
    }


def build_single_patient_subspaces(diff_list, dataset, mode_name):
    """为某个数据集构建全部单患者 SVD 子空间。"""
    subspaces = []
    for diff_data in tqdm(diff_list, total=len(diff_list), desc=f"{dataset} single SVD {mode_name}"):
        Vt = run_single_patient_svd(diff_data, N_COMPONENTS)
        subspaces.append(vt_to_subspace(Vt))
    return subspaces


def plot_mode_boxplots(results_by_ref, output_dir, mode_name):
    """按参考数据集绘制箱线图。"""
    for ref_dataset, source_results in results_by_ref.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for pc_idx in range(N_COMPONENTS):
            ax = axes[pc_idx]
            data_list = []
            labels = []

            for src_dataset, values in source_results.items():
                col_data = values[:, pc_idx]
                col_data = col_data[~np.isnan(col_data)]
                if len(col_data) == 0:
                    continue
                data_list.append(col_data)
                labels.append(f"{src_dataset}_single\nvs {ref_dataset}_joint")

            if data_list:
                bp = ax.boxplot(data_list, widths=0.6, patch_artist=True)
                colors = plt.cm.tab10.colors
                for idx, patch in enumerate(bp["boxes"]):
                    patch.set_facecolor(colors[idx % len(colors)])
                    patch.set_alpha(0.6)

            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("Angle (degrees)")
            ax.set_title(f"{mode_name} Mode - PC{pc_idx + 1}")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(0, 100)

        plt.suptitle(f"{mode_name} Mode: principal angles vs {ref_dataset} joint", fontsize=14)
        plt.tight_layout()

        output_path = output_dir / f"cross_dataset_boxplot_{mode_name}_ref_{ref_dataset}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"{mode_name} mode boxplot saved: {output_path}")


def main():
    print("=" * 60)
    print("Grassmann流形跨数据集分析")
    print("=" * 60)

    datasets = list_available_datasets()
    print(f"Datasets found: {datasets}")

    # ========== 加载数据 ==========
    print("\n--- 加载数据 ---")
    dataset_data = {}
    for dataset in datasets:
        dataset_data[dataset] = collect_dataset_diffs(dataset)
        print(f"{dataset} patients: {len(dataset_data[dataset]['patient_ids'])}")

    valid_datasets = [d for d in datasets if dataset_data[d]["patient_ids"]]
    if len(valid_datasets) < 2:
        raise RuntimeError("至少需要两个有有效患者的数据集才能进行跨数据集分析。")

    # ========== 计算联合SVD ==========
    print("\n--- 计算联合SVD ---")
    joint_subspaces = {"X": {}, "Y": {}}
    for dataset in valid_datasets:
        Vt_x, sigma_x = run_joint_svd(dataset_data[dataset]["diff_x"], N_COMPONENTS)
        Vt_y, sigma_y = run_joint_svd(dataset_data[dataset]["diff_y"], N_COMPONENTS)
        joint_subspaces["X"][dataset] = vt_to_subspace(Vt_x)
        joint_subspaces["Y"][dataset] = vt_to_subspace(Vt_y)
        print(f"{dataset} joint X: {Vt_x.shape}, sigma: {sigma_x[:3]}")
        print(f"{dataset} joint Y: {Vt_y.shape}, sigma: {sigma_y[:3]}")

    # ========== 计算每个单患者的 SVD ==========
    print("\n--- 计算单患者SVD ---")
    single_patient_subspaces = {"X": {}, "Y": {}}
    for dataset in valid_datasets:
        single_patient_subspaces["X"][dataset] = build_single_patient_subspaces(
            dataset_data[dataset]["diff_x"], dataset, "X"
        )
        single_patient_subspaces["Y"][dataset] = build_single_patient_subspaces(
            dataset_data[dataset]["diff_y"], dataset, "Y"
        )

    # ========== 计算主角度 ==========
    print("\n--- 计算主角度 ---")
    results = {"X": {}, "Y": {}}

    for mode in ["X", "Y"]:
        for ref_dataset in valid_datasets:
            results[mode][ref_dataset] = {}
            ref_joint = joint_subspaces[mode][ref_dataset]

            for src_dataset in valid_datasets:
                padded_angles = []
                for Q in tqdm(
                    single_patient_subspaces[mode][src_dataset],
                    desc=f"{src_dataset} single vs {ref_dataset} joint {mode}",
                ):
                    angles = compute_principal_angles_efficient(Q, ref_joint)
                    padded_angles.append(pad_angles(angles, N_COMPONENTS))
                results[mode][ref_dataset][src_dataset] = np.array(padded_angles)

    # ========== 可视化与保存 ==========
    print("\n--- 可视化与保存 ---")
    output_dir = OUTPUT_ROOT / "grassmann_cross_analysis_results"
    output_dir.mkdir(exist_ok=True)

    plot_mode_boxplots(results["X"], output_dir, "X")
    plot_mode_boxplots(results["Y"], output_dir, "Y")

    summary = {
        "datasets": valid_datasets,
        "n_components": N_COMPONENTS,
        "n_patients": {dataset: len(dataset_data[dataset]["patient_ids"]) for dataset in valid_datasets},
        "X_mode": {},
        "Y_mode": {},
    }

    for mode in ["X", "Y"]:
        mode_key = f"{mode}_mode"
        for ref_dataset, source_results in results[mode].items():
            summary[mode_key][ref_dataset] = {}
            for src_dataset, values in source_results.items():
                summary[mode_key][ref_dataset][src_dataset] = {
                    "mean": np.nanmean(values, axis=0).tolist(),
                    "std": np.nanstd(values, axis=0).tolist(),
                }

    results_file = output_dir / "cross_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {results_file}")

    print("\n" + "=" * 80)
    print("汇总表格 (mean ± std)")
    print("=" * 80)
    for mode in ["X", "Y"]:
        print(f"\n[{mode} mode]")
        for ref_dataset in valid_datasets:
            print(f"  Reference joint: {ref_dataset}")
            for src_dataset in valid_datasets:
                mean_pc1 = summary[f"{mode}_mode"][ref_dataset][src_dataset]["mean"][0]
                std_pc1 = summary[f"{mode}_mode"][ref_dataset][src_dataset]["std"][0]
                print(f"    {src_dataset}_single vs {ref_dataset}_joint: PC1={mean_pc1:.1f}° ± {std_pc1:.1f}°")

    print("\n解读:")
    print("- 同数据集 single vs joint 角度更小，说明该数据集内部主子空间更稳定。")
    print("- 异数据集 single vs joint 角度更大，说明不同来源间存在系统性差异。")
    print("- 现在输出按参考数据集拆分，可直接比较 XW 相对 IMR/TT 联合基的偏离程度。")


if __name__ == "__main__":
    main()
