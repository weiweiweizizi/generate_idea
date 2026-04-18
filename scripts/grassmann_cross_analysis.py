"""
Grassmann流形跨数据集分析
比较IMR和TT数据集之间的主模态相似性

分析内容：
1. 相对于IMR联合SVD的主模态：
   - IMR所有被试的单SVD的夹角
   - TT所有被试的单SVD的夹角
   - TT联合SVD的夹角

2. 相对于TT联合SVD的主模态：
   - IMR所有被试的单SVD的夹角
   - TT所有被试的单SVD的夹角
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# 配置
DATA_ROOT = Path("/home/weizilin/generate_idea/data/win20-step20")
N_COMPONENTS = 4
RANDOM_STATE = 42


def load_patient_windows(dataset_path, subj_id):
    """加载单个患者所有窗口的x和y矩阵"""
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
    """计算前后差分矩阵"""
    diffs = []
    for i in range(1, len(windows)):
        diff = windows[i] - windows[i - 1]
        diffs.append(diff)
    return np.array(diffs)


def run_single_patient_svd(diff_data, n_components):
    """对单个患者的数据做SVD"""
    n_samples = diff_data.shape[0]
    X = diff_data.reshape(n_samples, -1)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    U = svd.fit_transform(X)
    sigma = svd.singular_values_
    Vt = svd.components_

    return U, sigma, Vt


def run_joint_svd(all_diffs, n_components):
    """对所有患者的差分数据堆叠后做联合SVD"""
    X_list = [d.reshape(d.shape[0], -1) for d in all_diffs]
    X_stacked = np.vstack(X_list)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    _ = svd.fit_transform(X_stacked)
    sigma = svd.singular_values_
    Vt = svd.components_

    return Vt, sigma


def Vt_to_subspace(Vt, n_components):
    """将Vt矩阵转换为正交基"""
    Q = Vt.T  # shape (dim*dim, n_components)
    Q, _ = np.linalg.qr(Q)
    return Q


def compute_principal_angles_efficient(Q1, Q2):
    """高效计算两个子空间之间的principal angles"""
    Q1, _ = np.linalg.qr(Q1)
    Q2, _ = np.linalg.qr(Q2)

    M = Q1.T @ Q2  # shape (k, k)
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)

    return np.degrees(angles)


def main():
    print("=" * 60)
    print("Grassmann流形跨数据集分析")
    print("=" * 60)

    # ========== 加载数据 ==========
    print("\n--- 加载数据 ---")

    imr_diff_x = []
    imr_diff_y = []
    imr_patient_ids = []

    tt_diff_x = []
    tt_diff_y = []
    tt_patient_ids = []

    # 加载IMR数据
    imr_path = DATA_ROOT / "IMR"
    for subj_dir in sorted(imr_path.iterdir()):
        if not subj_dir.is_dir():
            continue
        windows_x, windows_y = load_patient_windows(imr_path, subj_dir.name)
        if windows_x is None:
            continue
        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)
        if diff_x.shape[0] >= 3:
            imr_diff_x.append(diff_x)
            imr_diff_y.append(diff_y)
            imr_patient_ids.append(subj_dir.name)

    print(f"IMR patients: {len(imr_patient_ids)}")

    # 加载TT数据
    tt_path = DATA_ROOT / "TT"
    for subj_dir in sorted(tt_path.iterdir()):
        if not subj_dir.is_dir():
            continue
        windows_x, windows_y = load_patient_windows(tt_path, subj_dir.name)
        if windows_x is None:
            continue
        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)
        if diff_x.shape[0] >= 3:
            tt_diff_x.append(diff_x)
            tt_diff_y.append(diff_y)
            tt_patient_ids.append(subj_dir.name)

    print(f"TT patients: {len(tt_patient_ids)}")

    # ========== 计算联合SVD ==========
    print("\n--- 计算联合SVD ---")

    # IMR联合SVD
    Vt_imr_x, sigma_imr_x = run_joint_svd(imr_diff_x, N_COMPONENTS)
    Vt_imr_y, sigma_imr_y = run_joint_svd(imr_diff_y, N_COMPONENTS)
    print(f"IMR joint X: {Vt_imr_x.shape}, sigma: {sigma_imr_x[:3]}")
    print(f"IMR joint Y: {Vt_imr_y.shape}, sigma: {sigma_imr_y[:3]}")

    # TT联合SVD
    Vt_tt_x, sigma_tt_x = run_joint_svd(tt_diff_x, N_COMPONENTS)
    Vt_tt_y, sigma_tt_y = run_joint_svd(tt_diff_y, N_COMPONENTS)
    print(f"TT joint X: {Vt_tt_x.shape}, sigma: {sigma_tt_x[:3]}")
    print(f"TT joint Y: {Vt_tt_y.shape}, sigma: {sigma_tt_y[:3]}")

    # ========== 计算每个单患者的SVD ==========
    print("\n--- 计算单患者SVD ---")

    # IMR单患者SVD
    imr_single_Vt_x = []
    imr_single_Vt_y = []
    for diff_x, diff_y in tqdm(zip(imr_diff_x, imr_diff_y), total=len(imr_diff_x), desc="IMR single SVD"):
        _, _, Vt_x = run_single_patient_svd(diff_x, N_COMPONENTS)
        _, _, Vt_y = run_single_patient_svd(diff_y, N_COMPONENTS)
        imr_single_Vt_x.append(Vt_x)
        imr_single_Vt_y.append(Vt_y)

    # TT单患者SVD
    tt_single_Vt_x = []
    tt_single_Vt_y = []
    for diff_x, diff_y in tqdm(zip(tt_diff_x, tt_diff_y), total=len(tt_diff_x), desc="TT single SVD"):
        _, _, Vt_x = run_single_patient_svd(diff_x, N_COMPONENTS)
        _, _, Vt_y = run_single_patient_svd(diff_y, N_COMPONENTS)
        tt_single_Vt_x.append(Vt_x)
        tt_single_Vt_y.append(Vt_y)

    # ========== 计算主角度 ==========
    print("\n--- 计算主角度 ---")

    Q_imr_joint_x = Vt_to_subspace(Vt_imr_x, N_COMPONENTS)
    Q_imr_joint_y = Vt_to_subspace(Vt_imr_y, N_COMPONENTS)
    Q_tt_joint_x = Vt_to_subspace(Vt_tt_x, N_COMPONENTS)
    Q_tt_joint_y = Vt_to_subspace(Vt_tt_y, N_COMPONENTS)

    results = {
        'X': {
            'imr_vs_imr_joint': [],
            'tt_vs_imr_joint': [],
            'tt_vs_tt_joint': [],
            'imr_vs_tt_joint': [],
        },
        'Y': {
            'imr_vs_imr_joint': [],
            'tt_vs_imr_joint': [],
            'tt_vs_tt_joint': [],
            'imr_vs_tt_joint': [],
        }
    }

    def pad_angles(angles, target_len=N_COMPONENTS):
        """将角度数组填充到目标长度，不足的用NaN填充"""
        padded = np.full(target_len, np.nan)
        n = min(len(angles), target_len)
        padded[:n] = angles[:n]
        return padded

    # IMR单患者 vs IMR联合 (X)
    print("Computing IMR single vs IMR joint (X)...")
    for Vt in tqdm(imr_single_Vt_x, desc="IMR vs IMR joint X"):
        Q = Vt_to_subspace(Vt, N_COMPONENTS)
        angles = compute_principal_angles_efficient(Q, Q_imr_joint_x)
        results['X']['imr_vs_imr_joint'].append(pad_angles(angles))

    # TT单患者 vs IMR联合 (X)
    print("Computing TT single vs IMR joint (X)...")
    for Vt in tqdm(tt_single_Vt_x, desc="TT vs IMR joint X"):
        Q = Vt_to_subspace(Vt, N_COMPONENTS)
        angles = compute_principal_angles_efficient(Q, Q_imr_joint_x)
        results['X']['tt_vs_imr_joint'].append(pad_angles(angles))

    # TT单患者 vs TT联合 (X)
    print("Computing TT single vs TT joint (X)...")
    for Vt in tqdm(tt_single_Vt_x, desc="TT vs TT joint X"):
        Q = Vt_to_subspace(Vt, N_COMPONENTS)
        angles = compute_principal_angles_efficient(Q, Q_tt_joint_x)
        results['X']['tt_vs_tt_joint'].append(pad_angles(angles))

    # IMR单患者 vs TT联合 (X)
    print("Computing IMR single vs TT joint (X)...")
    for Vt in tqdm(imr_single_Vt_x, desc="IMR vs TT joint X"):
        Q = Vt_to_subspace(Vt, N_COMPONENTS)
        angles = compute_principal_angles_efficient(Q, Q_tt_joint_x)
        results['X']['imr_vs_tt_joint'].append(pad_angles(angles))

    # Y模态
    print("Computing IMR single vs IMR joint (Y)...")
    for Vt in tqdm(imr_single_Vt_y, desc="IMR vs IMR joint Y"):
        Q = Vt_to_subspace(Vt, N_COMPONENTS)
        angles = compute_principal_angles_efficient(Q, Q_imr_joint_y)
        results['Y']['imr_vs_imr_joint'].append(pad_angles(angles))

    print("Computing TT single vs IMR joint (Y)...")
    for Vt in tqdm(tt_single_Vt_y, desc="TT vs IMR joint Y"):
        Q = Vt_to_subspace(Vt, N_COMPONENTS)
        angles = compute_principal_angles_efficient(Q, Q_imr_joint_y)
        results['Y']['tt_vs_imr_joint'].append(pad_angles(angles))

    print("Computing TT single vs TT joint (Y)...")
    for Vt in tqdm(tt_single_Vt_y, desc="TT vs TT joint Y"):
        Q = Vt_to_subspace(Vt, N_COMPONENTS)
        angles = compute_principal_angles_efficient(Q, Q_tt_joint_y)
        results['Y']['tt_vs_tt_joint'].append(pad_angles(angles))

    print("Computing IMR single vs TT joint (Y)...")
    for Vt in tqdm(imr_single_Vt_y, desc="IMR vs TT joint Y"):
        Q = Vt_to_subspace(Vt, N_COMPONENTS)
        angles = compute_principal_angles_efficient(Q, Q_tt_joint_y)
        results['Y']['imr_vs_tt_joint'].append(pad_angles(angles))

    # ========== 可视化 ==========
    print("\n--- 可视化 ---")

    output_dir = DATA_ROOT / "grassmann_cross_analysis_results"
    output_dir.mkdir(exist_ok=True)

    # 转换为numpy数组（已经是填充后的）
    for mode in ['X', 'Y']:
        for key in results[mode]:
            results[mode][key] = np.array(results[mode][key])

    # 创建汇总表格数据
    summary_data = []
    for mode in ['X', 'Y']:
        for pc_idx in range(N_COMPONENTS):
            row = {
                'mode': mode,
                'pc': pc_idx + 1,
                'IMR_single vs IMR_joint': f"{np.nanmean(results[mode]['imr_vs_imr_joint'][:, pc_idx]):.1f}° ± {np.nanstd(results[mode]['imr_vs_imr_joint'][:, pc_idx]):.1f}°",
                'TT_single vs IMR_joint': f"{np.nanmean(results[mode]['tt_vs_imr_joint'][:, pc_idx]):.1f}° ± {np.nanstd(results[mode]['tt_vs_imr_joint'][:, pc_idx]):.1f}°",
                'TT_single vs TT_joint': f"{np.nanmean(results[mode]['tt_vs_tt_joint'][:, pc_idx]):.1f}° ± {np.nanstd(results[mode]['tt_vs_tt_joint'][:, pc_idx]):.1f}°",
                'IMR_single vs TT_joint': f"{np.nanmean(results[mode]['imr_vs_tt_joint'][:, pc_idx]):.1f}° ± {np.nanstd(results[mode]['imr_vs_tt_joint'][:, pc_idx]):.1f}°",
            }
            summary_data.append(row)

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("汇总表格 (mean ± std)")
    print("=" * 80)
    print(f"{'Mode':<6} {'PC':<4} {'IMR_single':<20} {'TT_single':<20} {'TT_single':<20} {'IMR_single':<20}")
    print(f"{'':6} {'':4} {'vs IMR_joint':<20} {'vs IMR_joint':<20} {'vs TT_joint':<20} {'vs TT_joint':<20}")
    print("-" * 80)
    for row in summary_data:
        print(f"{row['mode']:<6} {row['pc']:<4} {row['IMR_single vs IMR_joint']:<20} {row['TT_single vs IMR_joint']:<20} {row['TT_single vs TT_joint']:<20} {row['IMR_single vs TT_joint']:<20}")

    # ========== 可视化 ==========

    # 图1: 按PC分面的箱线图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    from matplotlib.patches import Patch

    for pc_idx in range(N_COMPONENTS):
        row = pc_idx // 2
        col = pc_idx % 2
        ax = axes[row, col]

        # 每个比较的数据（过滤NaN）
        data_list = []
        labels = []
        colors_list = []

        for key, label_base, color in [
            ('imr_vs_imr_joint', 'IMR_single\nvs IMR_joint', '#3498db'),
            ('tt_vs_imr_joint', 'TT_single\nvs IMR_joint', '#e74c3c'),
            ('tt_vs_tt_joint', 'TT_single\nvs TT_joint', '#e74c3c'),
            ('imr_vs_tt_joint', 'IMR_single\nvs TT_joint', '#3498db'),
        ]:
            col_data = results['X'][key][:, pc_idx]
            col_data = col_data[~np.isnan(col_data)]
            if len(col_data) > 0:
                data_list.append(col_data)
                labels.append(label_base)
                colors_list.append(color)

        positions = np.arange(len(data_list))

        bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'X Mode - PC{pc_idx+1}')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)

    plt.suptitle('X Mode: Principal Angles Comparison', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / 'cross_dataset_boxplot_X.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nX mode boxplot saved: {output_path}")

    # Y模态
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for pc_idx in range(N_COMPONENTS):
        row = pc_idx // 2
        col = pc_idx % 2
        ax = axes[row, col]

        data_list = []
        labels = []
        colors_list = []

        for key, label_base, color in [
            ('imr_vs_imr_joint', 'IMR_single\nvs IMR_joint', '#3498db'),
            ('tt_vs_imr_joint', 'TT_single\nvs IMR_joint', '#e74c3c'),
            ('tt_vs_tt_joint', 'TT_single\nvs TT_joint', '#e74c3c'),
            ('imr_vs_tt_joint', 'IMR_single\nvs TT_joint', '#3498db'),
        ]:
            col_data = results['Y'][key][:, pc_idx]
            col_data = col_data[~np.isnan(col_data)]
            if len(col_data) > 0:
                data_list.append(col_data)
                labels.append(label_base)
                colors_list.append(color)

        positions = np.arange(len(data_list))

        bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'Y Mode - PC{pc_idx+1}')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)

    plt.suptitle('Y Mode: Principal Angles Comparison', fontsize=14)
    plt.tight_layout()

    output_path_y = output_dir / 'cross_dataset_boxplot_Y.png'
    plt.savefig(output_path_y, dpi=150)
    plt.close()
    print(f"Y mode boxplot saved: {output_path_y}")

    # ========== 保存结果 ==========
    save_results = {
        'n_imr_patients': len(imr_patient_ids),
        'n_tt_patients': len(tt_patient_ids),
        'n_components': N_COMPONENTS,
        'X_mode': {
            'IMR_single_vs_IMR_joint_mean': np.nanmean(results['X']['imr_vs_imr_joint'], axis=0).tolist(),
            'TT_single_vs_IMR_joint_mean': np.nanmean(results['X']['tt_vs_imr_joint'], axis=0).tolist(),
            'TT_single_vs_TT_joint_mean': np.nanmean(results['X']['tt_vs_tt_joint'], axis=0).tolist(),
            'IMR_single_vs_TT_joint_mean': np.nanmean(results['X']['imr_vs_tt_joint'], axis=0).tolist(),
        },
        'Y_mode': {
            'IMR_single_vs_IMR_joint_mean': np.nanmean(results['Y']['imr_vs_imr_joint'], axis=0).tolist(),
            'TT_single_vs_IMR_joint_mean': np.nanmean(results['Y']['tt_vs_imr_joint'], axis=0).tolist(),
            'TT_single_vs_TT_joint_mean': np.nanmean(results['Y']['tt_vs_tt_joint'], axis=0).tolist(),
            'IMR_single_vs_TT_joint_mean': np.nanmean(results['Y']['imr_vs_tt_joint'], axis=0).tolist(),
        }
    }

    results_file = output_dir / 'cross_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved: {results_file}")

    # ========== 结论 ==========
    print("\n" + "=" * 60)
    print("关键发现")
    print("=" * 60)

    print("\n1. 相对于IMR联合SVD:")
    print(f"   - IMR单患者 vs IMR联合: X PC1={save_results['X_mode']['IMR_single_vs_IMR_joint_mean'][0]:.1f}°")
    print(f"   - TT单患者 vs IMR联合:  X PC1={save_results['X_mode']['TT_single_vs_IMR_joint_mean'][0]:.1f}°")

    print("\n2. 相对于TT联合SVD:")
    print(f"   - TT单患者 vs TT联合:  X PC1={save_results['X_mode']['TT_single_vs_TT_joint_mean'][0]:.1f}°")
    print(f"   - IMR单患者 vs TT联合: X PC1={save_results['X_mode']['IMR_single_vs_TT_joint_mean'][0]:.1f}°")

    print("\n解读:")
    print("- '单患者 vs 同数据集联合' 角度小 → 该患者发现的基与同数据集联合基一致")
    print("- '单患者 vs 异数据集联合' 角度大 → 数据集间存在差异")
    print("- 如果IMR_single vs IMR_joint ≈ TT_single vs IMR_joint → 基是共享的")
    print("- 如果TT_single vs TT_joint > IMR_single vs IMR_joint → TT内部异质性更大")


if __name__ == "__main__":
    main()
