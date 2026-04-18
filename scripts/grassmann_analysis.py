"""
Grassmann流形分析 - 验证单患者SVD与多患者联合SVD的基是否相似
验证"共享基是强迫计算还是必然存在"

原理:
- 单患者SVD: 每个患者独立得到自己的V_i（主模态列空间）
- 多患者联合SVD: 堆叠后得到共同的V_all
- 如果V的列空间与各V_i的列空间相似 → 共享基是"必然存在"
- 如果差异很大 → 可能是"计算强迫"的结果

度量方法:
1. Principal Angles (主角度): 两个子空间之间的夹角
2. Subspace Projection F-norm: ||U_i U_i^T - U_all U_all^T||_F
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# 配置
DATA_ROOT = Path("/home/weizilin/generate_idea/data/win20-step20")
N_COMPONENTS = 4  # 用于比较的主成分数量
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
    """
    对单个患者的数据做SVD
    返回: U, sigma, Vt (其中Vt的每一行是一个基向量)
    """
    n_samples = diff_data.shape[0]
    X = diff_data.reshape(n_samples, -1)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    U = svd.fit_transform(X)
    sigma = svd.singular_values_
    Vt = svd.components_  # shape: (n_components, dim*dim)

    return U, sigma, Vt


def run_multi_patient_svd(all_diffs, n_components):
    """
    对所有患者的差分数据堆叠后做联合SVD
    all_diffs: list of diff matrices, each (n_windows_i, 341, 341)
    返回: Vt_all
    """
    # 堆叠所有患者的差分窗口
    X_list = [d.reshape(d.shape[0], -1) for d in all_diffs]
    X_stacked = np.vstack(X_list)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    _ = svd.fit_transform(X_stacked)
    sigma = svd.singular_values_
    Vt_all = svd.components_

    return Vt_all, sigma


def compute_principal_angles_efficient(Q1, Q2):
    """
    高效计算两个子空间之间的principal angles
    Q1, Q2: 正交基矩阵，shape (d, k) where k <= d

    原理: 如果Q1, Q2是正交基，则cos(θ_i) = σ_i(Q1^T Q2)，其中σ_i是奇异值

    返回: angles in degrees
    """
    # 确保Q1, Q2是列正交的
    Q1, _ = np.linalg.qr(Q1)
    Q2, _ = np.linalg.qr(Q2)

    # 计算Q1^T Q2，然后求奇异值
    M = Q1.T @ Q2  # shape (k, k)
    _, s, _ = np.linalg.svd(M)

    # s是cos(θ)，避免数值问题
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)

    return np.degrees(angles)


def compute_subspace_projection_norm_efficient(Q1, Q2):
    """
    高效计算子空间投影F范数: ||Q1 Q1^T - Q2 Q2^T||_F

    原理: ||Q1 Q1^T - Q2 Q2^T||_F^2 = 2 * sum(sin^2(θ_i))
    其中θ_i是principal angles

    这个值越小，两个子空间越接近
    """
    # 确保Q1, Q2是列正交的
    Q1, _ = np.linalg.qr(Q1)
    Q2, _ = np.linalg.qr(Q2)

    # 计算Q1^T Q2的奇异值（cosine of angles）
    M = Q1.T @ Q2  # shape (k, k)
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, 0, 1)

    # ||Q1 Q1^T - Q2 Q2^T||_F^2 = 2 * sum((1 - s_i^2))
    norm_sq = 2 * np.sum(1 - s ** 2)

    return np.sqrt(norm_sq)


def Vt_to_subspace(Vt, n_components):
    """
    将Vt矩阵转换为正交基
    Vt: shape (n_components, dim*dim)
    返回: Q, shape (dim*dim, n_components)
    """
    # Vt的每一行是一个基向量
    # 我们需要列空间，即Vt.T的列
    Q = Vt.T  # shape (dim*dim, n_components)
    # 正交化
    Q, _ = np.linalg.qr(Q)
    return Q


def compute_correlation_matrix(V_single_list, V_multi):
    """
    计算每个单患者基与多患者基之间的相关性矩阵
    V_single_list: list of Vt matrices, each (n_comp_i, dim*dim)
    V_multi: Vt matrix from multi-patient SVD

    返回: correlation matrix (n_patients, n_comp)
    """
    n_patients = len(V_single_list)
    n_comp = V_multi.shape[0]
    correlations = np.zeros((n_patients, n_comp))

    V_multi_vecs = V_multi[:n_comp]  # (n_comp, dim*dim)

    for i, V_single in enumerate(V_single_list):
        # 获取该患者实际能得到的最大components
        n_single = min(V_single.shape[0], n_comp)
        for j in range(n_single):
            # 计算第i个患者的第j个基与多患者第j个基的相关性
            corr, _ = pearsonr(V_single[j], V_multi_vecs[j])
            correlations[i, j] = corr
        # 剩余位置填NaN
        for j in range(n_single, n_comp):
            correlations[i, j] = np.nan

    return correlations


def visualize_principal_angles_heatmap(angles_dict, save_dir, n_show=5):
    """
    可视化principal angles热图
    angles_dict: dict, keys are 'X' and 'Y', values are (n_patients, n_angles)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (mode, angles) in enumerate(angles_dict.items()):
        n_patients, n_angles = angles.shape

        # 热图
        im = axes[idx].imshow(angles[:, :n_show], cmap='RdYlBu_r',
                               aspect='auto', vmin=0, vmax=90)
        axes[idx].set_xlabel('Principal Angle Index')
        axes[idx].set_ylabel('Patient Index')
        axes[idx].set_title(f'{mode} Mode: Principal Angles (degrees)')
        axes[idx].set_xticks(range(min(n_show, n_angles)))
        axes[idx].set_xticklabels([f'θ{i+1}' for i in range(min(n_show, n_angles))])
        axes[idx].set_yticks(range(n_patients))
        axes[idx].set_yticklabels([f'P{i}' for i in range(n_patients)], fontsize=8)
        plt.colorbar(im, ax=axes[idx], shrink=0.8, label='Degrees')

        # 添加数值标注
        for i in range(n_patients):
            for j in range(min(n_show, n_angles)):
                val = angles[i, j]
                color = 'white' if val > 45 else 'black'
                axes[idx].text(j, i, f'{val:.1f}', ha='center', va='center',
                              color=color, fontsize=7)

    plt.suptitle('Single-Patient vs Multi-Patient SVD: Principal Angles\n(Lower = More Similar)', fontsize=12)
    plt.tight_layout()

    output_path = save_dir / 'principal_angles_heatmap.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_projection_norm_heatmap(norm_dict, save_dir):
    """
    可视化子空间投影F范数热图
    norm_dict: dict, keys are 'X' and 'Y', values are (n_patients,)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (mode, norms) in enumerate(norm_dict.items()):
        n_patients = len(norms)

        axes[idx].barh(range(n_patients), norms, color='steelblue', alpha=0.8)
        axes[idx].set_xlabel('Subspace Projection F-norm')
        axes[idx].set_ylabel('Patient Index')
        axes[idx].set_title(f'{mode} Mode: Subspace Difference')
        axes[idx].set_yticks(range(n_patients))
        axes[idx].set_yticklabels([f'P{i}' for i in range(n_patients)], fontsize=8)
        axes[idx].grid(True, alpha=0.3, axis='x')

        # 添加数值标注
        for i, val in enumerate(norms):
            axes[idx].text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=8)

    plt.suptitle('Single-Patient vs Multi-Patient: Subspace Projection Norm\n(Lower = More Similar to Joint)', fontsize=12)
    plt.tight_layout()

    output_path = save_dir / 'projection_norm.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_correlation_heatmap(corr_dict, save_dir):
    """
    可视化单患者基与多患者基之间的相关性热图
    corr_dict: dict, keys are 'X' and 'Y', values are (n_patients, n_comp)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (mode, corr) in enumerate(corr_dict.items()):
        n_patients, n_comp = corr.shape

        im = axes[idx].imshow(corr, cmap='RdBu_r', aspect='auto',
                               vmin=-1, vmax=1)
        axes[idx].set_xlabel('Component Index')
        axes[idx].set_ylabel('Patient Index')
        axes[idx].set_title(f'{mode} Mode: Correlation (Single vs Multi)')
        axes[idx].set_xticks(range(n_comp))
        axes[idx].set_xticklabels([f'PC{i+1}' for i in range(n_comp)])
        axes[idx].set_yticks(range(n_patients))
        axes[idx].set_yticklabels([f'P{i}' for i in range(n_patients)], fontsize=8)
        plt.colorbar(im, ax=axes[idx], shrink=0.8, label='Pearson r')

    plt.suptitle('Single-Patient vs Multi-Patient SVD: Basis Correlation\n(Higher |r| = More Similar)', fontsize=12)
    plt.tight_layout()

    output_path = save_dir / 'correlation_heatmap.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_summary_barchart(results_x, results_y, save_dir):
    """
    可视化汇总对比柱状图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # X模式 - Principal Angles
    pc_labels = [f'PC{i+1}' for i in range(N_COMPONENTS)]
    x_angles_mean = results_x['angles_mean']
    x_angles_std = results_x['angles_std']

    axes[0, 0].bar(pc_labels, x_angles_mean, yerr=x_angles_std,
                   color='steelblue', alpha=0.8, capsize=3)
    axes[0, 0].set_xlabel('Component')
    axes[0, 0].set_ylabel('Mean Principal Angle (degrees)')
    axes[0, 0].set_title('X Mode: Mean Principal Angles')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Y模式 - Principal Angles
    y_angles_mean = results_y['angles_mean']
    y_angles_std = results_y['angles_std']

    axes[0, 1].bar(pc_labels, y_angles_mean, yerr=y_angles_std,
                   color='coral', alpha=0.8, capsize=3)
    axes[0, 1].set_xlabel('Component')
    axes[0, 1].set_ylabel('Mean Principal Angle (degrees)')
    axes[0, 1].set_title('Y Mode: Mean Principal Angles')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # X模式 - 投影F范数
    axes[1, 0].hist(results_x['proj_norms'], bins=10, color='steelblue',
                    alpha=0.8, edgecolor='black')
    axes[1, 0].set_xlabel('Subspace Projection F-norm')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('X Mode: Subspace Projection Norm Distribution')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Y模式 - 投影F范数
    axes[1, 1].hist(results_y['proj_norms'], bins=10, color='coral',
                    alpha=0.8, edgecolor='black')
    axes[1, 1].set_xlabel('Subspace Projection F-norm')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Y Mode: Subspace Projection Norm Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Grassmann Manifold Analysis Summary', fontsize=14)
    plt.tight_layout()

    output_path = save_dir / 'summary_barchart.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main():
    print("=" * 60)
    print("Grassmann流形分析 - 验证共享基是强迫还是必然存在")
    print("=" * 60)

    # 收集患者列表
    patients = []
    for dataset in ["IMR", "TT"]:
        dataset_path = DATA_ROOT / dataset
        if not dataset_path.exists():
            continue
        for subj_dir in sorted(dataset_path.iterdir()):
            if subj_dir.is_dir():
                patients.append((dataset, subj_dir.name))

    print(f"Total patients found: {len(patients)}")

    # 选择所有有足够窗口的患者（至少3个差分窗口）
    selected_patients = []
    for dataset, subj_id in patients:
        windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
        if windows_x is None:
            continue
        diff_x = compute_diff_matrix(windows_x)
        if diff_x.shape[0] >= 3:  # 至少3个差分窗口
            selected_patients.append((dataset, subj_id))

    print(f"Selected patients with >=3 windows: {len(selected_patients)}")

    # 输出目录
    output_dir = DATA_ROOT / "grassmann_analysis_results"
    output_dir.mkdir(exist_ok=True)

    # 加载所有患者的数据
    print("\n--- 加载患者数据 ---")
    all_diff_x = []
    all_diff_y = []
    patient_ids = []

    for dataset, subj_id in tqdm(selected_patients, desc="Loading"):
        windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
        if windows_x is None:
            continue

        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)

        if diff_x.shape[0] < 3:
            continue

        all_diff_x.append(diff_x)
        all_diff_y.append(diff_y)
        patient_ids.append(f"{dataset}/{subj_id}")

    print(f"Loaded {len(patient_ids)} patients")

    # ========== 联合SVD ==========
    print("\n--- 多患者联合SVD ---")
    Vt_multi_x, sigma_multi_x = run_multi_patient_svd(all_diff_x, N_COMPONENTS)
    Vt_multi_y, sigma_multi_y = run_multi_patient_svd(all_diff_y, N_COMPONENTS)

    print(f"X multi Vt shape: {Vt_multi_x.shape}, singular values: {sigma_multi_x[:3]}")
    print(f"Y multi Vt shape: {Vt_multi_y.shape}, singular values: {sigma_multi_y[:3]}")

    # ========== 单患者SVD + 比较 ==========
    print("\n--- 单患者SVD + Grassmann分析 ---")

    angles_x_list = []
    angles_y_list = []
    proj_norms_x_list = []
    proj_norms_y_list = []
    correlations_x_list = []
    correlations_y_list = []

    for i, (diff_x, diff_y) in enumerate(tqdm(zip(all_diff_x, all_diff_y), total=len(all_diff_x), desc="Analyzing")):
        # 单患者SVD
        _, _, Vt_single_x = run_single_patient_svd(diff_x, N_COMPONENTS)
        _, _, Vt_single_y = run_single_patient_svd(diff_y, N_COMPONENTS)

        # 转换为子空间基
        Q_single_x = Vt_to_subspace(Vt_single_x, N_COMPONENTS)
        Q_single_y = Vt_to_subspace(Vt_single_y, N_COMPONENTS)
        Q_multi_x = Vt_to_subspace(Vt_multi_x, N_COMPONENTS)
        Q_multi_y = Vt_to_subspace(Vt_multi_y, N_COMPONENTS)

        # Principal angles (efficient computation)
        angles_x = compute_principal_angles_efficient(Q_single_x, Q_multi_x)
        angles_y = compute_principal_angles_efficient(Q_single_y, Q_multi_y)

        # 统一长度（填充NaN）
        angles_x_padded = np.full(N_COMPONENTS, np.nan)
        angles_y_padded = np.full(N_COMPONENTS, np.nan)
        n_angles = min(len(angles_x), N_COMPONENTS)
        angles_x_padded[:n_angles] = angles_x[:n_angles]
        angles_y_padded[:n_angles] = angles_y[:n_angles]

        angles_x_list.append(angles_x_padded)
        angles_y_list.append(angles_y_padded)

        # 投影F范数 (efficient computation)
        norm_x = compute_subspace_projection_norm_efficient(Q_single_x, Q_multi_x)
        norm_y = compute_subspace_projection_norm_efficient(Q_single_y, Q_multi_y)
        proj_norms_x_list.append(norm_x)
        proj_norms_y_list.append(norm_y)

    # 转换为数组
    angles_x = np.array(angles_x_list)
    angles_y = np.array(angles_y_list)
    proj_norms_x = np.array(proj_norms_x_list)
    proj_norms_y = np.array(proj_norms_y_list)

    # 计算相关性矩阵
    corr_x = compute_correlation_matrix(
        [run_single_patient_svd(d, N_COMPONENTS)[2] for d in all_diff_x],
        Vt_multi_x
    )
    corr_y = compute_correlation_matrix(
        [run_single_patient_svd(d, N_COMPONENTS)[2] for d in all_diff_y],
        Vt_multi_y
    )

    # ========== 可视化 ==========
    print("\n--- 可视化 ---")

    # 1. Principal Angles热图
    angles_dict = {'X': angles_x, 'Y': angles_y}
    vis_angles = visualize_principal_angles_heatmap(angles_dict, output_dir)
    print(f"Principal angles heatmap: {vis_angles}")

    # 2. 投影F范数
    norm_dict = {'X': proj_norms_x, 'Y': proj_norms_y}
    vis_norms = visualize_projection_norm_heatmap(norm_dict, output_dir)
    print(f"Projection norm: {vis_norms}")

    # 3. 相关性热图
    corr_dict = {'X': corr_x, 'Y': corr_y}
    vis_corr = visualize_correlation_heatmap(corr_dict, output_dir)
    print(f"Correlation heatmap: {vis_corr}")

    # ========== 统计分析结果 ==========
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)

    results = {
        'X': {
            'angles_mean': angles_x.mean(axis=0),
            'angles_std': angles_x.std(axis=0),
            'angles_all': angles_x,
            'proj_norms': proj_norms_x,
            'proj_norms_mean': proj_norms_x.mean(),
            'proj_norms_std': proj_norms_x.std(),
            'correlations': corr_x
        },
        'Y': {
            'angles_mean': angles_y.mean(axis=0),
            'angles_std': angles_y.std(axis=0),
            'angles_all': angles_y,
            'proj_norms': proj_norms_y,
            'proj_norms_mean': proj_norms_y.mean(),
            'proj_norms_std': proj_norms_y.std(),
            'correlations': corr_y
        }
    }

    print("\n--- X模式 ---")
    print(f"  Principal Angles (mean±std):")
    for i in range(min(5, N_COMPONENTS)):
        print(f"    θ{i+1}: {results['X']['angles_mean'][i]:.2f}° ± {results['X']['angles_std'][i]:.2f}°")
    print(f"  Projection F-norm: {results['X']['proj_norms_mean']:.4f} ± {results['X']['proj_norms_std']:.4f}")
    corr_x_mean = np.nanmean(np.abs(results['X']['correlations']))
    print(f"  Mean |correlation|: {corr_x_mean:.4f} (ignoring NaN)")

    print("\n--- Y模式 ---")
    print(f"  Principal Angles (mean±std):")
    for i in range(min(5, N_COMPONENTS)):
        print(f"    θ{i+1}: {results['Y']['angles_mean'][i]:.2f}° ± {results['Y']['angles_std'][i]:.2f}°")
    print(f"  Projection F-norm: {results['Y']['proj_norms_mean']:.4f} ± {results['Y']['proj_norms_std']:.4f}")
    corr_y_mean = np.nanmean(np.abs(results['Y']['correlations']))
    print(f"  Mean |correlation|: {corr_y_mean:.4f} (ignoring NaN)")

    # ========== 结论 ==========
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)

    # Principal angles解释
    # 如果θ接近0°，说明两个子空间几乎相同
    # 如果θ接近90°，说明两个子空间几乎正交

    x_first_angle = results['X']['angles_mean'][0]
    y_first_angle = results['Y']['angles_mean'][0]

    if x_first_angle < 30 and y_first_angle < 30:
        conclusion = "共享基很可能是'必然存在'，而非计算强迫"
    elif x_first_angle < 60 or y_first_angle < 60:
        conclusion = "共享基部分相似，存在一定差异，可能需要进一步验证"
    else:
        conclusion = "共享基差异较大，可能是计算强迫的结果"

    print(f"\nX模式 PC1 主角度: {x_first_angle:.2f}°")
    print(f"Y模式 PC1 主角度: {y_first_angle:.2f}°")
    print(f"\n结论: {conclusion}")

    # 保存结果
    save_results = {
        'patients': patient_ids,
        'n_components': N_COMPONENTS,
        'X_mode': {
            'principal_angles_mean': results['X']['angles_mean'].tolist(),
            'principal_angles_std': results['X']['angles_std'].tolist(),
            'projection_norm_mean': float(results['X']['proj_norms_mean']),
            'projection_norm_std': float(results['X']['proj_norms_std']),
            'mean_correlation': float(np.nanmean(np.abs(results['X']['correlations'])))
        },
        'Y_mode': {
            'principal_angles_mean': results['Y']['angles_mean'].tolist(),
            'principal_angles_std': results['Y']['angles_std'].tolist(),
            'projection_norm_mean': float(results['Y']['proj_norms_mean']),
            'projection_norm_std': float(results['Y']['proj_norms_std']),
            'mean_correlation': float(np.nanmean(np.abs(results['Y']['correlations'])))
        },
        'conclusion': conclusion
    }

    results_file = output_dir / 'grassmann_results.json'
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n结果已保存到: {results_file}")

    return results


if __name__ == "__main__":
    results = main()
