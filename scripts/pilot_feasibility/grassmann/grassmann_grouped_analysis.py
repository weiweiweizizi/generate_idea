"""
Grassmann-like Analysis: Patient PC1 vs Group PC1

对比每个患者的单患者SVD PC1基与各分组模式下组级PC1基的主角度和相关性。

患者按分组类型排序（同一类型排在一起），以分组类型为列，
热图值为该患者PC1与对应组PC1的主角度或相关系数。
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win20-step20"
GROUPED_RESULTS_DIR = REPO_ROOT / "outputs" / "pilot_feasibility" / "svd" / "win20-step20" / "svd_multi_patient_grouped_results"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "pilot_feasibility" / "grassmann" / "win20-step20"
RANDOM_STATE = 42


def compute_principal_angle(V1, V2):
    """
    计算两个1-D子空间（PC1基）之间的主角度。
    V1, V2: shape (341, 341) or flattened (341*341,)
    返回: 角度（度）
    """
    v1 = V1.flatten()
    v2 = V2.flatten()

    # cos(theta) = |v1·v2| / (||v1|| * ||v2||)
    cos_theta = np.abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, 0, 1)
    return np.degrees(np.arccos(cos_theta))


def compute_correlation(V1, V2):
    """计算两个PC1向量之间的Pearson相关系数"""
    v1 = V1.flatten()
    v2 = V2.flatten()
    r, _ = pearsonr(v1, v2)
    return r


def load_patient_pc1(dataset, subj, base_dir):
    """加载单个患者的PC1基"""
    # 患者PC1存储在 {dataset}-SVD/{subj} 目录下
    pc1_dir = base_dir / f"{dataset}-SVD" / subj
    pc1_x_path = pc1_dir / "PC1_x.npy"
    pc1_y_path = pc1_dir / "PC1_y.npy"

    if not pc1_x_path.exists() or not pc1_y_path.exists():
        return None, None

    pc1_x = np.load(pc1_x_path)
    pc1_y = np.load(pc1_y_path)
    return pc1_x, pc1_y


def load_group_pc1(grouped_results_dir, mode, group_name):
    """加载某分组模式下特定组的PC1基"""
    pc1_x_path = grouped_results_dir / mode / group_name / "PC1_x.npy"
    pc1_y_path = grouped_results_dir / mode / group_name / "PC1_y.npy"

    if not pc1_x_path.exists() or not pc1_y_path.exists():
        return None, None

    pc1_x = np.load(pc1_x_path)
    pc1_y = np.load(pc1_y_path)
    return pc1_x, pc1_y


def visualize_grouped_heatmap(values_matrix, group_names, mode_name, mode_dir,
                             metric_type, cmap='RdYlBu_r', vmin=None, vmax=None):
    """
    绘制分组热图：
    - 行：患者（按组类型排序，无标签）
    - 列：组类型
    - 值：主角度或相关系数
    - 添加组边界线
    """
    n_patients, n_groups = values_matrix.shape

    fig, ax = plt.subplots(figsize=(8, max(10, n_patients * 0.15)))

    # 默认范围
    if vmin is None:
        vmin = values_matrix[~np.isnan(values_matrix)].min()
    if vmax is None:
        vmax = values_matrix[~np.isnan(values_matrix)].max()

    im = ax.imshow(values_matrix, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax)

    ax.set_xlabel('Group Type')
    ax.set_ylabel('Patient (sorted by group)')
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    ax.set_yticks([])  # 不显示患者索引

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label(metric_type)

    # 获取患者所属组的索引，用于绘制水平边界线
    # 我们需要知道每个患者属于哪个组，以便绘制水平分隔线
    return fig, ax


def plot_heatmap_with_boundaries(patient_group_indices, values_matrix, group_names,
                                 mode_name, mode_dir, metric_type, cmap, vmin, vmax):
    """
    绘制带边界线的热图
    patient_group_indices: list of group index for each patient (in sorted order)
    """
    n_patients, n_groups = values_matrix.shape

    fig, ax = plt.subplots(figsize=(6, max(6, n_patients * 0.006)))

    im = ax.imshow(values_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xlabel('Group Type')
    ax.set_ylabel('Patient')
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    ax.set_yticks([])

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label(metric_type)

    # 绘制水平分隔线（不同组之间）
    current_group = patient_group_indices[0]
    for i, gidx in enumerate(patient_group_indices[1:], 1):
        if gidx != current_group:
            ax.axhline(y=i - 0.5, color='black', linewidth=2, linestyle='--')
            current_group = gidx

    # 标题
    title = f"{mode_name} - {metric_type}"
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    save_path = mode_dir / f"{metric_type.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path


def main():
    print("=" * 60)
    print("Grassmann-like Analysis: Patient PC1 vs Group PC1")
    print("=" * 60)

    # 加载分组结果
    grouped_results_path = DATA_ROOT / "svd_multi_patient_grouped_results" / "all_grouped_results.json"
    with open(grouped_results_path, 'r') as f:
        all_grouped_results = json.load(f)

    # PC1基目录
    pc1_base_dir = DATA_ROOT
    grouped_results_dir = GROUPED_RESULTS_DIR

    # 输出目录
    output_dir = OUTPUT_ROOT / "grassmann_grouped_results"
    output_dir.mkdir(exist_ok=True)

    # 三种分组模式
    grouping_modes = ['by_side', 'by_severity', 'by_source']

    all_results = {}

    for mode in grouping_modes:
        print(f"\n{'='*60}")
        print(f"分组模式: {mode}")
        print("=" * 60)

        mode_dir = output_dir / mode
        mode_dir.mkdir(exist_ok=True)

        # 获取该模式下的所有组
        groups = all_grouped_results[mode]
        group_names = list(groups.keys())
        print(f"Groups: {group_names}")

        # 加载各组的PC1基
        group_pc1 = {}
        for group_name in group_names:
            pc1_x, pc1_y = load_group_pc1(grouped_results_dir, mode, group_name)
            if pc1_x is not None:
                group_pc1[group_name] = {'x': pc1_x, 'y': pc1_y}
                print(f"  Loaded PC1 for {group_name}")
            else:
                print(f"  WARNING: No PC1 found for {group_name}")

        if len(group_pc1) == 0:
            print(f"  No group PC1 data found, skipping")
            continue

        # 收集所有患者的信息，按组排序
        patients_sorted = []  # [(dataset, subj, group_name), ...]
        for group_name, group_data in groups.items():
            for patient_info in group_data['patient_info']:
                patients_sorted.append({
                    'dataset': patient_info['dataset'],
                    'subj': patient_info['subj'],
                    'group': group_name
                })

        print(f"Total patients: {len(patients_sorted)}")

        # 计算每个患者与每个组的PC1主角度和相关系数
        n_patients = len(patients_sorted)
        n_groups = len(group_names)
        group_name_to_idx = {g: i for i, g in enumerate(group_names)}

        angle_x = np.full((n_patients, n_groups), np.nan)
        angle_y = np.full((n_patients, n_groups), np.nan)
        corr_x = np.full((n_patients, n_groups), np.nan)
        corr_y = np.full((n_patients, n_groups), np.nan)

        patient_group_indices = []  # 记录每个患者属于哪个组（用于画边界线）

        for p_idx, patient in enumerate(tqdm(patients_sorted, desc=f"Computing {mode}")):
            dataset = patient['dataset']
            subj = patient['subj']
            patient_group = patient['group']
            patient_group_indices.append(group_name_to_idx[patient_group])

            # 加载患者PC1
            pc1_x, pc1_y = load_patient_pc1(dataset, subj, pc1_base_dir)
            if pc1_x is None:
                continue

            # 与每个组计算主角度和相关性
            for g_idx, group_name in enumerate(group_names):
                if group_name not in group_pc1:
                    continue

                group_pc1_x = group_pc1[group_name]['x']
                group_pc1_y = group_pc1[group_name]['y']

                angle_x[p_idx, g_idx] = compute_principal_angle(pc1_x, group_pc1_x)
                angle_y[p_idx, g_idx] = compute_principal_angle(pc1_y, group_pc1_y)
                corr_x[p_idx, g_idx] = compute_correlation(pc1_x, group_pc1_x)
                corr_y[p_idx, g_idx] = compute_correlation(pc1_y, group_pc1_y)

        # 绘制热图
        print(f"\n--- 可视化 ---")

        # X模态 - 主角度
        vmin, vmax = 0, 90
        plot_heatmap_with_boundaries(
            patient_group_indices, angle_x, group_names,
            f"{mode} X", mode_dir, "X_principal_angle", 'RdYlBu_r', vmin, vmax
        )
        print(f"  Saved X_principal_angle.png")

        # Y模态 - 主角度
        plot_heatmap_with_boundaries(
            patient_group_indices, angle_y, group_names,
            f"{mode} Y", mode_dir, "Y_principal_angle", 'RdYlBu_r', vmin, vmax
        )
        print(f"  Saved Y_principal_angle.png")

        # X模态 - 相关性
        vmin, vmax = -1, 1
        plot_heatmap_with_boundaries(
            patient_group_indices, corr_x, group_names,
            f"{mode} X", mode_dir, "X_correlation", 'RdBu_r', vmin, vmax
        )
        print(f"  Saved X_correlation.png")

        # Y模态 - 相关性
        plot_heatmap_with_boundaries(
            patient_group_indices, corr_y, group_names,
            f"{mode} Y", mode_dir, "Y_correlation", 'RdBu_r', vmin, vmax
        )
        print(f"  Saved Y_correlation.png")

        # 保存结果
        mode_results = {
            'n_patients': n_patients,
            'group_names': group_names,
            'patient_group_indices': patient_group_indices,
            'angle_x': angle_x.tolist(),
            'angle_y': angle_y.tolist(),
            'corr_x': corr_x.tolist(),
            'corr_y': corr_y.tolist(),
            'summary': {
                'angle_x_mean_by_group': {g: float(np.nanmean(angle_x[:, i])) for i, g in enumerate(group_names)},
                'angle_y_mean_by_group': {g: float(np.nanmean(angle_y[:, i])) for i, g in enumerate(group_names)},
                'corr_x_mean_by_group': {g: float(np.nanmean(corr_x[:, i])) for i, g in enumerate(group_names)},
                'corr_y_mean_by_group': {g: float(np.nanmean(corr_y[:, i])) for i, g in enumerate(group_names)},
            }
        }

        with open(mode_dir / "results.json", 'w') as f:
            json.dump(mode_results, f, indent=2)

        all_results[mode] = mode_results

    # 保存汇总结果
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印汇总统计
    print(f"\n{'='*60}")
    print("汇总统计")
    print("=" * 60)

    for mode, results in all_results.items():
        print(f"\n[{mode}]")
        summary = results['summary']
        print("  X模态 - 平均主角度(对角线):")
        for g, val in summary['angle_x_mean_by_group'].items():
            print(f"    {g}: {val:.2f}°")
        print("  X模态 - 平均相关性(对角线):")
        for g, val in summary['corr_x_mean_by_group'].items():
            print(f"    {g}: {val:.3f}")

    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
