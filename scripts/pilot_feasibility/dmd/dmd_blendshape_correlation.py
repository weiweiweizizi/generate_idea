"""
DMD模态 与 blendshape 相关性分析

使用TT联合DMD保存的主模态作为基
对所有患者（IMR和TT）计算时间系数（类似SVD的直接投影方式）
然后计算与blendshape的相关性

注意：blendshape现在以5帧为一个窗口，每两个窗口计算一次差分
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win5-step5"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "pilot_feasibility" / "dmd" / "win5-step5"
DMD_MODES_DIR = OUTPUT_ROOT / "dmd_multi_patient_results" / "saved_modes" / "TT"
BLENDSHAPE_ROOT = REPO_ROOT / "data" / "blendshape"
N_MODES = 5  # 使用前5个DMD模态

OUTPUT_DIR = OUTPUT_ROOT / "dmd_blendshape_correlation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_patient_windows(dataset_path, subj_id):
    """加载单个患者所有窗口的x和y矩阵"""
    subj_path = dataset_path / subj_id
    if not subj_path.exists():
        return None, None

    win_x_files = sorted(subj_path.glob("win_*_x.npy"))
    win_y_files = sorted(subj_path.glob("win_*_y.npy"))

    if len(win_x_files) == 0:
        return None, None

    windows_x = [np.load(f).astype(np.float32) for f in win_x_files]
    windows_y = [np.load(f).astype(np.float32) for f in win_y_files]

    return np.stack(windows_x, axis=0), np.stack(windows_y, axis=0)


def compute_diff_matrix(windows):
    """计算前后差分矩阵"""
    diffs = []
    for i in range(1, len(windows)):
        diff = windows[i] - windows[i-1]
        diffs.append(diff)
    return np.stack(diffs, axis=0)


def load_dmd_modes():
    """加载TT联合DMD保存的模态"""
    modes_x = np.load(DMD_MODES_DIR / "modes_x.npy")
    modes_y = np.load(DMD_MODES_DIR / "modes_y.npy")
    eigenvalues_x = np.load(DMD_MODES_DIR / "eigenvalues_x.npy")
    eigenvalues_y = np.load(DMD_MODES_DIR / "eigenvalues_y.npy")
    return modes_x, modes_y, eigenvalues_x, eigenvalues_y


def load_blendshape(subj_id, dataset="TT"):
    """加载blendshape数据"""
    bs_path = BLENDSHAPE_ROOT / dataset / subj_id / "blendshapes.csv"
    if not bs_path.exists():
        return None
    df = pd.read_csv(bs_path)
    return df


def compute_blendshape_diff_5frames(blendshape_df):
    """
    计算blendshape在每个5帧窗口的平均值，然后相邻窗口取差分

    1. 每5帧作为一个窗口（与DMD窗口一致）
    2. 计算这5帧内各blendshape的均值
    3. 相邻窗口均值相减：diff[i] = mean[i+1] - mean[i]
    4. 尾部不足5帧的部分忽略

    返回: (n_windows-1) x n_blendshape 的变化量矩阵
    """
    blend_cols = [c for c in blendshape_df.columns if c != 'frame']
    all_blends = blendshape_df[blend_cols].values
    n_frames = len(all_blends)

    WINDOW_SIZE = 5

    n_windows = n_frames // WINDOW_SIZE
    window_means = []
    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end = (i + 1) * WINDOW_SIZE
        window_blends = all_blends[start:end]
        mean_blend = np.mean(window_blends, axis=0)
        window_means.append(mean_blend)

    window_means = np.array(window_means)

    # 相邻窗口取差分
    diff_windows = []
    for i in range(1, len(window_means)):
        diff = window_means[i] - window_means[i-1]
        diff_windows.append(diff)

    return np.array(diff_windows), blend_cols


def compute_blendshape_window_5frames(blendshape_df):
    """
    计算blendshape在每个5帧窗口的平均值（不做差分）

    对齐方式：delta(1)=win(2)-win(1) 对应 blendshape(2)

    1. 每5帧作为一个窗口
    2. 计算这5帧内各blendshape的均值
    3. 返回: n_windows x n_blendshape 的窗口均值矩阵
    """
    blend_cols = [c for c in blendshape_df.columns if c != 'frame']
    all_blends = blendshape_df[blend_cols].values
    n_frames = len(all_blends)

    WINDOW_SIZE = 5

    n_windows = n_frames // WINDOW_SIZE
    window_means = []
    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end = (i + 1) * WINDOW_SIZE
        window_blends = all_blends[start:end]
        mean_blend = np.mean(window_blends, axis=0)
        window_means.append(mean_blend)

    return np.array(window_means), blend_cols


def project_to_dmd_basis(diff_data, modes):
    """
    将差分数据投影到DMD模态上得到时间系数
    类似SVD的展开方式：coef = X @ modes.T

    diff_data: shape (n_windows, 341, 341)
    modes: shape (341*341, n_modes)
    返回: shape (n_windows, n_modes)
    """
    n_windows = diff_data.shape[0]
    X_flat = diff_data.reshape(n_windows, -1)  # (n_windows, 341*341)
    # 时间系数 = X @ modes.T（类似SVD的投影）
    time_coef = X_flat @ modes
    return time_coef


def compute_correlation_matrix(time_coef, blend_diff, method='pearson'):
    """计算时间系数与blendshape变化的相关性矩阵"""
    n_comp = time_coef.shape[1]
    n_blend = blend_diff.shape[1]
    corr_matrix = np.zeros((n_comp, n_blend))

    for i in range(n_comp):
        for j in range(n_blend):
            tc = time_coef[:, i]
            bd = blend_diff[:, j]
            mask = ~(np.isnan(tc) | np.isnan(bd) | (tc == tc[0]) | (bd == bd[0]))
            if mask.sum() > 3:
                if method == 'pearson':
                    corr_matrix[i, j] = pearsonr(tc[mask], bd[mask])[0]
                else:
                    corr_matrix[i, j] = spearmanr(tc[mask], bd[mask])[0]
            else:
                corr_matrix[i, j] = np.nan

    return corr_matrix


def main(use_blend_diff=True):
    print("=" * 60)
    mode_name = "差分模式" if use_blend_diff else "原始值模式"
    print(f"DMD模态 与 Blendshape 相关性分析 ({mode_name})")
    print("=" * 60)

    # ========== 1. 加载DMD模态 ==========
    print("\n--- 1. 加载TT联合DMD模态 ---")
    modes_x, modes_y, eig_x, eig_y = load_dmd_modes()
    print(f"模态形状: X={modes_x.shape}, Y={modes_y.shape}")
    print(f"特征值: X={eig_x[:3]}, Y={eig_y[:3]}")
    print(f"使用前{N_MODES}个模态")

    modes_x = modes_x[:, :N_MODES]
    modes_y = modes_y[:, :N_MODES]

    # ========== 2. 收集所有患者数据 ==========
    print("\n--- 2. 加载患者数据 ---")

    all_patients = []
    for dataset in ["IMR", "TT"]:
        dataset_path = DATA_ROOT / dataset
        if not dataset_path.exists():
            continue
        for subj_dir in sorted(dataset_path.iterdir()):
            if subj_dir.is_dir():
                all_patients.append((dataset, subj_dir.name))

    print(f"总患者数: {len(all_patients)}")

    # 收集所有患者数据
    all_diff_x = []
    all_diff_y = []
    all_blend_diffs = []
    patient_info = []

    valid_count = 0
    for dataset, subj_id in tqdm(all_patients, desc="Processing patients"):
        # 加载窗口数据
        windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
        if windows_x is None or windows_x.shape[0] < 3:
            continue

        # 计算差分
        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)

        # 加载blendshape（只有TT有）
        blend_data = None
        if dataset == "TT":
            bs_df = load_blendshape(subj_id, "TT")
            if bs_df is not None:
                if use_blend_diff:
                    blend_data, blend_cols = compute_blendshape_diff_5frames(bs_df)
                    # 对齐窗口数
                    n_common = min(diff_x.shape[0], blend_data.shape[0])
                else:
                    blend_data, blend_cols = compute_blendshape_window_5frames(bs_df)
                    # 对齐窗口数：diff是n_windows-1，blend是n_windows
                    n_common = min(diff_x.shape[0], blend_data.shape[0] - 1)
                if n_common >= 3:
                    if use_blend_diff:
                        all_diff_x.append(diff_x[:n_common])
                        all_diff_y.append(diff_y[:n_common])
                        all_blend_diffs.append(blend_data[:n_common])
                    else:
                        # 不做差分时：diff(1)对齐blend(2)，diff(2)对齐blend(3)，...
                        # 即diff[i]对齐blend[i+1]
                        all_diff_x.append(diff_x[:n_common])
                        all_diff_y.append(diff_y[:n_common])
                        all_blend_diffs.append(blend_data[1:n_common+1])
                    patient_info.append({
                        "dataset": dataset,
                        "subj_id": subj_id,
                        "n_windows": n_common
                    })
                    valid_count += 1
        else:
            # IMR没有blendshape，只收集运动数据
            if diff_x.shape[0] >= 3:
                all_diff_x.append(diff_x)
                all_diff_y.append(diff_y)
                all_blend_diffs.append(None)
                patient_info.append({
                    "dataset": dataset,
                    "subj_id": subj_id,
                    "n_windows": diff_x.shape[0]
                })
                valid_count += 1

    print(f"有效患者数: {valid_count}")

    # ========== 3. 对所有患者计算时间系数 ==========
    print("\n--- 3. 计算DMD时间系数 ---")

    all_coefs_x = []
    all_coefs_y = []

    for i, info in enumerate(tqdm(patient_info, desc="Projecting")):
        diff_x = all_diff_x[i]
        diff_y = all_diff_y[i]

        # 投影到DMD模态
        coef_x = project_to_dmd_basis(diff_x, modes_x)
        coef_y = project_to_dmd_basis(diff_y, modes_y)

        all_coefs_x.append(coef_x)
        all_coefs_y.append(coef_y)

    # ========== 4. 计算与blendshape的相关性（仅TT患者） ==========
    print("\n--- 4. 计算相关性 ---")

    tt_indices = [i for i, info in enumerate(patient_info) if info["dataset"] == "TT"]
    print(f"TT患者数（有blendshape）: {len(tt_indices)}")

    all_corr_x = []
    all_corr_y = []

    for idx in tt_indices:
        blend_diff = all_blend_diffs[idx]
        coef_x = all_coefs_x[idx]
        coef_y = all_coefs_y[idx]

        n_wins = min(coef_x.shape[0], blend_diff.shape[0])
        if n_wins < 3:
            continue

        # 计算相关性
        corr_x = compute_correlation_matrix(coef_x[:n_wins], blend_diff[:n_wins])
        corr_y = compute_correlation_matrix(coef_y[:n_wins], blend_diff[:n_wins])

        all_corr_x.append(corr_x)
        all_corr_y.append(corr_y)

    all_corr_x = np.array(all_corr_x)
    all_corr_y = np.array(all_corr_y)
    print(f"相关性矩阵形状: X={all_corr_x.shape}, Y={all_corr_y.shape}")

    # ========== 5. 分析结果 ==========
    print("\n--- 5. 相关性分析结果 ---")

    # 平均绝对相关性
    mean_abs_corr_x = np.nanmean(np.abs(all_corr_x), axis=0)
    mean_abs_corr_y = np.nanmean(np.abs(all_corr_y), axis=0)

    # 标准差
    std_abs_corr_x = np.nanstd(np.abs(all_corr_x), axis=0)
    std_abs_corr_y = np.nanstd(np.abs(all_corr_y), axis=0)

    # 打印每个模态与blendshape的最高相关性
    print("\n=== DMD X模态 ===")
    for mode in range(N_MODES):
        top_idx = np.argsort(mean_abs_corr_x[mode])[::-1][:5]
        print(f"Mode{mode+1}: ", end="")
        for idx in top_idx:
            print(f"{blend_cols[idx]}={mean_abs_corr_x[mode, idx]:.3f} ", end="")
        print()

    print("\n=== DMD Y模态 ===")
    for mode in range(N_MODES):
        top_idx = np.argsort(mean_abs_corr_y[mode])[::-1][:5]
        print(f"Mode{mode+1}: ", end="")
        for idx in top_idx:
            print(f"{blend_cols[idx]}={mean_abs_corr_y[mode, idx]:.3f} ", end="")
        print()

    # ========== 6. 保存结果 ==========
    results = {
        "n_valid_patients": len(tt_indices),
        "blend_cols": blend_cols,
        "eigenvalues_x": [str(e) for e in eig_x[:N_MODES]],
        "eigenvalues_y": [str(e) for e in eig_y[:N_MODES]],
        "mean_abs_corr": {
            "X": mean_abs_corr_x.tolist(),
            "Y": mean_abs_corr_y.tolist(),
        },
        "std_abs_corr": {
            "X": std_abs_corr_x.tolist(),
            "Y": std_abs_corr_y.tolist(),
        }
    }

    # 根据模式选择输出目录
    if use_blend_diff:
        sub_dir = OUTPUT_DIR / "with_diff"
    else:
        sub_dir = OUTPUT_DIR / "without_diff"
    sub_dir.mkdir(parents=True, exist_ok=True)

    results_file = sub_dir / "dmd_blendshape_correlation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {results_file}")

    # ========== 7. 可视化 ==========
    # 热图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    titles = ["DMD X Mode", "DMD Y Mode"]
    matrices = [mean_abs_corr_x, mean_abs_corr_y]

    for ax, title, mat in zip(axes, titles, matrices):
        im = ax.imshow(np.abs(mat), aspect='auto', cmap='seismic',
                       norm=TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1))
        ax.set_title(title)
        ax.set_xlabel('blendshape index')
        ax.set_ylabel('DMD Mode')
        ax.set_yticks(range(N_MODES))
        ax.set_yticklabels([f'Mode{i+1}' for i in range(N_MODES)])
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Mean Absolute Correlation: DMD Time Coef vs Blendshape Diff (5-frame window)', fontsize=14)
    plt.tight_layout()
    plt.savefig(sub_dir / 'dmd_correlation_heatmap.png', dpi=150)
    plt.close()
    print(f"热图已保存: {sub_dir / 'dmd_correlation_heatmap.png'}")

    # 标准差热图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, title, mat in zip(axes, titles, [std_abs_corr_x, std_abs_corr_y]):
        im = ax.imshow(mat, aspect='auto', cmap='seismic', vmin=0, vmax=0.4)
        ax.set_title(title)
        ax.set_xlabel('blendshape index')
        ax.set_ylabel('DMD Mode')
        ax.set_yticks(range(N_MODES))
        ax.set_yticklabels([f'Mode{i+1}' for i in range(N_MODES)])
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Std of Absolute Correlation: DMD Time Coef vs Blendshape Diff', fontsize=14)
    plt.tight_layout()
    plt.savefig(sub_dir / 'dmd_correlation_std_heatmap.png', dpi=150)
    plt.close()
    print(f"标准差热图已保存: {sub_dir / 'dmd_correlation_std_heatmap.png'}")

    # 箱线图
    boxplot_data = []
    boxplot_labels = []
    boxplot_colors = []

    colors_map = {"X": "#3498db", "Y": "#2ecc71"}

    for mode in range(min(3, N_MODES)):  # 取前3个模态
        for mat_name, mat, color in [("X", all_corr_x, "#3498db"), ("Y", all_corr_y, "#2ecc71")]:
            top_idx = np.argsort(np.nanmean(np.abs(mat), axis=0)[mode])[::-1][0]
            patient_corrs = mat[:, mode, top_idx]
            patient_corrs = patient_corrs[~np.isnan(patient_corrs)]
            boxplot_data.append(np.abs(patient_corrs))
            boxplot_labels.append(f"DMD_{mat_name}\nMode{mode+1}\n{blend_cols[top_idx]}")
            boxplot_colors.append(color)

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], boxplot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('|Pearson r|', fontsize=12)
    ax.set_title('Distribution of Correlation Coefficients (Top Blendshapes per Mode)', fontsize=14)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='r=0.5')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", alpha=0.6, label='DMD_X'),
        Patch(facecolor="#2ecc71", alpha=0.6, label='DMD_Y'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(sub_dir / 'dmd_correlation_boxplot.png', dpi=150)
    plt.close()
    print(f"箱线图已保存: {sub_dir / 'dmd_correlation_boxplot.png'}")

    # ========== 8. 汇总统计 ==========
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)

    summary = []
    for mat_name, mat, std_mat in [("X", mean_abs_corr_x, std_abs_corr_x), ("Y", mean_abs_corr_y, std_abs_corr_y)]:
        for mode in range(N_MODES):
            top_idx = np.argsort(mat[mode])[::-1][0]
            top_corr = mat[mode, top_idx]
            top_std = std_mat[mode, top_idx]
            summary.append({
                "mode": f"DMD_{mat_name}",
                "dmd_mode": f"Mode{mode+1}",
                "top_blendshape": blend_cols[top_idx],
                "mean_abs_corr": float(top_corr),
                "std_abs_corr": float(top_std),
                "eigenvalue": str(eig_x[mode] if mat_name == "X" else eig_y[mode])
            })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    return results


if __name__ == "__main__":
    # 先运行原始值模式（不对blendshape做差分）
    print("\n" + "="*60)
    print("模式1: 不对blendshape做差分")
    print("="*60)
    results_no_diff = main(use_blend_diff=False)

    print("\n" + "="*60)
    print("模式2: 对blendshape做差分 (对比)")
    print("="*60)
    results_diff = main(use_blend_diff=True)
