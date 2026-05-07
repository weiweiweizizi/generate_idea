"""
blendshape 与 SVD 时间系数相关性分析

验证思路：
1. 使用 IM R联合SVD 和 TT联合SVD 的前两个主模态（PC1, PC2）
2. 对每个 TT 患者：
   - 将其数据投影到 IMR_joint 和 TT_joint 基上，得到时间系数
   - 加载对应的 blendshape 数据
   - 计算时间系数与 blendshape 变化的相关性
3. 分析哪种基（IMR_joint vs TT_joint）更能捕捉 blendshape 语义
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win20-step20"
BLENDSHAPE_ROOT = REPO_ROOT / "data" / "blendshape"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "pilot_feasibility" / "blendshape" / "win20-step20"
N_COMPONENTS = 4  # 分析前4个主成分
RANDOM_STATE = 42

OUTPUT_DIR = OUTPUT_ROOT / "blendshape_correlation_results"
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


def load_blendshape(subj_id, dataset="TT"):
    """加载blendshape数据"""
    bs_path = BLENDSHAPE_ROOT / dataset / subj_id / "blendshapes.csv"
    if not bs_path.exists():
        return None
    df = pd.read_csv(bs_path)
    return df


def compute_blendshape_diff(blendshape_df):
    """
    计算blendshape在每个窗口的平均值，然后相邻窗口取差分

    1. 每20帧作为一个窗口（与SVD窗口一致）
    2. 计算这20帧内各blendshape的均值
    3. 相邻窗口均值相减：diff[i] = mean[i+1] - mean[i]
    4. 尾部不足20帧的部分忽略

    返回: (n_windows-1) x n_blendshape 的变化量矩阵
    """
    # 获取所有帧的blendshape值
    blend_cols = [c for c in blendshape_df.columns if c != 'frame']
    all_blends = blendshape_df[blend_cols].values  # shape: (n_frames, n_blendshape)
    n_frames = len(all_blends)

    WINDOW_SIZE = 20

    # Step 1: 按固定20帧划分窗口，计算每个窗口的blendshape均值
    n_windows = n_frames // WINDOW_SIZE  # 向下取整，忽略尾部不足20帧的部分
    window_means = []
    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end = (i + 1) * WINDOW_SIZE
        window_blends = all_blends[start:end]
        mean_blend = np.mean(window_blends, axis=0)
        window_means.append(mean_blend)

    window_means = np.array(window_means)  # shape: (n_windows, n_blendshape)

    # Step 2: 相邻窗口取差分（与SVD的差分方式一致）
    diff_windows = []
    for i in range(1, len(window_means)):
        diff = window_means[i] - window_means[i-1]
        diff_windows.append(diff)

    return np.array(diff_windows), blend_cols


def project_to_joint_basis(diff_data, Vt):
    """
    将差分数据投影到联合SVD基上得到时间系数
    diff_data: shape (n_windows, 341, 341)
    Vt: shape (n_components, 341*341)
    返回: shape (n_windows, n_components)
    """
    n_windows = diff_data.shape[0]
    X_flat = diff_data.reshape(n_windows, -1)  # (n_windows, 341*341)
    # 时间系数 = X @ Vt.T
    time_coef = X_flat @ Vt.T
    return time_coef


def compute_correlation_matrix(time_coef, blend_diff, method='pearson'):
    """
    计算时间系数与blendshape变化的相关性矩阵
    time_coef: shape (n_windows, n_components)
    blend_diff: shape (n_windows, n_blendshape)
    返回: (n_components, n_blendshape) 相关矩阵
    """
    n_comp = time_coef.shape[1]
    n_blend = blend_diff.shape[1]
    corr_matrix = np.zeros((n_comp, n_blend))

    for i in range(n_comp):
        for j in range(n_blend):
            tc = time_coef[:, i]
            bd = blend_diff[:, j]
            # 过滤掉nan和常量
            mask = ~(np.isnan(tc) | np.isnan(bd) | (tc == tc[0]) | (bd == bd[0]))
            if mask.sum() > 3:
                if method == 'pearson':
                    corr_matrix[i, j] = pearsonr(tc[mask], bd[mask])[0]
                else:
                    corr_matrix[i, j] = spearmanr(tc[mask], bd[mask])[0]
            else:
                corr_matrix[i, j] = np.nan

    return corr_matrix


def main():
    print("=" * 60)
    print("blendshape 与 SVD 时间系数相关性分析")
    print("=" * 60)

    # ========== 1. 加载并计算联合SVD基 ==========
    print("\n--- 1. 计算联合SVD基 ---")

    imr_patients = []
    tt_patients = []

    for dataset in ["IMR", "TT"]:
        dataset_path = DATA_ROOT / dataset
        if not dataset_path.exists():
            continue
        for subj_dir in sorted(dataset_path.iterdir()):
            if subj_dir.is_dir():
                if dataset == "IMR":
                    imr_patients.append((dataset, subj_dir.name))
                else:
                    tt_patients.append((dataset, subj_dir.name))

    print(f"IMR patients: {len(imr_patients)}, TT patients: {len(tt_patients)}")

    # 收集IMR数据
    imr_diff_x, imr_diff_y = [], []
    imr_windows_per_patient = []
    for dataset, subj_id in tqdm(imr_patients, desc="Loading IMR"):
        windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
        if windows_x is None or windows_x.shape[0] < 3:
            continue
        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)
        imr_diff_x.append(diff_x)
        imr_diff_y.append(diff_y)
        imr_windows_per_patient.append(diff_x.shape[0])

    # 收集TT数据
    tt_diff_x, tt_diff_y = [], []
    tt_original_windows = []  # 原始窗口数（用于blendshape对齐）
    tt_diff_windows = []  # 差分窗口数（用于SVD投影）
    for dataset, subj_id in tqdm(tt_patients, desc="Loading TT"):
        windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
        if windows_x is None or windows_x.shape[0] < 3:
            continue
        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)
        tt_diff_x.append(diff_x)
        tt_diff_y.append(diff_y)
        tt_original_windows.append(windows_x.shape[0])  # 原始窗口数
        tt_diff_windows.append(diff_x.shape[0])  # 差分窗口数

    print(f"Valid IMR: {len(imr_diff_x)} patients, {sum(imr_windows_per_patient)} diff_windows")
    print(f"Valid TT: {len(tt_diff_x)} patients, {sum(tt_diff_windows)} diff_windows")

    # 计算联合SVD
    X_imr = np.vstack([d.reshape(d.shape[0], -1) for d in imr_diff_x])
    X_tt = np.vstack([d.reshape(d.shape[0], -1) for d in tt_diff_x])

    svd_imr = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    U_imr = svd_imr.fit_transform(X_imr)
    Vt_imr = svd_imr.components_
    sigma_imr = svd_imr.singular_values_

    svd_tt = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    U_tt = svd_tt.fit_transform(X_tt)
    Vt_tt = svd_tt.components_
    sigma_tt = svd_tt.singular_values_

    print(f"\nIMR joint SVD: Vt shape {Vt_imr.shape}, sigma: {sigma_imr[:2]}")
    print(f"TT joint SVD: Vt shape {Vt_tt.shape}, sigma: {sigma_tt[:2]}")

    # ========== 2. 对每个TT患者，计算时间系数与blendshape相关性 ==========
    print("\n--- 2. 计算每个患者的相关性 ---")

    # 重建TT患者与窗口的对应关系
    tt_patient_ids = [p[1] for p in tt_patients]

    all_corr_imr_x = []
    all_corr_imr_y = []
    all_corr_tt_x = []
    all_corr_tt_y = []
    valid_patients = []

    for idx, subj_id in enumerate(tt_patient_ids):
        # 加载blendshape
        bs_df = load_blendshape(subj_id, "TT")
        if bs_df is None:
            continue

        # 使用固定20帧窗口计算blendshape变化
        blend_diff, blend_cols = compute_blendshape_diff(bs_df)

        # 获取该患者的数据
        diff_x_patient = tt_diff_x[idx]  # shape (n_diff_win, 341, 341)
        diff_y_patient = tt_diff_y[idx]

        # blendshape diff和svd diff窗口数应该一致
        n_diff_wins = min(diff_x_patient.shape[0], blend_diff.shape[0])
        if n_diff_wins < 3:
            continue

        if n_diff_wins < 3:
            continue

        # 投影到IMR_joint基
        coef_imr_x = project_to_joint_basis(diff_x_patient[:n_diff_wins], Vt_imr)
        coef_imr_y = project_to_joint_basis(diff_y_patient[:n_diff_wins], Vt_imr)

        # 投影到TT_joint基
        coef_tt_x = project_to_joint_basis(diff_x_patient[:n_diff_wins], Vt_tt)
        coef_tt_y = project_to_joint_basis(diff_y_patient[:n_diff_wins], Vt_tt)

        # 计算相关性
        corr_imr_x = compute_correlation_matrix(coef_imr_x[:n_diff_wins], blend_diff[:n_diff_wins])
        corr_imr_y = compute_correlation_matrix(coef_imr_y[:n_diff_wins], blend_diff[:n_diff_wins])
        corr_tt_x = compute_correlation_matrix(coef_tt_x[:n_diff_wins], blend_diff[:n_diff_wins])
        corr_tt_y = compute_correlation_matrix(coef_tt_y[:n_diff_wins], blend_diff[:n_diff_wins])

        all_corr_imr_x.append(corr_imr_x)
        all_corr_imr_y.append(corr_imr_y)
        all_corr_tt_x.append(corr_tt_x)
        all_corr_tt_y.append(corr_tt_y)
        valid_patients.append(subj_id)

    print(f"Valid patients with blendshape: {len(valid_patients)}")

    if len(valid_patients) == 0:
        print("No valid patients found!")
        return

    # 堆叠相关性
    all_corr_imr_x = np.array(all_corr_imr_x)
    all_corr_imr_y = np.array(all_corr_imr_y)
    all_corr_tt_x = np.array(all_corr_tt_x)
    all_corr_tt_y = np.array(all_corr_tt_y)

    # ========== 3. 分析结果 ==========
    print("\n--- 3. 相关性分析结果 ---")

    # 计算平均相关性（取绝对值）和标准差
    mean_abs_corr_imr_x = np.nanmean(np.abs(all_corr_imr_x), axis=0)
    mean_abs_corr_imr_y = np.nanmean(np.abs(all_corr_imr_y), axis=0)
    mean_abs_corr_tt_x = np.nanmean(np.abs(all_corr_tt_x), axis=0)
    mean_abs_corr_tt_y = np.nanmean(np.abs(all_corr_tt_y), axis=0)

    # 计算标准差（对绝对值）
    std_abs_corr_imr_x = np.nanstd(np.abs(all_corr_imr_x), axis=0)
    std_abs_corr_imr_y = np.nanstd(np.abs(all_corr_imr_y), axis=0)
    std_abs_corr_tt_x = np.nanstd(np.abs(all_corr_tt_x), axis=0)
    std_abs_corr_tt_y = np.nanstd(np.abs(all_corr_tt_y), axis=0)

    # 打印每个PC与blendshape的最高相关性
    print("\n=== IMR_joint 基投影 (X mode) ===")
    for pc in range(N_COMPONENTS):
        top_idx = np.argsort(mean_abs_corr_imr_x[pc])[::-1][:5]
        print(f"PC{pc+1}: ", end="")
        for idx in top_idx:
            print(f"{blend_cols[idx]}={mean_abs_corr_imr_x[pc, idx]:.3f} ", end="")
        print()

    print("\n=== IMR_joint 基投影 (Y mode) ===")
    for pc in range(N_COMPONENTS):
        top_idx = np.argsort(mean_abs_corr_imr_y[pc])[::-1][:5]
        print(f"PC{pc+1}: ", end="")
        for idx in top_idx:
            print(f"{blend_cols[idx]}={mean_abs_corr_imr_y[pc, idx]:.3f} ", end="")
        print()

    print("\n=== TT_joint 基投影 (X mode) ===")
    for pc in range(N_COMPONENTS):
        top_idx = np.argsort(mean_abs_corr_tt_x[pc])[::-1][:5]
        print(f"PC{pc+1}: ", end="")
        for idx in top_idx:
            print(f"{blend_cols[idx]}={mean_abs_corr_tt_x[pc, idx]:.3f} ", end="")
        print()

    print("\n=== TT_joint 基投影 (Y mode) ===")
    for pc in range(N_COMPONENTS):
        top_idx = np.argsort(mean_abs_corr_tt_y[pc])[::-1][:5]
        print(f"PC{pc+1}: ", end="")
        for idx in top_idx:
            print(f"{blend_cols[idx]}={mean_abs_corr_tt_y[pc, idx]:.3f} ", end="")
        print()

    # ========== 4. 保存结果 ==========
    results = {
        "n_valid_patients": len(valid_patients),
        "blend_cols": blend_cols,
        "mean_abs_corr": {
            "IMR_joint_X": mean_abs_corr_imr_x.tolist(),
            "IMR_joint_Y": mean_abs_corr_imr_y.tolist(),
            "TT_joint_X": mean_abs_corr_tt_x.tolist(),
            "TT_joint_Y": mean_abs_corr_tt_y.tolist(),
        },
        "std_abs_corr": {
            "IMR_joint_X": std_abs_corr_imr_x.tolist(),
            "IMR_joint_Y": std_abs_corr_imr_y.tolist(),
            "TT_joint_X": std_abs_corr_tt_x.tolist(),
            "TT_joint_Y": std_abs_corr_tt_y.tolist(),
        }
    }

    results_file = OUTPUT_DIR / "blendshape_correlation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {results_file}")

    # ========== 5. 可视化 ==========
    # 热图：每个PC与blendshape的相关性
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    titles = ["IMR_joint X", "IMR_joint Y", "TT_joint X", "TT_joint Y"]
    matrices = [mean_abs_corr_imr_x, mean_abs_corr_imr_y, mean_abs_corr_tt_x, mean_abs_corr_tt_y]

    for ax, title, mat in zip(axes.flat, titles, matrices):
        im = ax.imshow(np.abs(mat), aspect='auto', cmap='seismic', norm=TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1))
        ax.set_title(title)
        ax.set_xlabel('blendshape index')
        ax.set_ylabel('PC')
        ax.set_yticks(range(N_COMPONENTS))
        ax.set_yticklabels([f'PC{i+1}' for i in range(N_COMPONENTS)])
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Mean Absolute Correlation: SVD Time Coef vs Blendshape Diff', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=150)
    plt.close()
    print(f"热图已保存: {OUTPUT_DIR / 'correlation_heatmap.png'}")

    # ========== 5b. 标准差热图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    std_matrices = [std_abs_corr_imr_x, std_abs_corr_imr_y, std_abs_corr_tt_x, std_abs_corr_tt_y]

    for ax, title, mat in zip(axes.flat, titles, std_matrices):
        im = ax.imshow(mat, aspect='auto', cmap='seismic', vmin=0, vmax=0.4)
        ax.set_title(title)
        ax.set_xlabel('blendshape index')
        ax.set_ylabel('PC')
        ax.set_yticks(range(N_COMPONENTS))
        ax.set_yticklabels([f'PC{i+1}' for i in range(N_COMPONENTS)])
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Std of Absolute Correlation: SVD Time Coef vs Blendshape Diff', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_std_heatmap.png', dpi=150)
    plt.close()
    print(f"标准差热图已保存: {OUTPUT_DIR / 'correlation_std_heatmap.png'}")

    # ========== 5c. 箱线图：PC1/PC2最高相关性blendshape的分布 ==========
    # 找出每个mode中PC1和PC2最高相关性的blendshape索引
    top_blendshape_info = []
    all_corr_mats = {
        "IMR_X": all_corr_imr_x,
        "IMR_Y": all_corr_imr_y,
        "TT_X": all_corr_tt_x,
        "TT_Y": all_corr_tt_y,
    }
    mean_mats = {
        "IMR_X": mean_abs_corr_imr_x,
        "IMR_Y": mean_abs_corr_imr_y,
        "TT_X": mean_abs_corr_tt_x,
        "TT_Y": mean_abs_corr_tt_y,
    }

    for mode_name in ["IMR_X", "IMR_Y", "TT_X", "TT_Y"]:
        mat = mean_mats[mode_name]
        for pc in range(2):  # PC1和PC2
            top_idx = np.argsort(mat[pc])[::-1][0]
            top_blendshape_info.append({
                "mode": mode_name,
                "pc": pc,
                "blendshape_idx": top_idx,
                "blendshape_name": blend_cols[top_idx]
            })

    # 准备箱线图数据
    boxplot_data = []
    boxplot_labels = []
    boxplot_colors = []

    colors_map = {
        "IMR_X": "#3498db",
        "IMR_Y": "#ffcc00",
        "TT_X": "#2ecc71",
        "TT_Y": "#9b59b6"
    }

    for info in top_blendshape_info:
        mode_name = info["mode"]
        blendshape_idx = info["blendshape_idx"]
        corr_mat = all_corr_mats[mode_name]
        # 提取每个患者在该blendshape上的相关系数
        patient_corrs = corr_mat[:, info["pc"], blendshape_idx]
        # 过滤NaN
        patient_corrs = patient_corrs[~np.isnan(patient_corrs)]
        boxplot_data.append(np.abs(patient_corrs))
        boxplot_labels.append(f"{mode_name}\nPC{info['pc']+1}\n{info['blendshape_name']}")
        boxplot_colors.append(colors_map[mode_name])

    # 画箱线图
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], boxplot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('|Pearson r|', fontsize=12)
    ax.set_title('Distribution of Correlation Coefficients (PC1 & PC2 Top Blendshapes)', fontsize=14)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='r=0.5 threshold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", alpha=0.6, label='IMR_X'),
        Patch(facecolor="#ffcc00", alpha=0.6, label='IMR_Y'),
        Patch(facecolor="#2ecc71", alpha=0.6, label='TT_X'),
        Patch(facecolor="#9b59b6", alpha=0.6, label='TT_Y'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_boxplot.png', dpi=150)
    plt.close()
    print(f"箱线图已保存: {OUTPUT_DIR / 'correlation_boxplot.png'}")

    # ========== 6. 汇总统计 ==========
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)

    # 每个PC的最高相关性blendshape（带标准差）
    summary = []
    std_mats = [("IMR_X", mean_abs_corr_imr_x, std_abs_corr_imr_x),
                ("IMR_Y", mean_abs_corr_imr_y, std_abs_corr_imr_y),
                ("TT_X", mean_abs_corr_tt_x, std_abs_corr_tt_x),
                ("TT_Y", mean_abs_corr_tt_y, std_abs_corr_tt_y)]
    for mode_name, mat, std_mat in std_mats:
        for pc in range(N_COMPONENTS):
            top_idx = np.argsort(mat[pc])[::-1][0]
            top_corr = mat[pc, top_idx]
            top_std = std_mat[pc, top_idx]
            summary.append({
                "mode": mode_name,
                "pc": f"PC{pc+1}",
                "top_blendshape": blend_cols[top_idx],
                "mean_abs_corr": float(top_corr),
                "std_abs_corr": float(top_std)
            })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    return results


if __name__ == "__main__":
    main()
