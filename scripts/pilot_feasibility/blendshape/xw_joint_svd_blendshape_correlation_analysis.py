"""
XW 联合 SVD 主模态与 blendshape 时间变化相关性分析

分析思路：
1. 使用 XW 数据集做联合 SVD，取前 4 个主模态作为参考基
2. 对有 blendshape 标注的 TT 患者：
   - 计算窗口差分矩阵
   - 投影到 XW_joint 的 X / Y 基上，得到时间系数
   - 计算时间系数与 blendshape 窗口差分的相关性
3. 将 TT 患者汇总成一个总体结果做可视化
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win20-step20"
BLENDSHAPE_ROOT = REPO_ROOT / "data" / "blendshape"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "pilot_feasibility" / "blendshape" / "win20-step20"
N_COMPONENTS = 4
RANDOM_STATE = 42
SOURCE_DATASETS = ["TT"]
REFERENCE_DATASET = "XW"
WINDOW_SIZE = 20

OUTPUT_DIR = OUTPUT_ROOT / "xw_joint_svd_blendshape_correlation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_patient_windows(dataset_path, subj_id):
    """加载单个患者所有窗口的 x 和 y 矩阵。"""
    subj_path = dataset_path / subj_id
    if not subj_path.exists():
        return None, None

    win_x_files = sorted(subj_path.glob("win_*_x.npy"))
    win_y_files = sorted(subj_path.glob("win_*_y.npy"))
    if len(win_x_files) == 0 or len(win_y_files) == 0:
        return None, None

    windows_x = [np.load(f) for f in win_x_files]
    windows_y = [np.load(f) for f in win_y_files]
    return np.array(windows_x), np.array(windows_y)


def compute_diff_matrix(windows):
    """计算前后差分矩阵。"""
    return np.array([windows[i] - windows[i - 1] for i in range(1, len(windows))])


def list_dataset_patients(dataset):
    """列出数据集下全部患者目录。"""
    dataset_path = DATA_ROOT / dataset
    if not dataset_path.exists():
        return []
    return [(dataset, subj_dir.name) for subj_dir in sorted(dataset_path.iterdir()) if subj_dir.is_dir()]


def load_blendshape(subj_id, dataset):
    """加载 blendshape 数据。"""
    bs_path = BLENDSHAPE_ROOT / dataset / subj_id / "blendshapes.csv"
    if not bs_path.exists():
        return None
    return pd.read_csv(bs_path)


def compute_blendshape_diff(blendshape_df):
    """
    以固定 20 帧窗口计算 blendshape 均值，再做相邻窗口差分。
    返回: (n_windows-1, n_blendshape), blend_cols
    """
    blend_cols = [c for c in blendshape_df.columns if c != "frame"]
    all_blends = blendshape_df[blend_cols].values
    n_frames = len(all_blends)

    n_windows = n_frames // WINDOW_SIZE
    window_means = []
    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end = (i + 1) * WINDOW_SIZE
        window_means.append(np.mean(all_blends[start:end], axis=0))

    window_means = np.array(window_means)
    diff_windows = np.array(
        [window_means[i] - window_means[i - 1] for i in range(1, len(window_means))]
    )
    return diff_windows, blend_cols


def project_to_joint_basis(diff_data, Vt):
    """将差分数据投影到联合 SVD 基上得到时间系数。"""
    n_windows = diff_data.shape[0]
    x_flat = diff_data.reshape(n_windows, -1)
    return x_flat @ Vt.T


def compute_correlation_matrix(time_coef, blend_diff):
    """计算时间系数与 blendshape 变化的相关性矩阵。"""
    n_comp = time_coef.shape[1]
    n_blend = blend_diff.shape[1]
    corr_matrix = np.full((n_comp, n_blend), np.nan)

    for i in range(n_comp):
        for j in range(n_blend):
            tc = time_coef[:, i]
            bd = blend_diff[:, j]
            mask = ~(np.isnan(tc) | np.isnan(bd))
            tc = tc[mask]
            bd = bd[mask]
            if len(tc) <= 3:
                continue
            if np.allclose(tc, tc[0]) or np.allclose(bd, bd[0]):
                continue
            corr_matrix[i, j] = pearsonr(tc, bd)[0]

    return corr_matrix


def compute_joint_svd_for_dataset(dataset):
    """对参考数据集做联合 SVD，返回 X/Y 模态基。"""
    dataset_patients = list_dataset_patients(dataset)
    all_diff_x = []
    all_diff_y = []

    for _, subj_id in tqdm(dataset_patients, desc=f"Loading {dataset}"):
        windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
        if windows_x is None or windows_y is None or windows_x.shape[0] < 3:
            continue
        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)
        if diff_x.shape[0] < 3 or diff_y.shape[0] < 3:
            continue
        all_diff_x.append(diff_x)
        all_diff_y.append(diff_y)

    if not all_diff_x:
        raise RuntimeError(f"{dataset} 没有可用于联合 SVD 的有效患者。")

    x_stacked = np.vstack([d.reshape(d.shape[0], -1) for d in all_diff_x])
    y_stacked = np.vstack([d.reshape(d.shape[0], -1) for d in all_diff_y])

    svd_x = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    svd_y = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    svd_x.fit(x_stacked)
    svd_y.fit(y_stacked)

    return {
        "Vt_x": svd_x.components_,
        "Vt_y": svd_y.components_,
        "sigma_x": svd_x.singular_values_,
        "sigma_y": svd_y.singular_values_,
        "n_patients": len(all_diff_x),
        "n_diff_windows_x": int(x_stacked.shape[0]),
        "n_diff_windows_y": int(y_stacked.shape[0]),
    }


def collect_patient_correlations(vt_x, vt_y):
    """汇总 TT 患者相对 XW 联合基的相关性。"""
    all_corr_x = []
    all_corr_y = []
    patient_records = []
    blend_cols_ref = None

    for dataset in SOURCE_DATASETS:
        dataset_blend_root = BLENDSHAPE_ROOT / dataset
        if not dataset_blend_root.exists():
            print(f"Blendshape directory missing for {dataset}, skipping")
            continue

        dataset_patients = list_dataset_patients(dataset)
        print(f"Candidate {dataset} patients: {len(dataset_patients)}")

        for _, subj_id in tqdm(dataset_patients, desc=f"Correlating {dataset}"):
            bs_df = load_blendshape(subj_id, dataset)
            if bs_df is None:
                continue

            windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
            if windows_x is None or windows_y is None or windows_x.shape[0] < 3:
                continue

            diff_x_patient = compute_diff_matrix(windows_x)
            diff_y_patient = compute_diff_matrix(windows_y)
            blend_diff, blend_cols = compute_blendshape_diff(bs_df)

            n_diff_wins = min(diff_x_patient.shape[0], diff_y_patient.shape[0], blend_diff.shape[0])
            if n_diff_wins < 3:
                continue

            coef_x = project_to_joint_basis(diff_x_patient[:n_diff_wins], vt_x)
            coef_y = project_to_joint_basis(diff_y_patient[:n_diff_wins], vt_y)

            corr_x = compute_correlation_matrix(coef_x, blend_diff[:n_diff_wins])
            corr_y = compute_correlation_matrix(coef_y, blend_diff[:n_diff_wins])

            all_corr_x.append(corr_x)
            all_corr_y.append(corr_y)
            patient_records.append(
                {
                    "dataset": dataset,
                    "subj_id": subj_id,
                    "n_diff_windows_used": int(n_diff_wins),
                }
            )
            if blend_cols_ref is None:
                blend_cols_ref = blend_cols

    if not patient_records:
        raise RuntimeError("没有找到可用于 blendshape 相关性分析的有效 TT 患者。")

    return np.array(all_corr_x), np.array(all_corr_y), patient_records, blend_cols_ref


def print_top_correlations(mean_abs_corr, blend_cols, mode_name):
    """打印每个 PC 对应的最高相关 blendshape。"""
    print(f"\n=== {mode_name} ===")
    for pc in range(N_COMPONENTS):
        top_idx = np.argsort(mean_abs_corr[pc])[::-1][:5]
        print(f"PC{pc + 1}: ", end="")
        for idx in top_idx:
            print(f"{blend_cols[idx]}={mean_abs_corr[pc, idx]:.3f} ", end="")
        print()


def plot_mean_heatmap(mean_abs_corr_x, mean_abs_corr_y):
    """绘制平均绝对相关性热图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    titles = ["XW_joint X", "XW_joint Y"]
    matrices = [mean_abs_corr_x, mean_abs_corr_y]

    for ax, title, mat in zip(axes.flat, titles, matrices):
        im = ax.imshow(
            np.abs(mat),
            aspect="auto",
            cmap="seismic",
            norm=TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1),
        )
        ax.set_title(title)
        ax.set_xlabel("blendshape index")
        ax.set_ylabel("PC")
        ax.set_yticks(range(N_COMPONENTS))
        ax.set_yticklabels([f"PC{i + 1}" for i in range(N_COMPONENTS)])
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Mean Absolute Correlation: XW-joint SVD Time Coef vs Blendshape Diff", fontsize=14)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"热图已保存: {output_path}")


def plot_std_heatmap(std_abs_corr_x, std_abs_corr_y):
    """绘制标准差热图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    titles = ["XW_joint X", "XW_joint Y"]
    matrices = [std_abs_corr_x, std_abs_corr_y]

    for ax, title, mat in zip(axes.flat, titles, matrices):
        im = ax.imshow(mat, aspect="auto", cmap="seismic", vmin=0, vmax=0.4)
        ax.set_title(title)
        ax.set_xlabel("blendshape index")
        ax.set_ylabel("PC")
        ax.set_yticks(range(N_COMPONENTS))
        ax.set_yticklabels([f"PC{i + 1}" for i in range(N_COMPONENTS)])
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Std of Absolute Correlation: XW-joint SVD Time Coef vs Blendshape Diff", fontsize=14)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "correlation_std_heatmap.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"标准差热图已保存: {output_path}")


def plot_boxplot(all_corr_x, all_corr_y, mean_abs_corr_x, mean_abs_corr_y, blend_cols, patient_records):
    """绘制 PC1/PC2 顶部 blendshape 的箱线图。"""
    boxplot_data = []
    boxplot_labels = []
    boxplot_colors = []
    mode_specs = [
        ("X", all_corr_x, mean_abs_corr_x, "#3498db"),
        ("Y", all_corr_y, mean_abs_corr_y, "#9b59b6"),
    ]

    for mode_name, corr_all, mean_mat, color in mode_specs:
        for pc in range(2):
            top_idx = np.argsort(mean_mat[pc])[::-1][0]
            patient_corrs = corr_all[:, pc, top_idx]
            valid_mask = ~np.isnan(patient_corrs)
            boxplot_data.append(np.abs(patient_corrs[valid_mask]))
            boxplot_labels.append(f"{mode_name}\nPC{pc + 1}\n{blend_cols[top_idx]}")
            boxplot_colors.append(color)

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], boxplot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("|Pearson r|", fontsize=12)
    ax.set_title("Distribution of Correlation Coefficients (PC1 & PC2 Top Blendshapes)", fontsize=14)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="r=0.5 threshold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    legend_elements = [
        Patch(facecolor="#3498db", alpha=0.6, label="XW_joint X"),
        Patch(facecolor="#9b59b6", alpha=0.6, label="XW_joint Y"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "correlation_boxplot.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"箱线图已保存: {output_path}")

    return None


def main():
    print("=" * 60)
    print("XW 联合 SVD 主模态与 blendshape 时间变化相关性分析")
    print("=" * 60)

    print("\n--- 1. 计算 XW 联合 SVD 基 ---")
    reference = compute_joint_svd_for_dataset(REFERENCE_DATASET)
    print(
        f"XW joint SVD: {reference['n_patients']} patients, "
        f"X sigma={reference['sigma_x'][:4]}, Y sigma={reference['sigma_y'][:4]}"
    )

    print("\n--- 2. 计算 TT 患者相关性 ---")
    all_corr_x, all_corr_y, patient_records, blend_cols = collect_patient_correlations(
        reference["Vt_x"], reference["Vt_y"]
    )
    print(f"Valid patients with blendshape: {len(patient_records)}")

    mean_abs_corr_x = np.nanmean(np.abs(all_corr_x), axis=0)
    mean_abs_corr_y = np.nanmean(np.abs(all_corr_y), axis=0)
    std_abs_corr_x = np.nanstd(np.abs(all_corr_x), axis=0)
    std_abs_corr_y = np.nanstd(np.abs(all_corr_y), axis=0)

    print("\n--- 3. 相关性分析结果 ---")
    print_top_correlations(mean_abs_corr_x, blend_cols, "XW_joint 基投影 (X mode)")
    print_top_correlations(mean_abs_corr_y, blend_cols, "XW_joint 基投影 (Y mode)")

    print("\n--- 4. 保存结果 ---")
    results = {
        "reference_dataset": REFERENCE_DATASET,
        "source_datasets": SOURCE_DATASETS,
        "n_valid_patients": len(patient_records),
        "patient_records": patient_records,
        "blend_cols": blend_cols,
        "reference_sigma": {
            "X": reference["sigma_x"].tolist(),
            "Y": reference["sigma_y"].tolist(),
        },
        "mean_abs_corr": {
            "XW_joint_X": mean_abs_corr_x.tolist(),
            "XW_joint_Y": mean_abs_corr_y.tolist(),
        },
        "std_abs_corr": {
            "XW_joint_X": std_abs_corr_x.tolist(),
            "XW_joint_Y": std_abs_corr_y.tolist(),
        },
    }

    results_file = OUTPUT_DIR / "xw_joint_blendshape_correlation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存: {results_file}")

    print("\n--- 5. 可视化 ---")
    plot_mean_heatmap(mean_abs_corr_x, mean_abs_corr_y)
    plot_std_heatmap(std_abs_corr_x, std_abs_corr_y)
    plot_boxplot(all_corr_x, all_corr_y, mean_abs_corr_x, mean_abs_corr_y, blend_cols, patient_records)

    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)

    summary = []
    for mode_name, mean_mat, std_mat in [
        ("XW_X", mean_abs_corr_x, std_abs_corr_x),
        ("XW_Y", mean_abs_corr_y, std_abs_corr_y),
    ]:
        for pc in range(N_COMPONENTS):
            top_idx = np.argsort(mean_mat[pc])[::-1][0]
            summary.append(
                {
                    "mode": mode_name,
                    "pc": f"PC{pc + 1}",
                    "top_blendshape": blend_cols[top_idx],
                    "mean_abs_corr": float(mean_mat[pc, top_idx]),
                    "std_abs_corr": float(std_mat[pc, top_idx]),
                }
            )

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    print(f"\nValid TT patients: {len(patient_records)}")

    return {
        "results_file": results_file,
        "summary_df": summary_df,
    }


if __name__ == "__main__":
    main()
