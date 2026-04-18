"""
Tucker分解多患者联合分解脚本
对IMR和TT数据集分别做Tucker分解，分解出身份基和运动基

数据筛选条件：
- 窗口数在5-9之间
- 使用差分形式
- Zero-pad到8窗口

张量结构: X ∈ R^(N_subjects × 8 × 341 × 341)
Tucker分解: X ≈ G ×_1 A ×_2 B ×_3 C ×_4 D
- A: 身份基 (N_subjects × r1)
- B: 运动基 (8 × r2)
- C, D: 空间基 (341 × r3, 341 × r4)
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import gc

# 配置
DATA_ROOT = Path("/home/weizilin/generate_idea/data/win20-step20")
N_COMPONENTS = 10
MAX_WINDOWS = 8  # 统一padding到8窗口
WINDOW_RANGE = (5, 9)  # 包含5、6、7、8、9
RANDOM_STATE = 42

# 语义区域边界
REGION_BOUNDARIES = [18, 38, 100, 130, 140, 188, 233, 307, 331, 341]
REGION_NAMES = [
    "forehead", "eyebrow", "eyehole", "eye_contour",
    "eye_iris", "nose", "around_mouth", "mouth", "cheek", "jaw"
]


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
        diff = windows[i] - windows[i-1]
        diffs.append(diff)
    return np.array(diffs)


def zero_pad_to(windows, target_len):
    """Zero-pad到目标长度"""
    n = windows.shape[0]
    if n >= target_len:
        return windows[:target_len]
    else:
        padding = np.zeros((target_len - n,) + windows.shape[1:], dtype=windows.dtype)
        return np.concatenate([windows, padding], axis=0)


def _region_centers(boundaries):
    """计算每个区域的中心位置"""
    centers = []
    prev = 0
    for b in boundaries:
        centers.append((prev + b) / 2.0)
        prev = b
    return centers


def visualize_motion_basis_heatmap(B, mode_name, save_dir, n_show=5):
    """
    可视化运动基 B (window mode)
    B shape: (8, r2) - 8个窗口位置，r2个基
    显示为热图：行=窗口，列=PC
    """
    n_components = min(n_show, B.shape[1])
    B_show = B[:, :n_components]

    fig, ax = plt.subplots(figsize=(4, 6))
    vmax = np.abs(B_show).max()
    im = ax.imshow(B_show, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Component")
    ax.set_ylabel("Window Index")
    ax.set_title(f"Motion Basis - {mode_name}")
    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f"PC{i+1}" for i in range(n_components)])
    ax.set_yticks(range(B.shape[0]))
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f"Tucker Motion Basis - {mode_name}", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"tucker_motion_basis_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_identity_basis_summary(A, patient_info, mode_name, save_dir, n_show=5):
    """
    可视化身份基 A 的系数分布
    A shape: (N_subjects, r1)
    显示为热图：行=被试，列=PC
    """
    n_components = min(n_show, A.shape[1])
    A_show = A[:, :n_components]

    fig, ax = plt.subplots(figsize=(4, 8))
    vmax = np.abs(A_show).max()
    im = ax.imshow(A_show, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Component")
    ax.set_ylabel("Subject Index")
    ax.set_title(f"Identity Basis - {mode_name}")
    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f"PC{i+1}" for i in range(n_components)])
    plt.colorbar(im, ax=ax, shrink=0.6)

    plt.suptitle(f"Tucker Identity Basis - {mode_name}", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"tucker_identity_dist_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_spatial_basis_heatmap(C, mode_name, save_dir, n_show=3):
    """
    可视化空间基 C (landmark mode)
    C shape: (341, r3) - 每列是一个spatial component的341维系数向量
    可视化为每个component的系数分布图（而不是341x341热图）
    """
    n = min(n_show, C.shape[1])

    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        basis = C[:, i]  # shape: (341,)
        axes[i].bar(range(341), basis, alpha=0.7)
        axes[i].set_xlabel("Landmark Index")
        axes[i].set_ylabel("Coefficient")
        axes[i].set_title(f"Spatial PC{i+1}")
        axes[i].grid(True, alpha=0.3)

        # 标记区域边界
        for boundary in REGION_BOUNDARIES[:-1]:
            axes[i].axvline(boundary, color="black", linewidth=0.8, alpha=0.5, linestyle='--')

    plt.suptitle(f"Tucker Spatial Basis - {mode_name}", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"tucker_spatial_basis_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def analyze_semantic_regions_from_spatial(C, n_show=5):
    """
    从空间基C分析语义区域
    C shape: (341, r3) - 每列是一个spatial component的341维系数向量
    分析每个component在语义区域上的系数强度分布
    """
    n = min(n_show, C.shape[1])
    results = []

    for i in range(n):
        basis = C[:, i]  # shape: (341,)
        total_abs = np.sum(np.abs(basis)) + 1e-6

        region_contributions = {}
        prev_boundary = 0
        for region_name, boundary in zip(REGION_NAMES, REGION_BOUNDARIES):
            indices = list(range(prev_boundary, boundary))
            region_abs = np.sum(np.abs(basis[indices]))
            region_contributions[region_name] = float(region_abs / total_abs * 100)
            prev_boundary = boundary

        max_region = max(region_contributions, key=region_contributions.get)
        max_contribution = region_contributions[max_region]

        results.append({
            "pc": i + 1,
            "dominant_region": max_region,
            "dominant_contribution": max_contribution,
            "all_regions": region_contributions
        })

    return results


def analyze_motion_basis(B):
    """分析运动基B的激活模式"""
    n_components = B.shape[1]
    results = []

    for i in range(n_components):
        basis = B[:, i]
        # 找出激活最强的窗口
        max_window = int(np.argmax(np.abs(basis)))
        max_val = float(basis[max_window])

        # 计算基的稀疏性（非零比例）
        sparsity = float(np.sum(np.abs(basis) < 1e-6) / len(basis))

        results.append({
            "pc": i + 1,
            "dominant_window": max_window,
            "dominant_value": max_val,
            "sparsity": sparsity,
            "all_windows": basis.tolist()
        })

    return results


def simple_tucker(X, n_components_per_mode):
    """
    简化的Tucker分解
    使用SVD做低秩近似

    X: input tensor of shape (N_subjects, N_windows, N_landmarks, N_landmarks)
    n_components_per_mode: dict with keys 'subject', 'window', 'spatial'

    返回: A, B, C, D, G (factor matrices and core tensor)
    """
    # 强制转为32位以节省内存
    X = X.astype(np.float32)
    N_subjects, N_windows, N_landmarks, _ = X.shape

    # Mode-1 (subject) unfolding: (N_subjects, N_windows * N_landmarks * N_landmarks)
    X1 = X.reshape(N_subjects, -1)
    svd1 = TruncatedSVD(n_components=n_components_per_mode['subject'], random_state=RANDOM_STATE)
    A = svd1.fit_transform(X1)  # (N_subjects, r1)
    del X1
    gc.collect()

    # Mode-2 (window) unfolding: (N_windows, N_subjects * N_landmarks * N_landmarks)
    X2 = X.reshape(N_subjects, N_windows, -1).transpose(1, 0, 2).reshape(N_windows, -1)
    svd2 = TruncatedSVD(n_components=n_components_per_mode['window'], random_state=RANDOM_STATE)
    B = svd2.fit_transform(X2)  # (N_windows, r2)
    del X2
    gc.collect()

    # Mode-3 (spatial) unfolding: (N_landmarks, N_subjects * N_windows * N_landmarks)
    X3 = X.reshape(N_subjects, N_windows, N_landmarks, N_landmarks).transpose(2, 0, 1, 3).reshape(N_landmarks, -1)
    # 使用全SVD，然后重建spatial patterns: X3 @ V = U @ S @ V^T @ V = U @ S
    # 所以 X3 @ V[:, i] = U[:, i] * s[i]，得到第i个spatial pattern的341x341近似
    U, s, Vt = np.linalg.svd(X3, full_matrices=False)
    r3 = n_components_per_mode['spatial']
    # C_spatial[i] = U[:, i] * s[i]，重建第i个spatial pattern的341维向量
    # 然后reshape成341x341矩阵
    C = U[:, :r3] * s[:r3]  # (N_landmarks, r3)，每个spatial pattern的系数向量
    del X3, U, s, Vt
    gc.collect()

    # Core tensor estimation (simplified)
    # G = A^T @ X1 @ (B ⊗ C ⊗ C)^T  (pseudo-inverse based)

    return A, B, C, None, None


def analyze_identity_coefficient_variance(A):
    """分析身份基系数的方差（跨被试）"""
    var = np.var(A, axis=0)
    mean = np.mean(A, axis=0)
    std = np.std(A, axis=0)

    results = []
    for i in range(len(var)):
        results.append({
            "pc": i + 1,
            "variance": float(var[i]),
            "mean": float(mean[i]),
            "std": float(std[i]),
            "cv": float(std[i] / (np.abs(mean[i]) + 1e-6))  # coefficient of variation
        })

    return results


def main():
    print("=" * 60)
    print("Tucker 多患者联合分解 - 筛选窗口5-9，Pad到8")
    print("=" * 60)

    # 收集符合条件的患者
    imr_patients = []
    tt_patients = []

    for dataset in ["IMR", "TT"]:
        dataset_path = DATA_ROOT / dataset
        if not dataset_path.exists():
            continue
        for subj_dir in sorted(dataset_path.iterdir()):
            if not subj_dir.is_dir():
                continue
            win_files = list(subj_dir.glob("win_*_x.npy"))
            n_wins = len(win_files)
            if WINDOW_RANGE[0] <= n_wins <= WINDOW_RANGE[1]:
                if dataset == "IMR":
                    imr_patients.append((dataset, subj_dir.name, n_wins))
                else:
                    tt_patients.append((dataset, subj_dir.name, n_wins))

    print(f"\nIMR 符合条件的患者数: {len(imr_patients)}")
    print(f"TT 符合条件的患者数: {len(tt_patients)}")

    # 输出目录
    output_dir = DATA_ROOT / "tucker_multi_patient_results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    # ========== 考虑到内存问题，暂时先只处理TT，不处理IMR ==========
    # for dataset_name, dataset_patients in [("IMR", imr_patients), ("TT", tt_patients)]:
    for dataset_name, dataset_patients in [("TT", tt_patients)]:
        if len(dataset_patients) == 0:
            print(f"\nNo {dataset_name} patients found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"处理 {dataset_name} 数据集 ({len(dataset_patients)} 患者)")
        print(f"{'='*60}")

        # 加载数据
        all_diff_x = []
        all_diff_y = []
        valid_patient_info = []

        for dataset, subj_id, n_wins in tqdm(dataset_patients, desc=f"Loading {dataset_name}"):
            windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
            if windows_x is None:
                continue

            # 计算差分
            diff_x = compute_diff_matrix(windows_x)  # (n_wins-1, 341, 341)
            diff_y = compute_diff_matrix(windows_y)

            # 只取前MAX_WINDOWS个差分
            diff_x = diff_x[:MAX_WINDOWS]
            diff_y = diff_y[:MAX_WINDOWS]

            # Zero-pad到8
            diff_x = zero_pad_to(diff_x, MAX_WINDOWS)
            diff_y = zero_pad_to(diff_y, MAX_WINDOWS)

            all_diff_x.append(diff_x)
            all_diff_y.append(diff_y)
            valid_patient_info.append((dataset, subj_id, n_wins))

        if len(valid_patient_info) == 0:
            print(f"No valid {dataset_name} patients")
            continue

        print(f"\n{dataset_name}: {len(valid_patient_info)} patients")
        print(f"Tensor shape: ({len(valid_patient_info)}, {MAX_WINDOWS}, 341, 341)")

        # 堆叠为4D张量
        X_x = np.array(all_diff_x)  # (N, 8, 341, 341)
        X_y = np.array(all_diff_y)

        # Tucker分解配置
        n_components = {
            'subject': min(10, len(valid_patient_info) - 1),
            'window': min(8, MAX_WINDOWS),
            'spatial': 10
        }

        print(f"\n--- {dataset_name} X模态 Tucker分解 ---")
        print(f"Components: subject={n_components['subject']}, window={n_components['window']}, spatial={n_components['spatial']}")

        # X模态
        A_x, B_x, C_x, _, _ = simple_tucker(X_x, n_components)

        print(f"\n身份基 A shape: {A_x.shape} (被试 × 基)")
        print(f"运动基 B shape: {B_x.shape} (窗口 × 基)")
        print(f"空间基 C shape: {C_x.shape} (landmarks × 基)")

        # 分析运动基
        motion_analysis_x = analyze_motion_basis(B_x)
        print(f"\n运动基分析 (X模态):")
        for m in motion_analysis_x[:5]:
            print(f"    PC{m['pc']}: dominant_window={m['dominant_window']}, sparsity={m['sparsity']:.2f}")

        # 分析空间基语义
        spatial_semantic_x = analyze_semantic_regions_from_spatial(C_x)
        print(f"\n空间基语义 (X模态):")
        for s in spatial_semantic_x[:5]:
            print(f"    PC{s['pc']}: dominant={s['dominant_region']} ({s['dominant_contribution']:.1f}%)")

        # 分析身份基方差
        identity_analysis_x = analyze_identity_coefficient_variance(A_x)
        print(f"\n身份基方差 (X模态):")
        for i in identity_analysis_x[:5]:
            print(f"    PC{i['pc']}: variance={i['variance']:.4f}, std={i['std']:.4f}")

        # Y模态
        print(f"\n--- {dataset_name} Y模态 Tucker分解 ---")
        A_y, B_y, C_y, _, _ = simple_tucker(X_y, n_components)

        motion_analysis_y = analyze_motion_basis(B_y)
        spatial_semantic_y = analyze_semantic_regions_from_spatial(C_y)
        identity_analysis_y = analyze_identity_coefficient_variance(A_y)

        print(f"\n运动基分析 (Y模态):")
        for m in motion_analysis_y[:5]:
            print(f"    PC{m['pc']}: dominant_window={m['dominant_window']}, sparsity={m['sparsity']:.2f}")

        print(f"\n空间基语义 (Y模态):")
        for s in spatial_semantic_y[:5]:
            print(f"    PC{s['pc']}: dominant={s['dominant_region']} ({s['dominant_contribution']:.1f}%)")

        # 可视化
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        print(f"\n--- 可视化 ---")

        # 运动基热图
        motion_x_path = visualize_motion_basis_heatmap(B_x, f"{dataset_name}_X", dataset_dir, n_show=5)
        motion_y_path = visualize_motion_basis_heatmap(B_y, f"{dataset_name}_Y", dataset_dir, n_show=5)
        print(f"运动基: {motion_x_path}, {motion_y_path}")

        # 身份基分布
        identity_x_path = visualize_identity_basis_summary(A_x, valid_patient_info, f"{dataset_name}_X", dataset_dir, n_show=5)
        identity_y_path = visualize_identity_basis_summary(A_y, valid_patient_info, f"{dataset_name}_Y", dataset_dir, n_show=5)
        print(f"身份基分布: {identity_x_path}, {identity_y_path}")

        # 空间基热图
        spatial_x_path = visualize_spatial_basis_heatmap(C_x, f"{dataset_name}_X", dataset_dir, n_show=3)
        spatial_y_path = visualize_spatial_basis_heatmap(C_y, f"{dataset_name}_Y", dataset_dir, n_show=3)
        print(f"空间基: {spatial_x_path}, {spatial_y_path}")

        # 保存结果
        all_results[dataset_name] = {
            "n_patients": len(valid_patient_info),
            "window_range": WINDOW_RANGE,
            "padded_to": MAX_WINDOWS,
            "patient_info": [
                {"dataset": p[0], "subj_id": p[1], "original_windows": p[2]} for p in valid_patient_info
            ],
            "n_components": n_components,
            "X_mode": {
                "A_shape": list(A_x.shape),
                "B_shape": list(B_x.shape),
                "C_shape": list(C_x.shape),
                "motion_analysis": motion_analysis_x,
                "spatial_semantic": spatial_semantic_x,
                "identity_variance": identity_analysis_x
            },
            "Y_mode": {
                "A_shape": list(A_y.shape),
                "B_shape": list(B_y.shape),
                "C_shape": list(C_y.shape),
                "motion_analysis": motion_analysis_y,
                "spatial_semantic": spatial_semantic_y,
                "identity_variance": identity_analysis_y
            }
        }

    # 保存汇总结果
    results_file = output_dir / "tucker_results_summary.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n结果已保存到: {results_file}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    for name, results in all_results.items():
        print(f"\n{name}:")
        print(f"  患者数: {results['n_patients']}")
        print(f"  X空间基PC1 dominant: {results['X_mode']['spatial_semantic'][0]['dominant_region']}")
        print(f"  Y空间基PC1 dominant: {results['Y_mode']['spatial_semantic'][0]['dominant_region']}")
        print(f"  X运动基PC1 dominant window: {results['X_mode']['motion_analysis'][0]['dominant_window']}")
        print(f"  Y运动基PC1 dominant window: {results['Y_mode']['motion_analysis'][0]['dominant_window']}")


if __name__ == "__main__":
    main()
