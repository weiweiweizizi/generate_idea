"""
DMD 单患者验证脚本 - 动态模式分解
对IMR和TT各5个患者分别做DMD差分分解
- 空间模态热图可视化（前3个）
- 特征值分布（动态特性：频率/衰减）
- 时间系数分析可视化
"""

import numpy as np
from scipy.linalg import svd, pinv
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# 配置
DATA_ROOT = Path("/home/weizilin/generate_idea/data/win5-step5")
N_MODES = 10
RANDOM_STATE = 42
N_IMR = 5  # IMR患者数
N_TT = 5   # TT患者数

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


def dmd(X, n_modes=None):
    """
    经典DMD（Dynamic Mode Decomposition）

    输入: X - shape (n_samples, n_features) 每行是一个snapshot
          或者 X 和 Y 分别是被移动前后的数据
    返回: modes (n_features, n_modes), eigenvalues (n_modes,), time_coeffs (n_samples, n_modes)

    DMD分解: X_1 = A X_0
    A = X_1 X_0^+, 然后对A做特征分解
    """
    if X.ndim == 3:
        n_samples, h, w = X.shape
        X = X.reshape(n_samples, -1)
    else:
        n_samples, n_features = X.shape

    if n_modes is None:
        n_modes = min(n_samples - 1, n_features, N_MODES)

    # 构建DMD数据矩阵
    # X0 = [x0, x1, ..., x_{m-2}], X1 = [x1, x2, ..., x_{m-1}]
    X0 = X[:-1].T  # (n_features, m-1)
    X1 = X[1:].T   # (n_features, m-1)

    # A = X1 * X0^+ (pseudo-inverse)
    A = X1 @ pinv(X0)

    # 特征分解 A * v = lambda * v
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # DMD模态 = X1 * pinv(X0) * v / lambda (refined modes)
    # 简化: 直接用特征向量作为模态基础
    # 取前n_modes个特征值/向量
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx[:n_modes]]
    eigenvectors = eigenvectors[:, idx[:n_modes]]

    # DMD modes: 归一化
    modes = eigenvectors
    for i in range(modes.shape[1]):
        norm = np.linalg.norm(modes[:, i])
        if norm > 1e-10:
            modes[:, i] = modes[:, i] / norm

    # 时间系数 (amplitudes): 通过最小二乘确定每个snapshot在模态上的系数
    # X ≈ modes @ diag(eigenvalues^t) @ b
    # 简化为: b = pinv(modes) @ X.mean(axis=1) 或用伪逆拟合
    time_coeffs = np.zeros((n_samples, n_modes), dtype=complex)

    # 使用 least-squares 求解 amplitudes
    # X[i] ≈ sum_j b_j * lambda_j^i * modes[:,j]
    for i in range(n_samples):
        for j in range(n_modes):
            time_coeffs[i, j] = eigenvalues[j] ** i

    # b = (modes^+ @ X[:, 0])  初始时刻系数
    # 简化：使用伪逆
    b, residuals, rank, s = np.linalg.lstsq(modes, X[0], rcond=None)
    time_coeffs = np.zeros((n_samples, n_modes), dtype=complex)
    for i in range(n_samples):
        for j in range(n_modes):
            time_coeffs[i, j] = b[j] * (eigenvalues[j] ** i)

    return modes, eigenvalues, time_coeffs


def dmd_with_svd_truncation(X, n_modes=None, rank=20):
    """
    截断DMD - 空间降维版本

    核心思想：
    - X: (n_samples, n_features) 其中 n_features = 341*341 = 116281
    - SVD截断: X ≈ U_r @ S_r @ V_r^T，其中 V_r^T 是 (rank, n_features)
    - V_r 的每行是一个空间基向量，列是时间方向
    - 但V_r.T的每列才是我们要的空间模态

    降维发生在**空间方向**（n_features方向），时间维度n_samples保持完整！

    参数:
        X: (n_samples, n_features) 每行是一个snapshot（flatten的341×341矩阵）
        n_modes: 返回的模态数量
        rank: SVD截断rank（空间降维维度）

    返回:
        modes: (n_features, n_modes) DMD模态（可reshape成341×341）
        eigenvalues: (n_modes,) 特征值
        time_coeffs: (n_samples, n_modes) 时间系数
    """
    if X.ndim == 3:
        n_samples, h, w = X.shape
        X = X.reshape(n_samples, -1)
    else:
        n_samples, n_features = X.shape

    if n_modes is None:
        n_modes = min(rank, N_MODES)

    # ========== 核心：SVD在空间方向截断 ==========
    # X = U @ S @ Vh
    # U: (n_samples, n_samples)
    # S: (min(n_samples, n_features),)
    # Vh: (n_features, n_features) 或 (rank, n_features) 如果 full_matrices=False

    # 取前rank个奇异值对应的右奇异向量（空间基）
    # Vh的形状是 (n_features, n_features)，取前rank行得到 (rank, n_features)
    # 但我们用 full_matrices=False 时，Vh就是 (rank, n_features)
    U, s, Vh = svd(X, full_matrices=False)

    # V_r: (rank, n_features) - 每行是一个空间基的权重向量
    # 把它转置一下方便理解: V_r.T: (n_features, rank)
    V_r = Vh[:rank, :]  # (rank, n_features)

    # X0和X1是时间相邻的snapshots
    # X0: (n_samples-1, n_features), X1: (n_samples-1, n_features)
    X0 = X[:-1, :]      # t=0 到 t=m-2
    X1 = X[1:, :]       # t=1 到 t=m-1

    # 在空间方向降维后的数据
    # X0_low: (n_samples-1, rank), X1_low: (n_samples-1, rank)
    X0_low = X0 @ V_r.T   # (n_samples-1, rank)
    X1_low = X1 @ V_r.T   # (n_samples-1, rank)

    # ========== 在降维后的空间做DMD ==========
    # A_low: (rank, rank) 降维后的动力学矩阵
    # x_{k+1} = A_low @ x_k（在低维空间）
    A_low = X1_low.T @ pinv(X0_low.T)  # 注意这里要转置一下使得维度对齐

    # 实际上更标准的做法：
    # X1_low ≈ A_low @ X0_low.T 的每一列是一个样本
    # 即 X1_low[:, k] ≈ A_low @ X0_low[:, k]
    # 所以 A_low = X1_low @ pinv(X0_low)

    # 让我重新理清：
    # X0: (n_samples-1, n_features) - 每行是一个snapshot
    # X0_low = X0 @ V_r.T: (n_samples-1, rank) - 每个snapshot投影到rank维空间

    # DMD: X1_low[k,:] ≈ A @ X0_low[k,:]
    # 所以 A = X1_low.T @ pinv(X0_low.T) = X1_low.T @ X0_low (pinv of tall matrix)
    # 或者直接: A = X1_low.T @ np.linalg.pinv(X0_low.T)

    # 改写成更清晰的形式：
    # X1_low: (m, r), X0_low: (m, r)
    # DMD: X1_low[k] = A @ X0_low[k], k=1..m
    # 所以 A = X1_low.T @ X0_low (当X0_low列满秩时)
    # 或 A = X1_low.T @ pinv(X0_low.T) = X1_low.T @ X0_low @ pinv(X0_low.T @ X0_low)

    # 实际上更常用：
    A_low = X1_low.T @ np.linalg.pinv(X0_low.T)  # (rank, rank)

    # ========== 特征分解 ==========
    eigenvalues, W = np.linalg.eig(A_low)  # (rank, rank)

    # 取前n_modes个最大的（按模）
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx[:n_modes]]
    W = W[:, idx[:n_modes]]

    # ========== DMD模态映射回高维 ==========
    # DMD模态 = V_r.T @ W （把低维特征向量映射回n_features维）
    modes = V_r.T @ W  # (n_features, n_modes)

    # 归一化
    for i in range(modes.shape[1]):
        norm = np.linalg.norm(modes[:, i])
        if norm > 1e-10:
            modes[:, i] = modes[:, i] / norm

    # ========== 时间系数 ==========
    # x_k ≈ sum_j b_j * λ_j^k * φ_j
    # b = pinv(modes) @ x_0
    b, _, _, _ = np.linalg.lstsq(modes, X[0], rcond=None)

    time_coeffs = np.zeros((n_samples, n_modes), dtype=complex)
    for k in range(n_samples):
        for j in range(n_modes):
            time_coeffs[k, j] = b[j] * (eigenvalues[j] ** k)

    return modes, eigenvalues, time_coeffs


def _region_centers(boundaries):
    """计算每个区域的中心位置"""
    centers = []
    prev = 0
    for b in boundaries:
        centers.append((prev + b) / 2.0)
        prev = b
    return centers


def visualize_dmd_heatmaps(modes, eigenvalues, mode_name, subj_id, dataset, save_dir, n_show=3):
    """可视化前n_show个DMD模态的热图，带区域分割线"""
    n = min(n_show, modes.shape[1])
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        mode = modes[:, i].reshape(341, 341)
        vmax = np.abs(mode).max()
        im = axes[i].imshow(mode.real, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"Mode{i+1} (λ={eigenvalues[i]:.3f})")
        axes[i].set_xlabel("landmark j")
        axes[i].set_ylabel("landmark i")
        plt.colorbar(im, ax=axes[i], shrink=0.8)

        # 绘制区域分割线
        for boundary in REGION_BOUNDARIES[:-1]:
            axes[i].axhline(boundary, color="black", linewidth=0.8, alpha=0.6)
            axes[i].axvline(boundary, color="black", linewidth=0.8, alpha=0.6)

        # 添加区域名称标签
        if i == 0:
            centers = _region_centers(REGION_BOUNDARIES)
            for name, c in zip(REGION_NAMES, centers):
                axes[i].text(c, -2, name, ha="center", va="bottom",
                           fontsize=6, rotation=45, transform=axes[i].transData)

    plt.suptitle(f"DMD Modes - {dataset}/{subj_id} ({mode_name})", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"{dataset}_{subj_id}_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_eigenvalues(eigenvalues, mode_name, subj_id, dataset, save_dir):
    """可视化特征值分布（复平面）"""
    fig, ax = plt.subplots(figsize=(6, 5))

    # 绘制单位圆
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label="Unit circle")

    # 绘制特征值
    ax.scatter(eigenvalues.real, eigenvalues.imag, s=100, c='red', marker='o', zorder=5)

    # 标注每个特征值
    for i, (re, im) in enumerate(zip(eigenvalues.real, eigenvalues.imag)):
        ax.annotate(f'λ{i+1}', (re, im), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Real part (decay/growth)")
    ax.set_ylabel("Imaginary part (oscillation)")
    ax.set_title(f"Eigenvalue Distribution - {dataset}/{subj_id} ({mode_name})")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = save_dir / f"{dataset}_{subj_id}_{mode_name}_eigenvalues.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_time_coeffs(time_coeffs, eigenvalues, mode_name, subj_id, dataset, save_dir, n_show=3):
    """可视化时间系数（取实部）"""
    n = min(n_show, time_coeffs.shape[1])
    n_samples = time_coeffs.shape[0]

    fig, axes = plt.subplots(1, n, figsize=(5*n, 3))
    if n == 1:
        axes = [axes]

    window_indices = range(1, n_samples + 1)

    for i in range(n):
        coeff = time_coeffs[:, i].real
        axes[i].plot(window_indices, coeff, 'o-', linewidth=2, markersize=6)
        axes[i].set_xlabel("Window Index")
        axes[i].set_ylabel("Coefficient (real)")
        axes[i].set_title(f"Mode{i+1} (λ={eigenvalues[i]:.3f})")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f"DMD Time Coefficients - {dataset}/{subj_id} ({mode_name})", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"{dataset}_{subj_id}_{mode_name}_timecoef.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def analyze_semantic_regions(modes, n_show=3):
    """分析每个DMD模态主要激活的语义区域"""
    n = min(n_show, modes.shape[1])
    results = []

    for i in range(n):
        mode = modes[:, i].reshape(341, 341)
        total_abs = np.sum(np.abs(mode))

        region_contributions = {}
        prev_boundary = 0
        for region_name, boundary in zip(REGION_NAMES, REGION_BOUNDARIES):
            indices = list(range(prev_boundary, boundary))
            sub = mode[np.ix_(indices, indices)]
            region_abs = np.sum(np.abs(sub))
            region_contributions[region_name] = float(region_abs / total_abs * 100)
            prev_boundary = boundary

        max_region = max(region_contributions, key=region_contributions.get)
        max_contribution = region_contributions[max_region]

        results.append({
            "mode": i + 1,
            "dominant_region": max_region,
            "dominant_contribution": max_contribution,
            "all_regions": region_contributions
        })

    return results


def analyze_mode_energy(modes, eigenvalues, n_show=3):
    """分析DMD模态的能量（基于特征值模）"""
    n = min(n_show, modes.shape[1])
    total_abs_eig = np.sum(np.abs(eigenvalues[:n]))

    energy_info = []
    for i in range(n):
        mode = modes[:, i]
        mode_energy = np.sum(mode ** 2)  # 空间能量
        eig_weight = np.abs(eigenvalues[i]) / total_abs_eig if total_abs_eig > 1e-10 else 0
        energy_info.append({
            "mode": i + 1,
            "eigenvalue": float(eigenvalues[i]),
            "eigenvalue_magnitude": float(np.abs(eigenvalues[i])),
            "spatial_energy": float(mode_energy),
        })

    return energy_info


def main():
    print("=" * 60)
    print("DMD 单患者验证 - 动态模式分解 (win5-step5)")
    print("=" * 60)

    # 收集患者列表
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

    print(f"IMR patients found: {len(imr_patients)}")
    print(f"TT patients found: {len(tt_patients)}")

    # 随机选择 N_IMR 和 N_TT
    np.random.seed(RANDOM_STATE)
    selected_imr = list(np.random.choice(len(imr_patients), size=min(N_IMR, len(imr_patients)), replace=False))
    selected_tt = list(np.random.choice(len(tt_patients), size=min(N_TT, len(tt_patients)), replace=False))

    selected_patients = [imr_patients[i] for i in selected_imr] + [tt_patients[i] for i in selected_tt]
    print(f"Selected patients: {selected_patients}")

    # 结果保存目录
    output_dir = DATA_ROOT / "dmd_single_patient_results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "x_heatmaps").mkdir(exist_ok=True)
    (output_dir / "y_heatmaps").mkdir(exist_ok=True)
    (output_dir / "x_eigenvalues").mkdir(exist_ok=True)
    (output_dir / "y_eigenvalues").mkdir(exist_ok=True)
    (output_dir / "x_timecoef").mkdir(exist_ok=True)
    (output_dir / "y_timecoef").mkdir(exist_ok=True)

    all_results = []

    for dataset, subj_id in tqdm(selected_patients, desc="Processing patients"):
        print(f"\n{'='*50}")
        print(f"--- {dataset}/{subj_id} ---")
        print(f"{'='*50}")

        # 加载数据
        windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
        if windows_x is None:
            print(f"  Skipping: no data found")
            continue

        n_windows = windows_x.shape[0]
        print(f"  Windows: {n_windows}")

        # 计算差分
        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)
        print(f"  Diff shape: {diff_x.shape}")

        if diff_x.shape[0] < 3:
            print(f"  Skipping: not enough windows for DMD")
            continue

        # 动态确定rank（基于窗口数）
        rank = min(diff_x.shape[0] - 1, 10)

        # 对x和y分别做DMD
        try:
            # X模态
            modes_x, eig_x, tc_x = dmd_with_svd_truncation(diff_x, n_modes=min(N_MODES, rank), rank=rank)
            print(f"\n  [X模态] 特征值: {eig_x[:5]}")
            print(f"  [X模态] 特征值模: {np.abs(eig_x[:5])}")

            # Y模态
            modes_y, eig_y, tc_y = dmd_with_svd_truncation(diff_y, n_modes=min(N_MODES, rank), rank=rank)
            print(f"\n  [Y模态] 特征值: {eig_y[:5]}")
            print(f"  [Y模态] 特征值模: {np.abs(eig_y[:5])}")

            # 分析语义区域
            semantic_x = analyze_semantic_regions(modes_x)
            semantic_y = analyze_semantic_regions(modes_y)

            # 分析能量
            energy_x = analyze_mode_energy(modes_x, eig_x)
            energy_y = analyze_mode_energy(modes_y, eig_y)

            print(f"\n  [X模态] 前3模态语义区域:")
            for s in semantic_x:
                print(f"      Mode{s['mode']}: 主要={s['dominant_region']} ({s['dominant_contribution']:.1f}%)")

            print(f"\n  [Y模态] 前3模态语义区域:")
            for s in semantic_y:
                print(f"      Mode{s['mode']}: 主要={s['dominant_region']} ({s['dominant_contribution']:.1f}%)")

            # 可视化 - 热图
            heatmap_x = visualize_dmd_heatmaps(modes_x, eig_x, "X", subj_id, dataset,
                                                output_dir / "x_heatmaps", n_show=3)
            heatmap_y = visualize_dmd_heatmaps(modes_y, eig_y, "Y", subj_id, dataset,
                                                output_dir / "y_heatmaps", n_show=3)

            # 可视化 - 特征值
            eigplot_x = visualize_eigenvalues(eig_x, "X", subj_id, dataset, output_dir / "x_eigenvalues")
            eigplot_y = visualize_eigenvalues(eig_y, "Y", subj_id, dataset, output_dir / "y_eigenvalues")

            # 可视化 - 时间系数
            tcplot_x = visualize_time_coeffs(tc_x, eig_x, "X", subj_id, dataset, output_dir / "x_timecoef", n_show=3)
            tcplot_y = visualize_time_coeffs(tc_y, eig_y, "Y", subj_id, dataset, output_dir / "y_timecoef", n_show=3)

            print(f"\n  [输出]")
            print(f"      热图(X): {heatmap_x}")
            print(f"      热图(Y): {heatmap_y}")
            print(f"      特征值(X): {eigplot_x}")
            print(f"      特征值(Y): {eigplot_y}")
            print(f"      时间系数(X): {tcplot_x}")
            print(f"      时间系数(Y): {tcplot_y}")

            all_results.append({
                "dataset": dataset,
                "subj_id": subj_id,
                "n_windows": n_windows,
                "n_diff_windows": diff_x.shape[0],
                "eigenvalues_x": [str(e) for e in eig_x],
                "eigenvalues_y": [str(e) for e in eig_y],
                "eigenvalue_magnitude_x": [float(np.abs(e)) for e in eig_x],
                "eigenvalue_magnitude_y": [float(np.abs(e)) for e in eig_y],
                "energy_x": energy_x,
                "energy_y": energy_y,
                "semantic_x": semantic_x,
                "semantic_y": semantic_y,
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总分析
    print("\n" + "=" * 60)
    print("汇总分析")
    print("=" * 60)

    # 统计dominant region分布
    from collections import Counter
    x_domains = [r["semantic_x"][0]["dominant_region"] for r in all_results if len(r["semantic_x"]) > 0]
    y_domains = [r["semantic_y"][0]["dominant_region"] for r in all_results if len(r["semantic_y"]) > 0]

    print(f"\nX模态 Mode1 dominant region分布:")
    for region, count in Counter(x_domains).most_common():
        print(f"    {region}: {count}/{len(x_domains)}")

    print(f"\nY模态 Mode1 dominant region分布:")
    for region, count in Counter(y_domains).most_common():
        print(f"    {region}: {count}/{len(y_domains)}")

    # 统计特征值模分布
    all_mag_x = []
    all_mag_y = []
    for r in all_results:
        all_mag_x.extend(r["eigenvalue_magnitude_x"])
        all_mag_y.extend(r["eigenvalue_magnitude_y"])

    print(f"\nX模态 特征值模: mean={np.mean(all_mag_x):.3f}, std={np.std(all_mag_x):.3f}")
    print(f"Y模态 特征值模: mean={np.mean(all_mag_y):.3f}, std={np.std(all_mag_y):.3f}")

    # 保存结果
    results_file = output_dir / "results_summary.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n结果已保存到: {output_dir}")
    print(f"汇总文件: {results_file}")

    print("\n" + "=" * 60)
    print("生成的文件:")
    print("  - x_heatmaps/*.png: X模态前3个空间模态的热图")
    print("  - y_heatmaps/*.png: Y模态前3个空间模态的热图")
    print("  - x_eigenvalues/*.png: X模态特征值复平面分布")
    print("  - y_eigenvalues/*.png: Y模态特征值复平面分布")
    print("  - x_timecoef/*.png: X模态时间系数曲线")
    print("  - y_timecoef/*.png: Y模态时间系数曲线")
    print("=" * 60)


if __name__ == "__main__":
    main()
