"""
DMD 多患者联合分解
- 参考 svd_multi_patient.py 的结构和可视化风格
- 使用 dmd_single_patient.py 的截断DMD实现（空间投影版）
- rank取50（窗口数量更多）
"""

import numpy as np
from scipy.linalg import pinv
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win5-step5"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "pilot_feasibility" / "dmd" / "win5-step5"
DMD_RANK = 50  # 多患者窗口更多，可以取更大的rank
N_MODES_SHOW = 5
RANDOM_STATE = 42

# 窗口数筛选条件
MIN_WINDOWS = 27
MAX_WINDOWS = 200

# 内存警告阈值 (GB)
MEMORY_THRESHOLD_GB = 8.0

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


def dmd_spatial_projected(X, n_modes=None, rank=50):
    """
    截断DMD - 空间投影版本（来自dmd_single_patient.py）

    在空间方向降维（n_features → rank），时间维度完整保留

    参数:
        X: (n_samples, n_features) 每行是一个snapshot
        n_modes: 返回的模态数量
        rank: SVD截断rank（空间降维维度）

    返回:
        modes: (n_features, n_modes) DMD模态
        eigenvalues: (n_modes,) 特征值
        time_coeffs: (n_samples, n_modes) 时间系数
    """
    if X.ndim == 3:
        n_samples, h, w = X.shape
        X = X.reshape(n_samples, -1)
    else:
        n_samples, n_features = X.shape

    if n_modes is None:
        n_modes = min(rank, N_MODES_SHOW, n_samples)

    # SVD截断（空间方向）- 使用randomized_svd避免full SVD的内存爆炸
    U, s, Vh = randomized_svd(X, n_components=rank)
    V_r = Vh[:rank, :]  # (rank, n_features)

    # 投影到低维空间
    X_low = X @ V_r.T  # (n_samples, rank)

    # DMD: X1_low = A @ X0_low
    X0_low = X_low[:-1]  # (n_samples-1, rank)
    X1_low = X_low[1:]   # (n_samples-1, rank)

    # 动力学矩阵
    A_low = X1_low.T @ pinv(X0_low.T)  # (rank, rank)

    # 特征分解
    eigenvalues, W = np.linalg.eig(A_low)

    # 取前n_modes个最大的（按模）
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx[:n_modes]]
    W = W[:, idx[:n_modes]]

    # DMD模态映射回高维
    modes = V_r.T @ W  # (n_features, n_modes)

    # 归一化
    for i in range(modes.shape[1]):
        norm = np.linalg.norm(modes[:, i])
        if norm > 1e-10:
            modes[:, i] = modes[:, i] / norm

    # 时间系数
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


def visualize_dmd_heatmaps(modes, eigenvalues, mode_name, save_dir, n_show=3):
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

        for boundary in REGION_BOUNDARIES[:-1]:
            axes[i].axhline(boundary, color="black", linewidth=0.8, alpha=0.6)
            axes[i].axvline(boundary, color="black", linewidth=0.8, alpha=0.6)

        if i == 0:
            centers = _region_centers(REGION_BOUNDARIES)
            for name, c in zip(REGION_NAMES, centers):
                axes[i].text(c, -2, name, ha="center", va="bottom",
                           fontsize=6, rotation=45, transform=axes[i].transData)

    plt.suptitle(f"DMD Multi-Patient Basis - {mode_name} mode", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"multi_dmd_heatmap_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_dmd_heatmaps_detailed(modes, eigenvalues, mode_name, save_dir, n_show=5):
    """可视化前n_show个DMD模态，每个模态单独一个大图"""
    n = min(n_show, modes.shape[1])

    for i in range(n):
        fig, ax = plt.subplots(figsize=(6, 5))
        mode = modes[:, i].reshape(341, 341)
        vmax = np.abs(mode).max()
        im = ax.imshow(mode.real, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Mode{i+1} - {mode_name} (λ={eigenvalues[i]:.3f})", fontsize=11)
        ax.set_xlabel("landmark j")
        ax.set_ylabel("landmark i")
        plt.colorbar(im, ax=ax, shrink=0.8)

        for boundary in REGION_BOUNDARIES[:-1]:
            ax.axhline(boundary, color="black", linewidth=0.8, alpha=0.6)
            ax.axvline(boundary, color="black", linewidth=0.8, alpha=0.6)

        centers = _region_centers(REGION_BOUNDARIES)
        for name, c in zip(REGION_NAMES, centers):
            ax.text(c, -2, name, ha="center", va="bottom",
                     fontsize=6, rotation=45, transform=ax.transData)

        plt.tight_layout()
        output_path = save_dir / f"multi_dmd_mode{i+1}_{mode_name}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

    return [save_dir / f"multi_dmd_mode{i+1}_{mode_name}.png" for i in range(n)]


def visualize_eigenvalues(eigenvalues, mode_name, save_dir):
    """可视化特征值分布（复平面）"""
    fig, ax = plt.subplots(figsize=(6, 5))

    # 绘制单位圆
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label="Unit circle")

    # 绘制特征值
    ax.scatter(eigenvalues.real, eigenvalues.imag, s=80, c='red', marker='o', zorder=5)

    # 标注每个特征值
    for i, (re, im) in enumerate(zip(eigenvalues.real, eigenvalues.imag)):
        ax.annotate(f'λ{i+1}', (re, im), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Real part (decay/growth)")
    ax.set_ylabel("Imaginary part (oscillation)")
    ax.set_title(f"Eigenvalue Distribution - {mode_name} mode")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = save_dir / f"multi_dmd_eigenvalues_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_time_coeffs(time_coeffs, eigenvalues, mode_name, patient_info, save_dir, n_show=3):
    """可视化时间系数（取实部）"""
    n = min(n_show, time_coeffs.shape[1])
    n_samples = time_coeffs.shape[0]

    fig, axes = plt.subplots(1, n, figsize=(5*n, 3))
    if n == 1:
        axes = [axes]

    for i in range(n):
        coeff = time_coeffs[:, i].real
        axes[i].plot(range(n_samples), coeff, 'o-', linewidth=1.5, markersize=4)
        axes[i].set_xlabel("Window Index (all patients)")
        axes[i].set_ylabel("Coefficient (real)")
        axes[i].set_title(f"Mode{i+1} (λ={eigenvalues[i]:.3f})")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f"DMD Time Coefficients - {mode_name} mode", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"multi_dmd_timecoef_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_time_coeffs_by_patient(time_coeffs, eigenvalues, mode_name, patient_info, save_dir, n_show=3):
    """按患者分段显示时间系数"""
    n = min(n_show, time_coeffs.shape[1])

    fig, axes = plt.subplots(1, n, figsize=(5*n, 3))
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for i in range(n):
        current_idx = 0
        for pidx, (dataset, subj_id, n_wins) in enumerate(patient_info):
            start = current_idx
            end = current_idx + n_wins
            window_indices = range(start, end)
            coeff = time_coeffs[start:end, i].real
            axes[i].plot(window_indices, coeff, 'o-',
                        color=colors[pidx % len(colors)],
                        linewidth=1.5, markersize=4,
                        label=f"{dataset}/{subj_id}")
            current_idx = end

        axes[i].set_xlabel("Window Index")
        axes[i].set_ylabel("Coefficient (real)")
        axes[i].set_title(f"Mode{i+1} (λ={eigenvalues[i]:.3f})")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            axes[i].legend(fontsize=6, loc='best')

    plt.suptitle(f"DMD Time Coefficients by Patient - {mode_name} mode", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"multi_dmd_timecoef_by_patient_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def analyze_semantic_regions(modes, n_show=5):
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


def analyze_mode_energy(modes, eigenvalues, n_show=5):
    """分析DMD模态的能量"""
    n = min(n_show, modes.shape[1])

    energy_info = []
    for i in range(n):
        mode = modes[:, i]
        mode_energy = np.sum(mode ** 2)
        energy_info.append({
            "mode": i + 1,
            "eigenvalue": complex(eigenvalues[i]),
            "eigenvalue_magnitude": float(np.abs(eigenvalues[i])),
            "spatial_energy": float(mode_energy),
        })

    return energy_info


def main():
    print("=" * 60)
    print("DMD 多患者联合分解 (win5-step5)")
    print("=" * 60)

    # 分别收集IMR和TT患者
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

    # ========== 内存预估 ==========
    def estimate_memory():
        """预估DMD分解所需内存"""
        total_windows = 0
        dataset_stats = {}

        for dataset in ["IMR", "TT"]:
            dataset_path = DATA_ROOT / dataset
            if not dataset_path.exists():
                continue
            count = 0
            for subj_dir in sorted(dataset_path.iterdir()):
                if not subj_dir.is_dir():
                    continue
                win_files = list(subj_dir.glob("win_*_x.npy"))
                n_wins = len(win_files)
                n_diff = n_wins - 1
                if n_diff < MIN_WINDOWS or n_diff > MAX_WINDOWS:
                    continue
                count += n_diff
            dataset_stats[dataset] = {"windows": count}
            total_windows += count

        # 内存计算 (GB): n_windows * 341 * 341 * 4 bytes * 2 (X+Y模态)
        per_window_bytes = 341 * 341 * 4
        total_bytes = total_windows * per_window_bytes * 2
        total_gb = total_bytes / (1024 ** 3)

        return total_gb, total_windows, dataset_stats

    est_gb, est_windows, stats = estimate_memory()
    print(f"\n[内存预估]")
    print(f"  筛选条件: 窗口数 ∈ [{MIN_WINDOWS}, {MAX_WINDOWS}]")
    print(f"  IMR: {stats['IMR']['windows']} 窗口")
    print(f"  TT: {stats['TT']['windows']} 窗口")
    print(f"  总计: {est_windows} 窗口")
    print(f"  预估内存: {est_gb:.2f} GB")
    print(f"  警告阈值: {MEMORY_THRESHOLD_GB:.1f} GB")

    if est_gb > MEMORY_THRESHOLD_GB:
        print(f"\n⚠️  警告: 预估内存 ({est_gb:.1f}GB) 超过阈值 ({MEMORY_THRESHOLD_GB:.1f}GB)")
        print(f"   建议: 降低 MIN_WINDOWS, 提高 MEMORY_THRESHOLD_GB, 或减少 DMD_RANK")

    # 输出目录
    output_dir = OUTPUT_ROOT / "dmd_multi_patient_results"
    output_dir.mkdir(exist_ok=True)

    # 用于存储所有结果
    all_results = {}

    # ========== 分别处理IMR和TT ==========
    for dataset_name, dataset_patients in [("IMR", imr_patients), ("TT", tt_patients)]:
        if len(dataset_patients) == 0:
            print(f"\nNo {dataset_name} patients found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"处理 {dataset_name} 数据集 ({len(dataset_patients)} 患者)")
        print(f"{'='*60}")

        # 加载该数据集所有患者数据
        all_diff_x = []
        all_diff_y = []
        patient_info = []

        # 第一遍扫描：筛选窗口数在[MIN_WINDOWS, MAX_WINDOWS]范围内的患者
        filtered_patients = []
        for dataset, subj_id in tqdm(dataset_patients, desc=f"Scanning {dataset_name}"):
            windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
            if windows_x is None:
                continue
            n_wins = windows_x.shape[0]
            if n_wins < MIN_WINDOWS or n_wins > MAX_WINDOWS:
                continue
            if windows_y is None or windows_x.shape != windows_y.shape:
                continue
            filtered_patients.append((dataset, subj_id))

        print(f"{dataset_name}: 筛选前={len(dataset_patients)}, 筛选后(窗口数{MIN_WINDOWS}-{MAX_WINDOWS})={len(filtered_patients)})")

        # 第二遍加载筛选后的患者数据
        all_diff_x = []
        all_diff_y = []
        patient_info = []

        for dataset, subj_id in tqdm(filtered_patients, desc=f"Loading {dataset_name}"):
            windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj_id)
            n_wins = windows_x.shape[0]

            diff_x = compute_diff_matrix(windows_x)
            diff_y = compute_diff_matrix(windows_y)

            all_diff_x.append(diff_x)
            all_diff_y.append(diff_y)
            patient_info.append((dataset, subj_id, diff_x.shape[0]))

        if len(patient_info) == 0:
            print(f"No valid {dataset_name} patients")
            continue

        total_windows = sum(info[2] for info in patient_info)
        print(f"\n{dataset_name}: {len(patient_info)} patients, {total_windows} diff windows")

        # X模态
        print(f"\n--- {dataset_name} X模态 ---")
        X_list = [d.reshape(d.shape[0], -1) for d in all_diff_x]
        X_stacked = np.vstack(X_list)
        print(f"X stacked shape: {X_stacked.shape}")

        modes_x, eig_x, tc_x = dmd_spatial_projected(X_stacked, n_modes=N_MODES_SHOW, rank=DMD_RANK)
        print(f"X eigenvalues: {eig_x[:5]}")

        energy_x = analyze_mode_energy(modes_x, eig_x)
        print(f"\n[{dataset_name} X模态] 前5模态能量:")
        for e in energy_x:
            print(f"    Mode{e['mode']}: |λ|={e['eigenvalue_magnitude']:.3f}, 能量={e['spatial_energy']:.3f}")

        semantic_x = analyze_semantic_regions(modes_x)
        print(f"\n[{dataset_name} X模态] 前5模态语义区域:")
        for s in semantic_x:
            print(f"    Mode{s['mode']}: 主要={s['dominant_region']} ({s['dominant_contribution']:.1f}%)")

        # Y模态
        print(f"\n--- {dataset_name} Y模态 ---")
        Y_list = [d.reshape(d.shape[0], -1) for d in all_diff_y]
        Y_stacked = np.vstack(Y_list)
        print(f"Y stacked shape: {Y_stacked.shape}")

        modes_y, eig_y, tc_y = dmd_spatial_projected(Y_stacked, n_modes=N_MODES_SHOW, rank=DMD_RANK)
        print(f"Y eigenvalues: {eig_y[:5]}")

        energy_y = analyze_mode_energy(modes_y, eig_y)
        print(f"\n[{dataset_name} Y模态] 前5模态能量:")
        for e in energy_y:
            print(f"    Mode{e['mode']}: |λ|={e['eigenvalue_magnitude']:.3f}, 能量={e['spatial_energy']:.3f}")

        semantic_y = analyze_semantic_regions(modes_y)
        print(f"\n[{dataset_name} Y模态] 前5模态语义区域:")
        for s in semantic_y:
            print(f"    Mode{s['mode']}: 主要={s['dominant_region']} ({s['dominant_contribution']:.1f}%)")

        # ========== 保存主模态（节省后续计算资源） ==========
        # 保存DMD模态和特征值，供后续分析使用
        modes_dir = output_dir / "saved_modes" / dataset_name
        modes_dir.mkdir(parents=True, exist_ok=True)

        np.save(modes_dir / "modes_x.npy", modes_x.astype(np.float32))
        np.save(modes_dir / "eigenvalues_x.npy", eig_x.astype(np.complex64))
        np.save(modes_dir / "time_coeffs_x.npy", tc_x.astype(np.complex64))

        np.save(modes_dir / "modes_y.npy", modes_y.astype(np.float32))
        np.save(modes_dir / "eigenvalues_y.npy", eig_y.astype(np.complex64))
        np.save(modes_dir / "time_coeffs_y.npy", tc_y.astype(np.complex64))

        print(f"\n[{dataset_name}] 模态已保存到: {modes_dir}")

        # 可视化
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        heatmap_x = visualize_dmd_heatmaps(modes_x, eig_x, f"{dataset_name}_X", dataset_dir, n_show=3)
        detailed_x = visualize_dmd_heatmaps_detailed(modes_x, eig_x, f"{dataset_name}_X", dataset_dir, n_show=N_MODES_SHOW)
        eigplot_x = visualize_eigenvalues(eig_x, f"{dataset_name}_X", dataset_dir)
        timecoef_x = visualize_time_coeffs(tc_x, eig_x, f"{dataset_name}_X", patient_info, dataset_dir, n_show=3)
        timecoef_x_by_patient = visualize_time_coeffs_by_patient(tc_x, eig_x, f"{dataset_name}_X", patient_info, dataset_dir, n_show=3)

        heatmap_y = visualize_dmd_heatmaps(modes_y, eig_y, f"{dataset_name}_Y", dataset_dir, n_show=3)
        detailed_y = visualize_dmd_heatmaps_detailed(modes_y, eig_y, f"{dataset_name}_Y", dataset_dir, n_show=N_MODES_SHOW)
        eigplot_y = visualize_eigenvalues(eig_y, f"{dataset_name}_Y", dataset_dir)
        timecoef_y = visualize_time_coeffs(tc_y, eig_y, f"{dataset_name}_Y", patient_info, dataset_dir, n_show=3)
        timecoef_y_by_patient = visualize_time_coeffs_by_patient(tc_y, eig_y, f"{dataset_name}_Y", patient_info, dataset_dir, n_show=3)

        print(f"\n[{dataset_name}] 输出: {dataset_dir}")

        # 保存结果
        all_results[dataset_name] = {
            "n_patients": len(patient_info),
            "total_diff_windows": total_windows,
            "dmd_rank": DMD_RANK,
            "patient_info": [
                {"dataset": p[0], "subj_id": p[1], "n_windows": p[2]} for p in patient_info
            ],
            "X_mode": {
                "stacked_shape": list(X_stacked.shape),
                "eigenvalues": [str(e) for e in eig_x],
                "eigenvalue_magnitudes": [float(np.abs(e)) for e in eig_x],
                "energy": energy_x,
                "semantic_regions": semantic_x
            },
            "Y_mode": {
                "stacked_shape": list(Y_stacked.shape),
                "eigenvalues": [str(e) for e in eig_y],
                "eigenvalue_magnitudes": [float(np.abs(e)) for e in eig_y],
                "energy": energy_y,
                "semantic_regions": semantic_y
            }
        }

    # ========== 保存汇总结果 ==========
    results_file = output_dir / "multi_dmd_by_dataset_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存到: {results_file}")

    # 打印对比汇总
    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name} ({results['n_patients']} patients, {results['total_diff_windows']} windows, rank={results['dmd_rank']}):")
        print(f"  X Mode1: |λ|={results['X_mode']['eigenvalue_magnitudes'][0]:.3f}, dominant: {results['X_mode']['semantic_regions'][0]['dominant_region']}")
        print(f"  Y Mode1: |λ|={results['Y_mode']['eigenvalue_magnitudes'][0]:.3f}, dominant: {results['Y_mode']['semantic_regions'][0]['dominant_region']}")


if __name__ == "__main__":
    main()
