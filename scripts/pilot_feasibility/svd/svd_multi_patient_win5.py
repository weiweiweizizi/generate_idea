"""
SVD 多患者联合分解 - win5-step5 TT数据
参考 svd_multi_patient.py 的可视化风格
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win5-step5"
OUTPUT_DIR = REPO_ROOT / "outputs" / "pilot_feasibility" / "svd" / "win5-step5" / "svd_multi_patient_results_win5"
OUTPUT_DIR.mkdir(exist_ok=True)
N_COMPONENTS = 5
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

    windows_x = [np.load(f).astype(np.float32) for f in win_x_files]
    windows_y = [np.load(f).astype(np.float32) for f in win_y_files]

    return np.stack(windows_x, axis=0), np.stack(windows_y, axis=0)


def compute_diff_matrix(windows):
    """计算前后差分矩阵"""
    diffs = []
    for i in range(1, len(windows)):
        diff = windows[i] - windows[i-1]
        diffs.append(diff.astype(np.float32))
    return np.stack(diffs, axis=0)


def _region_centers(boundaries):
    """计算每个区域的中心位置"""
    centers = []
    prev = 0
    for b in boundaries:
        centers.append((prev + b) / 2.0)
        prev = b
    return centers


def visualize_basis_heatmaps(Vt, mode_name, save_dir, n_show=3):
    """可视化前n_show个基的热图，带区域分割线"""
    n = min(n_show, Vt.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        basis = Vt[i].reshape(341, 341)
        vmax = np.abs(basis).max()
        im = axes[i].imshow(basis, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"PC{i+1}")
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

    plt.suptitle(f"SVD Multi-Patient Basis - {mode_name} mode (win5-step5 TT)", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"svd_heatmap_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_basis_heatmaps_detailed(Vt, mode_name, save_dir, n_show=5):
    """可视化前n_show个基的热图，每个基单独一个大图"""
    n = min(n_show, Vt.shape[0])

    for i in range(n):
        fig, ax = plt.subplots(figsize=(6, 5))
        basis = Vt[i].reshape(341, 341)
        vmax = np.abs(basis).max()
        im = ax.imshow(basis, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"PC{i+1} - {mode_name} (win5-step5 TT)", fontsize=11)
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
        output_path = save_dir / f"svd_pc{i+1}_{mode_name}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

    return [save_dir / f"svd_pc{i+1}_{mode_name}.png" for i in range(n)]


def visualize_time_coefficients(U, sigma, mode_name, patient_info, save_dir, n_show=3):
    """可视化时间系数 U 随窗口/患者的变化"""
    n = min(n_show, U.shape[1])
    n_windows = U.shape[0]

    fig, axes = plt.subplots(1, n, figsize=(5*n, 3))
    if n == 1:
        axes = [axes]

    for i in range(n):
        axes[i].plot(range(n_windows), U[:, i], 'o-', linewidth=1.5, markersize=4)
        axes[i].set_xlabel("Window Index (all patients)")
        axes[i].set_ylabel("Coefficient")
        axes[i].set_title(f"PC{i+1} (σ={sigma[i]:.2f})")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f"Time Coefficients - {mode_name} mode (win5-step5 TT)", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"svd_timecoef_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def visualize_time_coefficients_by_patient(U, sigma, mode_name, patient_info, save_dir, n_show=3):
    """按患者分段显示时间系数"""
    n = min(n_show, U.shape[1])

    fig, axes = plt.subplots(1, n, figsize=(5*n, 3))
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for i in range(n):
        current_idx = 0
        for pidx, (subj_id, n_wins) in enumerate(patient_info):
            start = current_idx
            end = current_idx + n_wins
            window_indices = range(start, end)
            axes[i].plot(window_indices, U[start:end, i], 'o-',
                        color=colors[pidx % len(colors)],
                        linewidth=1.5, markersize=4,
                        label=f"TT/{subj_id}")
            current_idx = end

        axes[i].set_xlabel("Window Index")
        axes[i].set_ylabel("Coefficient")
        axes[i].set_title(f"PC{i+1} (σ={sigma[i]:.2f})")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            axes[i].legend(fontsize=5, loc='best', ncol=2)

    plt.suptitle(f"Time Coefficients by Patient - {mode_name} mode (win5-step5 TT)", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"svd_timecoef_by_patient_{mode_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def analyze_semantic_regions(Vt, n_show=5):
    """分析每个基向量主要激活的语义区域"""
    n = min(n_show, Vt.shape[0])
    results = []

    for i in range(n):
        basis = Vt[i].reshape(341, 341)
        total_abs = np.sum(np.abs(basis))

        region_contributions = {}
        prev_boundary = 0
        for region_name, boundary in zip(REGION_NAMES, REGION_BOUNDARIES):
            indices = list(range(prev_boundary, boundary))
            sub = basis[np.ix_(indices, indices)]
            region_abs = np.sum(np.abs(sub))
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


def analyze_basis_energy(Vt, singular_values, n_show=5):
    """分析基向量的能量分布"""
    n = min(n_show, Vt.shape[0])
    total_energy = np.sum(singular_values ** 2)
    energy_info = []

    for i in range(n):
        basis = Vt[i].reshape(341, 341)
        basis_energy = singular_values[i] ** 2
        energy_ratio = basis_energy / total_energy * 100
        energy_info.append({
            "pc": i + 1,
            "singular_value": float(singular_values[i]),
            "energy_ratio": float(energy_ratio)
        })

    return energy_info


def main():
    print("=" * 60)
    print("SVD 多患者联合分解 - win5-step5 TT数据")
    print("=" * 60)

    # 收集TT患者
    tt_patients = sorted([d.name for d in (DATA_ROOT / "TT").iterdir() if d.is_dir()])
    print(f"TT patients found: {len(tt_patients)}")

    # 加载数据
    all_diff_x = []
    all_diff_y = []
    patient_info = []

    for subj_id in tqdm(tt_patients, desc="Loading TT"):
        windows_x, windows_y = load_patient_windows(DATA_ROOT / "TT", subj_id)
        if windows_x is None:
            continue

        n_wins = windows_x.shape[0]
        if n_wins < 3:
            continue

        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)

        all_diff_x.append(diff_x)
        all_diff_y.append(diff_y)
        patient_info.append((subj_id, diff_x.shape[0]))

    print(f"Valid patients: {len(patient_info)}")
    total_windows = sum(p[1] for p in patient_info)
    print(f"Total diff windows: {total_windows}")

    # X模态
    print("\n--- X模态 ---")
    X_list = [d.reshape(d.shape[0], -1) for d in all_diff_x]
    X_stacked = np.vstack(X_list).astype(np.float32)
    print(f"X stacked shape: {X_stacked.shape}")

    svd_x = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    U_x = svd_x.fit_transform(X_stacked)
    sigma_x = svd_x.singular_values_
    Vt_x = svd_x.components_

    print(f"X singular values: {sigma_x[:5]}")

    energy_x = analyze_basis_energy(Vt_x, sigma_x)
    print(f"\n[X模态] 前5基能量:")
    for e in energy_x:
        print(f"    PC{e['pc']}: σ={e['singular_value']:.3f}, 能量={e['energy_ratio']:.1f}%")

    semantic_x = analyze_semantic_regions(Vt_x)
    print(f"\n[X模态] 前5基语义区域:")
    for s in semantic_x:
        print(f"    PC{s['pc']}: 主要={s['dominant_region']} ({s['dominant_contribution']:.1f}%)")

    # Y模态
    print("\n--- Y模态 ---")
    Y_list = [d.reshape(d.shape[0], -1) for d in all_diff_y]
    Y_stacked = np.vstack(Y_list).astype(np.float32)
    print(f"Y stacked shape: {Y_stacked.shape}")

    svd_y = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    U_y = svd_y.fit_transform(Y_stacked)
    sigma_y = svd_y.singular_values_
    Vt_y = svd_y.components_

    print(f"Y singular values: {sigma_y[:5]}")

    energy_y = analyze_basis_energy(Vt_y, sigma_y)
    print(f"\n[Y模态] 前5基能量:")
    for e in energy_y:
        print(f"    PC{e['pc']}: σ={e['singular_value']:.3f}, 能量={e['energy_ratio']:.1f}%")

    semantic_y = analyze_semantic_regions(Vt_y)
    print(f"\n[Y模态] 前5基语义区域:")
    for s in semantic_y:
        print(f"    PC{s['pc']}: 主要={s['dominant_region']} ({s['dominant_contribution']:.1f}%)")

    # 可视化
    vis_dir = OUTPUT_DIR / "visualization"
    vis_dir.mkdir(exist_ok=True)

    heatmap_x = visualize_basis_heatmaps(Vt_x, "X", vis_dir, n_show=3)
    detailed_x = visualize_basis_heatmaps_detailed(Vt_x, "X", vis_dir, n_show=N_COMPONENTS)
    timecoef_x = visualize_time_coefficients(U_x, sigma_x, "X", patient_info, vis_dir, n_show=3)
    timecoef_x_by_patient = visualize_time_coefficients_by_patient(U_x, sigma_x, "X", patient_info, vis_dir, n_show=3)

    heatmap_y = visualize_basis_heatmaps(Vt_y, "Y", vis_dir, n_show=3)
    detailed_y = visualize_basis_heatmaps_detailed(Vt_y, "Y", vis_dir, n_show=N_COMPONENTS)
    timecoef_y = visualize_time_coefficients(U_y, sigma_y, "Y", patient_info, vis_dir, n_show=3)
    timecoef_y_by_patient = visualize_time_coefficients_by_patient(U_y, sigma_y, "Y", patient_info, vis_dir, n_show=3)

    print(f"\n可视化输出: {vis_dir}")

    # 保存结果
    np.save(vis_dir / "U_x.npy", U_x.astype(np.float32))
    np.save(vis_dir / "U_y.npy", U_y.astype(np.float32))
    np.save(vis_dir / "Vt_x.npy", Vt_x.astype(np.float32))
    np.save(vis_dir / "Vt_y.npy", Vt_y.astype(np.float32))
    np.save(vis_dir / "sigma_x.npy", sigma_x.astype(np.float32))
    np.save(vis_dir / "sigma_y.npy", sigma_y.astype(np.float32))

    results = {
        "n_patients": len(patient_info),
        "total_windows": total_windows,
        "patient_info": [{"subj_id": p[0], "n_windows": p[1]} for p in patient_info],
        "X_mode": {
            "stacked_shape": list(X_stacked.shape),
            "singular_values": sigma_x.tolist(),
            "energy": energy_x,
            "semantic_regions": semantic_x
        },
        "Y_mode": {
            "stacked_shape": list(Y_stacked.shape),
            "singular_values": sigma_y.tolist(),
            "energy": energy_y,
            "semantic_regions": semantic_y
        }
    }

    results_file = vis_dir / "svd_win5_tt_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存: {results_file}")

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"\nX PC1: {semantic_x[0]['dominant_region']} ({semantic_x[0]['dominant_contribution']:.1f}%), σ={sigma_x[0]:.2f}")
    print(f"Y PC1: {semantic_y[0]['dominant_region']} ({semantic_y[0]['dominant_contribution']:.1f}%), σ={sigma_y[0]:.2f}")


if __name__ == "__main__":
    main()
