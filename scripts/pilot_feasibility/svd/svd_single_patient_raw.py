"""
SVD 单患者验证脚本 - 原始距离矩阵版本（非差分）
对10个患者分别做SVD分解（不取差分）
用于对比：原始距离矩阵 vs 差分距离矩阵

与 svd_single_patient.py 的区别：
- 不调用 compute_diff_matrix()
- 直接对原始距离矩阵做SVD
- 可视化和结果汇报保持一致
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win20-step20"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "pilot_feasibility" / "svd" / "win20-step20"
N_COMPONENTS = 10
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


def run_svd(patient_data, n_components=10):
    """对患者数据做SVD"""
    n_samples = patient_data.shape[0]
    X = patient_data.reshape(n_samples, -1)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    U = svd.fit_transform(X)
    Vt = svd.components_

    X_reconstructed = svd.inverse_transform(U)
    recon_error = np.sum((X_reconstructed - X) ** 2)

    return U, svd.singular_values_, Vt, recon_error


def visualize_basis_heatmaps(Vt, subj_id, dataset, save_dir, n_show=3):
    """可视化前n_show个基的热图，带区域分割线"""
    n = min(n_show, Vt.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        basis = Vt[i].reshape(341, 341)
        vmax = np.abs(basis).max()
        im = axes[i].imshow(basis, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"PC{i+1} (σ={i+1:.2f})")
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

    plt.suptitle(f"SVD Basis (RAW) - {dataset}/{subj_id}", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"{dataset}_{subj_id}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _region_centers(boundaries):
    """计算每个区域的中心位置"""
    centers = []
    prev = 0
    for b in boundaries:
        centers.append((prev + b) / 2.0)
        prev = b
    return centers


def visualize_time_coefficients(U, singular_values, subj_id, dataset, save_dir, n_show=3):
    """可视化时间系数 U 随窗口的变化"""
    n = min(n_show, U.shape[1])
    n_windows = U.shape[0]

    fig, axes = plt.subplots(1, n, figsize=(5*n, 3))
    if n == 1:
        axes = [axes]

    window_indices = range(1, n_windows + 1)

    for i in range(n):
        axes[i].plot(window_indices, U[:, i], 'o-', linewidth=2, markersize=6)
        axes[i].set_xlabel("Window Index")
        axes[i].set_ylabel("Coefficient")
        axes[i].set_title(f"PC{i+1} (σ={singular_values[i]:.2f})")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f"Time Coefficients (RAW) - {dataset}/{subj_id}", fontsize=12)
    plt.tight_layout()

    output_path = save_dir / f"{dataset}_{subj_id}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def analyze_basis_energy(Vt, singular_values, n_show=3):
    """分析基向量的能量分布"""
    n = min(n_show, Vt.shape[0])
    energy_info = []

    total_energy = np.sum(singular_values ** 2)

    for i in range(n):
        basis = Vt[i].reshape(341, 341)
        basis_energy = singular_values[i] ** 2
        energy_ratio = basis_energy / total_energy * 100
        energy_info.append({
            "pc": i + 1,
            "singular_value": singular_values[i],
            "energy_ratio": energy_ratio
        })

    return energy_info


# 关键点区域定义
REGION_BOUNDARIES = [18, 38, 100, 130, 140, 188, 233, 307, 331, 341]
REGION_NAMES = [
    "forehead", "eyebrow", "eyehole", "eye_contour",
    "eye_iris", "nose", "around_mouth", "mouth", "cheek", "jaw"
]


def analyze_semantic_regions(Vt, n_show=3):
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


def main():
    print("=" * 60)
    print("SVD 单患者验证 - 原始距离矩阵（非差分）版本")
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

    # 随机选择10个患者
    np.random.seed(RANDOM_STATE)
    selected_indices = np.random.choice(len(patients), size=min(10, len(patients)), replace=False)
    selected_patients = [patients[i] for i in selected_indices]

    print(f"Selected patients: {selected_patients}")

    # 结果保存目录
    output_dir = OUTPUT_ROOT / "svd_single_patient_raw_results"
    output_dir.mkdir(exist_ok=True)

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
        print(f"  Raw shape: {windows_x.shape}")

        # 不做差分，直接使用原始距离矩阵
        if windows_x.shape[0] < 3:
            print(f"  Skipping: not enough windows for analysis")
            continue

        # 对x和y分别做SVD（原始矩阵，非差分）
        try:
            U_x, sigma_x, Vt_x, err_x = run_svd(windows_x, N_COMPONENTS)
            U_y, sigma_y, Vt_y, err_y = run_svd(windows_y, N_COMPONENTS)

            print(f"\n  [X模态] 奇异值: {sigma_x[:5]}")
            print(f"  [Y模态] 奇异值: {sigma_y[:5]}")

            # 分析能量分布
            energy_x = analyze_basis_energy(Vt_x, sigma_x)
            energy_y = analyze_basis_energy(Vt_y, sigma_y)

            # 分析语义区域
            semantic_x = analyze_semantic_regions(Vt_x)
            semantic_y = analyze_semantic_regions(Vt_y)

            print(f"\n  [X模态] 前3基能量占比:")
            for e in energy_x:
                print(f"      PC{e['pc']}: σ={e['singular_value']:.3f}, 能量={e['energy_ratio']:.1f}%")

            print(f"\n  [X模态] 前3基语义区域:")
            for s in semantic_x:
                print(f"      PC{s['pc']}: 主要={s['dominant_region']} ({s['dominant_contribution']:.1f}%)")

            print(f"\n  [Y模态] 前3基能量占比:")
            for e in energy_y:
                print(f"      PC{e['pc']}: σ={e['singular_value']:.3f}, 能量={e['energy_ratio']:.1f}%")

            print(f"\n  [Y模态] 前3基语义区域:")
            for s in semantic_y:
                print(f"      PC{s['pc']}: 主要={s['dominant_region']} ({s['dominant_contribution']:.1f}%)")

            # 可视化 - 热图
            (output_dir / "x_heatmaps").mkdir(exist_ok=True)
            heatmap_x = visualize_basis_heatmaps(Vt_x, subj_id, dataset, output_dir / "x_heatmaps", n_show=3)
            (output_dir / "y_heatmaps").mkdir(exist_ok=True)
            heatmap_y = visualize_basis_heatmaps(Vt_y, subj_id, dataset, output_dir / "y_heatmaps", n_show=3)

            # 可视化 - 时间系数
            (output_dir / "x_timecoef").mkdir(exist_ok=True)
            timecoef_x = visualize_time_coefficients(U_x, sigma_x, subj_id, dataset, output_dir / "x_timecoef", n_show=3)
            (output_dir / "y_timecoef").mkdir(exist_ok=True)
            timecoef_y = visualize_time_coefficients(U_y, sigma_y, subj_id, dataset, output_dir / "y_timecoef", n_show=3)

            print(f"\n  [输出]")
            print(f"      热图(X): {heatmap_x}")
            print(f"      热图(Y): {heatmap_y}")
            print(f"      时间系数(X): {timecoef_x}")
            print(f"      时间系数(Y): {timecoef_y}")

            # 时间系数统计
            print(f"\n  [时间系数统计]")
            print(f"      X PC1 范围: [{U_x[:, 0].min():.3f}, {U_x[:, 0].max():.3f}], 均值: {U_x[:, 0].mean():.3f}")
            print(f"      Y PC1 范围: [{U_y[:, 0].min():.3f}, {U_y[:, 0].max():.3f}], 均值: {U_y[:, 0].mean():.3f}")

            # 重构误差
            print(f"\n  [重构误差]")
            print(f"      X 重构误差: {err_x:.2f}")
            print(f"      Y 重构误差: {err_y:.2f}")

            all_results.append({
                "dataset": dataset,
                "subj_id": subj_id,
                "n_windows": n_windows,
                "sigma_x": sigma_x.tolist(),
                "sigma_y": sigma_y.tolist(),
                "energy_x": energy_x,
                "energy_y": energy_y,
                "semantic_x": semantic_x,
                "semantic_y": semantic_y,
                "U_x_range": [float(U_x[:, 0].min()), float(U_x[:, 0].max()), float(U_x[:, 0].mean())],
                "U_y_range": [float(U_y[:, 0].min()), float(U_y[:, 0].max()), float(U_y[:, 0].mean())],
                "recon_error_x": float(err_x),
                "recon_error_y": float(err_y),
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

    # 统计X模态PC1能量占比
    x_pc1_energies = [r["energy_x"][0]["energy_ratio"] for r in all_results if len(r["energy_x"]) > 0]
    y_pc1_energies = [r["energy_y"][0]["energy_ratio"] for r in all_results if len(r["energy_y"]) > 0]

    print(f"\nX模态 PC1 能量占比: {np.mean(x_pc1_energies):.1f}% ± {np.std(x_pc1_energies):.1f}%")
    print(f"Y模态 PC1 能量占比: {np.mean(y_pc1_energies):.1f}% ± {np.std(y_pc1_energies):.1f}%")

    # 统计dominant region分布
    from collections import Counter
    x_domains = [r["semantic_x"][0]["dominant_region"] for r in all_results if len(r["semantic_x"]) > 0]
    y_domains = [r["semantic_y"][0]["dominant_region"] for r in all_results if len(r["semantic_y"]) > 0]

    print(f"\nX模态 PC1 dominant region分布: {Counter(x_domains)}")
    print(f"Y模态 PC1 dominant region分布: {Counter(y_domains)}")

    # 保存结果
    results_file = output_dir / "results_summary.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n结果已保存到: {output_dir}")
    print(f"汇总文件: {results_file}")

    print("\n" + "=" * 60)
    print("生成的文件:")
    print("  - x_heatmaps/*.png: X模态前3个空间基的热图")
    print("  - y_heatmaps/*.png: Y模态前3个空间基的热图")
    print("  - x_timecoef/*.png: X模态时间系数曲线")
    print("  - y_timecoef/*.png: Y模态时间系数曲线")
    print("=" * 60)


if __name__ == "__main__":
    main()
