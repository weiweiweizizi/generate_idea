"""
SVD 多患者分组联合分解
- 按 by_side, by_severity, by_source 分组
- 提取每组 PC1~PC3 主模态
- 保存 npy 和 png
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win20-step20"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "pilot_feasibility" / "svd" / "win20-step20"
N_COMPONENTS = 10
RANDOM_STATE = 42

# 语义区域边界
REGION_BOUNDARIES = [18, 38, 100, 130, 140, 188, 233, 307, 331, 341]
REGION_NAMES = [
    "forehead", "eyebrow", "eyehole", "eye_contour",
    "eye_iris", "nose", "around_mouth", "mouth", "cheek", "jaw"
]


def list_available_svd_datasets():
    """返回已生成 <DATASET>-SVD 结果的数据集名。"""
    datasets = []
    for path in sorted(DATA_ROOT.iterdir()):
        if not path.is_dir() or not path.name.endswith("-SVD"):
            continue
        if (path / "metadata.csv").exists():
            datasets.append(path.name[:-4])
    return datasets


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


def _region_centers(boundaries):
    """计算每个区域的中心位置"""
    centers = []
    prev = 0
    for b in boundaries:
        centers.append((prev + b) / 2.0)
        prev = b
    return centers


def visualize_pc_heatmap(Vt_pc, pc_index, mode_name, save_path):
    """可视化指定 PC 基的热图"""
    fig, ax = plt.subplots(figsize=(6, 5))
    basis = Vt_pc.reshape(341, 341)
    vmax = np.abs(basis).max()
    im = ax.imshow(basis, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(f"PC{pc_index} - {mode_name}", fontsize=11)
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
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


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


def load_metadata():
    """加载所有已生成 SVD 结果的数据集 metadata。"""
    frames = []
    for dataset in list_available_svd_datasets():
        meta = pd.read_csv(DATA_ROOT / f"{dataset}-SVD" / "metadata.csv")
        meta["dataset"] = dataset
        frames.append(meta)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_side_group(row):
    """根据 score 和 side 判断侧别分组"""
    if row.get("dataset") == "XW":
        return None

    score = row['score']
    side = row['side']

    if score < 0:
        return None
    if score == 0:
        return 'bilateral_normal'
    elif score != 0 and side == 0:
        return 'left_affected'
    elif score != 0 and side == 1:
        return 'right_affected'
    return None


def get_severity_group(row):
    """根据 score 判断严重程度分组"""
    if row.get("dataset") == "XW":
        return None

    score = row['score']
    if score < 0:
        return None
    if score == 0:
        return 'normal'
    elif score == 1:
        return 'mild'
    elif score == 2:
        return 'severe'
    return None


def group_patients_by_mode(metadata, mode):
    """根据分组模式对患者进行分组"""
    groups = {}

    for _, row in metadata.iterrows():
        subj = str(row['subj'])

        if mode == 'by_side':
            group = get_side_group(row)
        elif mode == 'by_severity':
            group = get_severity_group(row)
        elif mode == 'by_source':
            group = row['dataset']
        else:
            continue

        if group is None:
            continue

        subj_raw = str(row["subj"])
        subj_padded = subj_raw.zfill(5)
        dataset_dir = DATA_ROOT / row["dataset"]
        if (dataset_dir / subj_raw).exists():
            subj_name = subj_raw
        elif (dataset_dir / subj_padded).exists():
            subj_name = subj_padded
        else:
            subj_name = subj_raw

        if group not in groups:
            groups[group] = []
        groups[group].append({
            'subj': subj_name,
            'dataset': row['dataset'],
            'score': row['score'],
            'side': row['side']
        })

    return groups


def run_grouped_svd(patients, group_name, mode_name, save_dir):
    """对一组患者进行联合 SVD 分解"""
    all_diff_x = []
    all_diff_y = []
    patient_info = []

    for p in patients:
        dataset = p['dataset']
        subj = p['subj']

        windows_x, windows_y = load_patient_windows(DATA_ROOT / dataset, subj)
        if windows_x is None:
            continue

        n_wins = windows_x.shape[0]
        if n_wins < 3:
            continue

        if windows_y is None or windows_x.shape != windows_y.shape:
            continue

        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)

        all_diff_x.append(diff_x)
        all_diff_y.append(diff_y)
        patient_info.append({
            'dataset': dataset,
            'subj': subj,
            'n_windows': diff_x.shape[0]
        })

    if len(patient_info) == 0:
        print(f"  [WARNING] No valid patients in group {group_name}")
        return None

    total_windows = sum(p['n_windows'] for p in patient_info)
    print(f"  {group_name}: {len(patient_info)} patients, {total_windows} diff windows")

    # X 模态
    X_list = [d.reshape(d.shape[0], -1) for d in all_diff_x]
    X_stacked = np.vstack(X_list)

    svd_x = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    U_x = svd_x.fit_transform(X_stacked)
    sigma_x = svd_x.singular_values_
    Vt_x = svd_x.components_

    # Y 模态
    Y_list = [d.reshape(d.shape[0], -1) for d in all_diff_y]
    Y_stacked = np.vstack(Y_list)

    svd_y = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    U_y = svd_y.fit_transform(Y_stacked)
    sigma_y = svd_y.singular_values_
    Vt_y = svd_y.components_

    # 分析
    energy_x = analyze_basis_energy(Vt_x, sigma_x)
    semantic_x = analyze_semantic_regions(Vt_x)
    energy_y = analyze_basis_energy(Vt_y, sigma_y)
    semantic_y = analyze_semantic_regions(Vt_y)

    # 保存 PC1~PC3 npy 和 heatmap
    for i in range(3):
        np.save(save_dir / f"PC{i+1}_x.npy", Vt_x[i].reshape(341, 341))
        np.save(save_dir / f"PC{i+1}_y.npy", Vt_y[i].reshape(341, 341))

        visualize_pc_heatmap(Vt_x[i], i + 1, f"{group_name}_X",
                             save_dir / f"PC{i+1}_heatmap_X.png")
        visualize_pc_heatmap(Vt_y[i], i + 1, f"{group_name}_Y",
                             save_dir / f"PC{i+1}_heatmap_Y.png")

    for i in range(3):
        print(f"    X PC{i+1}: σ={sigma_x[i]:.3f}, energy={energy_x[i]['energy_ratio']:.1f}%, dominant={semantic_x[i]['dominant_region']}")
    for i in range(3):
        print(f"    Y PC{i+1}: σ={sigma_y[i]:.3f}, energy={energy_y[i]['energy_ratio']:.1f}%, dominant={semantic_y[i]['dominant_region']}")

    return {
        'n_patients': len(patient_info),
        'total_diff_windows': total_windows,
        'patient_info': patient_info,
        'X_mode': {
            'stacked_shape': list(X_stacked.shape),
            'singular_values': sigma_x.tolist(),
            'explained_variance_ratio': svd_x.explained_variance_ratio_.tolist(),
            'energy': energy_x,
            'semantic_regions': semantic_x
        },
        'Y_mode': {
            'stacked_shape': list(Y_stacked.shape),
            'singular_values': sigma_y.tolist(),
            'explained_variance_ratio': svd_y.explained_variance_ratio_.tolist(),
            'energy': energy_y,
            'semantic_regions': semantic_y
        }
    }


def main():
    print("=" * 60)
    print("SVD 多患者分组联合分解")
    print("=" * 60)

    # 加载 metadata
    metadata = load_metadata()
    print(f"Total patients in metadata: {len(metadata)}")

    # 输出目录
    output_dir = OUTPUT_ROOT / "svd_multi_patient_grouped_results"
    output_dir.mkdir(exist_ok=True)

    # 分组模式
    grouping_modes = ['by_side', 'by_severity', 'by_source']

    all_results = {}

    for mode in grouping_modes:
        print(f"\n{'='*60}")
        print(f"分组模式: {mode}")
        print("=" * 60)

        groups = group_patients_by_mode(metadata, mode)
        print(f"Groups found: {list(groups.keys())}")

        mode_dir = output_dir / mode
        mode_dir.mkdir(exist_ok=True)

        mode_results = {}

        for group_name, patients in groups.items():
            print(f"\n--- {group_name} ---")
            group_dir = mode_dir / group_name
            group_dir.mkdir(exist_ok=True)

            result = run_grouped_svd(patients, group_name, f"{mode}_{group_name}", group_dir)

            if result is not None:
                # 保存该组的 results.json
                with open(group_dir / "results.json", "w") as f:
                    json.dump(result, f, indent=2)
                mode_results[group_name] = result

        all_results[mode] = mode_results

    # 保存汇总结果
    results_file = output_dir / "all_grouped_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("汇总")
    print("=" * 60)
    for mode, mode_results in all_results.items():
        print(f"\n[{mode}]")
        for group_name, result in mode_results.items():
            info = f"  {group_name}: {result['n_patients']}p"
            for i in range(3):
                x_dom = result['X_mode']['semantic_regions'][i]['dominant_region']
                x_en = result['X_mode']['energy'][i]['energy_ratio']
                y_dom = result['Y_mode']['semantic_regions'][i]['dominant_region']
                y_en = result['Y_mode']['energy'][i]['energy_ratio']
                info += f" | PC{i+1} X={x_dom}({x_en:.1f}%) Y={y_dom}({y_en:.1f}%)"
            print(info)

    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
