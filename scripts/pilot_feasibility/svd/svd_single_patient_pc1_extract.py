"""
提取所有患者的单患者SVD PC1~PC3主模态

为每个患者执行单患者SVD分解，提取PC1~PC3空间基（341×341）并保存。
输出目录结构:
    data/win20-step20/<DATASET>-SVD/

每个患者目录包含:
    - PC{i}_x.npy: X方向PC{i}空间基 (341, 341), i=1,2,3
    - PC{i}_y.npy: Y方向PC{i}空间基 (341, 341), i=1,2,3
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
from tqdm import tqdm
import json
import csv

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win20-step20"
RANDOM_STATE = 42


def list_available_datasets():
    """返回 data/win20-step20 下可用的原始数据集目录名。"""
    datasets = []
    for path in sorted(DATA_ROOT.iterdir()):
        if not path.is_dir() or path.name.endswith("-SVD"):
            continue
        if (path / "config.json").exists():
            datasets.append(path.name)
    return datasets


def load_patient_windows(dataset_path, subj_id):
    """加载单个患者所有窗口的x和y矩阵"""
    subj_path = dataset_path / subj_id
    if not subj_path.exists():
        return None, None, 0

    win_x_files = sorted(subj_path.glob("win_*_x.npy"))
    win_y_files = sorted(subj_path.glob("win_*_y.npy"))

    if len(win_x_files) == 0:
        return None, None, 0

    windows_x = [np.load(f) for f in win_x_files]
    windows_y = [np.load(f) for f in win_y_files]

    # 从文件名提取总帧数: win_{idx}_{x/y}.npy -> idx*step + window
    # 由于step=window=20, 总帧数 = (max_idx + 1) * 20
    max_idx = max(int(f.stem.split('_')[1]) for f in win_x_files)
    total_frames = (max_idx + 1) * 20

    return np.array(windows_x), np.array(windows_y), total_frames


def compute_diff_matrix(windows):
    """计算前后差分矩阵"""
    diffs = []
    for i in range(1, len(windows)):
        diff = windows[i] - windows[i-1]
        diffs.append(diff)
    return np.array(diffs)


def run_svd_single_pc(patient_data, n_components=3):
    """对患者数据做SVD，提取PC1~PC3"""
    n_samples = patient_data.shape[0]
    X = patient_data.reshape(n_samples, -1)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    U = svd.fit_transform(X)
    Vt = svd.components_  # shape: (n_components, 341*341)
    singular_values = svd.singular_values_

    ratios = [float(r) for r in svd.explained_variance_ratio_]

    components = [Vt[i].reshape(341, 341) for i in range(n_components)]
    return components, singular_values, ratios


def main():
    print("=" * 60)
    print("提取所有患者单患者SVD PC1主模态")
    print("=" * 60)

    datasets = list_available_datasets()
    print(f"Datasets found: {datasets}")

    # 创建输出目录
    for dataset in datasets:
        output_dir = DATA_ROOT / f"{dataset}-SVD"
        output_dir.mkdir(exist_ok=True)
        print(f"输出目录: {output_dir}")

    # 收集患者列表
    patients = []
    patient_metadata = {}  # {(dataset, subj_id): {side, score, label_5class}}

    for dataset in datasets:
        dataset_path = DATA_ROOT / dataset
        metadata_path = dataset_path / "metadata.csv"

        # 读取metadata获取患者临床信息
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj = row['subj']
                key = (dataset, subj)
                if key not in patient_metadata:
                    patient_metadata[key] = {
                        'side': row['side'],
                        'score': row['score'],
                        'label_5class': row['label_5class']
                    }

        # 获取患者目录列表
        for subj_dir in sorted(dataset_path.iterdir()):
            if subj_dir.is_dir():
                patients.append((dataset, subj_dir.name))

    print(f"总患者数: {len(patients)}")

    # 处理每个患者
    all_results = []

    for dataset, subj_id in tqdm(patients, desc="Processing patients"):
        # 加载数据
        windows_x, windows_y, total_frames = load_patient_windows(
            DATA_ROOT / dataset, subj_id
        )

        if windows_x is None:
            print(f"  跳过 {dataset}/{subj_id}: 无数据")
            continue

        n_windows = windows_x.shape[0]

        # 计算差分
        diff_x = compute_diff_matrix(windows_x)
        diff_y = compute_diff_matrix(windows_y)

        if diff_x.shape[0] < 1:
            print(f"  跳过 {dataset}/{subj_id}: 窗口数不足")
            continue

        # SVD分解
        try:
            comps_x, sigmas_x, ratios_x = run_svd_single_pc(diff_x)
            comps_y, sigmas_y, ratios_y = run_svd_single_pc(diff_y)
        except Exception as e:
            print(f"  错误 {dataset}/{subj_id}: {e}")
            continue

        # 保存PC1~PC3矩阵
        output_dir = DATA_ROOT / f"{dataset}-SVD" / subj_id
        output_dir.mkdir(exist_ok=True)

        for i, (cx, cy) in enumerate(zip(comps_x, comps_y)):
            np.save(output_dir / f"PC{i+1}_x.npy", cx)
            np.save(output_dir / f"PC{i+1}_y.npy", cy)

        # 获取患者临床信息
        meta = patient_metadata.get((dataset, subj_id), {
            'side': '-1', 'score': '0', 'label_5class': '0'
        })

        # 记录结果
        all_results.append({
            'dataset': dataset,
            'subj_id': subj_id,
            'n_windows': n_windows,
            'total_frames': total_frames,
            'side': meta['side'],
            'score': meta['score'],
            'label_5class': meta['label_5class'],
            **{f'sigma_x_{i+1}': float(s) for i, s in enumerate(sigmas_x)},
            **{f'sigma_y_{i+1}': float(s) for i, s in enumerate(sigmas_y)},
            **{f'pc{i+1}_ratio_x': r for i, r in enumerate(ratios_x)},
            **{f'pc{i+1}_ratio_y': r for i, r in enumerate(ratios_y)},
        })

    # 生成config.json
    for dataset in datasets:
        source_config = DATA_ROOT / dataset / "config.json"
        output_config = DATA_ROOT / f"{dataset}-SVD" / "config.json"

        with open(source_config, 'r') as f:
            config = json.load(f)

        # 添加SVD相关信息
        config['decomposition'] = 'SVD'
        config['n_components'] = 3
        config['component_stored'] = 'PC1-PC3'

        with open(output_config, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"已生成 config.json: {output_config}")

    # 生成metadata.csv
    for dataset in datasets:
        output_metadata = DATA_ROOT / f"{dataset}-SVD" / "metadata.csv"

        dataset_results = [r for r in all_results if r['dataset'] == dataset]

        with open(output_metadata, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['subj', 'window_idx', 'start_frame', 'end_frame',
                           'side', 'score', 'label_5class', 'matrix_size',
                           'n_windows',
                           'pc1_ratio_x', 'pc2_ratio_x', 'pc3_ratio_x',
                           'pc1_ratio_y', 'pc2_ratio_y', 'pc3_ratio_y'])

            for r in dataset_results:
                writer.writerow([
                    r['subj_id'],
                    -1,  # window_idx = -1 表示单患者SVD
                    0,   # start_frame
                    r['total_frames'],  # end_frame
                    r['side'],
                    r['score'],
                    r['label_5class'],
                    341,
                    r['n_windows'],
                    f"{r['pc1_ratio_x']:.4f}",
                    f"{r['pc2_ratio_x']:.4f}",
                    f"{r['pc3_ratio_x']:.4f}",
                    f"{r['pc1_ratio_y']:.4f}",
                    f"{r['pc2_ratio_y']:.4f}",
                    f"{r['pc3_ratio_y']:.4f}",
                ])

        print(f"已生成 metadata.csv: {output_metadata}")

    # 保存完整结果JSON
    results_file = DATA_ROOT / "svd_single_patient_pc1_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n完整结果: {results_file}")
    print(f"处理患者数: {len(all_results)}")

    # 统计
    for i in range(1, 4):
        x_ratios = [r[f'pc{i}_ratio_x'] for r in all_results]
        y_ratios = [r[f'pc{i}_ratio_y'] for r in all_results]
        print(f"\nPC{i} 解释方差比例统计:")
        print(f"  X: {np.mean(x_ratios):.3f} ± {np.std(x_ratios):.3f}")
        print(f"  Y: {np.mean(y_ratios):.3f} ± {np.std(y_ratios):.3f}")

    # 累计解释方差
    cum_x = [sum(r[f'pc{i}_ratio_x'] for i in range(1, 4)) for r in all_results]
    cum_y = [sum(r[f'pc{i}_ratio_y'] for i in range(1, 4)) for r in all_results]
    print(f"\nPC1~PC3 累计解释方差比例:")
    print(f"  X: {np.mean(cum_x):.3f} ± {np.std(cum_x):.3f}")
    print(f"  Y: {np.mean(cum_y):.3f} ± {np.std(cum_y):.3f}")


if __name__ == "__main__":
    main()
