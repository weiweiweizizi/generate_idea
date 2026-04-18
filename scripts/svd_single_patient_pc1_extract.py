"""
提取所有患者的单患者SVD PC1主模态

为每个患者执行单患者SVD分解，提取PC1空间基（341×341）并保存。
输出目录结构:
    data/win20-step20/IMR-SVD/
    data/win20-step20/TT-SVD/

每个患者目录包含:
    - PC1_x.npy: X方向PC1空间基 (341, 341)
    - PC1_y.npy: Y方向PC1空间基 (341, 341)
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
from tqdm import tqdm
import json
import csv

# 配置
DATA_ROOT = Path("/home/weizilin/generate_idea/data/win20-step20")
RANDOM_STATE = 42


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


def run_svd_single_pc(patient_data):
    """对患者数据做SVD，只提取PC1"""
    n_samples = patient_data.shape[0]
    X = patient_data.reshape(n_samples, -1)

    svd = TruncatedSVD(n_components=1, random_state=RANDOM_STATE)
    U = svd.fit_transform(X)
    Vt = svd.components_  # shape: (1, 341*341)
    singular_values = svd.singular_values_  # shape: (1,)

    # 计算PC1解释方差比例
    # 使用n_components=1时，explained_variance_ratio_只有第一个
    total_variance = np.sum(svd.explained_variance_ratio_) if hasattr(svd, 'explained_variance_ratio_') else 1.0
    pc1_ratio = float(svd.explained_variance_ratio_[0]) if hasattr(svd, 'explained_variance_ratio_') else float(singular_values[0]**2 / np.sum(singular_values**2))

    return Vt[0].reshape(341, 341), singular_values[0], pc1_ratio


def main():
    print("=" * 60)
    print("提取所有患者单患者SVD PC1主模态")
    print("=" * 60)

    # 创建输出目录
    for dataset in ["IMR", "TT"]:
        output_dir = DATA_ROOT / f"{dataset}-SVD"
        output_dir.mkdir(exist_ok=True)
        print(f"输出目录: {output_dir}")

    # 收集患者列表
    patients = []
    patient_metadata = {}  # {subj_id: {side, score, label_5class}}

    for dataset in ["IMR", "TT"]:
        dataset_path = DATA_ROOT / dataset
        metadata_path = dataset_path / "metadata.csv"

        # 读取metadata获取患者临床信息
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj = row['subj']
                if subj not in patient_metadata:
                    patient_metadata[subj] = {
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
            pc1_x, sigma_x, ratio_x = run_svd_single_pc(diff_x)
            pc1_y, sigma_y, ratio_y = run_svd_single_pc(diff_y)
        except Exception as e:
            print(f"  错误 {dataset}/{subj_id}: {e}")
            continue

        # 保存PC1矩阵
        output_dir = DATA_ROOT / f"{dataset}-SVD" / subj_id
        output_dir.mkdir(exist_ok=True)

        np.save(output_dir / "PC1_x.npy", pc1_x)
        np.save(output_dir / "PC1_y.npy", pc1_y)

        # 获取患者临床信息
        meta = patient_metadata.get(subj_id, {
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
            'sigma_x': float(sigma_x),
            'sigma_y': float(sigma_y),
            'pc1_ratio_x': ratio_x,
            'pc1_ratio_y': ratio_y
        })

    # 生成config.json
    for dataset in ["IMR", "TT"]:
        source_config = DATA_ROOT / dataset / "config.json"
        output_config = DATA_ROOT / f"{dataset}-SVD" / "config.json"

        with open(source_config, 'r') as f:
            config = json.load(f)

        # 添加SVD相关信息
        config['decomposition'] = 'SVD'
        config['n_components'] = 1
        config['component_stored'] = 'PC1'

        with open(output_config, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"已生成 config.json: {output_config}")

    # 生成metadata.csv
    for dataset in ["IMR", "TT"]:
        output_metadata = DATA_ROOT / f"{dataset}-SVD" / "metadata.csv"

        dataset_results = [r for r in all_results if r['dataset'] == dataset]

        with open(output_metadata, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['subj', 'window_idx', 'start_frame', 'end_frame',
                           'side', 'score', 'label_5class', 'matrix_size',
                           'n_windows', 'pc1_ratio_x', 'pc1_ratio_y'])

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
                    f"{r['pc1_ratio_y']:.4f}"
                ])

        print(f"已生成 metadata.csv: {output_metadata}")

    # 保存完整结果JSON
    results_file = DATA_ROOT / "svd_single_patient_pc1_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n完整结果: {results_file}")
    print(f"处理患者数: {len(all_results)}")

    # 统计
    x_ratios = [r['pc1_ratio_x'] for r in all_results]
    y_ratios = [r['pc1_ratio_y'] for r in all_results]
    print(f"\nPC1解释方差比例统计:")
    print(f"  X: {np.mean(x_ratios):.3f} ± {np.std(x_ratios):.3f}")
    print(f"  Y: {np.mean(y_ratios):.3f} ± {np.std(y_ratios):.3f}")


if __name__ == "__main__":
    main()
