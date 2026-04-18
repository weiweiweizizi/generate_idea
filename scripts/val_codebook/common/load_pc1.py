"""
加载PC1数据模块

从IMR-SVD和TT-SVD目录加载所有患者的PC1矩阵和标签
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("/home/weizilin/generate_idea/data/win20-step20")


def load_pc1_data():
    """
    加载所有患者的PC1数据

    Returns:
        X_x: dict, {subj_id: PC1_x.npy array (341, 341)}
        X_y: dict, {subj_id: PC1_y.npy array (341, 341)}
        metadata: DataFrame with columns subj, dataset, side, score, label_5class
    """
    X_x = {}
    X_y = {}
    rows = []

    for dataset in ["IMR-SVD", "TT-SVD"]:
        dataset_path = DATA_ROOT / dataset
        metadata_path = dataset_path / "metadata.csv"

        # 读取metadata - keep subj as string to preserve leading zeros
        df = pd.read_csv(metadata_path, dtype={'subj': str})
        df['dataset'] = dataset  # Add dataset identifier
        rows.append(df)

        # 加载每个患者的PC1矩阵
        for _, row in df.iterrows():
            subj_id = row['subj']  # Already string due to dtype
            subj_path = dataset_path / subj_id

            pc1_x_path = subj_path / "PC1_x.npy"
            pc1_y_path = subj_path / "PC1_y.npy"

            if pc1_x_path.exists() and pc1_y_path.exists():
                X_x[subj_id] = np.load(pc1_x_path)
                X_y[subj_id] = np.load(pc1_y_path)

    metadata = pd.concat(rows, ignore_index=True)
    return X_x, X_y, metadata


def get_X_matrix(X_dict, subj_ids, mode='full'):
    """
    根据mode提取矩阵并展平

    Args:
        X_dict: {subj_id: array (341, 341)} - keys are strings
        subj_ids: list of subject IDs (strings)
        mode: 'full' (341×341) or 'mouth' (119×119, indices 188:307)

    Returns:
        X: array (n_samples, n_features)
    """
    matrices = []
    for subj_id in subj_ids:
        subj_id_str = str(subj_id)  # Ensure string key
        pc1 = X_dict[subj_id_str]
        if mode == 'full':
            matrices.append(pc1.flatten())
        elif mode == 'mouth':
            # around_mouth (188:233) + mouth (233:307) → 119×119
            matrices.append(pc1[188:307, 188:307].flatten())
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return np.array(matrices)


if __name__ == "__main__":
    X_x, X_y, metadata = load_pc1_data()
    print(f"Loaded {len(X_x)} patients")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Columns: {metadata.columns.tolist()}")
    print(f"\nLabel distributions:")
    print(metadata['label_5class'].value_counts().sort_index())
