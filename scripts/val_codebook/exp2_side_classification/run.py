"""
实验2: 测别分类（左/右/正常）

验证患者左侧异常/右侧异常/正常在PC1模态上是否存在差异

标签映射:
- label_5class = 2: 正常 (Normal)
- label_5class < 2: 左侧异常 (Left)
- label_5class > 2: 右侧异常 (Right)
"""

import sys
sys.path.insert(0, '/home/weizilin/generate_idea/scripts/val_codebook')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np
from pathlib import Path
from common.load_pc1 import load_pc1_data, get_X_matrix
from common.classify import run_experiment
from common.visualize import plot_roc_curve, plot_confusion_matrix, save_results

# 配置
N_PCA = 50
N_SPLITS = 5
CLASSIFIER = 'lr'

DIRECTIONS = ['x', 'y']
MATRIX_TYPES = ['full', 'mouth']

CLASS_NAMES = ['Left', 'Normal', 'Right']


def create_side_labels(label_5class):
    """
    根据label_5class创建测别标签

    Args:
        label_5class: array of label_5class values

    Returns:
        array of 0 (Left), 1 (Normal), 2 (Right)
    """
    labels = np.zeros(len(label_5class), dtype=int)
    for i, v in enumerate(label_5class):
        if v < 2:
            labels[i] = 0  # Left
        elif v == 2:
            labels[i] = 1  # Normal
        else:
            labels[i] = 2  # Right
    return labels


def run_exp2():
    """运行实验2"""
    print("=" * 60)
    print("实验2: 测别分类 (Left/Normal/Right)")
    print("=" * 60)

    # 加载数据
    X_x, X_y, metadata = load_pc1_data()
    subj_ids = metadata['subj'].tolist()

    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    for direction in DIRECTIONS:
        X_dict = X_x if direction == 'x' else X_y

        for matrix_type in MATRIX_TYPES:
            config_name = f"{matrix_type}_{direction}"

            print(f"\n--- {config_name} ---")

            # 准备数据
            X = get_X_matrix(X_dict, subj_ids, mode=matrix_type)
            y = create_side_labels(metadata['label_5class'].values)

            print(f"X shape: {X.shape}, y shape: {y.shape}")
            print(f"Class distribution: Left={np.sum(y==0)}, Normal={np.sum(y==1)}, Right={np.sum(y==2)}")

            # 运行实验
            results = run_experiment(X, y, n_pca=N_PCA, classifier_type=CLASSIFIER, n_splits=N_SPLITS)

            # 保存可视化
            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression

            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            y_prob_all = np.zeros((len(y), 3))

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
                clf.fit(X_train_scaled, y_train)
                y_prob_all[test_idx] = clf.predict_proba(X_test_scaled)

            # ROC曲线 (多分类 OvR)
            roc_path = output_dir / f"roc_{config_name}.png"
            plot_roc_curve(y, y_prob_all, str(roc_path), class_names=CLASS_NAMES)
            print(f"ROC curve saved: {roc_path}")

            # 混淆矩阵
            cm_path = output_dir / f"confusion_{config_name}.png"
            plot_confusion_matrix(np.array(results['y_true_all']),
                                np.array(results['y_pred_all']),
                                str(cm_path), class_names=CLASS_NAMES)
            print(f"Confusion matrix saved: {cm_path}")

            # 保存结果
            results_path = output_dir / f"results_{config_name}.json"
            save_results(results, str(results_path))

            # 打印关键指标
            print(f"Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
            print(f"F1 (macro): {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")

            all_results[config_name] = results

    # 汇总表格
    print("\n" + "=" * 60)
    print("实验2汇总: Left/Normal/Right分类")
    print("=" * 60)
    print(f"{'Config':<20} {'Acc':<12} {'F1':<12}")
    print("-" * 60)
    for name, res in all_results.items():
        acc = f"{res['accuracy_mean']:.3f}±{res['accuracy_std']:.3f}"
        f1 = f"{res['f1_mean']:.3f}±{res['f1_std']:.3f}"
        print(f"{name:<20} {acc:<12} {f1:<12}")

    return all_results


if __name__ == "__main__":
    run_exp2()
