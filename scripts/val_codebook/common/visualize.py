"""
可视化模块

绘制ROC曲线和混淆矩阵
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import json
import warnings


def plot_roc_curve(y_true, y_prob, save_path, class_names=None):
    """
    绘制ROC曲线（二分类或多分类OvR）

    Args:
        y_true: array (n_samples,) - 真实标签
        y_prob: array (n_samples, n_classes) - 预测概率
        save_path: 保存路径
        class_names: list of class names
    """
    n_classes = y_prob.shape[1]

    if n_classes == 2:
        # 二分类
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        # 多分类：One-vs-Rest
        classes = np.arange(n_classes)
        y_true_bin = label_binarize(y_true, classes=classes)

        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(n_classes):
                if np.sum(y_true_bin[:, i]) == 0:
                    # 跳过没有样本的类别
                    continue
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                label = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                        label=f'{label} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (One-vs-Rest)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None, normalize=True):
    """
    绘制混淆矩阵

    Args:
        y_true: array (n_samples,) - 真实标签
        y_pred: array (n_samples,) - 预测标签
        save_path: 保存路径
        class_names: list of class names
        normalize: 是否归一化
    """
    n_classes = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='.2f' if normalize else 'd')

    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_results(results, save_path):
    """保存结果到JSON"""
    # 移除不可序列化的字段
    save_dict = {k: v for k, v in results.items()
                 if k not in ['y_true_all', 'y_pred_all', 'explained_variance_ratio']}
    save_dict['explained_variance_ratio'] = results.get('explained_variance_ratio', [])

    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=2)


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.random.rand(8, 2)
    y_prob[:, 1] = 1 - y_prob[:, 0]

    plot_roc_curve(y_true, y_prob, '/tmp/test_roc.png')
    print("ROC curve saved to /tmp/test_roc.png")
