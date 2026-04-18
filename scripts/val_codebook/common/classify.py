"""
分类模块

提供PCA + 分类器的评估框架
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import json


def pca_transform(X, n_components=50):
    """PCA降维"""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def evaluate_classifier(X, y, clf, n_splits=5, random_state=42):
    """
    使用Stratified K-Fold评估分类器

    Args:
        X: array (n_samples, n_features)
        y: array (n_samples,) - 标签
        clf: 分类器实例
        n_splits: K-Fold折数
        random_state: 随机种子

    Returns:
        results: dict 包含各类指标
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs = []
    f1s = []
    aucs = []
    cm_sum = np.zeros((len(np.unique(y)), len(np.unique(y))))

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 克隆分类器
        clf_fold = type(clf)(**clf.get_params())

        # 训练
        clf_fold.fit(X_train_scaled, y_train)

        # 预测
        y_pred = clf_fold.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        accs.append(acc)
        f1s.append(f1)

        # ROC-AUC (仅二分类)
        if len(np.unique(y)) == 2:
            if hasattr(clf_fold, 'predict_proba'):
                y_prob = clf_fold.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = clf_fold.decision_function(X_test_scaled)
            y_prob_all.extend(y_prob)
            try:
                auc = roc_auc_score(y_test, y_prob)
                aucs.append(auc)
            except:
                pass

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_sum += cm

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    results = {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        'auc_mean': np.mean(aucs) if aucs else None,
        'auc_std': np.std(aucs) if aucs else None,
        'confusion_matrix': cm_sum.tolist(),
        'y_true_all': y_true_all,
        'y_pred_all': y_pred_all,
    }

    if aucs:
        results['auc_mean'] = np.mean(aucs)
        results['auc_std'] = np.std(aucs)

    return results


def run_experiment(X, y, n_pca=50, classifier_type='lr', n_splits=5):
    """
    运行完整实验流程

    Args:
        X: array (n_samples, n_features)
        y: array (n_samples,) - 标签
        n_pca: PCA组件数
        classifier_type: 'lr', 'svm', 'rf', 'linear_svc'
        n_splits: K-Fold折数

    Returns:
        results: dict
    """
    # PCA降维
    X_pca, pca = pca_transform(X, n_components=n_pca)

    # 选择分类器
    if classifier_type == 'lr':
        clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    elif classifier_type == 'svm':
        clf = SVC(kernel='rbf', probability=True, random_state=42)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_type == 'linear_svc':
        clf = LinearSVC(max_iter=5000, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    # 评估
    results = evaluate_classifier(X_pca, y, clf, n_splits=n_splits)
    results['n_pca'] = n_pca
    results['classifier'] = classifier_type
    results['n_samples'] = len(y)
    results['n_features'] = X.shape[1]
    results['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()

    return results


if __name__ == "__main__":
    # 简单测试
    from load_pc1 import load_pc1_data, get_X_matrix

    X_x, X_y, metadata = load_pc1_data()
    subj_ids = metadata['subj'].tolist()

    X = get_X_matrix(X_x, subj_ids, mode='full')
    print(f"X shape: {X.shape}")

    # 测试分类 - IMR vs TT
    y = (metadata['dataset'] == 'TT-SVD').astype(int).values
    results = run_experiment(X, y, n_pca=50, classifier_type='lr')
    print(f"\nIMR vs TT (LR):")
    print(f"  Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"  F1: {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
    print(f"  AUC: {results['auc_mean']:.3f} ± {results['auc_std']:.3f}")
