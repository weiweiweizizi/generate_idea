"""
NMF Baseline验证 (Idea 1)
验证方向：标准NMF能否从关键点距离矩阵中浮现出语义可解释的基向量

数据：win20-step20，只使用win_x和win_y
"""

import numpy as np
import os
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from pathlib import Path

# 配置
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "win20-step20"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "pilot_feasibility" / "nmf" / "win20-step20"
N_COMPONENTS = 10  # 基向量数量，可调整
RANDOM_STATE = 42


def load_data():
    """加载所有患者的win_x和win_y矩阵"""
    all_x = []
    all_y = []
    subjects = []

    for dataset in ["IMR", "TT"]:
        dataset_path = DATA_ROOT / dataset
        if not dataset_path.exists():
            print(f"Warning: {dataset_path} not found, skipping")
            continue

        for subject_dir in sorted(dataset_path.iterdir()):
            if not subject_dir.is_dir():
                continue
            subj_id = subject_dir.name

            # 收集该患者所有窗口的x和y矩阵
            win_x_files = sorted(subject_dir.glob("win_*_x.npy"))
            win_y_files = sorted(subject_dir.glob("win_*_y.npy"))

            if len(win_x_files) == 0 or len(win_y_files) == 0:
                continue

            for wf in win_x_files:
                win_x = np.load(wf)
                all_x.append(win_x.flatten())  # 展平为向量
                subjects.append(f"{dataset}_{subj_id}")

            for wf in win_y_files:
                win_y = np.load(wf)
                all_y.append(win_y.flatten())

    X = np.array(all_x)  # (n_samples, 341*341)
    Y = np.array(all_y)
    subjects = np.array(subjects)

    print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Total subjects: {len(set(subjects))}")
    print(f"X min: {X.min():.4f}, max: {X.max():.4f}")
    print(f"Y min: {Y.min():.4f}, max: {Y.max():.4f}")

    return X, Y, subjects


def run_nmf(V, n_components, name="X"):
    """对矩阵V运行NMF分解 V ≈ WH"""
    print(f"\n{'='*60}")
    print(f"Running NMF on {name}, shape: {V.shape}")
    print(f"n_components: {n_components}")
    print(f"{'='*60}")

    nmf = NMF(
        n_components=n_components,
        init="nndsvd",  # 更好的初始化
        random_state=RANDOM_STATE,
        max_iter=500,
        alpha_W=0.1,
        alpha_H=0.1,
    )

    W = nmf.fit_transform(V)  # (n_samples, n_components) - 系数矩阵
    H = nmf.components_  # (n_components, n_features) - 基矩阵

    reconstruction_error = nmf.reconstruction_err_
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    print(f"W shape: {W.shape}, H shape: {H.shape}")

    return W, H, nmf


def visualize_basis(H, name="X", n_show=6):
    """可视化基向量（热图）"""
    n = min(n_show, H.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n):
        basis = H[i].reshape(341, 341)  # 恢复为方阵
        im = axes[i].imshow(basis, cmap="viridis", aspect="auto")
        axes[i].set_title(f"Basis {i+1}")
        axes[i].set_xlabel("landmark j")
        axes[i].set_ylabel("landmark i")
        plt.colorbar(im, ax=axes[i])

    plt.suptitle(f"NMF Basis Vectors ({name}), {H.shape[0]} components")
    plt.tight_layout()

    output_path = DATA_ROOT / f"nmf_basis_{name.lower()}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved basis visualization to {output_path}")
    plt.close()


def analyze_coefficients(W, subjects, name="X"):
    """分析系数矩阵W，查看不同患者的系数分布"""
    unique_subjects = sorted(set(subjects))
    print(f"\nUnique subjects: {len(unique_subjects)}")

    # 统计每个基在多少个不同患者上被激活
    n_components = W.shape[1]
    activation_counts = []

    for j in range(n_components):
        # 该基在所有样本上的系数
        coeffs = W[:, j]
        # 被激活（系数 > 0.1*max）的样本数
        activated = np.sum(coeffs > 0.1 * np.max(coeffs))
        activation_counts.append(activated)

    print(f"\nBasis activation statistics ({name}):")
    for j in range(n_components):
        print(f"  Basis {j+1}: {activation_counts[j]} samples activated (> 0.1 * max)")

    return activation_counts


def main():
    print("NMF Baseline Verification (Idea 1)")
    print("=" * 60)

    # 1. 加载数据
    X, Y, subjects = load_data()

    # 2. 对X和Y分别做NMF
    W_x, H_x, nmf_x = run_nmf(X, N_COMPONENTS, "X")
    W_y, H_y, nmf_y = run_nmf(Y, N_COMPONENTS, "Y")

    # 3. 可视化基向量
    visualize_basis(H_x, "X", n_show=6)
    visualize_basis(H_y, "Y", n_show=6)

    # 4. 分析系数
    activation_x = analyze_coefficients(W_x, subjects, "X")
    activation_y = analyze_coefficients(W_y, subjects, "Y")

    # 5. 保存结果
    output_dir = OUTPUT_ROOT / "nmf_results"
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / "W_x.npy", W_x)
    np.save(output_dir / "H_x.npy", H_x)
    np.save(output_dir / "W_y.npy", W_y)
    np.save(output_dir / "H_y.npy", H_y)

    print(f"\nSaved NMF results to {output_dir}")
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Check nmf_basis_X.png and nmf_basis_Y.png for basis visualization")
    print("2. Analyze if basis vectors have semantic meaning (e.g., mouth, eye)")
    print("3. If bases are semantically mixed, consider identity removal (Idea 2)")
    print("=" * 60)


if __name__ == "__main__":
    main()
