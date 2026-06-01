"""
PC1_x / PC1_y 频谱分析脚本。

对 6 组矩阵（IMR/TT/IMR+TT × x/y）分别：
  1. 对每个患者的 341×341 PC1 矩阵做 2D FFT，计算径向功率谱
  2. 按空间频率从小到大排序，归一化能量
  3. 画衰减曲线 + 累计能量曲线（均值±标准差带）
  4. 汇总统计表（达到各累计能量阈值所需的最大空间频率 index）

输出目录：outputs/pilot_feasibility/svd/pc1_fft/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT   = Path("/home/weizilin/generate_idea")
IMR_DIR = ROOT / "data/win20-step20/IMR-SVD"
TT_DIR  = ROOT / "data/win20-step20/TT-SVD"
OUT_DIR = ROOT / "outputs/pilot_feasibility/svd/pc1_fft"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.70, 0.80, 0.90, 0.95]
THRESH_LABELS = ["70%", "80%", "90%", "95%"]
THRESH_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

GROUP_COLORS = {
    "IMR_PC1_x":       "#1f77b4",
    "IMR_PC1_y":       "#ff7f0e",
    "TT_PC1_x":        "#2ca02c",
    "TT_PC1_y":        "#d62728",
    "IMR+TT_PC1_x":    "#9467bd",
    "IMR+TT_PC1_y":    "#8c564b",
}


def load_group(pc_dir: Path, kind: str):
    """加载目录下所有患者的 PC1_{kind}.npy，返回 list[np.ndarray]"""
    files = sorted(pc_dir.glob(f"*/PC1_{kind}.npy"))
    mats = [np.load(f) for f in files]
    print(f"  loaded {len(mats)} matrices from {pc_dir.name}/{kind}")
    return mats


def radial_power_spectrum(mat: np.ndarray):
    """
    对 341×341 矩阵做 2D FFT，取径向功率谱。
    返回：归一化能量谱（从小频到大频），长度 N//2。
    """
    N = mat.shape[0]
    max_r = int(np.ceil(np.sqrt(2) * N // 2))  # 角点处半径可能超过 N//2

    # 2D FFT + shift zero frequency to center
    F = np.fft.fftshift(np.fft.fft2(mat))
    power = np.abs(F) ** 2

    # 构建半径数组（像素到中心的距离）
    y, x = np.ogrid[:N, :N]
    r = np.sqrt((x - N // 2) ** 2 + (y - N // 2) ** 2)
    r_flat = r.ravel()
    p_flat = power.ravel()

    # 沿半径方向累加（每层环形区域的总能量）
    radial = np.zeros(max_r + 1)
    count  = np.zeros(max_r + 1)
    idx = r_flat.astype(int)
    np.add.at(radial, idx, p_flat)
    np.add.at(count,  idx, 1)
    # 取平均（每像素能量），避免环形面积不同
    nonzero = count > 0
    radial[nonzero] /= count[nonzero]
    radial = radial[1:]   # 去掉 r=0（DC 分量）
    # 归一化
    return radial / np.sum(radial)


def analyze_group(mats):
    """
    对一组矩阵逐个计算径向功率谱，返回：
      - spectra:    shape (n_patients, max_r)，每行归一化能量谱（按频率从小到大）
      - cumspectra: shape (n_patients, max_r)，累计能量谱
    """
    spectra = np.array([radial_power_spectrum(m) for m in mats])
    cumspectra = np.cumsum(spectra, axis=1)
    return spectra, cumspectra


def plot_decay_curves(all_spectra, all_cumspectra, groups, fname):
    """
    2行6列子图：
      - 上排：径向功率衰减曲线（log scale，半透明个体 + 均值±std）
      - 下排：累计能量曲线（阈值线标注）
    """
    fig, axes = plt.subplots(2, 6, figsize=(36, 10))
    fig.suptitle("PC1 radial power spectrum — per-group mean ± std band", fontsize=14, fontweight="bold")

    # 从任意一个 group 的 spectra 长度推断频率 bin 总数
    first_len = next(iter(all_spectra.values())).shape[1]
    x = np.arange(1, first_len + 1)

    row_specs = [
        (all_spectra,    "Power per freq bin  (log scale)", "energy share"),
        (None,           "Cumulative Energy Fraction",        "cum. energy"),
    ]

    for row_idx, (spectra, row_ylabel, _) in enumerate(row_specs):
        for col, label in enumerate(groups):
            ax = axes[row_idx, col]
            mean_s  = spectra[label].mean(axis=0) if spectra is not None else None
            std_s   = spectra[label].std(axis=0)  if spectra is not None else None
            mean_cs = all_cumspectra[label].mean(axis=0)
            std_cs  = all_cumspectra[label].std(axis=0)
            color   = GROUP_COLORS[label]

            if spectra is not None:
                for spectra_i in spectra[label]:
                    ax.plot(x, spectra_i, color=color, alpha=0.08, lw=0.5)
                ax.fill_between(x, np.maximum(mean_s - std_s, 1e-6),
                                mean_s + std_s, color=color, alpha=0.30)
                ax.plot(x, mean_s, color=color, lw=2.0)
                ax.set_yscale("log")
                ax.set_ylim(1e-6, 1)
                ax.set_ylabel(row_ylabel)
            else:
                for cum_i in all_cumspectra[label]:
                    ax.plot(x, cum_i, color=color, alpha=0.08, lw=0.5)
                ax.fill_between(x,
                                np.clip(mean_cs - std_cs, 0, 1),
                                np.clip(mean_cs + std_cs, 0, 1),
                                color=color, alpha=0.30)
                ax.plot(x, mean_cs, color=color, lw=2.0)
                for t, tl, tc in zip(THRESHOLDS, THRESH_LABELS, THRESH_COLORS):
                    cross = np.searchsorted(mean_cs, t)
                    ax.axhline(t, color=tc, linestyle="--", lw=1.0, alpha=0.8)
                    ax.axvline(cross + 1, color=tc, linestyle=":", lw=0.8, alpha=0.5)
                    ax.text(cross + 2, t + 0.02, f"{tl}", color=tc, fontsize=7)
                ax.set_ylim(0, 1.05)

            ax.set_title(label)
            ax.set_xlabel("spatial frequency index (radial bin)")
            ax.set_xlim(1, 170)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {fname}")


def plot_overview(all_cumspectra, groups, fname):
    """
    汇总图：6 组累计能量曲线叠画，便于横向比较。
    """
    first_len = next(iter(all_cumspectra.values())).shape[1]
    x = np.arange(1, first_len + 1)
    fig, ax = plt.subplots(figsize=(12, 7))
    for label in groups:
        mean_cs = all_cumspectra[label].mean(axis=0)
        std_cs  = all_cumspectra[label].std(axis=0)
        color   = GROUP_COLORS[label]
        ax.fill_between(x,
                        np.clip(mean_cs - std_cs, 0, 1),
                        np.clip(mean_cs + std_cs, 0, 1),
                        color=color, alpha=0.15)
        ax.plot(x, mean_cs, color=color, lw=2, label=label)

    for t, tl, tc in zip(THRESHOLDS, THRESH_LABELS, THRESH_COLORS):
        ax.axhline(t, color=tc, linestyle="--", lw=1.0, label=f"{tl} threshold")
    ax.set_xlim(1, 170)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("spatial frequency index (radial bin)")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_title("PC1 radial power — cumulative energy, all groups overlaid", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {fname}")


def main():
    print("=" * 60)
    print("PC1 radial power spectrum (2D FFT) analysis")
    print("=" * 60)

    # --- 1. 加载数据 ---
    print("\n[1/4] Loading matrices ...")
    imr_x = load_group(IMR_DIR, "x")
    imr_y = load_group(IMR_DIR, "y")
    tt_x  = load_group(TT_DIR,  "x")
    tt_y  = load_group(TT_DIR,  "y")
    print(f"  IMR patients: {len(imr_x)}, TT patients: {len(tt_x)}")

    # --- 2. FFT 分析 ---
    print("\n[2/4] Running 2D FFT radial spectrum ...")
    groups_cfg = [
        ("IMR_PC1_x",       imr_x),
        ("IMR_PC1_y",       imr_y),
        ("TT_PC1_x",        tt_x),
        ("TT_PC1_y",        tt_y),
    ]
    all_spectra    = {}
    all_cumspectra = {}
    for label, mats in groups_cfg:
        spectra, cumspectra = analyze_group(mats)
        all_spectra[label]    = spectra
        all_cumspectra[label] = cumspectra
        print(f"  {label}: done")

    # --- 3. 合并 IMR+TT ---
    print("\n[3/4] Computing IMR+TT combined ...")
    combined_x = imr_x + tt_x
    combined_y = imr_y + tt_y
    all_spectra["IMR+TT_PC1_x"],    all_cumspectra["IMR+TT_PC1_x"]    = analyze_group(combined_x)
    all_spectra["IMR+TT_PC1_y"],    all_cumspectra["IMR+TT_PC1_y"]    = analyze_group(combined_y)
    print("  IMR+TT done")

    # --- 4. 画图 + CSV ---
    print("\n[4/4] Plotting ...")
    groups = list(all_spectra.keys())

    plot_decay_curves(all_spectra, all_cumspectra, groups,
                      OUT_DIR / "fft_decay_curves.png")
    plot_overview(all_cumspectra, groups,
                  OUT_DIR / "fft_cum_overview.png")

    # --- 5. 汇总统计表 ---
    csv_lines = ["group,threshold,n_patients,mean_freq_count,median_freq_count,std_freq_count,min_freq_count,max_freq_count"]
    print("\n" + "=" * 85)
    print(f"{'Group':<20} {'Thresh':>7} {'N':>5} {'Mean':>8} {'Median':>8} {'Std':>7} {'Min':>5} {'Max':>5}")
    print("-" * 85)

    for label in groups:
        for t, tl in zip(THRESHOLDS, THRESH_LABELS):
            freq_counts = np.sum(all_cumspectra[label] < t, axis=1) + 1
            mean_r   = np.mean(freq_counts)
            median_r = np.median(freq_counts)
            std_r    = np.std(freq_counts)
            min_r    = int(np.min(freq_counts))
            max_r    = int(np.max(freq_counts))
            n        = len(freq_counts)
            print(f"{label:<20} {tl:>7} {n:>5} {mean_r:>8.1f} {median_r:>8.0f} {std_r:>7.1f} {min_r:>5} {max_r:>5}")
            csv_lines.append(
                f"{label},{t},{n},{mean_r:.2f},{median_r:.2f},{std_r:.2f},{min_r},{max_r}"
            )
        print()

    csv_path = OUT_DIR / "freq_rank_summary.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"  CSV saved: {csv_path}")

    print("\nDone. All figures in:", OUT_DIR)


if __name__ == "__main__":
    main()