"""
PC1_x / PC1_y 低秩性验证脚本。

对 6 组矩阵（IMR/TT/IMR+TT × x/y）分别：
  1. 对每个患者的 341×341 PC1 矩阵做 SVD，记录完整奇异值谱
  2. 归一化奇异值为能量（除以总能量）
  3. 对所有患者叠画奇异值衰减曲线 + 均值±标准差带
  4. 汇总统计表（同阈值下奇异值个数）

输出目录：outputs/pilot_feasibility/svd/pc1_lowrank/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT   = Path("/home/weizilin/generate_idea")
IMR_DIR = ROOT / "data/win20-step20/IMR-SVD"
TT_DIR  = ROOT / "data/win20-step20/TT-SVD"
OUT_DIR = ROOT / "outputs/pilot_feasibility/svd/pc1_lowrank"
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


def svd_spectrum(mat: np.ndarray):
    """对单个矩阵做 SVD，返回归一化能量谱（从大到小，sum=1）"""
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    return s**2 / np.sum(s**2)


def analyze_group(mats):
    """
    对一组矩阵逐个做 SVD，返回：
      - spectra: shape (n_patients, 341)，每行是一个患者归一化能量谱
      - cumspectra: shape (n_patients, 341)，每行是累计能量谱
    """
    spectra    = np.array([svd_spectrum(m) for m in mats])
    cumspectra = np.cumsum(spectra, axis=1)
    return spectra, cumspectra


def plot_decay_curves(all_spectra, all_cumspectra, groups, fname):
    """
    2行3列子图：
      - 上排：奇异值衰减曲线（半透明个体 + 均值±标准差带）
      - 下排：对应累计能量曲线（阈值线标注）
    """
    fig, axes = plt.subplots(2, 6, figsize=(36, 10))
    fig.suptitle("PC1 singular value decay — per-group mean ± std band", fontsize=14, fontweight="bold")

    x = np.arange(1, 342)  # SV index 1-based for readability

    row_specs = [
        (all_spectra,    "Normalized Energy per SV  (log scale)", "energy share"),
        (None,           "Cumulative Energy Fraction",              "cum. energy"),
    ]

    for row_idx, (spectra, row_ylabel, _label) in enumerate(row_specs):
        for col, label in enumerate(groups):
            ax = axes[row_idx, col]
            mean_s  = spectra[label].mean(axis=0) if spectra is not None else None
            std_s   = spectra[label].std(axis=0)  if spectra is not None else None
            mean_cs = all_cumspectra[label].mean(axis=0)
            std_cs  = all_cumspectra[label].std(axis=0)
            color   = GROUP_COLORS[label]

            if spectra is not None:
                # 上排：衰减曲线
                for spectra_i in spectra[label]:
                    ax.plot(x, spectra_i, color=color, alpha=0.08, lw=0.5)
                ax.fill_between(x, np.maximum(mean_s - std_s, 1e-12),
                                mean_s + std_s, color=color, alpha=0.30)
                ax.plot(x, mean_s, color=color, lw=2.0)
                ax.set_yscale("log")
                ax.set_ylim(1e-12, 1)
                ax.set_ylabel(row_ylabel)
            else:
                # 下排：累计能量曲线
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
            ax.set_xlabel("singular value index")
            ax.set_xlim(1, 10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {fname}")


def plot_overview(all_cumspectra, groups, fname):
    """
    汇总图：6 组累计能量曲线叠画在同一张图，便于横向比较。
    横轴前 50 个 SV，突出前段差异。
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(1, 342)
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
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_title("PC1 cumulative energy — all groups overlaid", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {fname}")


def main():
    print("=" * 60)
    print("PC1 low-rank verification (SVD)")
    print("=" * 60)

    # --- 1. 加载数据 ---
    print("\n[1/4] Loading matrices ...")
    imr_x = load_group(IMR_DIR, "x")
    imr_y = load_group(IMR_DIR, "y")
    tt_x  = load_group(TT_DIR,  "x")
    tt_y  = load_group(TT_DIR,  "y")
    print(f"  IMR patients: {len(imr_x)}, TT patients: {len(tt_x)}")

    # --- 2. SVD 分析 ---
    print("\n[2/4] Running SVD ...")
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
                      OUT_DIR / "sv_decay_curves.png")
    plot_overview(all_cumspectra, groups,
                  OUT_DIR / "sv_cum_overview.png")

    # --- 5. 汇总统计表（阈值秩） ---
    csv_lines = ["group,threshold,n_patients,mean_sv_count,median_sv_count,std_sv_count,min_sv_count,max_sv_count"]
    print("\n" + "=" * 80)
    print(f"{'Group':<20} {'Thresh':>7} {'N':>5} {'Mean':>8} {'Median':>8} {'Std':>7} {'Min':>5} {'Max':>5}")
    print("-" * 80)

    for label in groups:
        for t, tl in zip(THRESHOLDS, THRESH_LABELS):
            sv_counts = np.sum(all_cumspectra[label] < t, axis=1) + 1
            mean_r  = np.mean(sv_counts)
            median_r = np.median(sv_counts)
            std_r   = np.std(sv_counts)
            min_r   = int(np.min(sv_counts))
            max_r   = int(np.max(sv_counts))
            n       = len(sv_counts)
            print(f"{label:<20} {tl:>7} {n:>5} {mean_r:>8.1f} {median_r:>8.0f} {std_r:>7.1f} {min_r:>5} {max_r:>5}")
            csv_lines.append(
                f"{label},{t},{n},{mean_r:.2f},{median_r:.2f},{std_r:.2f},{min_r},{max_r}"
            )
        print()

    csv_path = OUT_DIR / "rank_summary.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"  CSV saved: {csv_path}")

    print("\nDone. All figures in:", OUT_DIR)


if __name__ == "__main__":
    main()