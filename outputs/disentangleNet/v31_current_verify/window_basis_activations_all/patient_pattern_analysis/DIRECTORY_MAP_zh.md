# `patient_pattern_analysis` 目录与脚本对应关系

这个目录现在可以分成四块：

## 1. 患者级画像汇总

目录：

- `patient_profile_summary/`

含义：

- 基于窗口级激活表，先按患者聚合，再输出患者级 usage / activation / coeff / entropy / dominant basis / pattern label 等摘要。

生成脚本：

- `scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py`

关键输入：

- `outputs/disentangleNet/v31_current_verify/window_basis_activations_all/window_basis_activations_wide.csv`

关键输出：

- `patient_activation_profiles.csv`
- `summary_by_side_label.csv`
- `summary_by_dataset.csv`
- `summary_by_dataset_and_side_label.csv`
- `patient_extreme_rankings.csv`
- `crosstab_*.csv`
- `report.md`
- `summary.json`

## 2. 共激活分析

目录：

- `coactivation/by_side_label_name/`
- `coactivation/by_dataset_name/`
- `coactivation/by_score/`
- `coactivation/by_label_5class/`

每个目录下再分：

- `activation/`
- `usage/`

含义：

- 在患者级向量上做 permutation MANOVA、类内相关矩阵、pairwise correlation difference test、heatmap 与 delta heatmap。

生成脚本：

- `scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py`

关键输入：

- `patient_profile_summary/patient_activation_profiles.csv`

关键输出：

- `basis_manifest.csv`
- `class_correlation_long.csv`
- `pairwise_corr_diff_tests.csv`
- `mean_<feature_family>_by_<class_col>.csv`
- `mean_<feature_family>_by_<class_col>.png`
- `correlation_heatmaps/`
- `delta_heatmaps/`
- `report.md`
- `summary.json`

## 3. t-SNE 可视化

### 3.1 全体患者

- `tsne/all/all_basis/`
  - 全部 basis
- `tsne/all/no_side/`
  - 忽略 side basis，等价于排除 `b8,b9,b10`
- `tsne/all/side_only/`
  - 只保留 side basis，等价于排除 `b0-b7`

### 3.2 IMR-only

- `tsne/imr/all_basis/`
- `tsne/imr/no_side/`
- `tsne/imr/side_only/`

### 3.3 TT-only

- `tsne/tt/all_basis/`
- `tsne/tt/no_side/`
- `tsne/tt/side_only/`

含义：

- 对患者级 `basis_usage` / `basis_activation` / `combined` 特征做 `2D` 和 `3D` t-SNE。
- 每套结果都导出：
  - `dataset_name` 着色图
  - `side_label_name` 着色图
  - `score` 着色图
  - `label_5class` 着色图
  - `combined` 图
- `3D combined` 还会额外导出旋转 GIF。

生成脚本：

- `scripts/disentangleNet/analysis/analyze_patient_tsne.py`

关键输入：

- 全体患者：
  - `patient_profile_summary/patient_activation_profiles.csv`
- IMR-only：
  - `patient_profile_summary/patient_activation_profiles_imr.csv`
- TT-only：
  - `patient_profile_summary/patient_activation_profiles_tt.csv`

目录规则：

- 每个结果根目录下都有：
  - `usage/`
  - `activation/`
  - `combined/`
  - `report.md`
  - `summary.json`
- 每个 `usage|activation|combined` 目录下都有：
  - `tsne_2d/`
  - `tsne_3d/`

## 4. t-SNE 统一索引页

目录：

- `tsne/index_pages/`

含义：

- 把 `All Patients / IMR Only / TT Only` 与 `All Basis / No Side / Side Only` 拼成统一 3x3 总览页。
- 目前输出 3 张索引图：
  - `combined_2d_index.png`
  - `usage_2d_index.png`
  - `activation_2d_index.png`

生成脚本：

- `scripts/disentangleNet/analysis/build_tsne_index_pages.py`

## 5. 旧结果归档

目录：

- `legacy_pre_naming_cleanup/`

含义：

- 清理命名前的旧结果保留区，只为追溯，不建议继续往这里写新结果。

## 当前建议入口

如果只想快速进入当前分析，优先看：

- `report_zh.md`
- `DIRECTORY_MAP_zh.md`
- `patient_profile_summary/report.md`
- `tsne/index_pages/`
