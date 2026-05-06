# `scripts/disentangleNet/analysis` 中文说明

这个目录存放的是 `disentangleNet` 的分析、导出与可视化脚本。  
它们的定位不是训练模型，而是围绕“训练好的 checkpoint / 导出的 basis 激活结果”做进一步解释、验证和整理。

## 一、脚本分组

### 1. Checkpoint 级分析

- [analyze_checkpoint.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_checkpoint.py)
  - 对单个 checkpoint 做总体分析。
  - 导出 free / side basis bank、group-level representation、线性 probe 结果、summary。

- [analyze_side_interpretability.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_side_interpretability.py)
  - 专门看 side basis 的可解释性。
  - 输出 side basis 统计、左右交换对称性、group-level side semantics。

- [analyze_kfold_report.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_kfold_report.py)
  - 在 subject 层面做 k-fold probe 评估。
  - 输出 fold 分配、每折指标、预测结果、Markdown 报告。

### 2. Basis 激活导出与患者级统计

- [export_window_basis_activations.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/export_window_basis_activations.py)
  - 核心导出脚本。
  - 从 checkpoint 重新跑一遍 grouped window，导出窗口级 `usage / coeff / activation`。
  - 输出宽表与长表，是后续患者级分析和 t-SNE 的基础输入。

- [analyze_patient_activation_patterns.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py)
  - 把窗口级结果聚合到患者级。
  - 输出 `patient_activation_profiles.csv`、分组汇总、极值排行、模式标签等。

- [analyze_class_coactivation_patterns.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py)
  - 在患者级向量上做共激活分析。
  - 支持 `side_label_name / dataset_name / score / label_5class` 等分组。
  - 输出相关矩阵、delta heatmap、permutation MANOVA、pairwise corr-diff test。

### 3. t-SNE 可视化

- [analyze_patient_tsne.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_patient_tsne.py)
  - 对患者级 `usage / activation / combined` 做 2D / 3D t-SNE。
  - 支持全 basis、去掉 side basis、只保留 side basis。
  - `3D combined` 会额外导出旋转 GIF。

- [build_tsne_index_pages.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/build_tsne_index_pages.py)
  - 把 `All / IMR / TT` 与 `All Basis / No Side / Side Only` 拼成 3x3 总览页。
  - 当前输出三张总览图：`combined_2d / usage_2d / activation_2d`。

### 4. Matrix-Vis 导出

- [export_matrix_vis_basis.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/export_matrix_vis_basis.py)
  - 导出 basis bundle，供矩阵可视化使用。

- [export_matrix_vis_patient.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/export_matrix_vis_patient.py)
  - 导出单个患者的窗口序列矩阵 bundle。
  - 包括每个窗口的 free/side coeff、usage、composed matrix、side prediction。

## 二、推荐执行顺序

如果目标是“从 checkpoint 出发，逐步做到患者级解释与可视化”，推荐顺序如下：

1. 用 `export_window_basis_activations.py` 导出窗口级 basis 激活。
2. 用 `analyze_patient_activation_patterns.py` 生成患者级摘要。
3. 用 `analyze_class_coactivation_patterns.py` 做不同标签体系下的共激活分析。
4. 用 `analyze_patient_tsne.py` 做患者级 t-SNE。
5. 用 `build_tsne_index_pages.py` 做总览页。

如果目标是“先快速看 checkpoint 本身是否学到了有意义结构”，推荐顺序如下：

1. `analyze_checkpoint.py`
2. `analyze_side_interpretability.py`
3. `analyze_kfold_report.py`

## 三、常用命令

下面的命令默认针对当前项目已有的 `v31_current_verify/best.pt`。

### 1. 导出窗口级 basis 激活

```bash
python scripts/disentangleNet/analysis/export_window_basis_activations.py export \
  --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \
  --split all
```

验证集版本：

```bash
python scripts/disentangleNet/analysis/export_window_basis_activations.py export \
  --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \
  --split val
```

### 2. 构建患者级摘要

```bash
python scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py analyze
```

### 3. 共激活分析

按 side label：

```bash
python scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py analyze \
  --class_col side_label_name \
  --feature_family activation \
  --n_perm 4000
```

按 dataset：

```bash
python scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py analyze \
  --class_col dataset_name \
  --feature_family usage \
  --n_perm 4000
```

按 score：

```bash
python scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py analyze \
  --class_col score \
  --feature_family activation \
  --n_perm 4000
```

按 label_5class：

```bash
python scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py analyze \
  --class_col label_5class \
  --feature_family activation \
  --n_perm 4000
```

### 4. 患者级 t-SNE

全 basis：

```bash
python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \
  --output_root outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne/all/all_basis
```

去掉 side basis：

```bash
python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \
  --output_root outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne/all/no_side \
  --exclude_basis_indices 8,9,10
```

只保留 side basis：

```bash
python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \
  --output_root outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne/all/side_only \
  --exclude_basis_indices 0,1,2,3,4,5,6,7
```

IMR-only 版本：

```bash
python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \
  --patient_profiles_csv outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/patient_profile_summary/patient_activation_profiles_imr.csv \
  --output_root outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne/imr/all_basis
```

TT-only 版本：

```bash
python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \
  --patient_profiles_csv outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/patient_profile_summary/patient_activation_profiles_tt.csv \
  --output_root outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne/tt/all_basis
```

### 5. t-SNE 总览页

```bash
python scripts/disentangleNet/analysis/build_tsne_index_pages.py build
```

### 6. Checkpoint 总体分析

```bash
python scripts/disentangleNet/analysis/analyze_checkpoint.py \
  outputs/disentangleNet/v31_current_verify/best.pt
```

### 7. Side interpretability

```bash
python scripts/disentangleNet/analysis/analyze_side_interpretability.py \
  outputs/disentangleNet/v31_current_verify/best.pt
```

### 8. K-fold probe 报告

```bash
python scripts/disentangleNet/analysis/analyze_kfold_report.py \
  outputs/disentangleNet/v31_current_verify/best.pt
```

### 9. Matrix-vis 导出

导出 basis：

```bash
python scripts/disentangleNet/analysis/export_matrix_vis_basis.py export \
  --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt
```

导出患者：

```bash
python scripts/disentangleNet/analysis/export_matrix_vis_patient.py export \
  --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \
  --subject 844697
```

## 四、当前输出目录约定

当前和患者级分析直接相关的结果主要落在：

- [patient_pattern_analysis](/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis)

其中目前整理后的主树是：

- `patient_profile_summary/`
- `coactivation/by_<class_col>/<feature_family>/`
- `tsne/<cohort>/<basis_scope>/`
- `tsne/index_pages/`
- `legacy_pre_naming_cleanup/`

如果你要查更细的目录含义，直接看：

- [DIRECTORY_MAP_zh.md](/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/DIRECTORY_MAP_zh.md)

## 五、补充说明

- 这些脚本默认都优先从 checkpoint 里的 `config` 读取 `data_roots / mode / region / levels / group_size` 等参数。
- 如果 checkpoint 里没有 `data_roots`，需要在 CLI 里显式提供。
- 患者级分析链路的基础输入是：
  - `window_basis_activations_wide.csv`
  - `patient_activation_profiles.csv`
- `analyze_patient_tsne.py` 与 `build_tsne_index_pages.py` 已经按当前目录层级整理成可复用脚本。
