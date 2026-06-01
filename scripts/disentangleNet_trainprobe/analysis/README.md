# `scripts/disentangleNet_trainprobe/analysis` 中文说明

这个目录存放 `disentangleNet_trainprobe` 的分析、导出与可视化脚本。

目标是和 `scripts/disentangleNet/analysis` 保持基本一致的调用方式、输出目录层级和产物命名，便于两条线并行比较。

## 一、脚本分组

### 1. Checkpoint 级分析

- `analyze_checkpoint.py`
  - 对单个 checkpoint 做总体分析。
  - 导出 free / side basis、group-level representation、线性 probe summary。

- `analyze_side_interpretability.py`
  - 分析 side basis 统计、左右交换对称性、group-level side semantics。

- `analyze_kfold_report.py`
  - 对 group-level 表征做 subject 级 k-fold probe 评估。
  - 输出 fold 分配、逐折指标、预测结果和 Markdown 报告。

### 2. Basis 激活导出与患者级统计

- `export_window_basis_activations.py`
  - 导出窗口级 `usage / coeff / activation` 的宽表和长表。
  - 支持 `export` 和 `export_oof`。

- `analyze_patient_activation_patterns.py`
  - 将窗口级结果聚合到患者级。
  - 输出 `patient_activation_profiles.csv`、group summary、排行和报告。

- `analyze_class_coactivation_patterns.py`
  - 在患者级向量上做分组共激活分析。
  - 输出相关矩阵、delta heatmap、置换检验和报告。

### 3. t-SNE 可视化

- `analyze_patient_tsne.py`
  - 对患者级 `usage / activation / combined` 做 2D / 3D t-SNE。

- `build_tsne_index_pages.py`
  - 将 `all / imr / tt` 与 `all_basis / no_side / side_only` 拼成总览页。

### 4. Matrix-Vis 导出

- `export_matrix_vis_basis.py`
  - 导出 basis bundle、manifest 和 summary。

- `export_matrix_vis_patient.py`
  - 导出患者窗口序列 bundle、side prediction CSV 和 summary。

## 二、推荐执行顺序

如果目标是从 checkpoint 到患者级解释与可视化，推荐：

1. `export_window_basis_activations.py export --split all`
2. `analyze_patient_activation_patterns.py analyze`
3. `analyze_class_coactivation_patterns.py analyze`
4. `analyze_patient_tsne.py analyze`
5. `build_tsne_index_pages.py build`

如果目标是先看 checkpoint 学到的结构，推荐：

1. `analyze_checkpoint.py`
2. `analyze_side_interpretability.py`
3. `analyze_kfold_report.py`

## 三、常用命令

下面默认以：

- `outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/best.pt`

作为示例 checkpoint。

### 1. 导出窗口级 basis 激活

```bash
python scripts/disentangleNet_trainprobe/analysis/export_window_basis_activations.py export \
  --checkpoint_path outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/best.pt \
  --split all
```

### 2. 构建患者级摘要

```bash
python scripts/disentangleNet_trainprobe/analysis/analyze_patient_activation_patterns.py analyze
```

### 3. 共激活分析

```bash
python scripts/disentangleNet_trainprobe/analysis/analyze_class_coactivation_patterns.py analyze \
  --class_col side_label_name \
  --feature_family activation \
  --n_perm 4000
```

### 4. 患者级 t-SNE

```bash
python scripts/disentangleNet_trainprobe/analysis/analyze_patient_tsne.py analyze \
  --output_root outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/window_basis_activations_all/patient_pattern_analysis/tsne/all/all_basis
```

### 5. t-SNE 总览页

```bash
python scripts/disentangleNet_trainprobe/analysis/build_tsne_index_pages.py build
```

### 6. Checkpoint 总体分析

```bash
python scripts/disentangleNet_trainprobe/analysis/analyze_checkpoint.py \
  outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/best.pt
```

### 7. Side interpretability

```bash
python scripts/disentangleNet_trainprobe/analysis/analyze_side_interpretability.py \
  outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/best.pt
```

### 8. K-fold probe 报告

```bash
python scripts/disentangleNet_trainprobe/analysis/analyze_kfold_report.py \
  outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/best.pt
```

### 9. Matrix-vis 导出

```bash
python scripts/disentangleNet_trainprobe/analysis/export_matrix_vis_basis.py export \
  --checkpoint_path outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/best.pt
```

```bash
python scripts/disentangleNet_trainprobe/analysis/export_matrix_vis_patient.py export \
  --checkpoint_path outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50/best.pt \
  --subject 844697
```

患者窗口重建默认行为：

- 初始脸部位置使用标准 canonical face mesh
- full341 布局默认锚点使用 `33,263,10,175`
- 单轴重建时，另一轴默认保持静止

如果要同时用 `x` / `y` 两个 checkpoint 生成三组患者预览：

1. `x` 固定不动，重建 `y`
2. `y` 固定不动，重建 `x`
3. `x` / `y` 都重建

可以直接运行：

```bash
python scripts/matrix_vis/scripts/run_disentanglenet_patient_dual_axis.py run \
  --subject 844697 \
  --x_checkpoint_path outputs/disentangleNet_trainprobe/v32_tri_region_private_win20_e30_gpu_bs16/best.pt \
  --y_checkpoint_path outputs/disentangleNet_trainprobe/v32_tri_region_masked_y_win20_e30_bs16_openmmlab/best.pt
```

输出默认在：

- `outputs/matrix_vis/patient_dual_axis/disentanglenet_trainprobe/<dataset_subject>/`

## 四、目录约定

关键输出层级和主线尽量对齐：

- `window_basis_activations_all/`
- `window_basis_activations_all/patient_pattern_analysis/patient_profile_summary/`
- `window_basis_activations_all/patient_pattern_analysis/coactivation/`
- `window_basis_activations_all/patient_pattern_analysis/tsne/`
- `window_basis_activations_all/patient_pattern_analysis/tsne/index_pages/`
- `analysis/`
- `kfold_report/`
- `matrix_vis_exports/basis/`
- `matrix_vis_exports/patients/`
