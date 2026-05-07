# `scripts/disentangleNet` 中文说明

这个目录是从 `scripts/lq` 冻结出来的 `v31` 训练与分析闭包。  
它的定位不是通用包，而是“可复现实验 + 可解释性分析 + 与 `matrix_vis` 桥接”的研究工作目录。

如果你要快速理解它，最重要的是先分清三条主线：

1. **训练主线**
   - 从 `train.py` 或现成 shell 脚本启动 `v31` 训练。
2. **分析主线**
   - 从 checkpoint 出发，做 basis / side / probe / 患者级共激活 / t-SNE。
3. **`matrix_vis` 桥接主线**
   - 把学到的 basis 或患者窗口系数组合矩阵导出，再投影回局部 landmark 轨迹。

## 一、目录定位

当前 `scripts/disentangleNet` 主要包含：

- `train.py`
  - `v31` 的冻结训练入口。
- `data/`、`model/`、`training/`
  - 训练和推理时需要的最小运行闭包。
- `init_basis/`
  - `v31` 训练时用到的 basis 初始化器。
- `analysis/`
  - checkpoint 级分析、窗口级导出、患者级统计、共激活、t-SNE、`matrix_vis` 导出。
- `tests/`
  - 与 `matrix_vis` 导出有关的测试。

当前目录**刻意不再保留**大量历史兼容层、旧实验入口和 `scripts/lq` 的 fallback import。  
它的目标是把 `v31` 路径本身压缩成一个可单独理解和复用的研究快照。

## 二、当前保留的 `v31` 关键设置

当前 README 所描述的分析链主要对应如下设定：

- `mode=x`
- `region=mouth`
- `levels=2,6`
- `quantizer_type=residual_fsq`
- `basis_orthogonalization=joint_global_qr`
- `side_semantic_enabled=True`
- `side_basis_count=3`
- `side_pooling=fixed_region2_contrast`
- `early_branch_factorization=True`

在这个设定下：

- free basis 共 `8` 个
- side basis 共 `3` 个
- 总 basis 数为 `11`

## 三、你通常会从哪里开始

### 1. 想复现训练

直接从现成脚本开始：

```bash
bash scripts/disentangleNet/run_train_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe.sh
```

### 2. 想快速判断 checkpoint 学到了什么

优先看 checkpoint 级分析：

```bash
python scripts/disentangleNet/analysis/analyze_checkpoint.py \
  outputs/disentangleNet/v31_current_verify/best.pt

python scripts/disentangleNet/analysis/analyze_side_interpretability.py \
  outputs/disentangleNet/v31_current_verify/best.pt

python scripts/disentangleNet/analysis/analyze_kfold_report.py \
  outputs/disentangleNet/v31_current_verify/best.pt
```

### 3. 想分析“每个患者/每个窗口具体激活了哪些 basis”

优先走患者级分析链：

1. `export_window_basis_activations.py`
2. `analyze_patient_activation_patterns.py`
3. `analyze_class_coactivation_patterns.py`
4. `analyze_patient_tsne.py`
5. `build_tsne_index_pages.py`

这条链是当前这批新增分析脚本的主线。

### 4. 想把 basis 或患者窗口重建成可视化轨迹

走 `matrix_vis` 桥接链，分成两种输入：

1. **basis 级轨迹重建**
2. **患者窗口级轨迹重建**

这两条链在下面第六节单独展开。

## 四、分析脚本全景

更细的脚本分组见 [analysis/README.md](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/README.md)。  
这里先给一个总览。

### 1. Checkpoint 级分析

- [analyze_checkpoint.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_checkpoint.py)
  - 对 checkpoint 做总体剖析。
  - 会看 reconstruction、free/side 表征、basis bank、线性 probe 等。

- [analyze_side_interpretability.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_side_interpretability.py)
  - 专门分析 side basis 的可解释性。

- [analyze_kfold_report.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_kfold_report.py)
  - 在患者级做 k-fold probe 评估。

### 2. 窗口级 basis 激活导出

- [export_window_basis_activations.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/export_window_basis_activations.py)
  - 当前最关键的导出脚本。
  - 它会重新按 `disentangleNet` 的 grouped window 逻辑跑一遍数据，
    只保留真实窗口，导出：
    - `basis_usage`
    - 原始 `free_coeff_l0 / free_coeff_l1 / side_coeff`
    - `basis_activation = usage * coeff`
  - 输出：
    - `window_basis_activations_wide.csv`
    - `window_basis_activations_long.csv`

### 3. 患者级模式分析

- [analyze_patient_activation_patterns.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py)
  - 从窗口级表聚合到患者级。
  - 输出 `patient_activation_profiles.csv` 及多种按 side / dataset 的摘要表。

- [analyze_class_coactivation_patterns.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py)
  - 在患者级 basis 向量上做共激活模式分析。
  - 当前支持：
    - `side_label_name`
    - `dataset_name`
    - `score`
    - `label_5class`

### 4. 患者级 t-SNE

- [analyze_patient_tsne.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/analyze_patient_tsne.py)
  - 对患者级 `usage / activation / combined` 做 2D / 3D t-SNE。
  - 支持三套 basis 范围：
    - `all_basis`
    - `no_side`
    - `side_only`
  - `3D combined` 额外导出旋转 GIF。

- [build_tsne_index_pages.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/build_tsne_index_pages.py)
  - 把 `All / IMR / TT` 和 `All Basis / No Side / Side Only` 拼成 3x3 总览图。

### 5. `matrix_vis` 桥接导出

- [export_matrix_vis_basis.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/export_matrix_vis_basis.py)
  - 导出 free / side basis bank 及 manifest，供 `matrix_vis` 重建 basis 轨迹。

- [export_matrix_vis_patient.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/export_matrix_vis_patient.py)
  - 导出单个患者每个真实窗口的系数组合矩阵序列，供 `matrix_vis` 重建窗口级轨迹。

## 五、推荐分析链

### 5.1 从 checkpoint 到患者级解释

如果你的目标是“理解模型到底学到了哪些 basis 以及不同患者如何激活它们”，建议按这个顺序跑：

### 第一步：导出窗口级 basis 激活

```bash
python scripts/disentangleNet/analysis/export_window_basis_activations.py export \
  --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \
  --split all
```

输出默认在：

- `outputs/disentangleNet/v31_current_verify/window_basis_activations_all/`

### 第二步：构建患者级摘要

```bash
python scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py analyze
```

输出默认在：

- `outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/patient_profile_summary/`

### 第三步：做不同标签体系下的共激活分析

示例一，按 `side_label_name`：

```bash
python scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py analyze \
  --class_col side_label_name \
  --feature_family activation \
  --n_perm 4000
```

示例二，按 `label_5class`：

```bash
python scripts/disentangleNet/analysis/analyze_class_coactivation_patterns.py analyze \
  --class_col label_5class \
  --feature_family activation \
  --n_perm 4000
```

输出默认在：

- `.../patient_pattern_analysis/coactivation/by_<class_col>/<feature_family>/`

### 第四步：做患者级 t-SNE

全 basis：

```bash
python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \
  --output_root outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne/all/all_basis
```

忽略 side basis：

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

IMR-only：

```bash
python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \
  --patient_profiles_csv outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/patient_profile_summary/patient_activation_profiles_imr.csv \
  --output_root outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne/imr/all_basis
```

TT-only：

```bash
python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \
  --patient_profiles_csv outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/patient_profile_summary/patient_activation_profiles_tt.csv \
  --output_root outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne/tt/all_basis
```

### 第五步：拼总览页

```bash
python scripts/disentangleNet/analysis/build_tsne_index_pages.py build
```

输出默认在：

- `.../patient_pattern_analysis/tsne/index_pages/`

### 5.2 只想快速做 checkpoint 级 sanity check

优先跑这三步：

```bash
python scripts/disentangleNet/analysis/analyze_checkpoint.py \
  outputs/disentangleNet/v31_current_verify/best.pt

python scripts/disentangleNet/analysis/analyze_side_interpretability.py \
  outputs/disentangleNet/v31_current_verify/best.pt

python scripts/disentangleNet/analysis/analyze_kfold_report.py \
  outputs/disentangleNet/v31_current_verify/best.pt
```

## 六、如何串联到 `matrix_vis`

当前 `disentangleNet -> matrix_vis` 桥接明确支持两类输入：

1. **basis-wise 轨迹重建**
2. **patient-wise 窗口轨迹重建**

桥接契约见：

- [docs/disentanglenet_matrix_vis_contract.md](/home/weizilin/generate_idea/docs/disentanglenet_matrix_vis_contract.md)

`matrix_vis` 侧的脚本说明见：

- [scripts/matrix_vis/scripts/README.md](/home/weizilin/generate_idea/scripts/matrix_vis/scripts/README.md)

### 6.1 Basis 对应的轨迹重建

这条链的目标是：  
把 `disentangleNet` 学到的每一个 free / side basis，单独解释成 mouth-region 上的一个局部运动轨迹。

### 步骤 A：从 checkpoint 导出 basis bundle

入口：

- [export_matrix_vis_basis.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/export_matrix_vis_basis.py)

命令：

```bash
python scripts/disentangleNet/analysis/export_matrix_vis_basis.py export \
  --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt
```

默认输出：

- `outputs/disentangleNet/v31_current_verify/matrix_vis_exports/basis/`

关键产物：

- `basis_bank_x.npy`
- `side_basis_bank_x.npy`
- `basis_manifest.json`

### 步骤 B：批量把每个 basis 投影成轨迹

**这是 basis 轨迹重建的主入口。**

入口：

- [run_disentanglenet_basis_batch.py](/home/weizilin/generate_idea/scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py)

命令：

```bash
python scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py run \
  --manifest_path outputs/disentangleNet/v31_current_verify/matrix_vis_exports/basis/basis_manifest.json
```

如果只想调试前几个 basis：

```bash
python scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py run \
  --manifest_path outputs/disentangleNet/v31_current_verify/matrix_vis_exports/basis/basis_manifest.json \
  --limit_basis 3
```

这一步会自动完成：

1. 生成 fixed-y 和逐 basis 的 axis-x YAML
2. 跑每个 basis 的 `axis_x` 重建
3. 用 fixed-y 解合成 mouth-region preview
4. 可选地再生成 no-motion-y 对照 preview

主要输出：

- 生成的 YAML：
  - `scripts/matrix_vis/configs/real/disentanglenet/generated/<run_name>/<anchor_tag>/`
- x 轴重建：
  - `outputs/matrix_vis/real/disentanglenet/<run_name>/<anchor_tag>/<basis_label>_<idx>/axis_x/`
- preview：
  - `outputs/matrix_vis/real_preview/disentanglenet/<run_name>/<anchor_tag>/fixed_y/...`
  - `outputs/matrix_vis/real_preview/disentanglenet/<run_name>/<anchor_tag>/no_motion_y/...`

### 手动拆开跑时的补充入口

如果你不想走一键批处理，也可以拆成两步：

1. 用 [generate_disentanglenet_basis_configs.py](/home/weizilin/generate_idea/scripts/matrix_vis/scripts/generate_disentanglenet_basis_configs.py) 先只生成 YAML
2. 再用 `matrix_vis` 的单轴 CLI：

```bash
python scripts/matrix_vis/cli/reconstruct_axis.py reconstruct --config <axis_x_yaml>
```

以及单独预览：

```bash
python scripts/matrix_vis/scripts/preview_real_mouth_regions.py \
  --x-solution <axis_x_solution.npz> \
  --y-solution <fixed_y_solution.npz> \
  --output-dir <preview_dir>
```

但对于 basis 批量分析来说，通常没有必要绕开 `run_disentanglenet_basis_batch.py`。

### 6.2 患者窗口对应的轨迹重建

这条链的目标是：  
把某个患者每个真实窗口的 free/side 系数组合矩阵导出，再把每个窗口的 `x` 观测矩阵重建成时序轨迹。

### 步骤 A：从 checkpoint 导出患者窗口 bundle

入口：

- [export_matrix_vis_patient.py](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/export_matrix_vis_patient.py)

命令：

```bash
python scripts/disentangleNet/analysis/export_matrix_vis_patient.py export \
  --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \
  --subject 844697
```

默认输出：

- `outputs/disentangleNet/v31_current_verify/matrix_vis_exports/patients/TT_844697/`

关键产物：

- `patient_844697_x_sequence.npz`
- `patient_844697_side_predictions.csv`
- `patient_844697_summary.json`

这里导出的是：

- 每个真实窗口的 `shared_basis_coeffs`
- `side_basis_coeffs`
- `free_path_usage`
- `side_path_usage`
- `composed_basis_matrices`

### 步骤 B：把患者窗口 bundle 重建成轨迹

**这是患者窗口轨迹重建的主入口。**

入口：

- [scripts/matrix_vis/cli/reconstruct_patient_sequence.py](/home/weizilin/generate_idea/scripts/matrix_vis/cli/reconstruct_patient_sequence.py)

命令：

```bash
python scripts/matrix_vis/cli/reconstruct_patient_sequence.py reconstruct \
  --patient_bundle_path outputs/disentangleNet/v31_current_verify/matrix_vis_exports/patients/TT_844697/patient_844697_x_sequence.npz
```

如果想指定输出目录：

```bash
python scripts/matrix_vis/cli/reconstruct_patient_sequence.py reconstruct \
  --patient_bundle_path outputs/disentangleNet/v31_current_verify/matrix_vis_exports/patients/TT_844697/patient_844697_x_sequence.npz \
  --output_dir outputs/matrix_vis/patient_sequence/TT_844697
```

这一步会按窗口序列逐个重建患者的 `x` 方向局部轨迹。  
当前第一版桥接里，`y` 仍然保持静止。

## 七、当前输出目录约定

### 7.1 患者级解释链

窗口级 basis 激活默认在：

- `outputs/disentangleNet/<run_name>/window_basis_activations_<split>/`

针对当前 `v31_current_verify/all`，核心目录是：

- [window_basis_activations_all](/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all)

患者级模式分析默认在：

- [patient_pattern_analysis](/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis)

目前整理后的主树是：

- `patient_profile_summary/`
- `coactivation/by_<class_col>/<feature_family>/`
- `tsne/<cohort>/<basis_scope>/`
- `tsne/index_pages/`
- `legacy_pre_naming_cleanup/`

更详细的结构说明见：

- [DIRECTORY_MAP_zh.md](/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/DIRECTORY_MAP_zh.md)

### 7.2 `matrix_vis` 桥接链

basis 导出默认在：

- `outputs/disentangleNet/<run_name>/matrix_vis_exports/basis/`

患者 bundle 导出默认在：

- `outputs/disentangleNet/<run_name>/matrix_vis_exports/patients/<dataset_subject>/`

basis 轨迹重建与 preview 默认在：

- `outputs/matrix_vis/real/disentanglenet/<run_name>/<anchor_tag>/...`
- `outputs/matrix_vis/real_preview/disentanglenet/<run_name>/<anchor_tag>/...`



如果站在第一次接手这个目录的人角度，这份 README 现在仍有两个小障碍：

1. `run_name` 和默认输出目录较多，第一次看时不一定能马上知道自己的结果会落在哪里。
2. `matrix_vis` 这条桥接链依赖另一个子系统，虽然入口已经写清，但如果完全没接触过 `matrix_vis`，还需要再跳去看：
   - [scripts/matrix_vis/README.md](/home/weizilin/generate_idea/scripts/matrix_vis/README.md)
   - [scripts/matrix_vis/scripts/README.md](/home/weizilin/generate_idea/scripts/matrix_vis/scripts/README.md)

