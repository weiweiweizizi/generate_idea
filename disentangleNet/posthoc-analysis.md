# DisentangleNet Posthoc Analysis

这份文档收纳 `disentangleNet/README.md` 提到的后处理说明：

- 分析与导出入口
- phase 闭包患者与 Basis 重建
- PhaseB Side 激活分析
- Basis 激活分析

只描述 posthoc 流程，不承载训练主线说明。

## 0. 先看这里

这份文档默认以下约定已经成立：

- 你已经完成了训练，且每个目标 `phase` 目录下都有对应的 `best.pt`
- 你希望的不是“训练怎么跑”，而是“训练完之后怎么导出、怎么分析、怎么生成矩阵可视化”
- 所有命令都默认在仓库根目录执行；如果直接跑 `python path/to/script.py`，通常需要先设置 `PYTHONPATH=/home/weizilin/generate_idea`

当前仓库里最常见的两类 run_root 命名是：

- 三阶段：`phaseA / phaseB / phaseC`
- 两阶段：`phaseAB / phaseAB_ft150 / phaseC_ft150`

如果你手头的目录不是这两个模式，先把目录名和本节对齐，再继续往下跑。

如果某个文档段落写的是 “phaseB” 但你实际目录是 `phaseAB`，通常表示：

- 这里说的是“带 side 的中间阶段”
- 具体目录名请以你的 `run_root` 下实际存在的文件夹为准
- 不要硬套 phase 名字，先看脚本是否是按目录逐个扫描的

---

## 一、分析与导出入口

### 导出 basis

```bash
python -m disentangleNet.cli.export_basis export \
  --checkpoint_path outputs/.../best.pt
```

### 导出患者 bundle

```bash
python -m disentangleNet.cli.export_patient export \
  --checkpoint_path outputs/.../best.pt \
  --subject TT_851519
```

分析层详细说明见：

- [analysis/README.md](./analysis/README.md)

---

## 二、phase 闭包患者与 Basis 重建

如果你已经完成了一个完整的 phase 闭包，并且希望把每个 phase 的结果统一导出成：

- 患者级 bundle
- 患者级 `x` 重建
- 患者级 `no_motion_y` preview
- basis 导出
- basis 级 `x` 重建
- basis 级 `no_motion_y` preview

推荐不要手工一条条拼 `export_*`、`generate_*`、`matrix_vis` 命令，而是直接用统一脚本：

### 标准命令

三阶段示例：

```bash
python scripts/matrix_vis/scripts/run_disentanglenet_phase_comparison.py run_all \
  --run_root outputs/disentangleNet_frame/reflex_pair_side_imr_tt \
  --patient_id TTMORECF_851519 \
  --include_initial_reference_window False \
  --lambda_laplacian 1.0 \
  --lambda_area_sign 1.0 \
  --area_barrier_margin 0.05
```

两阶段示例，`{x|y}` 表示把下面路径中的 `x` 替换成 `y`：

```bash
python scripts/matrix_vis/scripts/run_disentanglenet_phase_comparison.py run_all \
  --run_root outputs/disentangleNet_frame/reflex_pair_side_2Phase_TTMORECF_static_side_{x|y} \
  --patient_id TTMORECF-851519 \
  --lambda_laplacian 1.0 \
  --lambda_area_sign 1.0 \
  --area_barrier_margin 0.05
```

这条固定流程会自动完成：

1. 对每个 phase 导出患者 bundle
2. 对每个 phase 做患者级 `x` 重建
3. 使用患者自身初始 landmark 的静止 `y` 做患者级 preview
4. 对每个 phase 导出 basis
5. 对每个 phase 做 basis 级 `x` 重建
6. 使用标准脸模板静止 `y` 做 basis 级 preview
7. 默认不额外求解 `win_000`；如需兼容旧结果，可以显式把 `include_initial_reference_window` 设为 `True`

### 运行前检查清单

下面这些是运行前最好显式确认的项目。它们能减少大部分“命令能跑，但结果语义不对”的情况：

- `run_root` 下每个目标 phase 都存在 `best.pt`
- `patient_id` 在 `data/win20-step20/{数据集名称}/{patient_id}/` 下有唯一匹配目录
- 该目录下存在 `lmks_crop.label`
- 该目录下存在 `win_000_x.npy`
- 如果是第一次跑当前语义，旧的患者输出缓存已清理，或者改用了新的 `run_root`
- 如果直接调用底层 CLI，而不是 `run_all`，要手工传 `initial_landmark_source` 和 `initial_distance_matrix_source`
- 默认不额外求解 `win_000`；如果你需要旧版递推语义，可以显式打开 `include_initial_reference_window=True`
- 如果只想要新患者流程，默认不用打开 `include_standard_face_patient`
- `basis` 路径保持旧规则，不要把患者初值逻辑混进去
- 如果你是直接执行某个 `outputs/.../scripts/*.py`，先把仓库根目录加到 `PYTHONPATH`，否则 `import disentangleNet` 会失败

### 默认固定项

这条统一入口默认把能从仓库路径和患者 ID 推出来的内容都固定掉了，不需要运行者手工判断：

- 患者级 `anchor_point_ids` 固定为 `205, 425, 200`
- 患者级 preview 默认使用患者自身初始 landmark 的静止 `y`
- 患者级 preview 默认关闭 `align_to_anchor`
- 患者级背景灰点默认来自患者 `lmks_crop.label` 第一行归一化结果
- 患者级首窗 `D0` 默认来自同目录下的 `win_000_x.npy`
- 患者级首行 landmark 默认从 `data/win20-step20/{数据集名称}/{patient_id}/lmks_crop.label` 自动解析
- 患者级后续窗口默认沿用上一窗口末帧，并按 `D_{k+1} = D_k + \Delta D_k` 递推
- `phaseA / phaseB / phaseC` 默认作为三阶段闭包
- 两阶段闭包则默认按 `phaseAB / phaseC_ft150` 解释；如果存在 `phaseAB_ft150`，可以单独作为中间微调阶段加入额外对照
- basis 路径继续沿用旧版本模板语义，不受患者初值影响

如果需要额外保留旧的患者 `standardFace` 结果，可以显式开启 `include_standard_face_patient=True`；默认不会跑这条额外分支。

### 结果目录

每个 phase 的结果会固定写回各自目录：

- `phaseA/patient/`
- `phaseA/basis/`
- `phaseB/patient/`
- `phaseB/basis/`
- `phaseC/patient/`
- `phaseC/basis/`

其中：

- `patient/`
  - `patient_bundle/`
  - `matrix_vis_sequence_x_no_motion_y/`
  - `preview_x_no_motion_y/`
- `basis/`
  - `basis_export/`
  - `generated_configs/`
  - `reconstructions/`
  - `preview_x_no_motion_y/`

### 当前固定语义

这条流程现在已经固定了下面这些约束，不需要再手工补参数：

- 患者和 basis 都只重建 `x`
- 患者级 preview 使用患者自身初始 landmark 的静止 `y`；basis preview 继续使用标准脸模板静止 `y`
- 患者级 preview 默认不再对齐到标准模板锚点；背景灰点来自患者自身初始 landmark
- basis 导出继续沿用旧版本导出语义，不使用患者自身初值逻辑
- basis 级重建继续沿用当前 `matrix_vis` 既有模板语义：
  - basis 首窗初值仍使用标准脸模板
  - basis preview 仍使用标准脸模板静止 `y`
  - 这次患者自身初值改动只作用于患者级重建与患者级 preview，不作用于 basis 路径
- 患者级 `x` 重建中，第一个窗口不再使用标准脸模板作为初值，而是使用患者自身初值：
  - 患者初始窗口文件按 `win_000_{x/y}.npy` 命名；当前这条患者级流程只重建 `x`，因此这里使用 `win_000_x.npy`
  - 初始 landmark 使用对应患者目录下 `lmks_crop.label` 的第一行
  - `lmks_crop.label` 按图像坐标读取后，需要先转换到笛卡尔方向，再分别按脸宽 / 脸高归一化到 `[-0.5, 0.5]`
- 患者级首窗初值所在目录固定形如：
  - `data/win20-step20/{数据集名称}/{patient_id}/`
  - 其中数据集名称包括 `TT`、`TTMORE`、`TTMOREC`、`XW`、`IMR` 等
- 患者级后续窗口仍沿用上一窗口末帧作为下一窗口的 `x` 初值
- 患者级重建会把对应患者目录下的 `win_000_x.npy` 作为首个 `D0` 纳入重建；如果后续还有 `N` 个 delta，则会得到 `N+1` 个 `D` 状态，依次满足 `D1 = D0 + delta0`、`D2 = D1 + delta1`、...，并拼接为 `(N+1) * 20` 帧
- 例如有 3 个 delta 时，会重建 4 个 `D` 状态，对应 80 帧
- `run_all` 默认只跑患者新流程和 basis 旧流程；`standardFace` 患者分支是可选的额外对照
- `matrix_vis` 重建会显式打开：
  - graph Laplacian 正则
  - area sign barrier

### 三个 Phase 的区别

- `phaseA`
  - 患者重建只包含 shared 分支
  - basis 只导出 shared basis
- `phaseB`
  - 患者重建包含 shared + side
  - basis 导出 shared + side basis
- `phaseC`
  - 患者重建包含 shared + side + private residual
  - basis 仍只导出 shared + side basis，不单独导 private

更细的后处理说明见：

- [analysis/README.md](./analysis/README.md)
- [../scripts/matrix_vis/README.md](../scripts/matrix_vis/README.md)
- [../scripts/matrix_vis/scripts/README.md](../scripts/matrix_vis/scripts/README.md)

---

## 三、Side 激活分析

如果当前实验的 **带 side 的中间 phase** 使用了 `paired_competitive_side` 结构，并且希望检查：

- side basis 的激活是否携带 side 信息
- 只用激活相关特征能否把患者分成 `Left / Normal / Right`

当前固定做法是直接在目标 phase 目录下建立：

- `side_activation_analysis/`

并运行两步：

1. 提取特征并训练分类器
2. 整理结果并画 confusion matrix

一个已经整理好的参考目录是：

- `outputs/disentangleNet_frame/reflex_pair_side_phaseB_no_fsq_no_side_loss/side_activation_analysis/`

如果你的 run_root 是两阶段闭包，通常要分别对这两个 phase 都跑一遍，例如：

- `outputs/disentangleNet_frame/reflex_pair_side_2Phase_TTMORECF_static_side_x/phaseAB/side_activation_analysis/`
- `outputs/disentangleNet_frame/reflex_pair_side_2Phase_TTMORECF_static_side_x/phaseC_ft150/side_activation_analysis/`
- `outputs/disentangleNet_frame/reflex_pair_side_2Phase_TTMORECF_static_side_y/phaseAB/side_activation_analysis/`
- `outputs/disentangleNet_frame/reflex_pair_side_2Phase_TTMORECF_static_side_y/phaseC_ft150/side_activation_analysis/`

实现路径分成两层：

- 运行时脚本入口
  - `phaseAB/side_activation_analysis/scripts/analyze_side_basis_activations.py`
  - `phaseAB/side_activation_analysis/scripts/organize_side_activation_results.py`
  - `phaseC_ft150/side_activation_analysis/scripts/analyze_side_basis_activations.py`
  - `phaseC_ft150/side_activation_analysis/scripts/organize_side_activation_results.py`
- 这些脚本内部依赖的仓库模块
  - [analysis/loaders](./analysis/loaders/README.md)
  - [data](./data/__init__.py)
  - [training/data.py](./training/data.py)

输出物固定包括：

- `side_activation_group_features.csv`
- `side_activation_patient_features.csv`
- `side_activation_patient_classification_summary.json`
- `side_activation_analysis/figures/all_patients_confusion_matrix.png`
- `side_activation_analysis/figures/imr_patients_confusion_matrix.png`
- `side_activation_analysis/figures/tt_patients_confusion_matrix.png`
  - 如果数据集不是 `TT`，则第三张会对应实际 held-out 集名称，例如 `ttmorecf_patients_confusion_matrix.png`

这条分析链当前默认会尝试三种分类器：

- logistic regression
- ridge classifier
- MLP

并自动选择 accuracy 最好的模型来画 confusion matrix。

### 运行方式

建议直接对每个 phase 单独运行。下面以 `x/phaseAB` 为例，`y/phaseAB`、`x/phaseC_ft150`、`y/phaseC_ft150` 都是同样的跑法，只是把路径替换掉：

```bash
PYTHONPATH=/home/weizilin/generate_idea \
python -u outputs/disentangleNet_frame/reflex_pair_side_2Phase_TTMORECF_static_side_x/phaseAB/side_activation_analysis/scripts/analyze_side_basis_activations.py
python -u outputs/disentangleNet_frame/reflex_pair_side_2Phase_TTMORECF_static_side_x/phaseAB/side_activation_analysis/scripts/organize_side_activation_results.py
```

`analyze_*` 负责产出：

- `side_activation_group_features.csv`
- `side_activation_patient_features.csv`
- `side_activation_patient_classification_summary.json`

`organize_*` 负责把这些结果整理成：

- `README.md`
- `organized_summary.json`
- `figures/*.png`

如果你只跑了 `analyze_*`，那还不算完成；如果只跑了 `organize_*`，它会先找不到上一步的 CSV / JSON。

---

## 四、Basis 激活分析

如果当前实验已经完成了两阶段训练，例如：

- `phaseAB`
- `phaseC_ft150`

并且希望检查：

- shared basis 本身的取值分布
- side basis 本身的取值分布
- shared basis 的 usage / effective coefficient
- side activation / side coefficient

当前固定做法是在目标 `run_root` 下建立：

- `basis_activation_analysis/`

然后直接运行汇总脚本：

```bash
PYTHONPATH=/home/weizilin/generate_idea \
python -u outputs/disentangleNet_frame/reflex_pair_side_2Phase_TTMORECF_static_side_x/basis_activation_analysis/scripts/summarize_basis_stats.py
```

这条脚本会自动读取同一 `run_root` 下的：

- `phaseAB/best.pt`
- `phaseC_ft150/best.pt`

如果 `phaseAB_ft150` 也存在，它不会默认被纳入统计，除非你明确把脚本里的 `PHASES` 改成包含它。

并分别对两个 phase 做 basis 激活统计。一个已经整理好的参考目录是：

- `outputs/disentangleNet_frame/reflex_pair_side_2Phase_TTMORECF_static_side_x/basis_activation_analysis/`
- `outputs/disentangleNet_frame/reflex_pair_side_2Phase_TTMORECF_static_side_y/basis_activation_analysis/`

输出物固定包括：

- `phase_basis_activation_summary.json`
- `phase_basis_activation_detail.csv`

其中 summary 里会固定汇报这些统计项：

- `basis_value_stats/shared_basis`
- `basis_value_stats/side_basis`
- `activation_stats/shared_usage_frame`
- `activation_stats/shared_usage_group`
- `activation_stats/shared_effective_coeff_frame`
- `activation_stats/shared_effective_coeff_group`
- `activation_stats/side_activation_frame`
- `activation_stats/side_activation_group`
- `activation_stats/side_coeff_frame`
- `activation_stats/side_coeff_group`

这条分析链当前默认会：

- 直接从 checkpoint 载入模型
- 读取 `train_config.json` 中的 `group_size`、`levels`、`side_pair_count`
- 用 `valid_mask` 过滤无效帧
- 对 frame 级和 group 级分别汇总统计量
- 将 shared basis 的 usage 按 `levels` 展开成 effective coefficient 再统计
- 只按脚本里列出的 `PHASES` 扫描 checkpoint；当前文档对应的是 `phaseAB` 和 `phaseC_ft150`
- 如果你想把 `phaseAB_ft150` 也纳入统计，需要先确认该目录下确实有 `best.pt`，再显式改脚本中的 `PHASES`

更细的分析说明见：

- [analysis/loaders](./analysis/loaders/README.md)
- [data](./data/__init__.py)
- [training/data.py](./training/data.py)
