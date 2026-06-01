# DisentangleNet / Matrix Vis 桥接指南

本文档说明 `disentangleNet` 和 `scripts/matrix_vis` 之间的当前桥接方式。

这个桥接的目标很直接：

- `disentangleNet` 负责导出显式的 basis 和 patient bundle 产物
- `matrix_vis` 只消费这些产物，不再猜测模型家族或观测语义
- 即便后续模型家族变化，预览和重建仍然保持可复现

本文档是下面几项内容的正式参考：

- basis 导出
- patient bundle 导出
- bundle contract
- `matrix_vis` 患者序列重建
- 静态 `y` 的患者预览

---

## 1. 范围

当前桥接支持的流程是：

1. 训练或加载一个 `disentangleNet` checkpoint。
2. 导出用于可视化的 basis 产物。
3. 为某个患者导出 patient bundle。
4. 在 `matrix_vis` 中重建患者序列。
5. 生成最终 preview GIF，`y` 可以固定为标准模板的初始位置。

这个桥接是围绕当前的 `x/mouth` 和患者序列场景设计的，不是一个通用的 latent-space 求解接口。

暂时不在范围内的内容：

- 在 `disentangleNet` 内联合求解 `x` 和 `y`
- 没有显式布局映射产物的 full341 扩展
- 把 `matrix_vis` 直接当作潜变量推理后端
- 通过 checkpoint 名称静默推断语义

---

## 2. 目录结构

当前和桥接相关的代码分布在三个位置：

- `disentangleNet/bridge/`
  - 给下游消费者使用的桥接辅助函数
- `disentangleNet/analysis/contracts/`
  - 版本化的产物契约
- `scripts/matrix_vis/`
  - 消费端：序列重建、预览和组合

典型的桥接输出目录结构如下：

```text
outputs/disentangleNet_lowrank/<run_name>/matrix_vis_exports/
  basis/
    basis_bank_x.npy
    basis_manifest.json
  patients/
    TT_851519/
      patient_851519_x_sequence.npz
      patient_851519_side_predictions.csv
      patient_851519_summary.json
      matrix_vis_sequence_x/
        solution.npz
        window_001/
        window_003/
        ...
      preview_x_static_y/
        preview.gif
        snapshot_last_frame.png
        preview_summary.json
```

具体目录名可能会因 run_name 略有变化，但文件契约是稳定的。

---

## 3. 核心桥接对象

### 3.1 `CheckpointContract`

定义位置：

- `disentangleNet/analysis/contracts/checkpoints.py`

这个对象描述当前加载的 checkpoint 属于哪一类。

关键字段：

- `framework`
  - `disentangleNet` 或 `disentangleNet_lowrank`
- `model_family`
  - `legacy_v31`、`legacy_v6_reflex`、`lowrank`、`lowrank_reflex`
- `mode`
  - 通常是 `x`
- `region`
  - 通常是 `mouth`
- `levels`
  - checkpoint 使用的 basis layout levels
- `basis_size`
  - mouth bridge 通常是 `119`
- `side_branch_type`
  - `legacy_residual_side` 或 `structured_reflex_side`
- `side_basis_count`
  - 实际导出的 side basis 数量

loader 会同时看 checkpoint payload 和同目录下的 `model_config.json`，不会只依赖文件名。

### 3.2 `bundle_contract`

定义位置：

- `disentangleNet/analysis/contracts/patient_bundle.py`

这个 contract 会写入 `patient_*_summary.json`，并且是 `matrix_vis` 解释 patient bundle 的权威来源。

当前字段：

- `framework`
- `mode`
- `region`
- `matrix_size`
- `signed_normalize`
- `value_semantics`
- `observation_matrix_space`
- `observation_scale_semantics`
- `composition_rule`
- `includes_private_residual`

最重要的语义字段是：

- `value_semantics = mean_distance_delta`
  - 每个表格值表示一个带符号的距离增量
- `observation_matrix_space = normalized_input_space`
  - bundle 保存的是归一化值，求解前必须反归一化
- `observation_scale_semantics = per_window_restore_scale`
  - 每个窗口可以有自己的 scale 因子
- `composition_rule = shared_coeff_weighted_basis_plus_private_residual`
  - basis 矩阵由 shared coefficients 组合，必要时再加 private residual
- `includes_private_residual = true`
  - patient bundle 已经包含 private residual 贡献

### 3.3 `PatientBundleBridge`

定义位置：

- `disentangleNet/bridge/matrix_vis.py`

这是 `scripts/matrix_vis/pipelines/patient_sequence.py` 直接消费的运行时对象。

它暴露这些内容：

- `bundle_path`
- `data`
- `metadata`
- `contract`
- `mode`
- `matrix_size`
- `dataset_name`
- `subject`
- `group_ids`
- `observation_scales`

下游代码应该直接使用这个对象，而不是重复解析 summary JSON 或自行猜 contract。

---

## 4. basis 导出契约

basis 导出是第一个桥接产物。

当前 CLI：

```bash
python -m disentangleNet.cli.export_basis export \
  --checkpoint_path outputs/disentangleNet_lowrank/<run_name>/best.pt
```

期望产物：

- `basis_bank_x.npy`
- `basis_manifest.json`
- 可选的 heatmap 或诊断图

### 4.1 `basis_bank_x.npy`

- dtype：`float32`
- shape：`[K, 119, 119]`
- 语义：每个切片是一张导出的 basis 矩阵，默认仍在 normalized input space，除非导出器明确说明不同语义

### 4.2 `basis_manifest.json`

这个文件必须能自描述导出内容。

重要字段：

- `checkpoint_path`
- `framework`
- `model_family`
- `mode`
- `region`
- `matrix_size`
- `num_basis`
- `levels`
- `side_branch_type`
- `side_basis_count`
- `point_layout`
- `point_layout_region_names`
- `value_semantics`
- `exported_basis_path`

对于当前 mouth-region bridge：

- `mode = x`
- `region = mouth`
- `matrix_size = 119`
- `point_layout = face_regions_grouped`
- `point_layout_region_names = ["around_mouth", "mouth"]`

这个 manifest 是下游消费者的权威来源，`matrix_vis` 不应该只靠 `.npy` 自己推断 basis 数量。

---

## 5. patient bundle 导出契约

patient bundle 导出是第二个桥接产物。

当前 CLI：

```bash
python -m disentangleNet.cli.export_patient export \
  --checkpoint_path outputs/disentangleNet_lowrank/<run_name>/best.pt \
  --subject TT_851519
```

期望产物：

- `patient_<subject>_x_sequence.npz`
- `patient_<subject>_side_predictions.csv`
- `patient_<subject>_summary.json`

### 5.1 `patient_*_x_sequence.npz`

这是 `matrix_vis` 做序列重建时消费的 bundle。

重要数组：

- `window_indices`
  - 有效窗口 id，按升序排列
- `prev_window_indices`
  - 每个有效窗口对应的前一个窗口 id
- `side_pred`
  - 与窗口对齐的 side 预测
- `side_true`
  - 如果可用，则是 ground-truth side label
- `basis_coeffs`
  - 每个窗口的 basis 混合系数
- `basis_usage`
  - 可选，每个 basis 的使用情况汇总
- `composed_basis_matrices`
  - shape `[W, 119, 119]`
  - 组合后的观测矩阵，通常存的是 normalized input space
- `observation_scales`
  - 可选，每个窗口的 scale 因子，用于恢复物理值
- `group_id`
  - 可选，每个窗口的组标识

这个 bundle 是窗口级的，不是 latent vector dump。

### 5.2 `patient_*_side_predictions.csv`

这个文件主要用于检查和快速 sanity check。

典型列：

- `dataset_name`
- `subject`
- `window_idx`
- `prev_window_idx`
- `side_pred`
- `side_label_name`
- 以及一些系数汇总列（如果导出器提供）

### 5.3 `patient_*_summary.json`

这里是桥接契约真正落地的地方。

重要字段：

- `checkpoint_path`
- `dataset_name`
- `subject`
- `mode`
- `region`
- `matrix_size`
- `num_valid_windows`
- `point_layout`
- `point_layout_region_names`
- `composition_rule`
- `bundle_contract`

其中 `bundle_contract` 应该包含第 3.2 节列出的语义字段。

---

## 6. `matrix_vis` 如何消费桥接产物

### 6.1 序列重建

消费端：

- `scripts/matrix_vis/pipelines/patient_sequence.py`

当前流程：

1. 通过 `load_patient_bundle_bridge(...)` 读取 bundle。
2. 从 bridge 中读取 `bundle_contract` 和 `observation_scales`。
3. 构建当前 axis 对应的标准 mesh projection。
4. 对每个有效窗口：
   - 取 `composed_basis_matrices[t]`
   - 如果 bundle 属于 `normalized_input_space`，则先恢复物理尺度
   - 把矩阵转成 observation table
   - 必要时裁剪不可能的负距离目标
   - 求解 axis QP
5. 将每个窗口的轨迹拼成一个最终 solution 文件。

这里最重要的语义规则是：

- 如果 `observation_matrix_space = normalized_input_space`，则必须先用 `observation_scales[t]` 反归一化
- 如果 bundle 已经是物理空间，就不需要做 scale 恢复

负责这个恢复的 bridge helper 是：

- `disentangleNet.bridge.matrix_vis.restore_physical_observation_scale(...)`

### 6.2 静态 `y` 预览

消费端：

- `scripts/matrix_vis/pipelines/preview_real_mouth_regions.py`

preview 阶段会消费一条 `x` solution，再加上一条可选的 `y` solution。

如果 `static_y=True`：

- 不需要 `y_solution` 文件
- `y` 由 canonical mesh projection 生成
- `y` 轨迹在时间上固定在标准模板的初始位置

这就是用户想看 x-only 运动、同时把 y 轴冻住时使用的模式。

典型命令：

```bash
python scripts/matrix_vis/scripts/preview_real_mouth_regions.py \
  --x-solution outputs/disentangleNet_lowrank/<run_name>/matrix_vis_exports/patients/TT_851519/matrix_vis_sequence_x/solution.npz \
  --static-y \
  --anchor-point-ids 205,425,200 \
  --output-dir outputs/matrix_vis/TT_851519_preview_static_y
```

输出目录会包含：

- `preview.gif`
- `snapshot_last_frame.png`
- `aligned_subset_motion.npz`
- `preview_summary.json`
- 每一帧的 PNG

---

## 7. 端到端常用流程

### 7.1 训练或加载 checkpoint

使用新的 CLI 入口：

```bash
python -m disentangleNet.cli.train_reflex \
  --epochs=1 \
  --output_dir=outputs/disentangleNet_lowrank/demo_run
```

### 7.2 导出 basis

```bash
python -m disentangleNet.cli.export_basis export \
  --checkpoint_path outputs/disentangleNet_lowrank/demo_run/best.pt
```

### 7.3 导出 patient bundle

```bash
python -m disentangleNet.cli.export_patient export \
  --checkpoint_path outputs/disentangleNet_lowrank/demo_run/best.pt \
  --subject TT_851519
```

### 7.4 重建患者序列

```bash
python -m scripts.matrix_vis.cli.reconstruct_patient_sequence reconstruct \
  --patient_bundle_path outputs/disentangleNet_lowrank/demo_run/matrix_vis_exports/patients/TT_851519/patient_851519_x_sequence.npz
```

### 7.5 运行静态 `y` preview

```bash
python scripts/matrix_vis/scripts/preview_real_mouth_regions.py \
  --x-solution outputs/disentangleNet_lowrank/demo_run/matrix_vis_exports/patients/TT_851519/matrix_vis_sequence_x/solution.npz \
  --static-y \
  --anchor-point-ids 205,425,200 \
  --output-dir outputs/matrix_vis/TT_851519_preview_static_y
```

---

## 8. bridge helper API

如果你想在 Python 里直接读取 bundle，而不是通过 CLI：

```python
from disentangleNet.bridge.matrix_vis import load_patient_bundle_bridge

bridge = load_patient_bundle_bridge(
    "outputs/disentangleNet_lowrank/<run_name>/matrix_vis_exports/patients/TT_851519/patient_851519_x_sequence.npz"
)

print(bridge.metadata.get("framework"))
print(bridge.contract["value_semantics"])
print(bridge.matrix_size)
print(bridge.observation_scales)
```

以下场景建议优先使用 bridge 对象：

- 在重建前先检查语义
- 需要根据 `observation_matrix_space` 决定行为
- 想避免在多个位置重复解析 JSON

---

## 9. 常见失败模式

### 9.1 `x_solution` 传错文件

`preview_real_mouth_regions.py` 需要的是 axis solution 文件，不是原始 patient bundle `.npz`。

正确：

- `.../matrix_vis_sequence_x/solution.npz`

错误：

- `.../patient_851519_x_sequence.npz`

### 9.2 bundle 是归一化的，但没有恢复尺度

如果 bundle 标记为 `normalized_input_space`，但求解器把它当成物理距离增量直接使用，重建结果就可能跑出脸部区域。这个时候必须先做 scale 恢复。

### 9.3 误解 static `y`

`static_y=True` 不是“忽略 y”。

它的含义是：

- 用 canonical mesh projection 构建固定的 y 轨迹
- 让 y 在整个时间轴上保持模板初始位置

### 9.4 布局不匹配

当前桥接使用的是 grouped mouth layout：

- `around_mouth`
- `mouth`

如果后面引入 full341 或其他布局，桥接必须导出显式映射产物，不能继续默默复用 grouped layout。

### 9.5 缺少 bundle contract

如果 `patient_*_summary.json` 里没有 `bundle_contract`，bridge 会退回到一个安全的默认 contract。这样可以兼容旧产物，但新导出应该始终写入显式 contract。

---

## 10. 当前状态

当前桥接已经可以正常工作，并且已经验证过。

已验证的产物包括：

- 患者序列重建
- 静态 `y` preview 生成
- bundle contract 解析
- per-window observation scale 恢复

现在 bridge 不再只是一些临时 legacy helper，而是一个独立的包层。

后续工作应继续保持 contract 的显式性和版本化，不要再回退到靠 checkpoint 名称或旧脚本路径做推断。
