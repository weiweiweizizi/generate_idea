# DisentangleNet

`disentangleNet/` 是当前这套面部分解代码的主包。

它已经不再是对旧 `scripts/disentangleNet*` 的薄包装，而是主线训练、模型装配、分析导出、`matrix_vis` 桥接的统一入口。

当前主线目标有三件事：

1. 用显式配置描述模型和训练结构
2. 让 `reflex / lowrank reflex / v31` 通过同一套包内入口运行
3. 让 basis 导出、patient bundle 导出和 `matrix_vis` 重建走稳定契约

---

## 一、当前目录职责

- `config/`
  - 配置 schema
  - 包括 `ModelConfig / TrainConfig / PipelineConfig`
- `config_templates/`
  - 主线实验模板
  - 例如 `reflex_train_template.json`、`reflex_pipeline_template.json`
- `data/`
  - 数据规范、样本读取、dataset 构造、region 辅助
- `models/`
  - 模型层
  - 包括 `encoders / basis_pipeline / basis_modes / side_heads / reconstruction / families`
- `losses/`
  - loss runtime、loss weight builder、Laplacian 图平滑约束
- `training/`
  - 训练入口、训练引擎、checkpoint、validation、pipeline 调度
- `analysis/`
  - checkpoint contract、checkpoint loader、basis / patient 导出
- `bridge/`
  - `disentangleNet -> matrix_vis` 桥接层
- `cli/`
  - 轻量命令行入口
- `init_basis/`
  - 训练初始化 basis 文件
- `legacy/`
  - 归档实现
  - 仅用于回溯、兼容和历史参考，不再是主开发面

---

## 二、当前主线模型家族

当前包内主线主要覆盖两类：

- `v31`
  - semantic side branching 主线
- `modular reflex / lowrank reflex`
  - 共享 trunk + `free / side / private`
  - shared basis 走 `direct` 或 `lowrank`
  - side branch 当前主线是 `structured_reflex_side`

其中 `structured_reflex_side` 的语义是：

- 3 个 side basis
- 1 个 reflex basis
- 1 对 mirrored pair basis

---

## 三、现在的配置驱动边界

当前主线已经是三层配置：

- `ModelConfig`
  - 控模型结构
  - 例如 `basis_mode`、`reflex self/pair`、`side branch type`
- `TrainConfig`
  - 控单阶段训练
  - 例如 optimizer、scheduler、Laplacian / 频域 / side loss 权重、冻结策略、validation
- `PipelineConfig`
  - 控多阶段 pipeline
  - 例如 A/B/C 阶段训练编排

这意味着：

- 新实验优先改 config，不优先改脚本
- checkpoint 和导出阶段也会记录结构化配置

---

## 四、推荐阅读顺序

如果你要理解“当前版本到底怎么组织”，建议按这个顺序看：

1. [config/schema.py](./config/schema.py)
2. [models/README.md](./models/README.md)
3. [training/reflex.py](./training/reflex.py) 或 [training/v31.py](./training/v31.py)
4. [analysis/README.md](./analysis/README.md)
5. [bridge/README.md](./bridge/README.md)

如果你只想看模型结构，直接从：

- [models/README.md](./models/README.md)

开始就够了。

---

## 五、训练入口

### 单阶段 reflex / lowrank reflex

```bash
python -m disentangleNet.cli.train_reflex \
  --config_path=/path/to/reflex_train_config.json
```

### 三阶段 reflex pipeline

```bash
python -m disentangleNet.cli.train_reflex_pipeline \
  --config_path=/path/to/reflex_pipeline_config.json
```

### `v31`

```bash
python -m disentangleNet.cli.train_v31 \
  --config_path=/path/to/v31_train_config.json
```

---

## 六、后处理与分析

这部分已经单独抽到：

- [posthoc-analysis.md](./posthoc-analysis.md)

包括导出、患者与 basis 重建、PhaseB side 激活分析和 basis 激活分析，都在新文档里维护。

---

## 七、与 Matrix-Vis 的桥接

当前 patient bundle 到 `matrix_vis` 的桥接已经收进：

- [bridge/matrix_vis.py](./bridge/matrix_vis.py)

桥接层负责明确这些语义：

- 观测值是什么
- 观测矩阵在哪个空间
- 是否需要按 `observation_scale` 恢复物理量级
- bundle 是否已包含 `private_residual`

详细说明见：

- [bridge/README.md](./bridge/README.md)
- [../docs/disentanglenet_matrix_vis_contract.md](../docs/disentanglenet_matrix_vis_contract.md)

---

## 八、当前状态说明

现在 `disentangleNet/` 已经是主开发面：

- 新训练逻辑优先写这里
- 新模型结构优先写这里
- 新导出和桥接逻辑优先写这里

`legacy/` 仍保留，但它的定位已经是：

- 历史参考
- 旧 checkpoint / 旧分析脚本兼容
- 不再是推荐入口
