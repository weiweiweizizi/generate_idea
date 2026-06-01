# DisentangleNet 模型层

`disentangleNet/models/` 负责回答三件事：

1. 模型结构由哪些模块组成
2. 这些模块如何通过配置组装成不同 family
3. basis、encoder、side、reconstruction 分别落在哪一层


---

## 一、当前目录边界

### `assemblers.py`

配置到模型实例的统一装配入口。

它负责：

- 读取 `ModelConfig`
- 选择 family
- 选择 basis mode
- 选择 side branch
- 把 concrete basis runtime 注入模型

如果你想知道“配置最后会拼成什么模型”，先看这里。

### `config_builders.py`

旧训练配置到 `ModelConfig` 的整理层。

主要做：

- `v31` 旧风格配置转显式 schema
- 初始化 basis 路径兼容

### `families/`

模型 family 本体。

当前主要文件：

- `distnet.py`
  - `v31` 主线
- `v31_forward.py`
  - `v31` 的 forward 后半段 helper
- `v6_distnet.py`
  - `reflex / lowrank reflex` 共享底座
- `lowrank_distnet.py`
  - lowrank 版本
- `lowrank_reflex_distnet.py`
  - lowrank + reflex 版本

family 层负责：

- 保存模型参数
- 组织 forward
- 调用 encoder / basis / side / reconstruction 子模块

### `encoders/`

encoder 结构层。

当前主要包括：

- `basic_block.py`
  - 基础 block
- `builders.py`
  - encoder 零件构造器
- `branching.py`
  - `Layer2BranchingEncoder`
  - `free / side / private` 在 `layer2` 后分支
- `semantic_branching.py`
  - `SemanticBranchingEncoder`
  - 给 `v31` 用

如果你关心“共享 trunk 到哪里、三条 branch 从哪里分开”，重点看这里。

### `basis_pipeline/`

basis 的真实实现层。

当前是三段式流水线：

- `synthesis.py`
  - 合成 raw basis
  - lowrank 时是 `latent/factor -> matrix`
- `correction.py`
  - 结构修正
  - 唯一真实实现是 `project_symmetric_zero_diagonal`
- `reflex.py`
  - reflex / mirrored 组织
- `direct.py`
  - direct basis runtime
- `runtime.py`
  - lowrank basis runtime

这里是 basis 的唯一主实现层。

### `basis_modes/`

basis 模式选择层。

这层不再自己实现 provider 包装树，只做：

- 根据 `ModelConfig.basis.mode_type` 选择 `direct` 或 `lowrank`
- 返回 `basis_pipeline` 的 concrete runtime

当前主线关系是：

- `basis_pipeline/` = 真正实现
- `basis_modes/` = 模式分发
- `assemblers.py` = 唯一注入入口
- `families/*` = 只消费注入进来的 runtime

### `side_heads/`

side branch 层。

当前主要包括：

- `runtime.py`
  - residual side branch runtime
- `features.py`
  - `fold_mouth_chunk_features`
- `factories.py`
  - side branch 配置到 runtime 的轻量转换

当前主线 `structured_reflex_side` 的语义是：

- 3 个 side basis
- 1 个 reflex basis
- 1 对 mirrored pair basis

### `reconstruction/`

shared reconstruction 和 residual 组合层。

当前主要在：

- `shared.py`

负责：

- shared reconstruction
- side/private residual compose
- group-level output 组织

### `sequence_utils.py`

序列辅助工具：

- `flatten_sequence_labels`
- `reshape_sequence_tensor`
- `mean_pool_sequence_tensor`

### 其他公共模块

- `heads.py`
  - 各类 head 构造器
- `quantizers.py`
  - shared quantizer 和量化辅助
- `basis_ops.py`
  - basis 切分、初始化、orth/l1 等辅助
- `frequency.py`
  - 频域正则
- `registry.py`
  - family 分发辅助

---

## 二、推荐阅读顺序

如果你想从上往下理解模型结构，建议按这个顺序看：

1. `disentangleNet/config/schema.py`
2. `disentangleNet/models/assemblers.py`
3. `disentangleNet/models/families/`
4. `disentangleNet/models/encoders/`
5. `disentangleNet/models/basis_pipeline/`
6. `disentangleNet/models/side_heads/`
7. `disentangleNet/models/reconstruction/shared.py`

最短主线路径通常是：

1. `assemblers.py`
2. `families/v6_distnet.py` 或 `families/distnet.py`
3. `encoders/branching.py` 或 `encoders/semantic_branching.py`
4. `basis_pipeline/`
5. `reconstruction/shared.py`

---

## 三、当前两条主线

### `reflex / lowrank reflex`

结构主线：

1. 输入 `signed ΔD`
2. 共享 CNN trunk 到 `layer2`
3. `free / side / private` 三路分支
4. `free` 路走 shared basis reconstruction
5. `side` 路走 structured reflex side residual
6. `private` 路走 nuisance / identity residual
7. 最后做线性加和重建

对应核心文件：

- `families/v6_distnet.py`
- `families/lowrank_reflex_distnet.py`
- `encoders/branching.py`
- `basis_pipeline/runtime.py`
- `reconstruction/shared.py`

### `v31`

结构主线：

1. 输入 `signed ΔD`
2. semantic branching encoder
3. shared discrete basis reconstruction
4. semantic side basis / side coefficient
5. private residual
6. 最后合成输出

对应核心文件：

- `families/distnet.py`
- `families/v31_forward.py`
- `encoders/semantic_branching.py`


---

## 四、阅读时需要知道的现实点

### 1. `basis_ops.py` 不是 basis 主实现

`basis_ops.py` 现在主要是辅助层：

- split
- init load
- orth/l1
- structured basis helper

真正的 basis 生成与修正仍在 `basis_pipeline/`。

### 2. `registry.py` 不是主入口

实际主入口仍然是：

- `assemblers.py`

`registry.py` 只是 family 分发辅助，不是理解结构时的首要文件。
