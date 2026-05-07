# Repository Document Index

这份索引面向“快速找资料”和“快速判断去哪看”的场景。  
重点列出仓库里各子目录下已有的 `README` / 关键 `.md`，并补一行说明该目录大致负责什么。

---

## 1. 顶层总览

### 1.1 先看这些

- [AGENTS.md](/home/weizilin/generate_idea/AGENTS.md)
  - 仓库协作约定、目录组织、常用命令、数据与输出 hygiene。
- [CLAUDE.md](/home/weizilin/generate_idea/CLAUDE.md)
  - 项目背景、研究方向、主要脚本入口、关键实验结果摘要。
- [IDEA_REPORT.md](/home/weizilin/generate_idea/IDEA_REPORT.md)
  - 当前主线的高层研究判断与关键数据总结。
- [IDEA_EXPERIMENTS.md](/home/weizilin/generate_idea/IDEA_EXPERIMENTS.md)
  - 最详细的实验账本，适合追版本演化和结果出处。
- [RESEARCH_PROGRESS.md](/home/weizilin/generate_idea/RESEARCH_PROGRESS.md)
  - 当前有效结论、风险和下一步重点。

### 1.2 顶层目录作用

- `data/`
  - 原始数据与少量被视为数据本体的派生目录。
- `docs/`
  - 设计说明、过程笔记、计划与补充文档。
- `literature_notes/`
  - 文献阅读笔记。
- `outputs/`
  - 训练输出、分析结果、可视化产物。
- `pilot_feasibility_check/`
  - 整理前的旧版研究文档快照。
- `scripts/`
  - 代码主目录，包括早期 feasibility、`lq`、`disentangleNet`、`matrix_vis`、`val_codebook`。

---

## 2. `scripts/` 索引

### 2.1 `scripts/pilot_feasibility/`

- README: [scripts/pilot_feasibility/README.md](/home/weizilin/generate_idea/scripts/pilot_feasibility/README.md)
  - 说明早期 feasibility 实验已经按方法重组，输出统一到 `outputs/pilot_feasibility/...`。

子目录：

- `svd/`
  - 早期 SVD 单患者、多患者、raw-vs-diff、grouped 等脚本。
- `dmd/`
  - DMD 单患者、多患者、与 blendshape 相关性验证。
- `grassmann/`
  - Grassmann 基空间相似性与 grouped 对齐分析。
- `blendshape/`
  - SVD 时间系数与 blendshape/AU 相关性分析。
- `nmf/`
  - NMF baseline 尝试。
- `tucker/`
  - Tucker 多患者分解尝试。

### 2.2 `scripts/lq/`

- 无目录级 README。
- 这是 `disentangleNet` 之前的结构探索场，保留训练、模型、数据、初始化 basis 与辅助工具。

关键相关文档：

- [IDEA_REPORT.md](/home/weizilin/generate_idea/IDEA_REPORT.md)
  - 当前如何定位 `scripts/lq`。
- [IDEA_EXPERIMENTS.md](/home/weizilin/generate_idea/IDEA_EXPERIMENTS.md)
  - `v1-v32` 的详细演化。
- [docs/lq_progress.md](/home/weizilin/generate_idea/docs/lq_progress.md)
  - `lq` 主线进展记录。
- [docs/lq_train_presets.md](/home/weizilin/generate_idea/docs/lq_train_presets.md)
  - 各训练 preset 的说明与命令。
- [docs/lq_fsq_followup_notes.md](/home/weizilin/generate_idea/docs/lq_fsq_followup_notes.md)
  - FSQ 之后的结构探索记录。
- [docs/lq_dataset_refactor_checklist.md](/home/weizilin/generate_idea/docs/lq_dataset_refactor_checklist.md)
  - 数据读取与结构重构检查清单。

常见子目录：

- `data/`
  - `lq` 训练数据封装。
- `model/`
  - `DistNet` 及相关组件。
- `training/`
  - config、engine、loss、checkpoint。
- `init_basis/`
  - 初始化 basis 文件。
- `utils/`
  - 例如 action basis 初始化构建器等工具。

### 2.3 `scripts/disentangleNet/`

- README: [scripts/disentangleNet/README.md](/home/weizilin/generate_idea/scripts/disentangleNet/README.md)
  - 冻结后的 `v31` 训练闭包说明，强调训练主线、分析主线和 `matrix_vis` 桥接主线。
- 相关设计文档: [scripts/disentangleNet/structure.md](/home/weizilin/generate_idea/scripts/disentangleNet/structure.md)
  - 目录/结构层面的补充说明。

子目录：

- `analysis/`
  - post-hoc 分析、basis 激活导出、患者级统计、coactivation、t-SNE、`matrix_vis` 导出。
- `data/`
  - 冻结版训练数据读取逻辑。
- `model/`
  - 冻结版模型结构。
- `training/`
  - 训练循环、损失、配置、checkpoint。
- `init_basis/`
  - `v31` 使用的 basis 初始化文件。
- `tests/`
  - 与导出/桥接相关的测试。

### 2.4 `scripts/disentangleNet/analysis/`

- README: [scripts/disentangleNet/analysis/README.md](/home/weizilin/generate_idea/scripts/disentangleNet/analysis/README.md)
  - 按 checkpoint 分析、患者级统计、t-SNE、`matrix_vis` 导出四组脚本做总览，并给出推荐执行顺序。

适合什么时候看：

- 想知道一个 checkpoint 后处理要跑哪些脚本。
- 想知道 `window_basis_activations`、`patient_profile_summary`、coactivation、t-SNE 的生成链条。

### 2.5 `scripts/disentangleNet_trainprobe/`

- README: [scripts/disentangleNet_trainprobe/README.md](/home/weizilin/generate_idea/scripts/disentangleNet_trainprobe/README.md)
  - `disentangleNet` 的轻量训练闭包，用于快速比较结构，不带完整复杂分析链。

子目录大意：

- `analysis/`
  - 较轻量的导出/桥接脚本。
- `data/` / `model/` / `training/` / `init_basis/`
  - 与主线类似，但服务于快速训练与 probe 观察。

### 2.6 `scripts/matrix_vis/`

- README: [scripts/matrix_vis/README.md](/home/weizilin/generate_idea/scripts/matrix_vis/README.md)
  - `matrix_vis` 的核心定位、当前观测语义、算法骨架、模块边界。
- 英文外部使用指南: [scripts/matrix_vis/USAGE_GUIDE.md](/home/weizilin/generate_idea/scripts/matrix_vis/USAGE_GUIDE.md)
  - 对外使用、输入输出契约、CLI、端到端 dataflow。
- 中文外部使用指南: [scripts/matrix_vis/USAGE_GUIDE_CN.md](/home/weizilin/generate_idea/scripts/matrix_vis/USAGE_GUIDE_CN.md)
  - 上一份的中文版。
- 数学/建模说明: [scripts/matrix_vis/modeling.md](/home/weizilin/generate_idea/scripts/matrix_vis/modeling.md)
  - 更偏建模层的说明。

子目录：

- `cli/`
  - 用户入口命令。
- `configs/`
  - toy、landmarks、real 实验配置。
- `core/`
  - 稳定数据对象和轻量几何/投影逻辑。
- `io/`
  - 配置、mesh、basis、结果读写。
- `pipelines/`
  - 端到端流程编排。
- `qp/`
  - 优化问题装配与求解。
- `viz/`
  - 图、动画、预览输出。
- `tests/`
  - 配置和求解回归测试。
- `scripts/`
  - 围绕真实病例和 `disentangleNet` basis 的批处理辅助脚本。

### 2.7 `scripts/matrix_vis/scripts/`

- README: [scripts/matrix_vis/scripts/README.md](/home/weizilin/generate_idea/scripts/matrix_vis/scripts/README.md)
  - 说明 `generate_disentanglenet_basis_configs.py`、`run_disentanglenet_basis_batch.py`、preview、solver compare、toy data 生成等脚本的用途和推荐顺序。

### 2.8 `scripts/val_codebook/`

- README: [scripts/val_codebook/README.md](/home/weizilin/generate_idea/scripts/val_codebook/README.md)
  - 公共码本验证实验说明，覆盖 dataset / side / severity 三个分类任务。
- 结果说明: [scripts/val_codebook/RESULTS.md](/home/weizilin/generate_idea/scripts/val_codebook/RESULTS.md)
  - 已跑实验结果的汇总。

子目录：

- `common/`
  - 共享数据加载、可视化、分类辅助函数。
- `exp1_dataset_classification/`
  - IMR vs TT 分类。
- `exp2_side_classification/`
  - Left / Normal / Right 分类。
- `exp3_severity_classification/`
  - severity 分类。

---

## 3. `data/` 索引

### 3.1 `data/win20-step20/`

- README: [data/win20-step20/readme.md](/home/weizilin/generate_idea/data/win20-step20/readme.md)
  - 该数据层的具体说明。

常见子目录：

- `IMR/`, `TT/`
  - 主数据。
- `IMR-SVD/`, `TT-SVD/`
  - 被视为数据本体的单患者 SVD PC1 目录。
- `cal_diff/`, `cal_diff_mouth-only/`
  - 被保留在 `data/` 内的差分数据相关目录。

### 3.2 其他数据目录

- `data/win10-step10/`
  - 较短窗口配置的数据。
- `data/win5-step5/`
  - 更密窗口配置，常用于 DMD 和更稳定的相关性分析。
- `data/blendshape/`
  - AU / blendshape 标注数据。
- `data/toy/matrix_vis/`
  - `matrix_vis` toy 数据。

---

## 4. `outputs/` 索引

### 4.1 `outputs/lq/`

- README: [outputs/lq/README.md](/home/weizilin/generate_idea/outputs/lq/README.md)
  - 说明 `scripts/lq` 的保留运行目录统一放在 `outputs/lq/win20-step20/`，当前保留从 `v20` 到 `v32`。

### 4.2 `outputs/disentangleNet/`

- 无目录级 README。
- 主要放冻结版 `v31` 的训练输出和验证分析结果，例如：
  - `v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/`
  - `v31_current_verify/`

### 4.3 `outputs/disentangleNet/.../patient_pattern_analysis/`

- README: [outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/README.md](/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/README.md)
  - 患者级激活模式分析目录指南，说明 `patient_profile_summary/`、`coactivation/`、`tsne/`、`index_pages/` 的组织方式和再生成命令。

### 4.4 `outputs/matrix_vis/`

- 无目录级 README。
- 主要放：
  - `examples/` toy 输出
  - `real/` 真实重建结果
  - `real_preview/` 预览图/GIF
  - `real_compare/` 不同 solver 对比结果

### 4.5 `outputs/pilot_feasibility/`

- 无目录级 README，但与 [scripts/pilot_feasibility/README.md](/home/weizilin/generate_idea/scripts/pilot_feasibility/README.md) 配套。
- 主要存放按方法重组后的早期 feasibility 结果：
  - `svd/`
  - `dmd/`
  - `grassmann/`
  - `blendshape/`
  - `nmf/`
  - `tucker/`

### 4.6 `outputs/disentangleNet_trainprobe/`

- 无 README。
- 存放 trainprobe 分支的快速训练输出。

---

## 5. `docs/` 索引

关键文档：

- [docs/disentanglenet_matrix_vis_contract.md](/home/weizilin/generate_idea/docs/disentanglenet_matrix_vis_contract.md)
  - `disentangleNet` 与 `matrix_vis` 的桥接契约。
- [docs/lq_progress.md](/home/weizilin/generate_idea/docs/lq_progress.md)
  - `lq` 进展记录。
- [docs/lq_train_presets.md](/home/weizilin/generate_idea/docs/lq_train_presets.md)
  - `lq` 训练 preset 说明。
- [docs/lq_fsq_followup_notes.md](/home/weizilin/generate_idea/docs/lq_fsq_followup_notes.md)
  - FSQ 之后的探索笔记。
- [docs/lq_dataset_refactor_checklist.md](/home/weizilin/generate_idea/docs/lq_dataset_refactor_checklist.md)
  - 数据重构核对。
- `docs/superpowers/plans/`
  - 分步实施计划存档。
- `docs/superpowers/specs/`
  - 设计 spec 存档。

---

## 6. `literature_notes/` 索引

- 无单独 README。
- 这是论文阅读笔记目录，按编号组织，每篇一个 `.md`。
- 适合查：
  - facial motion / palsy 相关先行工作
  - FSQ / FactorVAE / sparse coding 等方法背景

---

## 7. `pilot_feasibility_check/` 索引

- 无目录级 README。
- 主要是整理前的旧版研究文档快照：
  - [pilot_feasibility_check/IDEA_REPORT.pre_update.md](/home/weizilin/generate_idea/pilot_feasibility_check/IDEA_REPORT.pre_update.md)
  - [pilot_feasibility_check/IDEA_EXPERIMENTS.pre_update.md](/home/weizilin/generate_idea/pilot_feasibility_check/IDEA_EXPERIMENTS.pre_update.md)
  - [pilot_feasibility_check/RESEARCH_PROGRESS.pre_update.md](/home/weizilin/generate_idea/pilot_feasibility_check/RESEARCH_PROGRESS.pre_update.md)
  - [pilot_feasibility_check/FORMER_RESEARCH_TIMELINE.md](/home/weizilin/generate_idea/pilot_feasibility_check/FORMER_RESEARCH_TIMELINE.md)

适合什么时候看：

- 想回溯整理前的写法和历史叙述。
- 想比对“当前主线文档”与“旧实验记录”的差异。

---

## 8. 推荐阅读路径

如果你是第一次进这个仓库，建议顺序：

1. [AGENTS.md](/home/weizilin/generate_idea/AGENTS.md)
2. [CLAUDE.md](/home/weizilin/generate_idea/CLAUDE.md)
3. [IDEA_REPORT.md](/home/weizilin/generate_idea/IDEA_REPORT.md)
4. [IDEA_EXPERIMENTS.md](/home/weizilin/generate_idea/IDEA_EXPERIMENTS.md)
5. 按你的目标进入：
   - 训练主线：`scripts/lq/` 或 `scripts/disentangleNet/`
   - 可视化解释：`scripts/matrix_vis/`
   - 早期分解对照：`scripts/pilot_feasibility/`
   - 公共码本验证：`scripts/val_codebook/`
