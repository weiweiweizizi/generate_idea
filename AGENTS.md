# 仓库协作指南

## 项目定位与目录组织

这个仓库是一个**面部运动分解研究工作区**，不是打包发布的通用库。

当前主要目录分工如下：

- `scripts/`
  - 代码主目录。
  - 早期分解实验已整理到 `scripts/pilot_feasibility/`，按方法分成：
    - `svd/`
    - `dmd/`
    - `grassmann/`
    - `blendshape/`
    - `nmf/`
    - `tucker/`
  - `scripts/lq/`
    - 前置结构探索原型。
  - `scripts/disentangleNet/`
    - 当前冻结主线 `v31` 的训练与分析闭包。
  - `scripts/matrix_vis/`
    - 后验轨迹重建与可视化工具。
  - `scripts/val_codebook/`
    - 公共码本验证实验。
- `data/`
  - 原始数据与少量被视为数据本体的派生目录。
- `outputs/`
  - 可复现实验输出、训练结果、分析结果、可视化产物。
- `docs/`
  - 设计说明、计划、补充笔记。
- `literature_notes/`
  - 文献阅读记录。
- 顶层研究文档：
  - `IDEA_REPORT.md`
  - `IDEA_EXPERIMENTS.md`
  - `RESEARCH_PROGRESS.md`

如果想快速看仓库里的 Markdown / README 分布，先看：

- `docs/repo_index.md`

## 环境与常用命令

默认使用 `CLAUDE.md` 中提到的 Conda 环境：

```bash
conda activate openmmlab
```

常用命令示例：

```bash
python scripts/pilot_feasibility/svd/svd_single_patient.py
python scripts/pilot_feasibility/dmd/dmd_blendshape_correlation.py
python scripts/val_codebook/sweep.py --run
python scripts/lq/train.py --data_roots=data/win10-step10/IMR,data/win10-step10/TT
```

这些命令分别对应：

- 单个早期分解实验
- DMD 与 blendshape 相关性分析
- 公共码本验证 sweep
- `lq` 训练

这个仓库当前**没有**根级 build system 或 Makefile。

## 代码风格

保持现有 `scripts/` 下的 Python 风格：

- 缩进：4 空格
- 常量：`UPPER_SNAKE_CASE`
- 函数：`snake_case`
- 文件名：尽量描述实验含义，例如 `svd_multi_patient_win5.py`

额外约定：

- 一个脚本尽量只做一件实验相关的事。
- 路径优先使用 `pathlib.Path`。
- 输出目录命名要显式包含方法名、窗口设置或实验名。
- 不要提交 `__pycache__` 等缓存文件。

## 测试与验证

当前没有统一的自动化单元测试套件。

修改后建议这样验证：

1. 在一个小而有代表性的数据切片上重跑受影响脚本。
2. 检查预期输出是否出现在对应目录：
   - `outputs/`
   - 或明确被保留在 `data/` 内的数据型目录
3. 如果改动涉及 `scripts/val_codebook/`：
   - 跑 `python scripts/val_codebook/sweep.py --summary`
   - 确认汇总指标正常
4. 需要时把手工验证结论记到相关研究笔记或 `docs/` 文档里。

## 提交与 PR 约定

当前工作区里无法从历史中可靠推断出统一 commit 规范，因此默认使用：

- 简短
- 祈使句
- 必要时加作用域

例如：

```text
scripts: refine win5 DMD correlation output
```

如果后续发 PR，建议在说明里写清：

- 研究目标是什么
- 改了哪些脚本 / 数据集 / 输出路径
- 新增了哪些结果目录
- 如果影响图或结论，附上图或截图

## 数据与输出卫生

优先把 `data/` 视为**数据层**，不要随意重命名其中的数据集目录。

新的分析结果默认放到：

- `outputs/`

除非某个目录被明确视为数据本体的一部分，例如：

- `data/win20-step20/TT-SVD`
- `data/win20-step20/IMR-SVD`
- `data/win20-step20/cal_diff`
- `data/win20-step20/cal_diff_mouth-only`

如果脚本里还写着绝对本机路径，提交前应改成仓库相对路径或由 `Path(__file__)` 推导的相对路径。
