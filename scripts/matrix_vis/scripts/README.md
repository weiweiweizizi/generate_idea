# `scripts/matrix_vis/scripts` 使用说明

这组脚本是 `matrix_vis` 的可执行入口，主要服务于两个场景：

1. 把真实患者窗口或 disentangleNet basis 投影回二维 landmark 轨迹，并做可视化预览。
2. 做一些辅助实验，例如 toy 数据生成、不同求解器结果对比。

## 目录分工

- `generate_disentanglenet_basis_configs.py`
  - 根据 `export_matrix_vis_basis.py` 导出的 manifest，批量生成 matrix_vis YAML 配置。
  - 只“产出配置”，不真正执行重建。

- `run_disentanglenet_basis_batch.py`
  - 读取 manifest，先生成配置，再批量运行每个 basis 的 `axis_x` 重建。
  - 可选生成 `fixed_y` 预览和 `no_motion_y` 预览。
  - 这是 disentangleNet basis matrix_vis 可视化链路的主入口。

- `run_disentanglenet_phase_comparison.py`
  - 针对一个 phase 闭包做统一导出与重建。
  - 既支持三阶段 `phaseA / phaseB / phaseC`，也支持两阶段闭包，例如 `phaseAB / phaseC_ft150`。
  - 会在每个 phase 目录下固定产出：
    - `patient/`
    - `basis/`
  - 患者级别默认使用患者自身初值：
    - `win_000_x.npy` 作为首窗 `D0`
    - `lmks_crop.label` 第一行作为静止 `y`
  - 旧的 `standardFace` 患者分支现在只在显式开启时才会额外生成
  - basis 级别默认也只重建 `x`，并配套 `no_motion_y` 预览。
  - 这个脚本是当前推荐的 phase 对比主入口。

- `preview_real_mouth_regions.py`
  - 针对 mouth-region 子集，读取 `axis_x/axis_y` 解并导出 preview 图/GIF。

- `preview_real_full341.py`
  - 针对 full341 子集，读取 `axis_x/axis_y` 解并导出 preview 图/GIF。

- `compare_full341_solvers.py`
  - 比较同一 full341 重建任务下，`OSQP` 与 `matrix_free_cg` 的差异。

- `generate_toy_double_crescent_data.py`
  - 生成一个 toy mouth-opening 数据集，用于快速验证 matrix_vis 直觉。

## 推荐执行顺序

### 1. disentangleNet basis 批量可视化

前提：已经有 basis 导出 manifest，例如：
`outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json`

先直接跑批处理主入口：

```bash
python scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py run \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json
```

如果只想先检查前几个 basis：

```bash
python scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py run \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json \
  --limit_basis 3
```

如果还想额外生成 `y` 不动的对照预览：

```bash
python scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py run \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json \
  --run_no_motion_y_preview True
```

### 2. 只生成配置，不立刻运行

```bash
python scripts/matrix_vis/scripts/generate_disentanglenet_basis_configs.py generate \
  --manifest_path outputs/disentangleNet/.../matrix_vis_exports/basis/exported_basis_manifest.json
```

适合场景：
- 先检查生成的 YAML 是否符合预期
- 想手动挑某几个 basis 单独跑

### 3. 单次预览已有重建结果

mouth regions:

```bash
python scripts/matrix_vis/scripts/preview_real_mouth_regions.py \
  --x-solution outputs/matrix_vis/.../axis_x/solution.npz \
  --y-solution outputs/matrix_vis/.../axis_y/solution.npz \
  --anchor-point-ids 205,425,200 \
  --output-dir outputs/disentangleNet/.../matrix_vis_exports/.../preview_anchor205_425_200
```

full341:

```bash
python scripts/matrix_vis/scripts/preview_real_full341.py \
  --x-solution outputs/matrix_vis/.../axis_x/solution.npz \
  --y-solution outputs/matrix_vis/.../axis_y/solution.npz \
  --output-dir outputs/matrix_vis/real_preview/.../preview
```

`full341` 这条是旧的全脸预览示例，仍保留在 `outputs/matrix_vis/real_preview/...`。
当前统一收口到 `outputs/disentangleNet_*` 的是 basis / patient 的 preview 流水线。

### 4. 求解器差异分析

```bash
python scripts/matrix_vis/scripts/compare_full341_solvers.py \
  --osqp-x outputs/matrix_vis/.../osqp/axis_x/solution.npz \
  --osqp-y outputs/matrix_vis/.../osqp/axis_y/solution.npz \
  --cg-x outputs/matrix_vis/.../matrix_free_cg/axis_x/solution.npz \
  --cg-y outputs/matrix_vis/.../matrix_free_cg/axis_y/solution.npz \
  --output-dir outputs/matrix_vis/real_compare/...
```

### 5. 生成 toy 数据

```bash
python scripts/matrix_vis/scripts/generate_toy_double_crescent_data.py
```

## 关键输入输出约定

### `run_disentanglenet_basis_batch.py`

主要输入：
- `manifest_path`
- 可选 `anchor_point_id`
- 可选 `fixed_y_config`
- 可选 `limit_basis`

主要输出：
- 生成的 YAML 配置：
  - `scripts/matrix_vis/configs/real/disentanglenet/generated/<run_name>/<anchor_tag>/`
- x 轴重建：
  - `outputs/matrix_vis/real/disentanglenet/<run_name>/<anchor_tag>/<basis_label>_<idx>/axis_x/`
- fixed-y 预览：
  - `outputs/disentangleNet_*/matrix_vis_exports/basis/preview_anchor205_425_200/fixed_y/...`
- no-motion-y 预览：
  - `outputs/disentangleNet_*/matrix_vis_exports/basis/preview_anchor205_425_200/no_motion_y/...`
- `no_motion_y` 的语义是：
  - `y` 在所有时间帧都固定为模板初始位置
  - 这份静态 `y` 解由模板直接生成，不复用动态 `fixed_y` 轨迹
- 汇总：
  - `<generated_dir>/batch_run_summary.json`

### `generate_disentanglenet_basis_configs.py`

主要输出：
- `fixed_y_axis_y.yaml`
- `basis_00_axis_x.yaml` 等
- `side_basis_00_axis_x.yaml` 等
- `generation_summary.json`

## 复用建议

- 想改 anchor 时，优先从 CLI 传 `--anchor_point_id`，不要直接手改一堆 YAML。
- 想快速调试 basis 批跑是否通，先用 `--limit_basis 1` 或 `--limit_basis 3`。
- `run_disentanglenet_basis_batch.py` 是最适合复用的主入口；除非你只想检查配置文件，否则一般不需要先手动跑 `generate_disentanglenet_basis_configs.py`。

## phase 闭包统一对比重建

如果你要对训练后的 phase 闭包做统一导出、患者重建、basis 重建和 `no_motion_y` 预览，直接使用：

```bash
python scripts/matrix_vis/scripts/run_disentanglenet_phase_comparison.py run_all \
  --run_root outputs/disentangleNet_frame/reflex_pair_side_imr_tt \
  --patient_id TT_851519 \
  --lambda_laplacian 1.0 \
  --lambda_area_sign 1.0 \
  --area_barrier_margin 0.05
```

这个流程会对每个 phase 固定产出：

- `phaseA/patient/`
- `phaseA/basis/`
- `phaseB/patient/`
- `phaseB/basis/`
- `phaseC/patient/`
- `phaseC/basis/`

如果你的实际目录是两阶段闭包，例如：

- `phaseAB`
- `phaseC_ft150`

那么只要把 `--run_root` 指向对应目录即可。脚本会按 `run_root` 下实际存在的 phase 目录执行，不要求目录名必须是 `phaseB`。

### 运行前检查清单

运行前建议先确认：

- `run_root` 下每个目标 phase 都存在 `best.pt`
- `patient_id` 在 `data/win20-step20/{数据集名称}/{patient_id}/` 下有唯一匹配目录
- 该目录下存在 `lmks_crop.label`
- 该目录下存在 `win_000_x.npy`
- 如果你是直接执行脚本，而不是用 `python -m` 或仓库内的统一入口，先把仓库根目录加到 `PYTHONPATH`

患者级 `x` 重建当前语义是：

- 首窗 `D0` 使用患者目录下的 `win_000_x.npy`
- 后续窗口沿用上一窗口末帧，并按 `D_{k+1} = D_k + \Delta D_k` 递推
- 患者级 preview 使用患者自身初始 landmark 的静止 `y`
- 默认不再额外跑旧的 `standardFace` 患者分支；如需对照，可显式开启

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

当前这条流程还固定打开：

- graph Laplacian 正则
- area sign barrier

并且在 `phaseC` 的患者导出里，会把 `private residual` 一并合入最终患者级观测矩阵；
basis 导出仍然只保留 shared / side basis，不单独导 private。
