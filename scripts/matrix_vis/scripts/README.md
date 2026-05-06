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
  --output-dir outputs/matrix_vis/real_preview/.../preview
```

full341:

```bash
python scripts/matrix_vis/scripts/preview_real_full341.py \
  --x-solution outputs/matrix_vis/.../axis_x/solution.npz \
  --y-solution outputs/matrix_vis/.../axis_y/solution.npz \
  --output-dir outputs/matrix_vis/real_preview/.../preview
```

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
  - `outputs/matrix_vis/real_preview/disentanglenet/<run_name>/<anchor_tag>/fixed_y/...`
- no-motion-y 预览：
  - `outputs/matrix_vis/real_preview/disentanglenet/<run_name>/<anchor_tag>/no_motion_y/...`
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
