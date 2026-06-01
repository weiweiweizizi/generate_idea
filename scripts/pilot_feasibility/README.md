# 早期可行性探索脚本

按方法分组的早期可行性实验：

- `svd/`
- `dmd/`
- `grassmann/`
- `blendshape/`
- `nmf/`
- `tucker/`

目录规范：

- 源数据集统一放在 `data/` 下
- 实验输出放到 `outputs/pilot_feasibility/<method>/<window-setting>/`
- 仍在 `data/` 中的类数据集目录包括：
  - `data/win20-step20/TT-SVD`
  - `data/win20-step20/IMR-SVD`
  - `data/win20-step20/cal_diff`
  - `data/win20-step20/cal_diff_mouth-only`
- 如果某个可行性脚本后续接到 `matrix_vis` 预览，统一使用锚点
  `205,425,200`，并把预览结果放回对应的 `outputs/pilot_feasibility/...`
  或 `outputs/disentangleNet_*` 结果树下，不再单独散落到 `outputs/matrix_vis/`。
