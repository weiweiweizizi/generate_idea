# TTMORECF matrix_vis 重建负值距离问题分析

## 问题现象

`reflex_pair_side_2Phase_TTMORECF_static_side_x/phaseAB/patient/TTMORECF-851519` 在进行 matrix_vis 重建时，
第一个窗口的 landmark 很快触顶（触碰 box constraint 上界），而同类实验 `TTMOREC_static_side` 对同一患者
重建正常。

## 初步诊断

怀疑负值距离矩阵（`reference + delta < 0`）导致截断。

## 发现：`build_target_distance_matrix` 单位空间错配

### 问题位置

`scripts/matrix_vis/pipelines/patient_sequence.py`，第 371-382 行主循环：

```python
observation_matrix = data["composed_basis_matrices"][item_idx]   # 归一化空间
delta_matrix = observation_matrix                                 # 仍是归一化空间！
target_distance_matrix = build_target_distance_matrix(
    reference_distance_matrix=reference_distance_matrix,          # 物理空间 (D0 from file)
    delta_matrix=delta_matrix,                                    # 归一化空间 ← 问题所在
)
```

`_solve_single_window` 内部（第 271-277 行）才对 observation_matrix 做 denormalize：

```python
working_observation_matrix = restore_physical_observation_scale(
    working_observation_matrix, observation_scale, ...)
```

**结果**：`target_distance_matrix = D_k（物理）+ delta_k（归一化）`，单位不一致。
denormalize 发生在 distance recursion 之后，target 从未被正确 denormalize。

### 数值验证

| | TTMORECF (有问题) | TTMOREC (正常) |
|---|---|---|
| `observation_scale` | 0.0169 | 0.0342 |
| `composed` (归一化) | [-0.852, 0.900] | [-0.396, 0.986] |
| `delta_physical` (denormalize 后) | [-0.014, 0.015] | [-0.014, 0.034] |
| `D0+composed`（实际传给 solver） | [-0.709, 1.197] | [-0.308, 1.359] |
| 负值 entry 数 | 2404 (17.0%) | 1134 (8.0%) |
| `D0+delta_phys`（正确值） | [-0.004, 0.666] | — |
| 正确负值 entry 数 | 仅 20 (0.1%) | — |
| target 被放大倍数 | 59x | ~29x |

### 因果链

```
composed matrices (归一化空间)
        ↓
build_target_distance_matrix 未 denormalize
        ↓
target = D0 (物理 ~0.177) + delta (归一化 ~[-0.85, 0.90])
        ↓
target 范围 [-0.709, 1.197]，17% entry 为负
        ↓
QP objective 直接使用 target 作为 observation_targets
        ↓
solver 被拉向"距离≈-0.7"的不可能目标
        ↓
触碰 box constraint → landmark 触顶 / 轨迹截断
```

## TTMORECF 比 TTMOREC 更严重的原因

1. `observation_scale` 更小：TTMORECF = 0.0169，TTMOREC = 0.0342。scale 越小，归一化 delta
   denormalize 后越小，但 mixed-space target 偏离物理意义越严重。

2. basis norm 更大：TTMORECF 的 `residual_fsq` 量化器 + `levels="2,6"`（更粗糙的量化级别），
   导致两个 shared basis 的 Frobenius norm 达到 ~3.1（TTMOREC 全部 ≤ 1.07）。
   更大的 basis norm → 更大的归一化 delta → mixed-space target 偏移更极端。

## 修复（已应用）

修复位置：`scripts/matrix_vis/pipelines/patient_sequence.py` 第 376-388 行

在 `build_target_distance_matrix` 调用前对 `delta_matrix` denormalize：

```python
delta_matrix = observation_matrix.astype(np.float32, copy=False)
if renormalize_observations and observation_scale is not None:
    delta_matrix = restore_physical_observation_scale(
        delta_matrix, observation_scale,
        observation_matrix_space=observation_matrix_space,
    )
target_distance_matrix = build_target_distance_matrix(
    reference_distance_matrix=reference_distance_matrix,
    delta_matrix=delta_matrix,  # 现在是物理空间
)
```

修复后，TTMORECF 负值 entry 从 17.0% 降至 0.1%。

此修复同时解决：
1. target 本身正确：D_k(物理) + delta_k(物理) → 物理空间 target
2. reference 递推正确：effective_target（物理空间）带入下一轮 → 后续窗口 reference 也是物理空间

## 已确认：bundle 中 composed_matrices 是归一化空间

导出代码 `disentangleNet/analysis/exporters/patient.py` 第 236-277 行确认：

- `composed` 由 `einsum(shared_weights, shared_basis_bank) + side_recon` 直接计算，未乘以 observation_scale
- 合约明确声明 `"observation_matrix_space": "normalized_input_space"`
- `observation_scale` 被存入 bundle 作为元数据，留给 `patient_sequence.py` 在运行时恢复

因此 `patient_sequence.py` 主循环中 `build_target_distance_matrix` 收到的 `delta_matrix`
确实是归一化空间，mixed-space 错配问题成立。

## 待确认

- [ ] TTMORECF 所有患者的 observation_scale 分布是否普遍偏小（导致 mixed-space 偏差更大）
