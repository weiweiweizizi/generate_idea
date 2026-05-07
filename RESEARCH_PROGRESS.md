# 研究进展记录

[TOC]

**最后更新**: 2026-04-29

---

## 一、当前研究目标

当前主线目标已经收敛为三件事：

1. 学习一组跨被试共享、可解释的面部运动基
2. 把侧别信息尽量路由到显式的 side branch
3. 用后验轨迹重建把 learned basis / 距离差观测解释成可视运动

当前不再把“同时比较多种候选分解法”作为主任务。代码与结果都表明，研究重心已经转移到：

- `scripts/lq`
  - 负责前置结构探索
- `scripts/disentangleNet`
  - 负责当前接受版 `v31`
- `scripts/matrix_vis`
  - 负责后验解释

---

## 二、当前有效结论

## 2.1 `scripts/lq` 已完成主结构探索

当前 `scripts/lq` 的运行结果目录已经统一收纳到：

- `outputs/lq/win20-step20/`

当前实际保留的目录索引从 `v20` 开始，覆盖：

- `v20-v21` private-cap tradeoff
- `v22-v23` side semantic bank
- `v24-v28` side 路由结构探测
- `v29-v31` laterality / joint-QR 收敛
- `v32` severity 辅助分支

更早的 `v10-v19` 仍然保留为文档层结论，但不再作为 `outputs/` 顶层目录索引。

当前可以稳定保留的结论：

1. 早期 collapse 的真正缓解来自官方 `FSQ` 替换，而不是简单调 loss
2. 只加强 basis 约束并不能自动提升解释性，模型会把收益转移回 private residual
3. `residual_fsq + shared_recon supervision` 是后续主线的关键结构基础
4. `v19` 是 side-aware 分支引入前最稳的 interpretability baseline

不再成立或不再有价值的旧表述：

- “继续优先比较很多早期 baseline”
- “只要继续调 residual weight 就能解决问题”

## 2.2 side-aware 路由已经被证明可行

从 `v22-v31` 的结果看，当前已经能明确说：

1. side branch 不是空转分支，确实会被实际使用
2. laterality 可以被 side branch 显式承载
3. 旧的 `subspace_orth` 路线无效，因为 side / free 两个 latent 几乎线性等价

因此，当前关于 side 分离的主判断是：

> side 语义可以被单独路由出来，这条路已经走通。

## 2.3 `scripts/disentangleNet` 的冻结 `v31` 是当前主线

当前 `disentangleNet/v31` 的固定结构是：

- `levels=2,6`
- `quantizer_type=residual_fsq`
- `basis_orthogonalization=joint_global_qr`
- `side_basis_count=3`
- `side_pooling=fixed_region2_contrast`
- `early_branch_factorization=True`

full post-hoc k-fold 结果表明：

1. side branch 对侧别有很强的判别力
   - `side_from_side_rep` accuracy 约 `0.93`
2. free branch 的侧别判别力明显更弱
   - `side_from_free_rep` accuracy 约 `0.47`
3. private branch 仍然最强地保留 dataset 信息
   - `dataset_from_private_rep` accuracy 约 `0.88-0.89`

因此当前最准确的总结是：

> `v31` 已经实现了强 side-routing，但还没有实现强 dataset-invariance。

## 2.4 severity 还不是当前主线的强结果

`v32` 的结果只能支持一个保守判断：

1. side-aware 中层表示对 severity 可能有弱相关
2. 这种相关还不够强，也不够稳
3. 当前不能把 severity 说成已经被有效解耦

如果后续文档或汇报要提 severity，建议只写：

- 当前是弱阳性方向
- 还不是主贡献

## 2.5 `scripts/matrix_vis` 已经是可信的解释工具

当前保留的关键结论：

1. 在当前观测定义下，`y` 轴明显比 `x` 轴容易重建
2. matrix-free solver 与 OSQP 在 toy 和 real full341 case 上几乎一致
3. mouth subset 的运行代价远低于 full341，适合快速后验可视化

因此：

> `matrix_vis` 不是附属脚本，而是当前研究解释层的核心工具之一。

---

## 三、当前主风险

## 3.1 free branch code usage 仍不健康

虽然 side 路由已经成功，但 `v31` 的 free quantizer usage 仍明显集中。

这意味着：

- 当前问题已经从“side 能不能分出来”转向“free 表示是否足够丰富”
- 后续若继续优化主线，code usage 健康度会是优先级很高的问题

## 3.2 dataset leakage 仍明显存在

当前 full probe 最值得警惕的结论不是 accuracy 高，而是：

- `dataset_from_side_rep` 仍然不低
- `dataset_from_free_rep` 更高
- `dataset_from_private_rep` 最高

这说明：

- side、free、private 三条路径都还没有彻底摆脱 dataset 偏差
- 其中 private 是最大的 dataset carrier

## 3.3 当前验证集仍偏小

不少 checkpoint analysis 的验证指标仍基于较小的 val split。

这不影响当前的定性结论，但意味着：

- 对细小指标差异不应过度解读
- 当前更适合保留结构性判断，而不是过分强调小数点后的增益

---

## 四、当前阶段结论

### 4.1 可以明确写进阶段总结的内容

1. `lq` 已经完成从 collapse 到 side-aware 的主要结构搜索
2. `disentangleNet/v31` 是当前接受版训练栈
3. side 信息已经能被稳定压进 side branch
4. free / private 的 dataset leakage 仍未解决
5. severity 仍不是当前强结果
6. `matrix_vis` 已经能够提供可信的后验轨迹解释

### 4.2 不应再继续沿用的旧说法

1. 不应再把 `scripts/lq` 与 `scripts/disentangleNet` 写成同一阶段
2. 不应再把“多种分解候选 idea 排名”当成当前重点
3. 不应再把 severity 写成已完成结果
4. 不应再把 `matrix_vis` 写成单纯 toy 工具

---

## 五、下一步

当前最值得继续推进的方向只有三条：

1. 降低 `v31` 中 free / private 的 dataset leakage
2. 改善 free quantizer 的 usage spread，避免 level 内单码集中
3. 把 `matrix_vis` 与 learned basis 的解释链条进一步对齐，形成更直接的可视证据

如果必须给出一句当前项目状态总结，可以写成：

> 当前已经建立了一条 side-aware、可后验解释的共享运动分解主线；  
> 主线在侧别路由上表现强，但在 dataset-invariance 与 severity 表示上仍有明确缺口。
