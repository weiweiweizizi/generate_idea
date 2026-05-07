# V31 全量 K-Fold 后验分析报告（中文）

## 1. 分析目标

本报告面向 `v31` 版本模型的后验汇报，目标是回答以下几个问题：

1. `v31` 学到的 side 语义是否稳定、可解释，且主要集中在 side 分支而不是 free 分支。
2. `v31` 的 shared / free / private / side 各分支分别携带了多少 laterality 信息与 dataset 信息。
3. 在扩大评估样本量之后，先前基于单次 val split 得到的结论是否仍然成立。

本次分析不重新训练模型，而是对冻结后的 `v31` checkpoint 做全量表征抽取，再进行 subject-aware 的 `5-fold` OOF（out-of-fold）线性 probe 评估。

- checkpoint: [best.pt](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/best.pt)
- 总报告目录: [kfold_report](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report)
- 英文简版: [report.md](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/report.md)
- 结构化汇总: [summary.json](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/summary.json)

## 2. 方法说明

### 2.1 样本来源

分析样本来自 `data/win20-step20/IMR` 与 `data/win20-step20/TT` 的 `train + val` 全量 group，而不是只看单个 val split。

本次用于 probe 的总样本规模为：

- 总 group 数：`293`
- 总 subject 数：`267`
- 数据集分布：`IMR = 228`，`TT = 65`
- side 分布：`Left = 83`，`Normal = 111`，`Right = 99`

对应文件：

- 全量 group 表征: [full_group_level_representations.npz](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/full_group_level_representations.npz)
- 全量 side 语义表: [full_group_side_semantics.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/full_group_side_semantics.csv)

### 2.2 Fold 划分方式

这次不是普通的随机 sample-level 划分，而是 **subject-aware 的 5-fold 划分**：

1. 同一个 subject 的多个 group 只会出现在同一个 fold 中，避免 train/test 泄漏。
2. stratify 不是只按 side_label，而是按 `joint side + dataset` 做联合分层。
3. 每个 fold 的 group 数量和类别分布都比较均衡。

fold 分布如下：

| fold | subject 数 | group 数 | side 分布 | dataset 分布 |
| --- | --- | --- | --- | --- |
| 0 | 54 | 59 | Left 17 / Normal 22 / Right 20 | IMR 46 / TT 13 |
| 1 | 54 | 58 | Left 17 / Normal 21 / Right 20 | IMR 45 / TT 13 |
| 2 | 54 | 58 | Left 16 / Normal 22 / Right 20 | IMR 45 / TT 13 |
| 3 | 51 | 59 | Left 17 / Normal 23 / Right 19 | IMR 46 / TT 13 |
| 4 | 54 | 59 | Left 16 / Normal 23 / Right 20 | IMR 46 / TT 13 |

对应文件：

- fold 汇总: [fold_summary.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/fold_summary.csv)
- subject 到 fold 的映射: [subject_fold_assignments.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/subject_fold_assignments.csv)

### 2.3 Probe 任务定义

本次总共做了两大类 probe。

第一类是“分支表征 probe”：

- `side_from_side_rep`
- `side_from_free_rep`
- `dataset_from_side_rep`
- `dataset_from_free_rep`
- `dataset_from_private_rep`

第二类是“side 语义可解释性 probe”：

- `side_from_usage`
- `side_from_coeff`
- `side_from_usage_coeff`
- `dataset_from_usage`
- `dataset_from_coeff`
- `dataset_from_usage_coeff`

其中：

- `usage` 表示 side basis usage 向量
- `coeff` 表示 side coefficient 标量
- `usage_coeff` 表示二者拼接后的 4 维特征

所有 probe 都采用 OOF 方式：

1. 每次用 4 个 fold 训练线性分类器
2. 在剩余 1 个 fold 上预测
3. 最后将 5 个 fold 的 test 预测拼成全量 OOF 结果
4. 再统一计算 accuracy / balanced accuracy / macro-F1 / confusion matrix

## 3. 总体结论

如果只看一句话，本次 `v31` 的后验结论可以总结为：

**`v31` 已经能把 laterality 语义稳定地压进 side branch 和 side semantic path（usage / coeff），同时 free branch 对 side 的可分性明显较低；但是 dataset 信息尚未完全被赶出 side/free，尤其 private branch 仍然最强地携带 dataset 信息。**

更具体地说：

1. `side` 信息在 `side_rep`、`usage`、`coeff`、`usage+coeff` 上都非常强。
2. `free_rep` 对 side 的预测显著较弱，说明 side/free 解耦方向是成立的。
3. `usage+coeff` 的 side 预测最好，说明 side basis 的“选哪个基”与“激活强度”都包含有用语义。
4. `dataset` 信息在 `private_rep` 上最强，这符合“private 分支承载个体/域信息”的预期。
5. `dataset` 信息在 `side_rep` 和 `free_rep` 上也仍可被读出，说明 dataset leakage 还没有完全解决。
6. `usage / coeff` 对 dataset 的预测接近退化，说明 side semantic path 本身较少直接编码 dataset。

## 4. 核心结果总表

完整表见：

- probe 汇总: [probe_summary.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/probe_summary.csv)
- 各 fold 指标: [probe_fold_metrics.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/probe_fold_metrics.csv)

这里先给出最关键的任务结果：

| 任务 | accuracy | balanced accuracy | macro-F1 | 解释 |
| --- | ---: | ---: | ---: | --- |
| `side_from_side_rep` | 0.9215 | 0.9163 | 0.9181 | side 分支显式携带 laterality |
| `side_from_free_rep` | 0.4061 | 0.3766 | 0.3146 | free 分支对 laterality 的可分性明显较弱 |
| `side_from_usage` | 0.9249 | 0.9236 | 0.9226 | side basis usage 本身有很强 laterality 语义 |
| `side_from_coeff` | 0.9283 | 0.9244 | 0.9271 | side coefficient 也强烈编码 laterality |
| `side_from_usage_coeff` | 0.9386 | 0.9370 | 0.9370 | usage + coeff 联合是当前最佳 side 读出器 |
| `dataset_from_side_rep` | 0.8225 | 0.6495 | 0.6761 | side 分支仍有 dataset 泄漏 |
| `dataset_from_free_rep` | 0.8123 | 0.6594 | 0.6809 | free 分支也仍有 dataset 泄漏 |
| `dataset_from_private_rep` | 0.8874 | 0.8286 | 0.8341 | private 分支最强携带 dataset 信息 |
| `dataset_from_usage` | 0.7782 | 0.5000 | 0.4376 | usage 几乎退化为只预测 IMR |
| `dataset_from_coeff` | 0.7782 | 0.5000 | 0.4376 | coeff 同样几乎不具备 dataset 判别性 |
| `dataset_from_usage_coeff` | 0.7406 | 0.4869 | 0.4499 | usage+coeff 对 dataset 依旧较弱 |

## 5. Side 语义相关结果

### 5.1 side 分支表征可以稳定预测 laterality

`side_from_side_rep` 的整体结果：

- accuracy = `0.9215`
- balanced accuracy = `0.9163`
- macro-F1 = `0.9181`
- fold accuracy = `0.8814 / 0.9310 / 0.9483 / 0.9322 / 0.9153`
- fold accuracy 标准差约为 `0.0227`

这说明：

1. laterality 信息确实被稳定编码在 side 分支中。
2. 不同 fold 间波动不大，结果不是偶然来自某个特定 split。
3. 这与 `v31` 的设计目标一致，即让 side branch 成为 laterality 专用语义通路。

对应图表：

- confusion matrix 图: [side_from_side_rep_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_side_rep_confusion.png)
- confusion matrix 数值: [side_from_side_rep_confusion.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_side_rep_confusion.csv)
- 行归一化混淆矩阵: [side_from_side_rep_confusion_normalized.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_side_rep_confusion_normalized.csv)

从行归一化混淆矩阵看：

- Left: `84.34%` 被正确分类为 Left，`4.82%` 混到 Normal，`10.84%` 混到 Right
- Normal: `94.59%` 被正确分类为 Normal
- Right: `95.96%` 被正确分类为 Right

这说明当前最难的是 `Left`，而 `Normal` 和 `Right` 已经非常稳定。

### 5.2 free 分支对 laterality 的可分性显著下降

`side_from_free_rep` 的结果：

- accuracy = `0.4061`
- balanced accuracy = `0.3766`
- macro-F1 = `0.3146`
- fold accuracy = `0.4068 / 0.4310 / 0.3621 / 0.3729 / 0.4576`

对应图表：

- confusion matrix 图: [side_from_free_rep_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_free_rep_confusion.png)
- confusion matrix 数值: [side_from_free_rep_confusion.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_free_rep_confusion.csv)
- 行归一化混淆矩阵: [side_from_free_rep_confusion_normalized.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_free_rep_confusion_normalized.csv)

这个 confusion matrix 很有代表性：

- Left 一条都没有被预测成 Left
- Left 的样本主要被判成 Normal (`55.42%`) 或 Right (`44.58%`)
- Normal 有 `59.46%` 被判成 Normal
- Right 有 `53.54%` 被判成 Right

这说明 free 分支中几乎不存在稳定的 “Left 专用方向”。从解耦角度看，这反而是一个正面信号，因为 free 分支本来就不应该主承载 laterality 语义。

### 5.3 usage 与 coeff 都已经携带强 laterality 语义

三个 side semantic path probe 的结果如下：

| 任务 | accuracy | balanced accuracy | macro-F1 |
| --- | ---: | ---: | ---: |
| `side_from_usage` | 0.9249 | 0.9236 | 0.9226 |
| `side_from_coeff` | 0.9283 | 0.9244 | 0.9271 |
| `side_from_usage_coeff` | 0.9386 | 0.9370 | 0.9370 |

这三组结果说明：

1. `usage` 不是噪声，而是直接携带可读的 laterality 信息。
2. `coeff` 也同样携带强 laterality 信息。
3. 二者拼接后进一步提升，说明 “选哪个基” 和 “激活强度多大” 是互补信息。

最值得汇报的是 `side_from_usage_coeff`：

- accuracy = `0.9386`
- balanced accuracy = `0.9370`
- macro-F1 = `0.9370`
- fold accuracy = `0.8983 / 0.9483 / 0.9655 / 0.9322 / 0.9492`

对应图表：

- 图: [side_from_usage_coeff_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_usage_coeff_confusion.png)
- 数值: [side_from_usage_coeff_confusion.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_usage_coeff_confusion.csv)

它的行归一化混淆矩阵为：

- Left: `91.57%` 正确
- Normal: `94.59%` 正确
- Right: `94.95%` 正确

从汇报角度看，这个结果最适合作为 “`v31` 已经把 side 语义压进可解释通路” 的核心证据。

## 6. Dataset 信息相关结果

### 6.1 private 分支仍然是 dataset 信息最强承载者

`dataset_from_private_rep` 的结果：

- accuracy = `0.8874`
- balanced accuracy = `0.8286`
- macro-F1 = `0.8341`
- fold accuracy = `0.8305 / 0.8621 / 0.9483 / 0.8814 / 0.9153`

对应图表：

- 图: [dataset_from_private_rep_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_private_rep_confusion.png)
- 数值: [dataset_from_private_rep_confusion.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_private_rep_confusion.csv)

其行归一化结果为：

- IMR: `93.42%` 正确分类为 IMR
- TT: `72.31%` 正确分类为 TT

这非常符合当前结构设想，即 private 分支承担更多个体/域偏置。

### 6.2 side / free 分支仍然存在 dataset leakage

`dataset_from_side_rep` 与 `dataset_from_free_rep` 的结果分别是：

| 任务 | accuracy | balanced accuracy | macro-F1 |
| --- | ---: | ---: | ---: |
| `dataset_from_side_rep` | 0.8225 | 0.6495 | 0.6761 |
| `dataset_from_free_rep` | 0.8123 | 0.6594 | 0.6809 |

对应图表：

- [dataset_from_side_rep_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_side_rep_confusion.png)
- [dataset_from_free_rep_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_free_rep_confusion.png)

这两个任务的一个共同特点是：

- 对 IMR 的识别都很高
- 对 TT 的召回都明显偏低

具体来说：

`dataset_from_side_rep`

- IMR 召回：`96.05%`
- TT 召回：`33.85%`

`dataset_from_free_rep`

- IMR 召回：`93.42%`
- TT 召回：`38.46%`

这说明两件事：

1. side/free 两个分支都还残留着 dataset 相关模式。
2. 这种 dataset 读出更偏向于“把大多数样本判成 IMR”，因此虽然 accuracy 不低，但 balanced accuracy 与 macro-F1 没有那么高。

也就是说，如果从“完全去域化”的标准来看，`v31` 还没有完成这一目标。

### 6.3 usage / coeff 几乎不直接携带 dataset 信息

`dataset_from_usage` 与 `dataset_from_coeff` 的结果完全一致：

- accuracy = `0.7782`
- balanced accuracy = `0.5000`
- macro-F1 = `0.4376`

这个结果本质上接近 “永远预测多数类 IMR”：

- IMR 召回 = `100%`
- TT 召回 = `0%`

`dataset_from_usage_coeff` 也没有明显提升：

- accuracy = `0.7406`
- balanced accuracy = `0.4869`
- macro-F1 = `0.4499`

对应图表：

- [dataset_from_usage_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_usage_confusion.png)
- [dataset_from_coeff_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_coeff_confusion.png)
- [dataset_from_usage_coeff_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_usage_coeff_confusion.png)

这说明 `side semantic path` 本身较少直接编码 dataset 信息。换句话说，dataset leakage 主要还在 side/free/private 的连续表征空间，而不在 side usage / coeff 这条更离散、更解释性的路径上。

## 7. 各任务结果的结构性解读

### 7.1 关于 laterality 学习

本次结果最强的一条主线是：

`side_rep`、`usage`、`coeff`、`usage+coeff` 都能高精度读出 laterality，而 `free_rep` 明显不能。

这意味着：

1. `v31` 的 laterality 语义确实已经集中到 side 路径。
2. 这种 laterality 不是只能从一个黑盒高维连续向量里读出来，而是已经能从低维、可解释的 `usage / coeff` 中读出来。
3. 从解释性角度看，这是比单看 `side_rep` 更重要的结果。

### 7.2 关于 side/free 解耦

如果只看 `side_from_side_rep`，我们知道 side 分支有用；但如果再看 `side_from_free_rep = 0.4061`，就可以进一步说明：

- free 分支没有学成另一个 side 分支
- side/free 的功能分工已经开始形成

虽然 `0.4061` 仍高于三分类随机水平，但它和 `0.9215` 相比差距非常大，因此可以合理认为 `v31` 的 side/free 解耦已经取得了阶段性结果。

### 7.3 关于 dataset leakage

本次结果同样明确指出：

- private 分支最强地携带 dataset 信息，这在当前架构下是合理的。
- side/free 分支仍然都能一定程度上预测 dataset，说明 leakage 还在。
- 但 usage/coeff 路径对 dataset 几乎不可分，这说明解释性路径本身相对“干净”。

因此更准确的说法不是 “dataset 信息已经解决”，而是：

**`v31` 已经把 side semantic path 清理得比连续表征更干净，但 continuous representation space 中的 dataset leakage 仍需要后续处理。**

## 8. 汇报时建议重点展示的图

如果你需要控制汇报页数，我建议优先展示以下 6 张图：

1. [side_from_usage_coeff_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_usage_coeff_confusion.png)  
   用来证明 side semantic path 已经具备强 laterality 语义。

2. [side_from_side_rep_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_side_rep_confusion.png)  
   用来证明 laterality 信息稳定存在于 side branch。

3. [side_from_free_rep_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_free_rep_confusion.png)  
   用来证明 free branch 对 laterality 的可分性明显较弱。

4. [dataset_from_private_rep_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_private_rep_confusion.png)  
   用来证明 private 分支仍是 dataset 信息主承载者。

5. [dataset_from_side_rep_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_side_rep_confusion.png)  
   用来说明 side branch 仍有 dataset leakage。

6. [dataset_from_free_rep_confusion.png](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_free_rep_confusion.png)  
   用来说明 free branch 也仍有 dataset leakage。

如果还需要一张总览表，建议直接使用：

- [probe_summary.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/probe_summary.csv)

## 9. 当前阶段可支持的结论

基于本次全量 `5-fold` 后验结果，我认为目前可以较稳妥地支持以下结论：

1. `v31` 的 side semantic path 已经学习到了稳定的 laterality 语义。
2. laterality 信息主要位于 side branch，而不是 free branch。
3. `usage` 和 `coeff` 都是有意义的、可解释的 side 语义载体。
4. `usage + coeff` 联合后对 laterality 的可分性最强，是当前最适合汇报的 side 语义读出结果。
5. private 分支最强承载 dataset 信息，这符合结构预期。
6. side/free 分支仍残留 dataset leakage，说明结构上仍有继续去域化的空间。

## 10. 当前阶段不宜过度声称的结论

同样需要明确，目前还不适合直接声称：

1. dataset 信息已经完全被排到 private 分支之外。
2. shared/free/side 三者已经彻底解耦。
3. 当前 side basis 与 side semantic path 已经对应到明确的临床语义。

更准确的说法应该是：

- `v31` 在 **laterality 可解释性** 上已经明显前进；
- 在 **dataset 去耦合** 上取得了部分进展，但还没有彻底解决；
- 在 **临床语义映射** 上，当前结果更像是“形成了稳定结构”，还需要下一轮后验关联分析去命名这些语义。

## 11. 相关文件索引

- 中文报告: [report_zh.md](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/report_zh.md)
- 英文报告: [report.md](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/report.md)
- 汇总 JSON: [summary.json](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/summary.json)
- probe 汇总表: [probe_summary.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/probe_summary.csv)
- 各 fold 指标: [probe_fold_metrics.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/probe_fold_metrics.csv)
- OOF 预测明细: [probe_predictions.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/probe_predictions.csv)
- fold 分配: [subject_fold_assignments.csv](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/subject_fold_assignments.csv)

