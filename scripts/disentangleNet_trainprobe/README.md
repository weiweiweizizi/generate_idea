# `scripts/disentangleNet_trainprobe`

这是从 `scripts/disentangleNet` 派生出来的轻量训练闭包，目标是做结构迭代时的快速比较。

这个目录只关注三件事：

- 训练
- 验证
- probe 观测

它**不包含**原 `disentangleNet` 那套复杂分析链，也不承担 `matrix_vis`、患者级统计、t-SNE、共激活分析等职责。

## 当前约束

- 默认任务是 `side`
- `num_side_classes=3`
- side supervision 使用 3 类交叉熵
- `probe` 只是观测指标，不参与总 loss，不影响梯度

## 目录组成

- `train.py`
  - 训练入口
- `data/`
  - 数据集与样本读取
- `model/`
  - `DistNet` 及相关模块
- `training/`
  - 配置、loss、probe、engine、checkpoint
- `init_basis/`
  - 初始化 basis

## 运行方式

```bash
bash scripts/disentangleNet_trainprobe/run_train_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe.sh
```

或直接：

```bash
python scripts/disentangleNet_trainprobe/train.py \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT
```

## 指标约定

训练和验证日志分成两路：

- `loss_metrics`
  - 用于优化和 checkpoint 选择
- `probe_metrics`
  - 只用于比较和观察

当前默认 probe 关注：

- `side_acc`
- `group_side_acc`
