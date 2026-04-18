# 公共码本验证实验 (val_codebook)

验证单患者SVD提取的PC1主模态是否可以作为公共码本，用于区分患者的不同属性。

## 实验任务

### 实验1: IMR vs TT 数据集分类
验证IMR和TT两个数据集的PC1模态是否存在显著差异。

**标签**: 0=IMR, 1=TT

### 实验2: 测别分类 (Left/Normal/Right)
验证患者面瘫侧别（左侧/正常/右侧）在PC1模态上是否存在差异。

**标签映射** (基于label_5class):
- label_5class = 2 → Normal (正常)
- label_5class < 2 → Left (左侧异常)
- label_5class > 2 → Right (右侧异常)

### 实验3: 严重度分类 (Normal/Mild/Severe)
验证面瘫严重程度在PC1模态上是否存在差异（忽略测别）。

**标签映射** (基于score):
- score = 0 → Normal (正常)
- score = 1 → Mild (轻微)
- score = 2 → Severe (严重)

## 实验配置

每个任务以2×2配置运行（共12个配置）:

| 矩阵类型 | 方向 | 说明 |
|---------|------|------|
| Full (341×341) | X | 完整面部矩阵，X方向PC1 |
| Full (341×341) | Y | 完整面部矩阵，Y方向PC1 |
| Mouth (119×119) | X | 截断矩阵（around_mouth + mouth, indices 188:307），X方向 |
| Mouth (119×119) | Y | 截断矩阵，Y方向 |

**Mouth区域边界** (来自svd_single_patient.py):
- around_mouth: indices 188-233 (45 landmarks)
- mouth: indices 233-307 (74 landmarks)
- **截断区域**: indices 188-307 → 119×119 submatrix

## 算法

由于样本量小(n=269)而特征维度高(p=116,281)，采用PCA降维 + 简单分类器:

1. **PCA**: 降维到50个主成分
2. **Logistic Regression (L2)**: 主分类器，避免过拟合

## 数据来源

- **PC1矩阵**: `data/win20-step20/IMR-SVD/` 和 `data/win20-step20/TT-SVD/`
- **患者数**: IMR=227, TT=42, 共269

## 输出结构

```
scripts/val_codebook/
├── exp1_dataset_classification/output/
│   ├── roc_full_x.png
│   ├── roc_full_y.png
│   ├── roc_mouth_x.png
│   ├── roc_mouth_y.png
│   ├── confusion_full_x.png
│   ├── confusion_full_y.png
│   ├── confusion_mouth_x.png
│   ├── confusion_mouth_y.png
│   └── results_*.json
├── exp2_side_classification/output/
│   └── ...
├── exp3_severity_classification/output/
│   └── ...
└── summary_results.csv
```

## 运行方式

```bash
# 运行所有实验
python sweep.py --run

# 生成汇总表
python sweep.py --summary

# 单独运行某个实验
cd exp1_dataset_classification && python run.py
```

## 评价指标

- **Accuracy**: 5折交叉验证平均准确率
- **F1 (macro)**: 多分类F1分数
- **AUC**: 二分类ROC-AUC / 多分类OvR平均AUC

## 预期结果解读

| 结果 | 解读 |
|------|------|
| IMR vs TT 准确率高 | 两个数据集PC1模态差异大，可能需要分开建模 |
| 测别分类准确率高 | PC1模态包含左右不对称的特征 |
| Mouth区域效果好 | 嘴部区域是主要的判别区域 |
| X/Y方向效果不同 | 水平和垂直方向捕捉不同类型的运动信息 |
