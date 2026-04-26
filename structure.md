下面是按 `v31` 实际 checkpoint 整理出的详细结构图。这个版本对应：

- checkpoint: [best.pt](/home/weizilin/generate_idea/outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/best.pt)
- 主模型实现: [distnet.py](/home/weizilin/generate_idea/scripts/lq/model/distnet.py#L131)
- CNN encoder: [encoder.py](/home/weizilin/generate_idea/scripts/lq/model/encoder.py#L14)
- `BasicBlock`: [BasicBlock.py](/home/weizilin/generate_idea/scripts/lq/model/BasicBlock.py#L24)
- 各 head: [heads.py](/home/weizilin/generate_idea/scripts/lq/model/heads.py#L22)
- quantizer: [quantizers.py](/home/weizilin/generate_idea/scripts/lq/model/quantizers.py#L17)
- basis 约束: [basis.py](/home/weizilin/generate_idea/scripts/lq/model/basis.py#L103)

**一、V31 实际配置**
`v31` 当前用到的关键结构参数是：

```text
mode = x
region = mouth
input = [B, T, 1, 119, 119]
group_size T = 4

hidden_dim = 32
levels = [2, 6]
side_basis_count = 3

early_branch_factorization = True
quantizer_type = residual_fsq
basis_orthogonalization = joint_global_qr

free_pool_size = 2
side_pooling = fixed_region2_contrast
private_pool_size = 1

free_z_dim = 32
side_z_dim = 32
private_dim = 32

private_adapter_enabled = False
private_residual_weight = 0.05
private_residual_max_l1 = 0.5

shared_basis_soft_mixing = True
shared_basis_anchor_bias = 2.0
shared_basis_topk = 2
```

---

**二、总图**

设：
- 输入 batch 为 `[B, T, 1, 119, 119]`
- 内部先展平时间维，记 `N = B * T`
- 所以主干 CNN 内部实际处理的是 `[N, 1, 119, 119]`

整体结构可以画成：

```text
Input: x
[B, T, 1, 119, 119]
    |
    | flatten sequence
    v
[N=B*T, 1, 119, 119]
    |
    | Shared CNN trunk
    v
[N, 32, 15, 15]
    |-------------------------------|------------------------------|
    |                               |                              |
    | Free branch                   | Side branch                  | Private branch
    |                               |                              |
    v                               v                              v
free_adapter                    side_adapter                  trunk feats directly
[N,32,15,15]                   [N,32,15,15]                  [N,32,15,15]
    |                               |                              |
free_pool 2x2                   fixed_region2_contrast         private_pool 1x1
[N,32,2,2]                     [N,64]                         [N,32,1,1]
    |                               |                              |
flatten                         side_head                       flatten
[N,128]                         [N,32]                          [N,32]
    |                               |                              |
free_head                        side_latent                     private_head
[N,32]=free_raw                  [N,32]                          [N,32]=private_z
    |                               |                              |
residual FSQ                     side semantic heads             private_decoder
2 stages: [2] + [6]             usage logits [N,3]              [N,14161]
    |                               |                              |
free_quantized [N,32]            softmax -> usage [N,3]          reshape
indices [N,2]                    coeff [N,1]                     [N,119,119]
stage q1,q2 [N,32]               side basis mix                  sym+zero_diag
    |                               |                              |
2 shared levels recon             shared_side_recon               private_residual
[N,119,119]                       [N,119,119]                     [N,119,119]
    |_______________________________|______________________________|
                                    |
                                    v
                        shared_recon + 0.05 * private_residual
                                    |
                                    v
                             reconstructed
                          [N,1,119,119]
                                    |
                              reshape back
                                    v
                         [B, T, 1, 119, 119]
```

---

**三、CNN 主干：逐 Block 维度变化**

实现位置：
- [encoder.py](/home/weizilin/generate_idea/scripts/lq/model/encoder.py#L14)
- [BasicBlock.py](/home/weizilin/generate_idea/scripts/lq/model/BasicBlock.py#L24)

输入：

```text
x: [B, T, 1, 119, 119]
-> flatten
x_flat: [N, 1, 119, 119]
```

### 1. `initial_conv`
代码： [encoder.py:17](/home/weizilin/generate_idea/scripts/lq/model/encoder.py#L17)

结构：

```text
Conv2d(1 -> 8, kernel=7, stride=2, padding=3, bias=False)
BatchNorm2d(8)
ReLU
```

维度：

```text
[N, 1, 119, 119]
-> [N, 8, 60, 60]
```

### 2. `layer1 = BasicBlock(8 -> 16, stride=2)`
代码：
- [encoder.py:22](/home/weizilin/generate_idea/scripts/lq/model/encoder.py#L22)
- [BasicBlock.py:27](/home/weizilin/generate_idea/scripts/lq/model/BasicBlock.py#L27)

主支路：

```text
conv1: Conv2d(8 -> 16, 3x3, stride=2, padding=1)
bn1
relu
conv2: Conv2d(16 -> 16, 3x3, stride=1, padding=1)
bn2
```

残差支路：

```text
downsample:
Conv2d(8 -> 16, 1x1, stride=2)
BatchNorm2d(16)
```

维度：

```text
input:        [N, 8, 60, 60]
main conv1:   [N,16,30,30]
main conv2:   [N,16,30,30]
skip branch:  [N,16,30,30]
add + relu:   [N,16,30,30]
```

### 3. `layer2 = BasicBlock(16 -> 32, stride=2)`
代码： [encoder.py:31](/home/weizilin/generate_idea/scripts/lq/model/encoder.py#L31)

主支路：

```text
conv1: Conv2d(16 -> 32, 3x3, stride=2, padding=1)
bn1
relu
conv2: Conv2d(32 -> 32, 3x3, stride=1, padding=1)
bn2
```

残差支路：

```text
downsample:
Conv2d(16 -> 32, 1x1, stride=2)
BatchNorm2d(32)
```

维度：

```text
input:        [N,16,30,30]
main conv1:   [N,32,15,15]
main conv2:   [N,32,15,15]
skip branch:  [N,32,15,15]
add + relu:   [N,32,15,15]
```

### 4. `layer3 = BasicBlock(32 -> 32, stride=1)`
代码： [encoder.py:40](/home/weizilin/generate_idea/scripts/lq/model/encoder.py#L40)

主支路：

```text
conv1: Conv2d(32 -> 32, 3x3, stride=1, padding=1)
bn1
relu
conv2: Conv2d(32 -> 32, 3x3, stride=1, padding=1)
bn2
```

残差支路：
- 无 downsample，直接 identity

维度：

```text
input:        [N,32,15,15]
main conv1:   [N,32,15,15]
main conv2:   [N,32,15,15]
skip branch:  [N,32,15,15]
add + relu:   [N,32,15,15]
```

### 5. 主干输出

```text
trunk feats = [N,32,15,15]
```

---

**四、Early Branch Factorization：三路分支**

`v31` 走的是 `early_branch_factorization=True`，所以不是旧版“先 pooled 再切 side/free”，而是直接在 `15x15` 特征图上分三路。

实现位置：
- 分支创建： [distnet.py:265](/home/weizilin/generate_idea/scripts/lq/model/distnet.py#L265)
- early branch forward： [distnet.py:1070](/home/weizilin/generate_idea/scripts/lq/model/distnet.py#L1070)

---

**五、Free Branch 详细结构**

### 1. `free_adapter`
代码： [encoder.py:45](/home/weizilin/generate_idea/scripts/lq/model/encoder.py#L45)

结构：

```text
Conv2d(32 -> 32, 3x3, stride=1, padding=1, bias=False)
BatchNorm2d(32)
ReLU
```

维度：

```text
[N,32,15,15]
-> [N,32,15,15]
```

### 2. `free_pool`
代码： [encoder.py:55](/home/weizilin/generate_idea/scripts/lq/model/encoder.py#L55)

结构：

```text
AdaptiveAvgPool2d((2,2))
```

维度：

```text
[N,32,15,15]
-> [N,32,2,2]
-> flatten -> [N,128]
```

### 3. `free_head`
代码： [heads.py:22](/home/weizilin/generate_idea/scripts/lq/model/heads.py#L22)

结构：

```text
Linear(128 -> 32)
ReLU
Linear(32 -> 32)
```

维度：

```text
[N,128]
-> [N,32]
```

这个输出记为：

```text
free_raw = [N,32]
```

---

**六、Free Branch 的量化器：Residual FSQ**

实现位置： [quantizers.py:49](/home/weizilin/generate_idea/scripts/lq/model/quantizers.py#L49)

`v31` 用的是：

```text
quantizer_type = residual_fsq
levels = [2, 6]
dim = 32
```

所以它不是一个单级 codebook，而是两个串联的 FSQ stage：

### 1. Stage 1
```text
FSQ(levels=[2], dim=32)
input residual: [N,32]
output stage_quantized_1: [N,32]
output stage_index_1: [N]
```

### 2. Stage 2
```text
FSQ(levels=[6], dim=32)
input residual: [N,32]   # residual = free_raw - stage_quantized_1.detach()
output stage_quantized_2: [N,32]
output stage_index_2: [N]
```

### 3. 汇总输出

```text
free_quantized = stage_quantized_1 + stage_quantized_2
shape = [N,32]

indices = stack([idx1, idx2], dim=-1)
shape = [N,2]

stage_quantized = stack([q1, q2], dim=1)
shape = [N,2,32]
```

在 `v31` early branch 中：

```text
free_latent = free_quantized = [N,32]
```

注意这里输出字段名里仍然叫 `shared_quantized`，但在 `v31` 的 early-branch 路径里，它实际承载的是 free branch 的量化结果。

---

**七、Shared Free Reconstruction：两级 basis 路径**

实现位置：
- basis bank： [distnet.py:322](/home/weizilin/generate_idea/scripts/lq/model/distnet.py#L322)
- routing 主体： [distnet.py:1098](/home/weizilin/generate_idea/scripts/lq/model/distnet.py#L1098)

### 1. Shared basis bank
`levels=[2,6]`，所以 shared basis 总数为：

```text
2 + 6 = 8
shared basis bank: [8,119,119]
```

### 2. Side basis bank
另外 side basis 有：

```text
side basis bank: [3,119,119]
```

### 3. Joint QR 正交化
实现： [basis.py:103](/home/weizilin/generate_idea/scripts/lq/model/basis.py#L103)

`v31` 用的是：

```text
basis_orthogonalization = joint_global_qr
```

所以每次前向时，会先对：

```text
shared 8 bases + side 3 bases = 共 11 个 basis
```

一起做：
- 对称化
- 零对角
- joint QR 正交化

因此前向中真正参与重建的是：

```text
shared basis: [8,119,119]
side basis:   [3,119,119]
```

且两者全局联合正交。

---

### 4. Level 1：`2` 个 shared basis
使用的是第 1 个 FSQ stage 的量化向量 `q1=[N,32]`。

#### 4.1 `shared_basis_head[0]`
代码： [heads.py:64](/home/weizilin/generate_idea/scripts/lq/model/heads.py#L64)

结构：

```text
Linear(32 -> 32)
ReLU
Linear(32 -> 2)
```

输出：

```text
level1_logits: [N,2]
```

然后做：
- anchor bias
- topk=2 裁剪
- softmax

得到：

```text
level1_weights: [N,2]
```

#### 4.2 `shared_coeff_head[0]`
代码： [heads.py:46](/home/weizilin/generate_idea/scripts/lq/model/heads.py#L46)

结构：

```text
Linear(32 -> 32)
ReLU
Linear(32 -> 1)
```

输出：

```text
level1_coeff: [N,1]
```

#### 4.3 Level 1 basis mix
第 1 级 basis 子集：

```text
basis_level1: [2,119,119]
```

加权组合：

```text
selected_basis_level1 = einsum("bl,lxy->bxy")
[N,2] x [2,119,119] -> [N,119,119]
```

乘系数：

```text
level1_recon = coeff * selected_basis_level1
[N,1,1] * [N,119,119] -> [N,119,119]
```

---

### 5. Level 2：`6` 个 shared basis
使用的是第 2 个 FSQ stage 的量化向量 `q2=[N,32]`。

#### 5.1 `shared_basis_head[1]`

```text
Linear(32 -> 32)
ReLU
Linear(32 -> 6)
```

输出：

```text
level2_logits: [N,6]
-> softmax weights: [N,6]
```

#### 5.2 `shared_coeff_head[1]`

```text
Linear(32 -> 32)
ReLU
Linear(32 -> 1)
```

输出：

```text
level2_coeff: [N,1]
```

#### 5.3 Level 2 basis mix
第 2 级 basis 子集：

```text
basis_level2: [6,119,119]
```

组合后：

```text
selected_basis_level2: [N,119,119]
level2_recon: [N,119,119]
```

---

### 6. Shared Free Reconstruction 汇总

```text
shared_free_recon
= level1_recon + level2_recon
shape = [N,119,119]
```

同时保留分析相关中间量：

```text
free_path_usage:
concat([level1_weights, level2_weights], dim=1)
shape = [N, 2+6] = [N,8]

free_path_coefficients:
concat([coeff1, coeff2], dim=1)
shape = [N,2]

free_path_representation:
concat([level1_weights*coeff1, level2_weights*coeff2], dim=1)
shape = [N,8]
```

---

**八、Side Branch 详细结构**

### 1. `side_adapter`
结构和 free_adapter 一样：

```text
Conv2d(32 -> 32, 3x3, padding=1)
BatchNorm2d(32)
ReLU
```

维度：

```text
[N,32,15,15]
-> [N,32,15,15]
```

### 2. `fixed_region2_contrast` 池化
实现： [distnet.py:495](/home/weizilin/generate_idea/scripts/lq/model/distnet.py#L495)

它不是普通 pooling，而是先从 `15x15` 特征图取 4 个固定对角区域：

```text
block 0: [0:3,  0:3 ]   -> around_left
block 1: [3:6,  3:6 ]   -> around_right
block 2: [6:10, 6:10]   -> mouth_left
block 3: [10:15,10:15]  -> mouth_right
```

每个 block 对空间做 mean：

```text
around_left:  [N,32]
around_right: [N,32]
mouth_left:   [N,32]
mouth_right:  [N,32]
```

然后构造显式左右差分：

```text
around_contrast = around_left - around_right   -> [N,32]
mouth_contrast  = mouth_left  - mouth_right    -> [N,32]
concat -> side_pooled = [N,64]
```

所以 `v31` 的 side 分支不是从全局平均池化来读 laterality，而是从 trunk 的 `15x15` 特征图上显式提取两个左右对比 token。

### 3. `side_head`
代码： [heads.py:30](/home/weizilin/generate_idea/scripts/lq/model/heads.py#L30)

结构：

```text
Linear(64 -> 32)
ReLU
Linear(32 -> 32)
```

输出：

```text
side_latent = [N,32]
```

---

**九、Side Semantic Path：3 个 side basis**

实现位置： [distnet.py:1142](/home/weizilin/generate_idea/scripts/lq/model/distnet.py#L1142)

### 1. `side_semantic_basis_head`

```text
Linear(32 -> 32)
ReLU
Linear(32 -> 3)
```

输出：

```text
side_basis_logits: [N,3]
side_path_usage = softmax(logits): [N,3]
```

### 2. `side_semantic_coeff_head`

```text
Linear(32 -> 32)
ReLU
Linear(32 -> 1)
```

输出：

```text
side_coeff: [N,1]
```

### 3. Side basis mix
side basis bank：

```text
[3,119,119]
```

组合后：

```text
selected_side_basis:
[N,3] x [3,119,119] -> [N,119,119]

shared_side_recon:
side_coeff * selected_side_basis
[N,1,1] * [N,119,119] -> [N,119,119]
```

同时保留：

```text
side_path_representation = side_path_usage * side_coeff
shape = [N,3]
```

这 3 维就是后面 `side_from_usage` / `side_from_coeff` / `side_from_usage_coeff` 分析里最核心的可解释路径。

---

**十、Private Branch 详细结构**

### 1. 输入
因为：

```text
private_adapter_enabled = False
```

所以 private branch 直接拿 trunk feats：

```text
private_feats = [N,32,15,15]
```

### 2. `private_pool`

```text
AdaptiveAvgPool2d((1,1))
```

维度：

```text
[N,32,15,15]
-> [N,32,1,1]
-> flatten -> [N,32]
```

### 3. `private_head`
代码： [heads.py:14](/home/weizilin/generate_idea/scripts/lq/model/heads.py#L14)

结构：

```text
Linear(32 -> 32)
ReLU
Linear(32 -> 32)
```

输出：

```text
private_z = [N,32]
```

### 4. `private_decoder`
代码： [heads.py:82](/home/weizilin/generate_idea/scripts/lq/model/heads.py#L82)

结构：

```text
Linear(32 -> 64)
ReLU
Linear(64 -> 14161)
```

因为：

```text
119 * 119 = 14161
```

所以：

```text
[N,32]
-> [N,64]
-> [N,14161]
-> reshape -> [N,119,119]
```

然后再经过：
- 对称化 + 零对角
- `private_residual_max_l1 = 0.5` 的幅值裁剪

得到：

```text
private_residual = [N,119,119]
```

---

**十一、重建路径**

### 1. Shared 部分
```text
shared_recon = shared_free_recon + shared_side_recon
shape = [N,119,119]
```

### 2. Private 残差缩放注入
```text
recon = shared_recon + 0.05 * private_residual
shape = [N,119,119]
```

### 3. 最终矩阵约束
再次做：
- symmetric
- zero diagonal

然后：

```text
recon.unsqueeze(1)
-> [N,1,119,119]
```

### 4. 恢复序列维
如果输入原本是 `[B,T,1,119,119]`，则输出恢复为：

```text
reconstructed:         [B,T,1,119,119]
action_reconstruction: [B,T,1,119,119]
shared_side_reconstruction: [B,T,1,119,119]
shared_free_reconstruction: [B,T,1,119,119]
private_residual:      [B,T,1,119,119]
```

---

**十二、训练/分析头在 V31 中的实际输入维度**

这部分很关键，因为很多名字容易混。

### 1. Side 分类头
代码： [heads.py:95](/home/weizilin/generate_idea/scripts/lq/model/heads.py#L95)

结构：

```text
Linear(32 -> 32)
ReLU
Linear(32 -> 3)
```

在 `v31` early branch 中，它作用在：

```text
side_latent: [N,32]
```

即：

```text
side_logits = side_classifier(side_latent)
```

### 2. Free-side adversary
同样是：

```text
Linear(32 -> 32)
ReLU
Linear(32 -> 3)
```

作用在：

```text
free_latent: [N,32]
```

通过 `grad_reverse` 反向梯度。

### 3. Group side classifier
代码： [heads.py:123](/home/weizilin/generate_idea/scripts/lq/model/heads.py#L123)

结构：

```text
Linear(3 -> 32)
ReLU
Linear(32 -> 3)
```

它作用在组级别的：

```text
mean_t(side_path_representation)
[B,3]
```

不是作用在 32 维 `side_latent` 上。

---

**十三、分析输出里几个“side_rep”名字的区别**

这个地方最容易混淆，我单独写清楚。

### 1. 帧级 `side_path_representation`
这是：

```text
side_path_usage * side_coeff
shape = [B,T,3]
```

表示 3 个 side basis 的“使用量 x 强度”。

### 2. 帧级 `side_latent`
这是：

```text
side_head(side_pooled)
shape = [B,T,32]
```

表示 side 分支的连续隐变量。

### 3. `analyze_checkpoint.py` 里用于 `side_from_side_rep` 的 group 表征
对 `v31` 来说，`group_side_rep` 被强制从：

```text
side_path_representation: [B,T,3]
```

做 masked mean 得到，所以它其实是：

```text
group_pooled_side_rep_for_probe = [B,3]
```

因此 `side_from_side_rep` 在 `v31` 的后验 probe 里，实际上读的是 3 维 side semantic path，而不是 32 维 `side_latent`。

### 4. `return_group_pooled=True` 时模型返回的 `group_pooled_side_rep`
在 `v31` early branch 路径里，这个字段实际上是：

```text
mean_t(side_latent)
shape = [B,32]
```

所以：
- 模型返回字段里的 `group_pooled_side_rep` 是 32 维
- 分析脚本 `side_from_side_rep` 实际使用的是 3 维 `side_path_representation` 的时间平均

这两个名字很像，但不是同一个东西。

---

**十四、V31 的完整维度流图**

下面给你一版可以直接汇报/画图用的完整结构图。

```text
Input
[B, T, 1, 119, 119]
T = 4
|
| flatten time
v
[N=B*T, 1, 119, 119]

==================== Shared CNN Trunk ====================

(1) initial_conv
    Conv2d 1->8, k7,s2,p3
    BN
    ReLU
    [N,1,119,119] -> [N,8,60,60]

(2) BasicBlock layer1
    conv1: 8->16, k3,s2,p1       [N,16,30,30]
    bn1 + relu
    conv2: 16->16, k3,s1,p1      [N,16,30,30]
    bn2
    skip:  1x1 conv 8->16,s2     [N,16,30,30]
    add + relu
    output: [N,16,30,30]

(3) BasicBlock layer2
    conv1: 16->32, k3,s2,p1      [N,32,15,15]
    bn1 + relu
    conv2: 32->32, k3,s1,p1      [N,32,15,15]
    bn2
    skip:  1x1 conv 16->32,s2    [N,32,15,15]
    add + relu
    output: [N,32,15,15]

(4) BasicBlock layer3
    conv1: 32->32, k3,s1,p1      [N,32,15,15]
    bn1 + relu
    conv2: 32->32, k3,s1,p1      [N,32,15,15]
    bn2
    skip: identity               [N,32,15,15]
    add + relu
    output: [N,32,15,15]

Trunk output feats = [N,32,15,15]

==================== Branch Split ====================

                 feats [N,32,15,15]
                 /         |         \
                /          |          \
               v           v           v

----------- Free Branch -----------  ----------- Side Branch -----------  ----------- Private Branch -----------

free_adapter                         side_adapter                         private path uses feats directly
Conv3x3 32->32                       Conv3x3 32->32
BN + ReLU                            BN + ReLU
[N,32,15,15]                         [N,32,15,15]                        [N,32,15,15]

free_pool 2x2                        fixed_region2_contrast              private_pool 1x1
[N,32,2,2]                           around_left  [N,32]                 [N,32,1,1]
flatten -> [N,128]                   around_right [N,32]                 flatten -> [N,32]
                                     mouth_left   [N,32]
                                     mouth_right  [N,32]
                                     around_contrast = left-right [N,32]
                                     mouth_contrast  = left-right [N,32]
                                     concat -> [N,64]

free_head                            side_head                            private_head
Linear 128->32                       Linear 64->32                        Linear 32->32
ReLU                                 ReLU                                 ReLU
Linear 32->32                        Linear 32->32                        Linear 32->32
output free_raw=[N,32]               output side_latent=[N,32]            output private_z=[N,32]

----------- Quantizer -----------

Residual FSQ with levels [2,6], dim=32

Stage 1:
    q1=[N,32], idx1=[N]
Stage 2:
    q2=[N,32], idx2=[N]

free_quantized = q1 + q2 = [N,32]
indices = [N,2]
decoded_indices = [idx1, idx2]

----------- Shared Free Reconstruction -----------

Shared basis bank:
    total = 8 = 2 + 6
    shape [8,119,119]

Side basis bank:
    shape [3,119,119]

All 11 bases jointly:
    symmetric + zero diagonal
    joint_global_qr

Level 1 path:
    input stage q1 = [N,32]
    basis_head_1: 32->32->2        logits [N,2]
    + anchor bias
    + topk=2
    softmax -> weights [N,2]
    coeff_head_1: 32->32->1        coeff [N,1]
    basis subset [2,119,119]
    weighted sum -> [N,119,119]
    scaled by coeff -> level1_recon [N,119,119]

Level 2 path:
    input stage q2 = [N,32]
    basis_head_2: 32->32->6        logits [N,6]
    + anchor bias
    + topk=2
    softmax -> weights [N,6]
    coeff_head_2: 32->32->1        coeff [N,1]
    basis subset [6,119,119]
    weighted sum -> [N,119,119]
    scaled by coeff -> level2_recon [N,119,119]

shared_free_recon = level1_recon + level2_recon
shape [N,119,119]

free_path_usage = concat([w1,w2]) = [N,8]
free_path_rep   = concat([w1*c1,w2*c2]) = [N,8]

----------- Side Semantic Reconstruction -----------

side_basis_head: 32->32->3
side_basis_logits = [N,3]
softmax -> side_path_usage = [N,3]

side_coeff_head: 32->32->1
side_coeff = [N,1]

side basis bank [3,119,119]
weighted sum -> selected_side_basis [N,119,119]
shared_side_recon = side_coeff * selected_side_basis
shape [N,119,119]

side_path_representation = side_path_usage * side_coeff
shape [N,3]

----------- Private Residual -----------

private_decoder:
    Linear 32->64
    ReLU
    Linear 64->14161
    reshape -> [N,119,119]
    symmetric + zero diagonal
    L1 clip to max 0.5

private_residual = [N,119,119]

----------- Final Reconstruction -----------

shared_recon = shared_free_recon + shared_side_recon
             = [N,119,119]

final_matrix = shared_recon + 0.05 * private_residual
             = [N,119,119]

enforce symmetric + zero diagonal again
unsqueeze channel -> [N,1,119,119]

reshape back -> reconstructed [B,T,1,119,119]
```

---

**十五、最终输出张量一览**

真实 shape tracing 得到：

```text
shared_quantized            [B,T,32]
side_latent                 [B,T,32]
free_latent                 [B,T,32]
private_z                   [B,T,32]

side_path_usage             [B,T,3]
free_path_usage             [B,T,8]

side_path_representation    [B,T,3]
free_path_representation    [B,T,8]

action_reconstruction       [B,T,1,119,119]
private_residual            [B,T,1,119,119]
reconstructed               [B,T,1,119,119]

indices                     [B,T,2]
decoded_indices             [ [B,T], [B,T] ]
```

---

**十六、最简结论**

如果压缩成一句话，`v31` 当前结构就是：

**一个浅层 CNN trunk，把 `119x119` 的对称差分矩阵编码成 `15x15x32` 特征图；在这个特征图上提前分成 free / side / private 三路，其中 free 路走两级 residual FSQ + 8 个 shared basis 重建动作主成分，side 路走显式左右对比池化 + 3 个 side basis 提取 laterality 语义，private 路走 32 维残差隐变量解码个体/域残差，最后三者在矩阵空间相加得到重建。**

如果你要，我下一步可以继续把这份结构整理成：
1. 一张更适合 PPT 的单页框图  
2. 一张“训练时哪些 loss 打在哪些张量上”的结构图  
3. 一张“v29 / v30 / v31 结构对比图”