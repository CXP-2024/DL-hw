# Coding Project 2 — SE-ResNet-18 模型设计与代码详解

## 一、整体设计思路

### 1.1 任务分析

- **数据集**: Tiny ImageNet — 200 类，64×64 RGB 图像，训练集 10 万张，验证集 1 万张
- **约束**: 参数量 ≤ 20M，不能使用预训练权重，训练时间 ≤ 30 分钟（16GB GPU）
- **目标**: 验证集 top-1 准确率 ≥ 45%（满分线）

### 1.2 为什么选择 SE-ResNet-18

| 候选架构 | 参数量 | 优劣 |
|---|---|---|
| 简单 CNN (VGG-style) | ~5-15M | 缺乏残差连接，深层梯度消失，难以训练深网络 |
| ResNet-18 | ~11.2M | 残差连接解决梯度消失，经典可靠 |
| **SE-ResNet-18** | **~11.4M** | ResNet-18 + 通道注意力，微小参数开销换来 1-2% 准确率提升 |
| ResNet-34/50 | 21-25M | 超出 20M 参数限制 |

最终选择 **SE-ResNet-18**：在参数预算内，兼顾深度、残差学习和注意力机制。

### 1.3 针对 64×64 小图像的适配

原始 ResNet 为 224×224 的 ImageNet 设计，开头用 7×7 大卷积 + MaxPool 快速将分辨率从 224 降到 56。但我们的输入只有 64×64，如果照搬会丢失过多信息。因此：

- **替换开头**: 用两个 3×3 卷积 + 一个 2×2 MaxPool（64→32），而非 7×7 conv + 3×3 MaxPool（224→56）
- **保留后续结构**: 4 个残差阶段的设计不变

---

## 二、模型架构详解 (`modules.py`)

### 2.1 网络总览

```
输入: (B, 3, 64, 64)
│
├── Stem ──────────────────── 两层 3×3 Conv + MaxPool
│   Conv2d(3→64, 3×3) → BN → ReLU
│   Conv2d(64→64, 3×3) → BN → ReLU
│   MaxPool2d(2×2)
│   输出: (B, 64, 32, 32)
│
├── Layer1 ────────────────── 2× BasicBlock(64→64)
│   输出: (B, 64, 32, 32)      不下采样
│
├── Layer2 ────────────────── 2× BasicBlock(64→128, stride=2)
│   输出: (B, 128, 16, 16)     空间尺寸减半
│
├── Layer3 ────────────────── 2× BasicBlock(128→256, stride=2)
│   输出: (B, 256, 8, 8)       空间尺寸减半
│
├── Layer4 ────────────────── 2× BasicBlock(256→512, stride=2)
│   输出: (B, 512, 4, 4)       空间尺寸减半
│
├── AdaptiveAvgPool2d(1) ──── 全局平均池化
│   输出: (B, 512, 1, 1)
│
├── Flatten ───────────────── 展平
│   输出: (B, 512)
│
├── Dropout(0.2) ──────────── 随机丢弃 20% 神经元
│
└── Linear(512→200) ───────── 全连接分类层
    输出: (B, 200)              200 类的 logits
```

**参数分布**:

| 模块 | 参数量 | 占比 |
|---|---|---|
| Stem | 38,848 | 0.3% |
| Layer1 (64ch) | 148,992 | 1.3% |
| Layer2 (128ch) | 529,664 | 4.6% |
| Layer3 (256ch) | 2,116,096 | 18.6% |
| Layer4 (512ch) | 8,459,264 | 74.2% |
| FC Head | 102,600 | 0.9% |
| **总计** | **11,395,464** | **100%** |

### 2.2 SEBlock — 通道注意力模块

```python
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = channels // reduction  # 瓶颈维度（如 64//16=4）
        self.squeeze = nn.AdaptiveAvgPool2d(1)          # 全局平均池化
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),        # 降维 FC
            nn.ReLU(inplace=True),                       # 非线性
            nn.Linear(mid, channels, bias=False),        # 升维 FC
            nn.Sigmoid(),                                # 输出 0~1 的权重
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)       # (B,C,H,W) → (B,C)  提取全局信息
        y = self.excitation(y).view(b, c, 1, 1)  # (B,C) → (B,C,1,1) 学习注意力权重
        return x * y                          # 按通道加权原始特征图
```

**原理**: 不同通道对应不同特征（边缘、纹理、语义等），SE Block 让网络学会"关注哪些通道更重要"。

**计算流程**:

```
输入特征图 (B, C, H, W)
     │
     ▼ Squeeze: 全局平均池化
    (B, C, 1, 1) → (B, C)      每个通道一个标量
     │
     ▼ Excitation: FC→ReLU→FC→Sigmoid
    (B, C) → (B, C/r) → (B, C) → (B, C)    学习通道间关系
     │
     ▼ Scale: 逐通道相乘
    输出 (B, C, H, W)          增强重要通道，抑制不重要通道
```

**参数开销**: 以 C=512, r=16 为例，仅 2×512×32 = 32,768 个参数，相比整层的数百万参数可以忽略不计。

### 2.3 BasicBlock — 残差块

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, se_reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, se_reduction)
        self.downsample = downsample  # 用于匹配维度的 1×1 卷积

    def forward(self, x):
        identity = x                          # 保存输入（用于残差连接）

        out = self.conv1(x)                   # 第一个 3×3 卷积（可能下采样）
        out = self.bn1(out)                   # 批归一化
        out = self.relu(out)                  # 激活

        out = self.conv2(out)                 # 第二个 3×3 卷积
        out = self.bn2(out)                   # 批归一化（注意：这里不加 ReLU）
        out = self.se(out)                    # SE 通道注意力

        if self.downsample is not None:       # 如果维度不匹配
            identity = self.downsample(x)     # 用 1×1 卷积调整 identity

        out += identity                       # 残差连接: F(x) + x
        out = self.relu(out)                  # 最后再激活
        return out
```

**残差连接的核心思想**: 网络学习的是 `F(x) = H(x) - x`（残差），而非直接学习 `H(x)`。这让梯度可以通过 shortcut 直接回传，解决深层网络的梯度消失问题。

**数据流图** (以 Layer2 第一个块为例: 64→128, stride=2):

```
输入 x: (B, 64, 32, 32)
     │                    │
     ▼ 主路径              ▼ Shortcut (downsample)
  Conv2d(64→128, 3×3,    Conv2d(64→128, 1×1,
         stride=2)               stride=2)
  → (B, 128, 16, 16)    → (B, 128, 16, 16)
     │                    │
  BN → ReLU              BN
     │                    │
  Conv2d(128→128, 3×3)   │
  → (B, 128, 16, 16)    │
     │                    │
  BN                      │
     │                    │
  SE Attention            │
     │                    │
     └────── + ───────────┘   残差相加
             │
           ReLU
             │
  输出: (B, 128, 16, 16)
```

### 2.4 CustomModel — 完整网络

```python
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        # === Stem ===
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 3→64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), # 64→64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64×64 → 32×32
        )

        # === 4 个残差阶段 ===
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)    # 32×32
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)   # → 16×16
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)  # → 8×8
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)  # → 4×4

        # === 分类头 ===
        self.avgpool = nn.AdaptiveAvgPool2d(1)    # 4×4 → 1×1
        self.dropout = nn.Dropout(p=0.2)          # 防过拟合
        self.fc = nn.Linear(512, 200)             # 200 类输出
```

**`_make_layer` 方法**:

```python
def _make_layer(self, in_channels, out_channels, num_blocks, stride):
    downsample = None
    if stride != 1 or in_channels != out_channels:
        # 当分辨率改变或通道数改变时，shortcut 需要 1×1 卷积适配
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    # 第一个块：可能下采样
    layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
    # 后续块：维度不变
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)
```

### 2.5 权重初始化策略

```python
# 1) Kaiming 初始化 — 适配 ReLU 的方差缩放
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)   # γ = 1
        nn.init.constant_(m.bias, 0)     # β = 0
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# 2) 零初始化残差分支最后的 BN
for m in self.modules():
    if isinstance(m, BasicBlock):
        nn.init.constant_(m.bn2.weight, 0)   # γ₂ = 0
```

**为什么零初始化 bn2.weight？**

- 残差块输出 = ReLU(BN₂(Conv₂(…)) + identity)
- 当 BN₂ 的 γ=0 时，BN₂ 输出全为 0，残差块退化为 identity：output = ReLU(0 + x) = ReLU(x)
- 这意味着训练初期，网络等效于一个浅层网络，逐渐学习残差
- 论文 [Goyal et al., 2017] 证明这能提升 0.2~0.3% 的准确率

---

## 三、训练策略详解 (`train.py`)

### 3.1 超参数配置

```python
BATCH_SIZE = 128          # 每批样本数
NUM_EPOCHS = 100          # 最大训练轮数
LR = 0.1                  # 初始学习率
WEIGHT_DECAY = 5e-4       # L2 正则化系数
WARMUP_EPOCHS = 5         # 学习率预热轮数
LABEL_SMOOTHING = 0.1     # 标签平滑系数
MIXUP_ALPHA = 0.2         # Mixup 的 Beta 分布参数
MAX_TRAIN_MINUTES = 25    # 时间安全限制
```

### 3.2 数据增强流水线

```python
train_transform = Compose([
    ToDtype(torch.float32, scale=True),                          # ①
    RandomCrop(64, padding=8, padding_mode="reflect"),           # ②
    RandomHorizontalFlip(),                                       # ③
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # ④
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # ⑤
    RandomErasing(p=0.25),                                       # ⑥
])
```

| 步骤 | 变换 | 作用 |
|---|---|---|
| ① `ToDtype` | uint8 [0,255] → float32 [0,1] | 归一化像素值到 [0,1] |
| ② `RandomCrop` | 先 reflect 填充 8px（64→80），再随机裁剪回 64 | 模拟平移变换，增加位置多样性 |
| ③ `RandomHorizontalFlip` | 50% 概率水平翻转 | 几乎所有图像分类任务的标配 |
| ④ `ColorJitter` | 随机调整亮度/对比度/饱和度/色调 | 模拟不同光照和拍摄条件 |
| ⑤ `Normalize` | 用 ImageNet 均值和标准差标准化 | 使输入分布与网络初始化匹配 |
| ⑥ `RandomErasing` | 25% 概率随机遮挡一块矩形区域 | 迫使网络不依赖局部特征，提升鲁棒性 |

**为什么用 `reflect` 填充而非零填充？** 零填充会引入黑色边框，是一种不自然的伪特征。reflect 填充镜像边缘像素，更加自然。

### 3.3 Mixup — 样本混合

```python
def mixup_data(x, y):
    lam = Beta(0.2, 0.2).sample()           # 从 Beta(0.2, 0.2) 采样混合比例
    index = torch.randperm(x.size(0))        # 随机打乱 batch 内顺序
    mixed_x = lam * x + (1-lam) * x[index]  # 像素级线性插值
    return mixed_x, y, y[index], lam

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)  # 损失也按比例混合
```

**直觉**: 把一张"猫"和一张"狗"按 7:3 混合，标签也变成 0.7×猫 + 0.3×狗。这样做：
- 提供了无限多的"虚拟训练样本"
- 在类别之间创建平滑过渡，防止模型对单一类别过度自信
- 是非常强力的正则化，尤其对小数据集效果显著

**Beta(0.2, 0.2) 分布的特点**: 大部分时候 λ 接近 0 或 1（轻微混合），偶尔产生 λ≈0.5 的强混合。这比均匀分布 U(0,1) 更温和。

```
Beta(0.2, 0.2) 分布形状（U 形，两端概率高）:

概率
 ▲
 │█                                        █
 │██                                      ██
 │███                                    ███
 │█████                                █████
 │████████████████████████████████████████
 └─────────────────────────────────────────▶ λ
  0                  0.5                    1
```

### 3.4 优化器 — SGD + Nesterov 动量

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,                # 初始学习率
    momentum=0.9,          # 动量系数
    weight_decay=5e-4,     # L2 正则化
    nesterov=True,         # Nesterov 加速
)
```

**为什么用 SGD 而非 Adam？**
- 在图像分类任务上，SGD + 动量的泛化性能通常优于 Adam
- Adam 收敛更快但容易陷入 sharp minima（泛化差）
- SGD 的 flat minima 泛化更好

**Nesterov 动量 vs 普通动量**:
- 普通动量: 先累积梯度，再移动
- Nesterov: 先"预判"移动到估计位置，再计算梯度修正
- 相当于"先开枪再瞄准"，收敛更快更稳定

### 3.5 学习率调度 — Warmup + Cosine Annealing

```python
def lr_lambda(epoch):
    if epoch < 5:  # Warmup 阶段
        return (epoch + 1) / 5              # 0.02 → 0.04 → 0.06 → 0.08 → 0.10
    progress = (epoch - 5) / (100 - 5)      # 归一化到 [0, 1]
    return 0.5 * (1.0 + cos(π * progress))  # 余弦退火: 1.0 → 0.0
```

```
学习率变化曲线:

LR
0.10 ┤         ╭──╮
     │        ╱    ╲
0.08 ┤       ╱      ╲
     │      ╱        ╲
0.06 ┤     ╱          ╲
     │    ╱             ╲
0.04 ┤   ╱               ╲
     │  ╱                  ╲
0.02 ┤ ╱                     ╲
     │╱                        ╲___
0.00 ┼─────────────────────────────────
     0    5   20   40   60   80   100  epoch
     ├warmup┤     cosine annealing
```

**Warmup 的作用**: 训练初期权重随机，梯度方向不稳定。如果直接用 0.1 的大学习率，可能导致训练发散。先用小 LR 让网络"热身"，找到一个合理的方向后再加速。

**Cosine Annealing 的优势**: 相比阶梯式衰减（step decay），余弦曲线更平滑，避免了 LR 突降时的训练震荡。

### 3.6 标签平滑 (Label Smoothing)

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

不使用标签平滑时，真实标签是 one-hot（例如 [0, 0, 1, 0, ...]），模型被迫输出极端的 logits 来逼近这个分布。

使用 label_smoothing=0.1 后：
- 正确类别的目标概率: 1 - 0.1 + 0.1/200 = **0.9005**
- 其他类别的目标概率: 0.1/200 = **0.0005**

这防止模型过度自信，改善泛化性能。

### 3.7 AMP 混合精度训练

```python
use_amp = device_type == "cuda"
scaler = torch.amp.GradScaler(enabled=use_amp)

# 训练时
with torch.amp.autocast(device_type=device_type, enabled=use_amp):
    outputs = model(mixed_images)           # 前向传播用 FP16
    loss = mixup_criterion(outputs, ...)    # 损失计算用 FP16

scaler.scale(loss).backward()   # 缩放 loss 再反向传播（防止 FP16 梯度下溢）
scaler.step(optimizer)          # 先 unscale 梯度，再更新参数
scaler.update()                 # 动态调整缩放因子
```

**原理**: 前向和反向传播用 FP16（16位浮点数），减少显存占用，同时利用 GPU 的 Tensor Core 加速计算（约 2× 提速）。参数更新仍用 FP32 保证精度。

### 3.8 最佳模型追踪

```python
best_acc = 0.0
best_state = {}

for epoch in range(NUM_EPOCHS):
    # ... 训练 + 验证 ...
    if val_acc > best_acc:
        best_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

# 训练结束后恢复最佳权重
model.load_state_dict(best_state)
```

由于训练后期可能出现过拟合（验证准确率下降），我们保存验证准确率最高时的权重，训练结束后恢复到那个状态。这确保最终保存的 checkpoint 是最优的。

---

## 四、各组件协同工作流程

```
┌─────────────────────────────────────────────────────────┐
│                    main() 入口                           │
│  1. 加载训练集（基础 transform）                           │
│  2. 创建 CustomModel → 移到 GPU                          │
│  3. 检查参数量 ≤ 20M                                     │
│  4. 调用 train(model, dataset)                           │
│  5. 保存 checkpoint                                      │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   train() 训练循环                        │
│                                                          │
│  准备阶段:                                               │
│  ├── 创建增强训练集 (RandomCrop, Flip, ColorJitter, ...)  │
│  ├── 创建验证集（仅 Normalize）                           │
│  ├── 创建 DataLoader (batch=128, shuffle)                │
│  ├── 创建 SGD 优化器 + Cosine+Warmup 调度器               │
│  └── 创建 AMP GradScaler                                │
│                                                          │
│  每个 Epoch:                                             │
│  ├── 检查时间限制（25 分钟）                               │
│  ├── 训练阶段:                                           │
│  │   └── for batch in train_loader:                     │
│  │       ├── Mixup 混合图像和标签                         │
│  │       ├── AMP autocast 前向传播                        │
│  │       ├── 计算 Mixup 交叉熵损失（含标签平滑）            │
│  │       ├── GradScaler 反向传播                          │
│  │       └── 更新参数                                    │
│  ├── 更新学习率                                          │
│  ├── 验证阶段:                                           │
│  │   └── 计算验证集 top-1 准确率                          │
│  └── 保存最佳模型权重                                     │
│                                                          │
│  结束: 恢复最佳权重到 model                               │
└─────────────────────────────────────────────────────────┘
```

---

## 五、预期性能

| 指标 | 预期值 |
|---|---|
| 参数量 | 11,395,464 (~11.4M) |
| 验证准确率 | 50-60%（远超 45% 满分线）|
| 单 Epoch 时间 (16GB GPU) | ~20-30 秒 |
| 总训练时间 (100 epochs) | ~25-30 分钟 |

### 各技术对准确率的贡献估计

| 技术 | 单独贡献 |
|---|---|
| 基础 ResNet-18 | ~45-50% |
| + SE 注意力 | +1-2% |
| + Mixup | +2-3% |
| + 数据增强 (Crop+Flip+ColorJitter+Erasing) | +3-5% |
| + Label Smoothing | +0.5-1% |
| + Cosine Annealing + Warmup | +1-2% |
| + BN 零初始化 | +0.2-0.3% |
| **综合** | **~55-60%** |
