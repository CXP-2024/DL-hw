# 条件变分自编码器 (CVAE) 详解说明书

> 本文档基于 `vae.ipynb` 的实现，面向入门学习者，从零开始讲解 VAE 和 CVAE 的原理、架构和代码实现。

---

## 目录

1. [什么是自编码器？——从 AE 到 VAE](#1-什么是自编码器从-ae-到-vae)
2. [VAE 的数学原理](#2-vae-的数学原理)
3. [条件 VAE (CVAE) 扩展](#3-条件-vae-cvae-扩展)
4. [重参数化技巧](#4-重参数化技巧)
5. [损失函数详解](#5-损失函数详解)
6. [CVAE 模型架构](#6-cvae-模型架构)
7. [编码器 (Encoder) 详解](#7-编码器-encoder-详解)
8. [解码器 (Decoder) 详解](#8-解码器-decoder-详解)
9. [完整数据流图解](#9-完整数据流图解)
10. [训练过程](#10-训练过程)
11. [采样与生成](#11-采样与生成)
12. [FID 评估指标](#12-fid-评估指标)
13. [超参数配置](#13-超参数配置)
14. [VAE vs Flow vs GAN 对比](#14-vae-vs-flow-vs-gan-对比)

---

## 1. 什么是自编码器？——从 AE 到 VAE

### 1.1 普通自编码器 (AE)

自编码器是一种"先压缩、再还原"的结构：

```
输入 x ──[编码器]──> 潜在表示 z ──[解码器]──> 重建 x̂
  │                    │                       │
 784维              很小(如20维)              784维
(28×28图片)        (压缩后的特征)          (重建的图片)
```

训练目标：让 $\hat{x}$ 尽可能接近 $x$（最小化重建误差）。

**问题**：AE 的潜在空间 $z$ 没有结构，不能随意采样。如果你随便取一个 $z$ 送入解码器，大概率会生成垃圾。

### 1.2 变分自编码器 (VAE) 的改进

VAE 的核心改进：**让潜在空间 $z$ 服从一个已知分布（标准正态分布）**。

```
           编码器                             解码器
输入 x ──────────> μ, σ ──采样──> z ──────────> 重建 x̂
                    │              │
                 不直接输出 z    z ~ N(μ, σ²)
                 而是输出分布参数  从分布中采样
```

这样，采样时我们可以直接从 $\mathcal{N}(0, I)$ 采样 $z$，送入解码器就能生成新的图片。

### 1.3 直觉理解

想象 VAE 是一个"概念压缩器"：

- **编码器**：看一张手写数字 "3"，说"这像一个 3，笔画粗细中等，略微右倾"（输出一个均值和方差描述的分布）
- **潜在空间**：这些描述被编码成一个连续的、有结构的空间，相近的数字在空间中靠近
- **解码器**：给定一个描述（从分布中采样），画出对应的数字

---

## 2. VAE 的数学原理

### 2.1 概率图模型视角

VAE 假设数据 $x$ 的生成过程是：

1. 从先验分布采样潜在变量：$z \sim p(z) = \mathcal{N}(0, I)$
2. 从条件分布采样数据：$x \sim p_\theta(x|z)$

我们的目标是最大化数据的**边际似然**（evidence）：

$$\log p_\theta(x) = \log \int p_\theta(x|z) p(z) dz$$

**问题**：这个积分无法直接计算（对所有可能的 $z$ 求积分）。

### 2.2 变分推断：引入近似后验

引入一个**近似后验分布** $q_\phi(z|x)$（由编码器参数化），来近似真实后验 $p_\theta(z|x)$。

通过数学推导可以得到 **ELBO（Evidence Lower BOund）**：

$$\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建项}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL 散度项}}$$

$$= \text{ELBO}(\theta, \phi; x)$$

### 2.3 两个 loss 项的直觉

| 项 | 公式 | 直觉 |
|----|------|------|
| **重建项** | $\mathbb{E}_{q(z\|x)}[\log p(x\|z)]$ | 编码再解码后，重建结果要像原图 |
| **KL 散度项** | $D_{KL}(q(z\|x) \| p(z))$ | 编码器输出的分布要接近标准正态 |

- **重建项**推动模型学会"记住信息"
- **KL 项**推动模型"忘掉无关细节"，让潜在空间规整

两者的博弈使 VAE 学到有意义且结构化的潜在表示。

### 2.4 为什么叫"变分"？

"变分"来自变分推断（Variational Inference），是一种将**概率推断问题转化为优化问题**的方法。我们不直接求后验 $p(z|x)$，而是用一个简单的分布族 $q_\phi(z|x)$（这里选高斯族）去逼近它。

---

## 3. 条件 VAE (CVAE) 扩展

### 3.1 动机

普通 VAE 只能随机生成图片，不能指定"我想生成数字 3"。CVAE 通过引入**条件信息 $y$**（类别标签）解决这个问题。

### 3.2 CVAE 的概率模型

$$p(x|y) = \int p_\theta(x|z, y) p(z) dz$$

- 先验：$p(z) = \mathcal{N}(0, I)$（与类别无关）
- 编码器（近似后验）：$q_\phi(z|x, y)$（输入同时看图片和标签）
- 解码器（似然）：$p_\theta(x|z, y)$（同时根据潜在变量和标签生成）

### 3.3 CVAE 的 ELBO

$$\log p(x|y) \geq \mathbb{E}_{q_\phi(z|x,y)}[\log p_\theta(x|z,y)] - D_{KL}(q_\phi(z|x,y) \| p(z))$$

**注意**：先验 $p(z)$ 保持为标准正态，不依赖于 $y$。

### 3.4 类别标签的编码

标签 $y$ 用 **one-hot 编码**表示：

```
数字 0 → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
数字 1 → [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
数字 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
...
数字 9 → [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

---

## 4. 重参数化技巧 (Reparameterization Trick)

### 4.1 问题

训练 VAE 需要对 loss 求梯度并反向传播。但 $z$ 是从 $q_\phi(z|x,y) = \mathcal{N}(\mu, \sigma^2)$ 中**随机采样**的，采样操作不可导！

```
x ──[编码器]──> μ, σ ──[采样 z ~ N(μ,σ²)]──> z ──[解码器]──> x̂
                              ↑
                         这里不可导!
                       梯度无法传回编码器
```

### 4.2 解决方案

把随机性"分离"出来：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\epsilon$ 是从标准正态分布采样的噪声（与参数无关）。

```
x ──[编码器]──> μ, σ ──┐
                       ├──> z = μ + σ·ε ──[解码器]──> x̂
ε ~ N(0,I) ────────────┘
       ↑
  外部随机性（不需要求梯度）
```

现在 $z$ 是 $\mu$ 和 $\sigma$ 的确定性函数（加上一个固定的随机数 $\epsilon$），梯度可以顺利传播到编码器！

### 4.3 代码实现

```python
def reparamaterize(self, mu: torch.Tensor, logstd: torch.Tensor):
    std = torch.exp(logstd)          # σ = exp(log σ)
    eps = torch.randn_like(std)      # ε ~ N(0, I)，与 std 同形状
    z = mu + eps * std               # z = μ + ε·σ
    return z
```

**为什么输出 `logstd` 而不是 `std`？**

- $\sigma$ 必须为正，但神经网络输出可以是任意实数
- 用 $\log\sigma$ 作为输出：$\sigma = e^{\log\sigma}$ 自动保证正值
- 数值更稳定（避免 $\sigma$ 太大或太小的问题）

---

## 5. 损失函数详解

### 5.1 公式分解

CVAE 的损失函数（负 ELBO）= **重建损失** + $\beta$ × **KL 散度**

$$\mathcal{L} = \underbrace{-\mathbb{E}_{q(z|x,y)}[\log p(x|z,y)]}_{\text{重建损失 (Reconstruction Loss)}} + \beta \cdot \underbrace{D_{KL}(q(z|x,y) \| p(z))}_{\text{KL 散度 (KL Divergence)}}$$

### 5.2 重建损失

因为像素值在 $[0, 1]$，用**二元交叉熵 (BCE)** 衡量重建质量：

$$\text{BCE} = -\sum_{i=1}^{784} \left[ x_i \log \hat{x}_i + (1 - x_i) \log (1 - \hat{x}_i) \right]$$

其中 $\hat{x}_i$ 是解码器输出的像素值（经过 Sigmoid 保证在 $(0,1)$）。

**直觉**：BCE 衡量"重建图片和原图有多像"，越小越好。

### 5.3 KL 散度

当 $q(z|x,y) = \mathcal{N}(\mu, \sigma^2)$，$p(z) = \mathcal{N}(0, I)$ 时，KL 散度有解析公式：

$$D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

$$= -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + 2\log\sigma_j - \mu_j^2 - e^{2\log\sigma_j} \right)$$

**直觉**：
- $\mu_j^2$ 项：均值偏离 0 越远，惩罚越大
- $\sigma_j^2$ 项：方差偏离 1 越远，惩罚越大
- 总之，推动编码器输出的分布靠近标准正态

### 5.4 $\beta$-VAE

损失函数中的 $\beta$ 参数控制重建和 KL 的权衡：

- $\beta = 1$：标准 VAE（ELBO）
- $\beta > 1$：更强的正则化，潜在空间更解耦（disentangled）
- $\beta < 1$：更好的重建质量，但生成多样性可能降低

### 5.5 代码实现

```python
def compute_vae_loss(vae_model, x, y, beta=1):
    # 标签转 one-hot
    if y.dim() == 1:
        y = F.one_hot(y, num_classes=vae_model.label_size).float()
    # y: (batch_size, 10)

    # 编码：得到后验分布参数
    mu, logstd = vae_model.encode_param(x, y)
    # mu:     (batch_size, latent_size=100)
    # logstd: (batch_size, latent_size=100)

    # 重参数化采样
    z = vae_model.reparamaterize(mu, logstd)
    # z: (batch_size, 100)

    # 解码：重建图片
    recon = vae_model.decode(z, y)
    # recon: (batch_size, 784)

    # 重建损失：BCE（逐像素，再对像素维求和）
    recon_loss = F.binary_cross_entropy(recon, x, reduction='none').sum(-1)
    # recon_loss: (batch_size,)  每个样本一个值

    # KL 散度（解析公式，逐维度，再求和）
    kl_loss = -0.5 * (1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp()).sum(-1)
    # kl_loss: (batch_size,)

    # 总损失
    loss = recon_loss + beta * kl_loss
    # loss: (batch_size,)
    return loss
```

### 5.6 损失项的数值级别

以训练后的模型为例（`val_loss ≈ 108.6`）：

```
总 loss ≈ 108.6 (每个样本)
  ├── 重建 loss ≈ 80-90 (784 个像素的 BCE 之和)
  └── KL loss   ≈ 20-30 (100 个潜在维度的 KL 之和)
```

---

## 6. CVAE 模型架构

### 6.1 整体结构图

```
┌─────────────────── 训练时 ───────────────────┐
│                                               │
│  图片 x (784维)  ──┐                          │
│                    ├─[img_fc]─┐               │
│                    │          ├─[concat]─[encoder]─┬─[fc_mu]──> μ      │
│  标签 y (10维)  ───┤          │                    │                   │
│                    ├─[label_fc_enc]─┘               └─[fc_logstd]──> log σ │
│                                                                        │
│                                         z = μ + ε·σ                    │
│                                           │                            │
│  标签 y (10维)  ──┐                       │                            │
│                    ├─[label_fc_dec]─┐      │                            │
│                    │               ├─[concat]─[decoder]──> x̂ (784维)  │
│                    ├─[latent_fc]───┘                                    │
│                    │                                                    │
│               z (100维)                                                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

┌─────────────────── 生成时 ───────────────────┐
│                                               │
│  z ~ N(0, I) ──[latent_fc]──┐                 │
│                              ├─[concat]─[decoder]──> 新图片          │
│  指定标签 y ──[label_fc_dec]─┘                 │
│                                               │
└───────────────────────────────────────────────┘
```

### 6.2 维度总结

| 组件 | 输入维度 | 输出维度 |
|------|---------|---------|
| `img_fc` | 784 | 256 |
| `label_fc_enc` | 10 | 256 |
| `encoder` | 512 (concat) | 256 |
| `fc_mu` | 256 | 100 |
| `fc_logstd` | 256 | 100 |
| `latent_fc` | 100 | 256 |
| `label_fc_dec` | 10 | 256 |
| `decoder` | 512 (concat) | 784 |

---

## 7. 编码器 (Encoder) 详解

### 7.1 功能

编码器的任务是：给定图片 $x$ 和标签 $y$，输出后验分布 $q(z|x,y) = \mathcal{N}(\mu, \sigma^2)$ 的参数 $\mu$ 和 $\log\sigma$。

### 7.2 结构

```
图片 x (batch, 784) ──[img_fc: 784→256]──[ReLU]──┐
                                                   ├──[concat: 512]──[encoder: 512→256, ReLU]──┬──[fc_mu: 256→100]──> μ
标签 y (batch, 10)  ──[label_fc_enc: 10→256]──[ReLU]┘                                          └──[fc_logstd: 256→100]──> log σ
```

### 7.3 代码详解

```python
# 定义网络组件（__init__ 中）
self.img_fc = nn.Linear(img_dim, hidden_size)        # 784 → 256: 图片特征提取
self.label_fc_enc = nn.Linear(label_size, hidden_size) # 10 → 256: 标签特征提取
self.encoder = nn.Sequential(
    nn.Linear(2 * hidden_size, hidden_size),          # 512 → 256: 融合特征
    nn.ReLU(),
)
self.fc_mu = nn.Linear(hidden_size, latent_size)      # 256 → 100: 输出均值
self.fc_logstd = nn.Linear(hidden_size, latent_size)  # 256 → 100: 输出对数标准差

# 前向传播（encode_param 中）
def encode_param(self, x, y):
    h_img = F.relu(self.img_fc(x))        # (batch, 784) → (batch, 256)
    h_label = F.relu(self.label_fc_enc(y)) # (batch, 10) → (batch, 256)
    h = torch.cat([h_img, h_label], dim=1) # (batch, 512)  拼接图片和标签特征
    h = self.encoder(h)                    # (batch, 512) → (batch, 256)
    mu = self.fc_mu(h)                     # (batch, 256) → (batch, 100)
    logstd = self.fc_logstd(h)             # (batch, 256) → (batch, 100)
    return mu, logstd
```

### 7.4 数据流示例

```
输入图片 x: 一张手写数字 "3" 的图片
  │
  ▼ img_fc + ReLU
h_img = [0.23, 0.0, 0.87, ..., 0.45]  (256维特征，捕捉了笔画粗细、倾斜等)

输入标签 y: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  (one-hot: 数字3)
  │
  ▼ label_fc_enc + ReLU
h_label = [0.0, 0.56, 0.0, ..., 0.12]  (256维特征，编码了"这是数字3")

  ▼ concat
h = [0.23, 0.0, 0.87, ..., 0.45, | 0.0, 0.56, 0.0, ..., 0.12]  (512维)

  ▼ encoder (Linear + ReLU)
h = [0.11, 0.0, 0.43, ..., 0.28]  (256维，融合后的特征)

  ▼ fc_mu                          ▼ fc_logstd
μ = [0.5, -0.3, ..., 0.1]  (100维)   log σ = [-1.2, -0.8, ..., -0.5]  (100维)

  ▼ 重参数化
z = μ + exp(log σ) · ε = [0.5 + 0.30·ε₁, -0.3 + 0.45·ε₂, ...]  (100维)
```

---

## 8. 解码器 (Decoder) 详解

### 8.1 功能

解码器的任务是：给定潜在变量 $z$ 和标签 $y$，输出重建的图片 $\hat{x}$。

### 8.2 结构

```
潜在变量 z (batch, 100) ──[latent_fc: 100→256]──[ReLU]──┐
                                                          ├──[concat: 512]──[decoder: 512→256→784, Sigmoid]──> x̂
标签 y (batch, 10)  ──[label_fc_dec: 10→256]──[ReLU]──────┘
```

### 8.3 代码详解

```python
# 定义网络组件（__init__ 中）
self.latent_fc = nn.Linear(latent_size, hidden_size)    # 100 → 256: 潜在变量特征提取
self.label_fc_dec = nn.Linear(label_size, hidden_size)  # 10 → 256: 标签特征提取
self.decoder = nn.Sequential(
    nn.Linear(2 * hidden_size, hidden_size),            # 512 → 256: 融合特征
    nn.ReLU(),
    nn.Linear(hidden_size, img_dim),                    # 256 → 784: 输出像素
    nn.Sigmoid(),                                       # 映射到 (0, 1)
)

# 前向传播（decode 中）
def decode(self, z, y):
    h_z = F.relu(self.latent_fc(z))        # (batch, 100) → (batch, 256)
    h_y = F.relu(self.label_fc_dec(y))     # (batch, 10) → (batch, 256)
    h = torch.cat([h_z, h_y], dim=1)       # (batch, 512)
    recon = self.decoder(h)                # (batch, 512) → (batch, 784)
    return recon  # 像素值 ∈ (0, 1)
```

### 8.4 为什么用 Sigmoid？

- 解码器输出代表像素值的概率（或灰度值），必须在 $[0, 1]$ 范围内
- Sigmoid 函数 $\sigma(x) = \frac{1}{1+e^{-x}}$ 把任意实数映射到 $(0, 1)$
- 这使得 BCE 损失有意义（BCE 要求输入在 $(0,1)$）

---

## 9. 完整数据流图解

### 9.1 训练时完整流程

```
Step 1: 数据准备
========================================
图片: (batch=128, 1, 28, 28)
  ▼ flatten
x: (128, 784)    像素值 ∈ (0.0001, 0.9999)

标签: (128,)  如 [3, 7, 1, 0, ...]
  ▼ one-hot
y: (128, 10)  如 [[0,0,0,1,...], [0,0,0,0,0,0,0,1,...], ...]


Step 2: 编码
========================================
x (128, 784)                        y (128, 10)
     │                                   │
     ▼ img_fc + ReLU                     ▼ label_fc_enc + ReLU
  (128, 256)                          (128, 256)
     │                                   │
     └──────────┬────────────────────────┘
                ▼ concat
            (128, 512)
                │
                ▼ encoder (Linear 512→256 + ReLU)
            (128, 256)
                │
        ┌───────┴───────┐
        ▼               ▼
     fc_mu          fc_logstd
   (128, 100)      (128, 100)
        │               │
        │   μ           │   log σ
        │               │
        └───┬───────────┘
            ▼ 重参数化: z = μ + exp(log σ) · ε
        z (128, 100)


Step 3: 解码
========================================
z (128, 100)                        y (128, 10)
     │                                   │
     ▼ latent_fc + ReLU                  ▼ label_fc_dec + ReLU
  (128, 256)                          (128, 256)
     │                                   │
     └──────────┬────────────────────────┘
                ▼ concat
            (128, 512)
                │
                ▼ decoder
                │  Linear 512→256 + ReLU
                │  Linear 256→784 + Sigmoid
                ▼
        recon (128, 784)  ← 重建的图片，像素值 ∈ (0, 1)


Step 4: 计算损失
========================================
重建损失 = BCE(recon, x)
         = -Σᵢ [xᵢ log(reconᵢ) + (1-xᵢ) log(1-reconᵢ)]
  → (128,)  每个样本一个值

KL 散度 = -0.5 * Σⱼ (1 + 2·log σⱼ - μⱼ² - σⱼ²)
  → (128,)

总 loss = recon_loss + β · kl_loss
  → (128,)

  ▼ mean()
标量 loss
  │
  ▼ backward() + optimizer.step()
更新编码器和解码器的参数
```

### 9.2 生成时

```
Step 1: 指定要生成的数字（如 "3"）
========================================
y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  (one-hot)

Step 2: 从先验分布采样
========================================
z ~ N(0, I)  → z: (1, 100)

Step 3: 解码
========================================
z (1, 100)                          y (1, 10)
     │                                   │
     ▼ latent_fc + ReLU                  ▼ label_fc_dec + ReLU
  (1, 256)                           (1, 256)
     │                                   │
     └──────────┬────────────────────────┘
                ▼ concat → decoder
        新图片 (1, 784)
                │
                ▼ reshape
        (1, 1, 28, 28)  ← 一张手写数字 "3" 的图片！
```

---

## 10. 训练过程

### 10.1 训练循环代码解读

```python
def train(n_epochs, vae_model, train_loader, val_loader, optimizer, beta=1, ...):
    for epoch in range(n_epochs):
        for i, (x, y) in enumerate(train_loader):
            vae_model.train()
            x = x.view(x.shape[0], -1).to(device)  # (batch, 784)
            y = y.to(device)                         # (batch,) 整数标签

            # 计算 VAE loss（内部会自动做 one-hot 转换）
            loss = compute_vae_loss(vae_model, x, y, beta)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
```

### 10.2 模型选择

每 `save_interval` 个 epoch 评估一次验证集 loss，保存最优模型：

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_model('./vae/vae_best.pth', vae_model)
```

### 10.3 训练曲线理解

从训练日志可以看到：
```
Epoch 1:  Train Loss ≈ 219.5   (一开始重建很差，KL 很大)
Epoch 10: Train Loss ≈ 130.2   (快速下降)
Epoch 30: Train Loss ≈ 112.1   (缓慢收敛)
Epoch 50: Train Loss ≈ 108.4   (基本收敛)
```

---

## 11. 采样与生成

### 11.1 条件生成

```python
@torch.no_grad()
def sample_images(self, label, save=True, save_dir='./vae'):
    self.eval()
    n_samples = label.shape[0]
    # 从标准正态分布采样 z
    z = torch.randn(n_samples, self.latent_size).to(label.device)  # (n, 100)
    # 用解码器生成图片
    samples = self.decode(z, label)  # (n, 784)
    imgs = samples.view(n_samples, 1, 28, 28).clamp(0., 1.)
    return imgs
```

### 11.2 生成 10×10 样本网格

```python
# 每个类别 10 张图片，共 100 张
label = torch.eye(10).repeat(10, 1).to(device)
# label 形状: (100, 10)
# 第 0-9 张: 数字 0
# 第 10-19 张: 数字 1
# ...
# 第 90-99 张: 数字 9

model.sample_images(label, save=True)
```

### 11.3 生成数据集（用于 FID 评估）

```python
@torch.no_grad()
def make_dataset(self, n_samples_per_class=100, ...):
    for i in range(10):  # 对每个数字类别
        # 创建 one-hot 标签
        label = torch.zeros(100, 10, device=device)
        label[:, i] = 1  # 全部设为第 i 类

        # 采样 z 并解码
        z = torch.randn(100, 100).to(device)
        samples = self.decode(z, label)

        # 保存为单独的 PNG 文件
        # ./vae/generated/3/3_042.png
```

---

## 12. FID 评估指标

### 12.1 什么是 FID？

**FID (Fréchet Inception Distance)** 衡量生成图片和真实图片的"距离"。越低越好。

### 12.2 计算过程

1. 用预训练的 Inception-V3 模型提取真实图片和生成图片的特征
2. 分别计算两组特征的均值 $\mu_r, \mu_g$ 和协方差矩阵 $\Sigma_r, \Sigma_g$
3. 计算 FID：

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

### 12.3 该项目的 FID 结果

从输出可以看到每个类别的 FID：
```
类别 0: 20.87    (较好)
类别 1: 11.58    (很好)
类别 2: 83.13    (较差)
类别 3: 38.30
类别 4: 81.74    (较差)
类别 5: 153.45   (很差)
类别 6: 43.96
类别 7: 60.12
类别 8: 51.12
类别 9: 40.47
```

某些数字（如 1）结构简单，容易生成；某些数字（如 5）变化多，难以生成。

---

## 13. 超参数配置

### 13.1 模型超参数

| 超参数 | 值 | 含义 |
|--------|-----|------|
| `img_size` | (1, 28, 28) | MNIST 图片大小 |
| `label_size` | 10 | 类别数（0-9） |
| `latent_size` | 100 | 潜在空间维度 |
| `hidden_size` | 256 | MLP 隐藏层维度 |

### 13.2 训练超参数

| 超参数 | 值 | 含义 |
|--------|-----|------|
| `batch_size` | 128 | 每批样本数 |
| `lr` | 2e-4 | 学习率 |
| `n_epochs` | 50 | 训练轮数 |
| `optimizer` | Adam | 优化器 |
| `beta` | 1 | KL 散度权重（标准 VAE） |
| `save_interval` | 10 | 每 10 轮评估 |

### 13.3 可调节方向

如果想提升 FID：

1. **增大隐藏层维度**：`hidden_size = 512` → 更强的表达力
2. **加深网络**：encoder/decoder 加更多层
3. **调整 $\beta$**：$\beta < 1$ 可能提升重建质量
4. **增大潜在维度**：`latent_size = 200` → 更丰富的潜在表示
5. **更多训练轮数**：`n_epochs = 200`
6. **学习率调度**：使用 `lr_scheduler` 逐步降低学习率

---

## 14. VAE vs Flow vs GAN 对比

### 14.1 架构对比

| | VAE | Flow | GAN |
|--|-----|------|-----|
| **编码器** | 有（输出分布参数） | 本身就是编码器（可逆） | 无 |
| **解码器/生成器** | 有（从 z 生成 x） | 编码器的逆（可逆） | 有（从 z 生成 x） |
| **判别器** | 无 | 无 | 有 |
| **可逆性** | 不可逆 | 可逆 | 不可逆 |

### 14.2 训练目标对比

| | VAE | Flow | GAN |
|--|-----|------|-----|
| **目标** | 最大化 ELBO | 最大化精确似然 | 极小极大博弈 |
| **能否算似然** | 只有下界 (ELBO) | 精确值 | 不能 |
| **训练稳定性** | 稳定 | 稳定 | 不稳定 |

### 14.3 优缺点对比

| | 优点 | 缺点 |
|--|------|------|
| **VAE** | 训练稳定，有潜在空间，可做条件生成 | 生成图片模糊（因为 BCE/MSE 损失） |
| **Flow** | 精确似然，可用于 inpainting | 架构受限（必须可逆），参数量大 |
| **GAN** | 生成质量最高，图片清晰 | 训练不稳定，模式崩塌 |

### 14.4 CVAE 的独特优势

CVAE 最大的优势是**可控生成**：

```python
# 想生成数字 "7"？只需要：
label = torch.zeros(1, 10)
label[0, 7] = 1  # one-hot: 数字 7
z = torch.randn(1, 100)
generated_7 = model.decode(z, label)

# 想生成数字 "3"？换个标签就行：
label[0, 7] = 0
label[0, 3] = 1  # one-hot: 数字 3
generated_3 = model.decode(z, label)  # 用同一个 z，但生成不同数字！
```

**同一个潜在变量 $z$，配上不同的标签 $y$，会生成不同类别但"风格相似"的图片**（比如笔画粗细、倾斜角度相似的不同数字）。

---

## 附录 A: 关键概念速查表

| 概念 | 符号 | 含义 |
|------|------|------|
| 先验分布 | $p(z) = \mathcal{N}(0, I)$ | 潜在变量的预设分布（标准正态） |
| 近似后验 | $q_\phi(z\|x, y)$ | 编码器输出的分布（看到 x 和 y 后对 z 的推断） |
| 似然 | $p_\theta(x\|z, y)$ | 给定 z 和 y 时生成 x 的概率（解码器） |
| ELBO | $\text{ELBO} = \mathbb{E}[\log p(x\|z,y)] - D_{KL}$ | 边际似然的下界（训练目标） |
| 重建损失 | BCE / MSE | 衡量重建图片和原图的差异 |
| KL 散度 | $D_{KL}(q \| p)$ | 衡量后验分布和先验的差异 |
| 重参数化 | $z = \mu + \sigma \cdot \epsilon$ | 使采样操作可导的技巧 |
| One-hot | $[0,0,1,0,...,0]$ | 类别标签的向量表示 |

## 附录 B: 常见问题

### Q: 为什么 VAE 生成的图片比 GAN 模糊？

A: VAE 使用 BCE/MSE 作为重建损失，这些损失对每个像素独立计算，倾向于生成"平均"的结果。当存在不确定性时，VAE 会生成所有可能性的模糊平均。GAN 使用判别器来衡量"像不像真的"，更注重全局质量。

### Q: 潜在维度 (latent_size=100) 怎么选？

A: 这是个经验值。
- 太小（如 2-10）：信息不够，重建差，但可以可视化潜在空间
- 适中（如 20-100）：通常效果好
- 太大（如 1000）：训练困难，KL 项可能太大导致"后验坍塌"

### Q: 什么是后验坍塌 (Posterior Collapse)？

A: 编码器直接输出 $\mu=0, \sigma=1$（标准正态），忽略输入 $x$，KL=0。解码器完全靠标签 $y$ 生成，不利用 $z$ 中的信息。这时 $z$ 变得无意义。解决方法：使用 KL annealing（逐渐增大 $\beta$）、更强的解码器等。
