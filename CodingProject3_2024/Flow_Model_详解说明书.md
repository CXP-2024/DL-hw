# Real NVP 流模型 (Flow-Based Model) 详解说明书

> 本文档基于 `flow.ipynb` 的实现，面向入门学习者，从零开始讲解流模型的原理、架构和代码实现。

---

## 目录

1. [什么是流模型？——从直觉出发](#1-什么是流模型从直觉出发)
2. [数学基础：变量替换公式](#2-数学基础变量替换公式)
3. [Real NVP 架构总览](#3-real-nvp-架构总览)
4. [核心模块一：CouplingLayer（仿射耦合层）](#4-核心模块一couplinglayer仿射耦合层)
5. [核心模块二：BatchNormFlow（批归一化层）](#5-核心模块二batchnormflow批归一化层)
6. [核心模块三：Shuffle（置换层）](#6-核心模块三shuffle置换层)
7. [FlowSequential：整体流水线](#7-flowsequential整体流水线)
8. [数据预处理：Logit 变换](#8-数据预处理logit-变换)
9. [训练过程：最大似然](#9-训练过程最大似然)
10. [采样过程：反向生成](#10-采样过程反向生成)
11. [Inpainting（图像修复）](#11-inpainting图像修复)
12. [完整数据流图解](#12-完整数据流图解)
13. [超参数与模型配置](#13-超参数与模型配置)

---

## 1. 什么是流模型？——从直觉出发

### 1.1 生成模型的目标

生成模型的目标是：**学习数据的概率分布 $p(x)$**。一旦学到了这个分布，我们就可以：
- 从中**采样**：生成新的数据（比如新的手写数字图片）
- 计算**似然**：判断一个数据点有多"像"训练数据

### 1.2 流模型的核心思路

流模型的思路非常直观：

> 找一个**可逆函数** $f$，把复杂的数据分布（比如 MNIST 图片）变换成一个简单的分布（标准正态分布）。

```
复杂分布（图片 x）  ──f──>  简单分布（噪声 z ~ N(0,I)）
                    <──f⁻¹──
```

- **正向** $z = f(x)$：把图片"编码"成正态分布的噪声
- **反向** $x = f^{-1}(z)$：从噪声"解码"出图片

### 1.3 类比理解

想象你有一团橡皮泥（复杂形状的数据分布），流模型就像一系列"揉搓操作"，把它揉成一个标准的球形（正态分布）。关键是：
- 每一步操作都是**可逆的**（能揉回去）
- 每一步操作都能算出**体积变化量**（Jacobian 行列式）

---

## 2. 数学基础：变量替换公式

### 2.1 一维情况

假设 $z = f(x)$ 是一个可逆变换，$x$ 的概率密度 $p_X(x)$ 和 $z$ 的概率密度 $p_Z(z)$ 之间的关系是：

$$p_X(x) = p_Z(f(x)) \cdot \left| \frac{df}{dx} \right|$$

直觉：如果 $f$ 在某处"拉伸"了空间（导数绝对值 > 1），那么这个区域的概率密度就要相应"摊薄"。

### 2.2 多维情况

对于多维向量 $\mathbf{x} \in \mathbb{R}^D$，公式变成：

$$p_X(\mathbf{x}) = p_Z(f(\mathbf{x})) \cdot \left| \det \frac{\partial f}{\partial \mathbf{x}} \right|$$

其中 $\frac{\partial f}{\partial \mathbf{x}}$ 是 **Jacobian 矩阵**（$D \times D$ 的矩阵，每个元素是 $\frac{\partial f_i}{\partial x_j}$）。

### 2.3 对数似然

取对数（更方便计算和优化）：

$$\log p_X(\mathbf{x}) = \log p_Z(f(\mathbf{x})) + \log \left| \det \frac{\partial f}{\partial \mathbf{x}} \right|$$

### 2.4 多层组合

如果 $f = f_K \circ f_{K-1} \circ \cdots \circ f_1$（多个可逆变换的组合），那么：

$$\log p_X(\mathbf{x}) = \log p_Z(\mathbf{z}) + \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial \mathbf{h}_{k-1}} \right|$$

其中 $\mathbf{h}_0 = \mathbf{x}$，$\mathbf{h}_K = \mathbf{z}$。

**这就是为什么叫"流"模型——数据像水流一样，经过一系列变换"流"向简单分布。**

---

## 3. Real NVP 架构总览

本 notebook 实现的是简化版 **Real NVP**（Real-valued Non-Volume Preserving transformations），论文：[Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)。

### 整体架构

模型由 **8 个相同结构的 block** 堆叠而成，每个 block 包含 3 层：

```
输入 x (784维向量, 即 28×28 的 MNIST 图片展平)
  │
  ├─── Block 1 ───┐
  │  CouplingLayer │  ← 仿射耦合变换（核心！）
  │  BatchNormFlow │  ← 批归一化（稳定训练）
  │  Shuffle       │  ← 随机置换维度
  ├────────────────┘
  │
  ├─── Block 2 ───┐
  │  CouplingLayer │
  │  BatchNormFlow │
  │  Shuffle       │
  ├────────────────┘
  │
  ⋮ (重复 8 次)
  │
  └──> 输出 z (784维向量, 服从标准正态分布)
```

总共 24 层 = 8 × (CouplingLayer + BatchNormFlow + Shuffle)

---

## 4. 核心模块一：CouplingLayer（仿射耦合层）

这是 Real NVP 最核心、最精妙的组件。

### 4.1 设计动机

问题：我们需要一个变换 $f$，满足：
1. **可逆**：能算 $f^{-1}$
2. **Jacobian 行列式好算**：$D \times D$ 矩阵的行列式一般需要 $O(D^3)$，对 $D=784$ 太慢了
3. **表达力强**：能学到复杂的变换

CouplingLayer 的天才设计同时满足了这三个条件。

### 4.2 核心思想：分而治之

把输入 $\mathbf{x}$ 分成两部分：
- **"条件"部分**（被 mask 覆盖的）：保持不变，作为条件
- **"变换"部分**（未被 mask 覆盖的）：根据条件部分做仿射变换

```
mask = [1, 0, 1, 0, 1, 0, ...]   (1 = 保持不变, 0 = 要变换)

输入 x:     [x₁, x₂, x₃, x₄, x₅, x₆, ...]
              │    │    │    │    │    │
mask部分:   [x₁,  ·, x₃,  ·, x₅,  ·, ...]  ──> 送入神经网络 ──> 输出 s 和 t
              │    │    │    │    │    │
输出 z:     [x₁, x₂·s₂+t₂, x₃, x₄·s₄+t₄, x₅, x₆·s₆+t₆, ...]
              ↑       ↑        ↑       ↑        ↑       ↑
           不变    仿射变换   不变    仿射变换   不变    仿射变换
```

### 4.3 数学公式

给定二值 mask $\mathbf{b} \in \{0, 1\}^D$：

**正向变换**（$\mathbf{x} \to \mathbf{z}$）：

$$\mathbf{z} = \mathbf{x} \odot \mathbf{s}(\mathbf{x} \odot \mathbf{b}) + \mathbf{t}(\mathbf{x} \odot \mathbf{b}) \odot (1 - \mathbf{b})$$

其中：
- $\mathbf{s}(\cdot)$ 是 scale 网络，输出缩放因子
- $\mathbf{t}(\cdot)$ 是 translate 网络，输出平移量
- $\odot$ 表示逐元素乘法

注意：被 mask 的部分 ($b_i = 1$) 保持不变（$z_i = x_i$），未被 mask 的部分 ($b_i = 0$) 做仿射变换（$z_i = x_i \cdot s_i + t_i$）。

**反向变换**（$\mathbf{z} \to \mathbf{x}$）：

$$\mathbf{x} = \frac{\mathbf{z} - \mathbf{t}(\mathbf{z} \odot \mathbf{b}) \odot (1 - \mathbf{b})}{\mathbf{s}(\mathbf{z} \odot \mathbf{b})}$$

**关键洞察**：反向变换只需要把乘法变除法、加法变减法，**不需要对神经网络本身求逆**！因为条件部分（$\mathbf{b}$ 对应位置）在正向和反向中是一样的。

### 4.4 Jacobian 行列式

这个变换的 Jacobian 矩阵是**三角矩阵**：

$$\frac{\partial z_i}{\partial x_j} = \begin{cases} s_i & \text{if } i = j \text{ and } b_i = 0 \\ 1 & \text{if } i = j \text{ and } b_i = 1 \\ \text{其他} & \text{if } i \neq j \end{cases}$$

三角矩阵的行列式 = 对角元素之积，所以：

$$\log|\det J| = \sum_{i: b_i=0} \log |s_i| = \sum_i \log s_i \cdot (1 - b_i)$$

**计算复杂度从 $O(D^3)$ 降到了 $O(D)$！**

### 4.5 代码实现详解

```python
class CouplingLayer(nn.Module):
    def __init__(self, num_inputs, num_hidden, mask, s_act=nn.Tanh(), t_act=nn.ReLU()):
        super(CouplingLayer, self).__init__()
        self.num_inputs = num_inputs  # 784 (28×28)
        self.mask = mask              # 二值 mask，形状 (784,)

        # scale 网络: 输入 masked_x → 输出 log_s
        # 用 Tanh 激活（限制 log_s 在 [-1, 1]，防止数值不稳定）
        self.scale_net = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),   # 784 → 512
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),   # 512 → 512
            nn.ReLU(),
            nn.Linear(num_hidden, num_inputs),   # 512 → 784
            s_act                                # Tanh: 输出范围 (-1, 1)
        )

        # translate 网络: 输入 masked_x → 输出 t
        # 没有最终激活（平移量可以是任意值）
        self.translate_net = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),   # 784 → 512
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),   # 512 → 512
            nn.ReLU(),
            nn.Linear(num_hidden, num_inputs)    # 512 → 784
        )
```

**正向传播**：

```python
    def forward(self, x, mode='direct'):
        mask = self.mask  # (784,)

        # Step 1: 提取条件部分
        masked_inputs = x * mask  # 只保留 mask=1 的像素，mask=0 的位置变成 0

        # Step 2: 用条件部分计算 s 和 t
        log_s = self.scale_net(masked_inputs) * (1 - mask)
        # 注意：× (1-mask) 确保 mask=1 位置的 log_s = 0（即 s = 1，不缩放）
        s = torch.exp(log_s)  # s > 0
        t = self.translate_net(masked_inputs)

        if mode == 'direct':
            # 正向: x → z
            z = x * s + t * (1 - mask)
            # mask=1 的位置: z_i = x_i * 1 + t_i * 0 = x_i （不变）
            # mask=0 的位置: z_i = x_i * s_i + t_i      （仿射变换）
            logdet = log_s.sum(-1, keepdim=True)  # (batch, 1)
            return z, logdet
        else:
            # 反向: z → x
            x_rec = (x - t * (1 - mask)) / s
            # mask=1 的位置: x_i = (z_i - 0) / 1 = z_i  （不变）
            # mask=0 的位置: x_i = (z_i - t_i) / s_i     （逆仿射变换）
            logdet = -log_s.sum(-1, keepdim=True)  # 反向的 logdet 是正向的负值
            return x_rec, logdet
```

### 4.6 数据流示意（以 4 维简化为例）

```
输入:       x = [0.3, 0.7, 0.5, 0.9]
mask:       b = [1,   0,   1,   0  ]

masked_x:   [0.3, 0.0, 0.5, 0.0]  ← x * mask

scale_net(masked_x) = [*, 0.2, *, -0.1]  (原始输出)
× (1-mask)          = [0, 0.2, 0, -0.1]  ← log_s

s = exp(log_s)      = [1, 1.22, 1, 0.90]

translate_net(masked_x) = [*, 0.1, *, 0.3]  (原始输出)

正向输出:
z[0] = x[0] * s[0] + t[0] * (1-b[0]) = 0.3 * 1    + * * 0 = 0.3    (不变)
z[1] = x[1] * s[1] + t[1] * (1-b[1]) = 0.7 * 1.22  + 0.1 * 1 = 0.954
z[2] = x[2] * s[2] + t[2] * (1-b[2]) = 0.5 * 1    + * * 0 = 0.5    (不变)
z[3] = x[3] * s[3] + t[3] * (1-b[3]) = 0.9 * 0.90  + 0.3 * 1 = 1.11

logdet = log_s.sum() = 0 + 0.2 + 0 + (-0.1) = 0.1
```

---

## 5. 核心模块二：BatchNormFlow（批归一化层）

### 5.1 作用

和普通神经网络中的 BatchNorm 类似，BatchNormFlow 的作用是：
- **稳定训练**：避免中间层的数值爆炸或消失
- **加速收敛**

但它和普通 BatchNorm 不同的是：**它也是一个可逆变换，也需要计算 logdet**。

### 5.2 数学公式

**正向变换**：

$$y_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2}} \cdot \exp(\gamma_i) + \beta_i$$

其中 $\mu, \sigma^2$ 是 batch 统计量（训练时）或 running 统计量（推断时），$\gamma, \beta$ 是可学习参数。

$$\log|\det J| = \sum_i \left( \gamma_i - \frac{1}{2}\log \sigma_i^2 \right)$$

**反向变换**：

$$x_i = \frac{y_i - \beta_i}{\exp(\gamma_i)} \cdot \sqrt{\sigma_i^2} + \mu_i$$

### 5.3 代码详解

```python
class BatchNormFlow(nn.Module):
    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()
        # 可学习参数
        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))  # 初始为 0 → γ=1
        self.beta = nn.Parameter(torch.zeros(num_inputs))       # 初始为 0
        self.momentum = momentum  # EMA 动量系数
        self.eps = eps

        # 运行统计量（推断时使用）
        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, mode='direct', **kwargs):
        if mode == 'direct':
            if self.training:
                # 训练时：用 batch 统计量
                self.batch_mean = inputs.mean(0)
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps
                # 更新 running 统计量（指数移动平均）
                self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))
                mean, var = self.batch_mean, self.batch_var
            else:
                # 推断时：用 running 统计量
                mean, var = self.running_mean, self.running_var

            x_hat = (inputs - mean) / var.sqrt()                        # 标准化
            y = torch.exp(self.log_gamma) * x_hat + self.beta           # 仿射变换
            logdet = (self.log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)
            return y, logdet
        else:
            # 反向：解出 x
            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)
            y = x_hat * var.sqrt() + mean
            logdet = (-self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)
            return y, logdet
```

---

## 6. 核心模块三：Shuffle（置换层）

### 6.1 为什么需要 Shuffle？

CouplingLayer 有一个局限：**mask=1 的位置永远不变**。如果不做 Shuffle，那么某些维度永远不会被变换，模型表达力受限。

Shuffle 随机打乱维度顺序，使得每次 CouplingLayer 变换不同的维度组合。

### 6.2 数学性质

置换操作的 Jacobian 矩阵是**置换矩阵**（正交矩阵），其行列式为 ±1，所以：

$$\log|\det J| = \log 1 = 0$$

### 6.3 代码详解

```python
class Shuffle(nn.Module):
    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        self.perm = np.random.permutation(num_inputs)   # 随机置换 [0, 1, ..., 783]
        self.inv_perm = np.argsort(self.perm)            # 逆置换（用于反向）

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            z = inputs[:, self.perm]       # 按 perm 重排列
            logdet = torch.zeros(batch_size, 1, device=inputs.device)  # logdet = 0
            return z, logdet
        else:
            x = inputs[:, self.inv_perm]   # 按逆 perm 恢复原顺序
            logdet = torch.zeros(batch_size, 1, device=inputs.device)
            return x, logdet
```

**示例**：
```
perm = [2, 0, 3, 1]

正向: x = [a, b, c, d] → z = [c, a, d, b]   (按 perm 取)
反向: z = [c, a, d, b] → x = [a, b, c, d]   (按 inv_perm 取)
```

---

## 7. FlowSequential：整体流水线

### 7.1 正向传播（计算似然）

```python
def forward(self, inputs, mode='direct', logdets=None, **kwargs):
    if logdets is None:
        logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)  # 初始 logdet = 0

    if mode == 'direct':
        # 预处理：logit 变换（见下一节）
        if kwargs.get("pre_process", True):
            inputs = self._pre_process(inputs)  # x → logit(x)

        # 依次通过所有层，累加 logdet
        for module in self._modules.values():
            inputs, logdet = module(inputs, mode)
            logdets += logdet
    else:
        # 反向：逆序通过所有层
        for module in reversed(self._modules.values()):
            inputs, logdet = module(inputs, mode, **kwargs)
            logdets += logdet

    return inputs, logdets
```

### 7.2 计算对数似然

```python
def log_probs(self, inputs, pre_process=True):
    u, log_jacob = self(inputs, pre_process=pre_process)
    # u 是变换后的向量（应该近似标准正态）
    # log_jacob 是所有层的 logdet 之和

    # log p(x) = log p_Z(z) + log|det J|
    # 其中 p_Z(z) = N(0, I)，所以 log p_Z(z) = sum_i log N(z_i; 0, 1)
    log_probs = self.prior.log_prob(u).sum(-1, keepdim=True) + log_jacob
    return log_probs  # (batch_size, 1)
```

### 7.3 采样

```python
@torch.no_grad()
def sample_images(self, n_samples=100, save=True, save_dir='./flow'):
    self.eval()
    # Step 1: 从标准正态分布采样 z
    z = self.prior.sample([n_samples, 28 * 28]).squeeze(-1)  # (100, 784)
    # Step 2: 反向通过所有层
    samples, _ = self.forward(z, mode='inverse')
    # Step 3: sigmoid 恢复到 [0, 1] 像素值
    imgs = torch.sigmoid(samples).view(n_samples, 1, 28, 28)
    return imgs
```

---

## 8. 数据预处理：Logit 变换

### 8.1 为什么需要？

MNIST 像素值在 $[0, 1]$，但标准正态分布的支撑集是 $(-\infty, +\infty)$。直接建模会有问题。

### 8.2 Logit 变换

$$\text{logit}(x) = \log\frac{x}{1-x}$$

- 当 $x \to 0$ 时，$\text{logit}(x) \to -\infty$
- 当 $x = 0.5$ 时，$\text{logit}(x) = 0$
- 当 $x \to 1$ 时，$\text{logit}(x) \to +\infty$

这样把 $(0, 1)$ 映射到 $(-\infty, +\infty)$，与正态分布匹配。

### 8.3 逆变换（采样时使用）

$$\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}$$

代码中采样时用 `torch.sigmoid(samples)` 把输出映射回像素值。

### 8.4 数据集的预处理

在 `utils.py` 中，数据集做了以下处理：
```python
transform = transforms.Compose([
    transforms.ToTensor(),                                          # 转为张量 [0, 1]
    transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(256.))),  # 加均匀噪声（去量化）
    transforms.Lambda(lambda x: rescale(x, 0.0001, 0.9999)),       # 缩放到 (0.0001, 0.9999)
])
```

加噪声是为了避免 logit(0) 和 logit(1) 出现 $\pm\infty$。

---

## 9. 训练过程：最大似然

### 9.1 目标函数

训练目标是**最大化数据的对数似然**：

$$\max_\theta \frac{1}{N}\sum_{i=1}^{N} \log p_\theta(\mathbf{x}_i)$$

等价于**最小化负对数似然**（代码中的 loss）：

$$\text{loss} = -\log p_\theta(\mathbf{x}) = -\left[\log p_Z(f_\theta(\mathbf{x})) + \log|\det J_\theta(\mathbf{x})|\right]$$

### 9.2 训练代码解读

```python
def train(n_epochs, flow_model, train_loader, val_loader, optimizer, ...):
    for epoch in range(n_epochs):
        for i, (x, _) in enumerate(train_loader):
            flow_model.train()
            x = x.to(device)

            # 计算负对数似然作为 loss
            loss = -flow_model.log_probs(x.reshape(x.shape[0], -1))
            # loss 形状: (batch_size, 1)

            optimizer.zero_grad()
            loss.mean().backward()  # 对 batch 取平均后反向传播
            optimizer.step()
```

### 9.3 训练流程图

```
输入图片 x (batch_size, 1, 28, 28)
    │
    ▼ reshape
x (batch_size, 784)
    │
    ▼ _pre_process (logit变换)
logit(x) (batch_size, 784)
    │
    ▼ 24 层正向变换
z (batch_size, 784) + 累积 logdet
    │
    ▼ 计算 log p(x) = log N(z; 0,I) + logdet
log_probs (batch_size, 1)
    │
    ▼ loss = -log_probs.mean()
标量 loss
    │
    ▼ loss.backward() + optimizer.step()
更新参数
```

---

## 10. 采样过程：反向生成

```
从标准正态分布采样 z ~ N(0, I)
z (n_samples, 784)
    │
    ▼ 24 层反向变换（逆序）
    │  Shuffle⁻¹ → BatchNorm⁻¹ → CouplingLayer⁻¹
    │  (重复 8 次)
    │
logit_x (n_samples, 784)
    │
    ▼ sigmoid
x (n_samples, 784) ∈ (0, 1)
    │
    ▼ reshape
图片 (n_samples, 1, 28, 28)
```

---

## 11. Inpainting（图像修复）

### 11.1 问题定义

给定一张部分被遮挡的图片（下半部分被噪声替换），利用流模型恢复被遮挡的部分。

### 11.2 方法：朗之万动力学（Langevin Dynamics）

核心思想：**沿着概率密度增大的方向迭代更新被遮挡的像素**。

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \alpha \cdot \nabla_\mathbf{x} \log p(\mathbf{x}_t)$$

其中 $\alpha$ 是步长。这个梯度 $\nabla_\mathbf{x} \log p(\mathbf{x})$ 叫做 **score function**。

对于被 mask 遮挡的部分，用梯度更新；对于可见的部分，保持原值不变。

### 11.3 代码详解

```python
def inpainting(model, inputs, mask, device, ...):
    # mask: 1 = 可见像素, 0 = 被遮挡像素

    # Step 1: 预处理，转到 logit 空间
    inputs = inputs.log() - (1. - inputs).log()  # logit 变换

    for i in range(1000):  # 迭代 1000 步
        alpha = 0.2  # 步长

        # Step 2: 计算 score（对数概率关于输入的梯度）
        inputs.requires_grad_()
        log_probs = model.log_probs(inputs, pre_process=False)
        dx = torch.autograd.grad([log_probs.sum()], [inputs])[0]
        # dx = ∇_x log p(x)，即 score function
        dx = torch.clip(dx, -10, 10)  # 梯度裁剪，防止数值爆炸

        with torch.no_grad():
            # Step 3: 更新
            # mask=1（可见）的位置：保持原值
            # mask=0（遮挡）的位置：沿梯度方向更新
            inputs = inputs * mask + (1 - mask) * (inputs + alpha * dx).clip(-10, 10)

    # Step 4: sigmoid 恢复像素值
    imgs = torch.sigmoid(inputs.view(num_samples, 1, 28, 28))
    return imgs
```

### 11.4 直觉理解

想象流模型学到了"手写数字长什么样"的概率分布。当图片下半部分缺失时：

1. 一开始，下半部分是随机噪声，整张图的概率很低
2. 计算梯度：模型告诉我们"往哪个方向修改能让图片更像真实数字"
3. 沿梯度方向小步更新
4. 重复多次后，下半部分逐渐变成与上半部分匹配的内容

```
迭代 0:    ┌──────────┐     迭代 500:  ┌──────────┐     迭代 1000: ┌──────────┐
           │  数字上半  │               │  数字上半  │               │  数字上半  │
           │  (清晰)   │               │  (清晰)   │               │  (清晰)   │
           ├──────────┤               ├──────────┤               ├──────────┤
           │  随机噪声  │               │  模糊轮廓  │               │  清晰下半  │
           │  ██████   │               │  ▓▓▓▓▓▓   │               │  (恢复)   │
           └──────────┘               └──────────┘               └──────────┘
```

### 11.5 corruption 函数

在 `utils.py` 中定义了遮挡方式：

```python
def corruption(x, type_='flow', noise_scale=0.3):
    mask = torch.zeros_like(x)
    if type_ == 'flow':
        mask[..., :mask.shape[-2] // 2, :] = 1  # 上半部分可见（mask=1），下半部分遮挡
    broken_data = x * mask + (1 - mask) * noise_scale * torch.randn_like(x)
    broken_data = torch.clip(broken_data, 1e-4, 1 - 1e-4)
    return broken_data, mask
```

---

## 12. 完整数据流图解

### 12.1 训练时

```
 MNIST 图片 (batch, 1, 28, 28)
       │
       ▼ flatten
 (batch, 784)  像素值 ∈ (0.0001, 0.9999)
       │
       ▼ logit 变换: log(x/(1-x))
 (batch, 784)  实数值 ∈ (-∞, +∞)
       │
       ▼ ═══════════════════════════════════════
       │  CouplingLayer_1 (mask: 棋盘格 [1,0,1,0,...])
       │    ├─ masked_x = x * mask
       │    ├─ log_s = scale_net(masked_x) * (1-mask)
       │    ├─ t = translate_net(masked_x)
       │    ├─ z = x * exp(log_s) + t * (1-mask)
       │    └─ logdet₁ = sum(log_s)
       │
       │  BatchNormFlow_1
       │    ├─ 标准化 + 仿射
       │    └─ logdet₂
       │
       │  Shuffle_1
       │    ├─ 随机置换维度
       │    └─ logdet₃ = 0
       │  ─────────────────────────────────────
       │  (重复 8 次, 每次使用不同的 mask)
       │  ═══════════════════════════════════════
       ▼
 z (batch, 784)  ← 应该接近 N(0, I)
       │
       ▼ 计算 log p(z) = Σᵢ log N(zᵢ; 0, 1) = -½Σᵢ(zᵢ² + log(2π))
       │
       ▼ log p(x) = log p(z) + Σlogdet
       │
       ▼ loss = -log p(x).mean()
       │
       ▼ 反向传播, 更新参数
```

### 12.2 采样时

```
 z ~ N(0, I)  (n_samples, 784)
       │
       ▼ ═══════════════════════════════════════
       │  逆序通过所有层:
       │
       │  Shuffle_8⁻¹ (逆置换)
       │  BatchNormFlow_8⁻¹ (逆归一化)
       │  CouplingLayer_8⁻¹ (逆仿射耦合)
       │  ...
       │  Shuffle_1⁻¹
       │  BatchNormFlow_1⁻¹
       │  CouplingLayer_1⁻¹
       │  ═══════════════════════════════════════
       ▼
 logit_x (n_samples, 784)
       │
       ▼ sigmoid: 1/(1+exp(-logit_x))
       │
 图片 (n_samples, 1, 28, 28)  像素值 ∈ (0, 1)
```

---

## 13. 超参数与模型配置

### 13.1 模型超参数

| 超参数 | 值 | 含义 |
|--------|-----|------|
| `num_inputs` | 784 | 输入维度 (28×28) |
| `num_hidden` | 512 | 隐藏层维度 |
| `num_blocks` | 8 | CouplingLayer 的数量 |
| `momentum` | 0.1 | BatchNorm EMA 动量 |

### 13.2 Mask 设计

```python
# mask 类型 1: 棋盘格
mask = torch.arange(0, 784) % 2        # [0,1,0,1,...] 和 [1,0,1,0,...]

# mask 类型 2: 前后半
mask2 = torch.zeros(784)
mask2[:392] = 1                          # [1,1,...,1,0,0,...,0] 和 [0,0,...,0,1,1,...,1]

# 4 种 mask 交替使用
masks = [mask, 1-mask, mask2, 1-mask2]
# Block i 使用 masks[i % 4]
```

### 13.3 训练超参数

| 超参数 | 值 | 含义 |
|--------|-----|------|
| `batch_size` | 128 | 每批样本数 |
| `lr` | 1e-4 | 学习率 |
| `weight_decay` | 1e-6 | 权重衰减 |
| `n_epochs` | 100 | 训练轮数 |
| `optimizer` | Adam | 优化器 |
| `save_interval` | 10 | 每 10 轮评估一次 |

### 13.4 Inpainting 超参数

| 超参数 | 值 | 含义 |
|--------|-----|------|
| `alpha` | 0.2 | 朗之万动力学步长 |
| `迭代次数` | 1000 | 梯度更新步数 |
| `梯度裁剪` | [-10, 10] | 防止数值爆炸 |

---

## 附录：流模型 vs 其他生成模型

| 特性 | 流模型 | VAE | GAN | 扩散模型 |
|------|--------|-----|-----|----------|
| 精确似然 | ✅ | ❌ (ELBO) | ❌ | ❌ |
| 采样速度 | ✅ 快（一次前向） | ✅ 快 | ✅ 快 | ❌ 慢 |
| 训练稳定性 | ✅ | ✅ | ❌ | ✅ |
| 可逆性 | ✅ | ❌ | ❌ | ✅ |
| 表达力限制 | ❌ 架构受限 | ❌ 后验近似 | ✅ 强 | ✅ 强 |
