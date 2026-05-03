# Assignment 7 学习笔记：DDPM 与扩散模型

---

## Problem 1：扩散模型能生成文本吗？（True/False）

**答案：True**

### 背景

标准扩散模型是为**连续数据**（如图像像素）设计的，但**文本是离散的**（由具体的词/token 组成），不能直接加高斯噪声。

### 解决方案：在 Embedding 空间加噪

以 **Diffusion-LM**（斯坦福）为代表的方法：

1. **映射到连续空间**：通过 Embedding 矩阵将离散 token 映射为连续向量
2. **前向扩散（加噪）**：在连续的 Embedding 向量上逐步加高斯噪声
3. **反向去噪（生成）**：从纯噪声开始，一步步去噪
4. **条件重采样 / Rounding**：将生成的连续向量映射回离散词汇表中最近的词

所以题目描述的"向词嵌入添加噪声 → 条件重采样生成新序列"是正确的。

---

## DDPM（Denoising Diffusion Probabilistic Models）详解

### 一、核心直觉：先破坏，再修复

1. **前向过程**：不断往数据上加高斯噪声，直到变成纯噪声
2. **反向过程**：训练神经网络学会去噪，从纯噪声中恢复出新数据

---

### 二、前向过程（Forward / Diffusion Process）

#### 定义

给定真实数据 $x_0$，定义 $T$ 步马尔可夫链：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\; \sqrt{\alpha_t}\, x_{t-1},\; (1-\alpha_t) I)$$

- $\alpha_t \in (0,1)$：预定义的噪声调度参数
- 每一步把上一步结果"缩小"（乘 $\sqrt{\alpha_t}$），再加新噪声

#### 重参数化技巧：一步到位（Problem 2.1）

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$。

**意义**：训练时不需要一步步跑马尔可夫链，直接对任意 $t$ 采样即可。

#### 深入理解：两个高斯噪声如何合并为一个？

这是 Problem 2.1 证明中最关键的数学技巧。下面一步步拆解。

##### 出发点：连续两步的展开

前向过程每一步是：

$$x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\, \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, I)$$

把 $x_{t-1}$ 也用它自己的公式展开：

$$x_{t-1} = \sqrt{\alpha_{t-1}}\, x_{t-2} + \sqrt{1-\alpha_{t-1}}\, \epsilon_{t-2}, \quad \epsilon_{t-2} \sim \mathcal{N}(0, I)$$

代入第一个式子：

$$x_t = \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}}\, x_{t-2} + \sqrt{1-\alpha_{t-1}}\, \epsilon_{t-2} \right) + \sqrt{1-\alpha_t}\, \epsilon_{t-1}$$

$$= \sqrt{\alpha_t \alpha_{t-1}}\, x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\, \epsilon_{t-2} + \sqrt{1-\alpha_t}\, \epsilon_{t-1}}_{\text{两个独立高斯噪声之和，如何合并？}}$$

##### 核心问题：为什么两个噪声可以合并成一个？

现在我们有两个**独立的**标准高斯随机变量 $\epsilon_{t-2} \sim \mathcal{N}(0, I)$ 和 $\epsilon_{t-1} \sim \mathcal{N}(0, I)$，以及它们的线性组合：

$$W = a \cdot \epsilon_{t-2} + b \cdot \epsilon_{t-1}, \quad a = \sqrt{\alpha_t(1-\alpha_{t-1})}, \; b = \sqrt{1-\alpha_t}$$

根据高斯分布的一个基本性质：

> **独立高斯变量的线性组合仍然是高斯变量。** 如果 $X \sim \mathcal{N}(0, \sigma_1^2 I)$，$Y \sim \mathcal{N}(0, \sigma_2^2 I)$，且 $X, Y$ 独立，则：
> $$aX + bY \sim \mathcal{N}(0,\; (a^2 \sigma_1^2 + b^2 \sigma_2^2) I)$$

由于 $\epsilon_{t-2}, \epsilon_{t-1}$ 都是标准正态（方差为 $I$），我们只需要计算合并后的方差：

$$\text{Var}(W) = a^2 \cdot I + b^2 \cdot I = \left( \alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) \right) I$$

展开：

$$= \left( \alpha_t - \alpha_t \alpha_{t-1} + 1 - \alpha_t \right) I = (1 - \alpha_t \alpha_{t-1}) I$$

所以：

$$W \sim \mathcal{N}(0,\; (1 - \alpha_t \alpha_{t-1}) I) = \sqrt{1 - \alpha_t \alpha_{t-1}} \cdot \bar{\epsilon}, \quad \bar{\epsilon} \sim \mathcal{N}(0, I)$$

##### 合并结果

$$x_t = \sqrt{\alpha_t \alpha_{t-1}}\, x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}}\, \bar{\epsilon}$$

**中间变量 $x_{t-1}$ 被完全消除了！** 我们直接用 $x_{t-2}$ 表达了 $x_t$。

##### 为什么可以继续递推？

注意到上式的结构和原来完全一样——"$\sqrt{\text{某个系数}} \cdot x_{\text{更早}} + \sqrt{1 - \text{那个系数}} \cdot \text{新噪声}$"。所以我们可以继续代入 $x_{t-2}$ 的表达式，合并噪声，消掉 $x_{t-2}$，得到用 $x_{t-3}$ 表达的公式……一直递推到 $x_0$：

$$x_t = \sqrt{\alpha_t \alpha_{t-1} \cdots \alpha_1}\, x_0 + \sqrt{1 - \alpha_t \alpha_{t-1} \cdots \alpha_1}\, \epsilon = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$$

每一步的合并都用到同一个性质：**独立高斯的线性组合仍是高斯，方差直接相加**。

##### 直觉类比

可以把这个过程想象成**调混合颜料**：

- 你有一杯纯色颜料（$x_0$，原始数据）
- 每一步往里倒一点随机的灰色颜料（高斯噪声），搅拌均匀
- 问：经过 $t$ 步之后，杯子里颜料的"随机灰度"是多少？

你不需要真的倒 $t$ 次——因为不管你分多少次倒，最终杯子里"纯色"和"灰色"的比例只取决于**总共倒了多少灰色颜料**。这就是为什么 $t$ 步噪声可以合并成一步。

> **关键前提**：每步加的噪声必须是**独立的高斯噪声**。如果噪声之间有相关性，方差就不能简单相加，这个合并技巧就不成立了。

#### 噪声调度的效果

| 时刻 | $\bar{\alpha}_t$ | 效果 |
|------|-------------------|------|
| $t = 0$ | $\approx 1$ | 原始数据，几乎无噪声 |
| $t$ 较小 | 接近 1 | 数据略有模糊 |
| $t$ 较大 | 接近 0 | 数据几乎全是噪声 |
| $t = T$ | $\approx 0$ | 纯高斯噪声 |

#### 噪声调度 $\alpha_t$ 是怎么设定的？

在 DDPM 中，我们通常不直接定义 $\alpha_t$，而是先定义 $\beta_t$（每一步的噪声强度），然后令 $\alpha_t = 1 - \beta_t$。

##### 1. $\beta_t$ 和 $\alpha_t$ 的关系

$$\beta_t = 1 - \alpha_t, \quad \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i = \prod_{i=1}^{t} (1 - \beta_i)$$

- $\beta_t$ 越大 → 这一步加的噪声越多 → $\alpha_t$ 越小
- $\bar{\alpha}_t$ 是累乘，随 $t$ 单调递减，从接近 1 下降到接近 0

##### 2. 线性调度（Linear Schedule）—— DDPM 原论文

Ho et al. 2020 使用最简单的做法：让 $\beta_t$ 从一个很小的值**线性增长**到一个稍大的值：

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

**原论文的具体数值**：
- $\beta_1 = 10^{-4}$（第一步几乎不加噪声）
- $\beta_T = 0.02$（最后一步也只加很少噪声）
- $T = 1000$（总步数）

对应的 $\alpha_t$：
- $\alpha_1 = 1 - 10^{-4} = 0.9999$
- $\alpha_T = 1 - 0.02 = 0.98$

看起来每一步的 $\alpha_t$ 都非常接近 1，但是 $\bar{\alpha}_t$ 是连乘 1000 次，所以整体效果是：

$$\bar{\alpha}_T \approx \prod_{t=1}^{1000} (1 - \beta_t) \approx e^{-\sum \beta_t} \approx e^{-10} \approx 0.0000454$$

到最后一步，原始信号几乎完全消失了。

##### 3. 余弦调度（Cosine Schedule）—— Improved DDPM

Nichol & Dhariwal 2021 发现线性调度有个问题：在扩散过程的**中间阶段**，$\bar{\alpha}_t$ 下降太快，导致信息损失过早。他们提出了**余弦调度**：

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

其中 $s = 0.008$ 是一个小偏移量，防止 $\beta_t$ 在 $t=0$ 附近太小。

**直觉**：余弦函数在中间区域变化平缓，使得 $\bar{\alpha}_t$ 的下降更均匀，信息不会过早丢失。

##### 4. 对比

| 调度方式 | $\bar{\alpha}_t$ 下降曲线 | 特点 |
|----------|---------------------------|------|
| 线性 | 中间下降快，两端平坦 | 简单但信息损失不均匀 |
| 余弦 | 均匀下降，呈 S 形 | 信息保留更均匀，生成质量更好 |

##### 5. 设计原则

不管用什么调度方式，核心原则是：

1. **起始时** $\bar{\alpha}_1 \approx 1$：第一步几乎不破坏数据
2. **终止时** $\bar{\alpha}_T \approx 0$：最后一步数据变成纯噪声
3. **中间过渡要平滑**：信噪比应该渐进式下降，不能太突然
4. **每步噪声要足够小**：保证反向过程的高斯近似成立（$\beta_t$ 通常在 $10^{-4}$ 到 $0.02$ 之间）
5. **步数 $T$ 要足够大**：通常取 $T = 1000$，确保每步变化微小

##### 6. Python 示例

```python
import numpy as np

T = 1000

# 线性调度
beta_min, beta_max = 1e-4, 0.02
betas_linear = np.linspace(beta_min, beta_max, T)
alphas_linear = 1.0 - betas_linear
alpha_bar_linear = np.cumprod(alphas_linear)
# alpha_bar_linear[0] ≈ 0.9999, alpha_bar_linear[-1] ≈ 0.0000454

# 余弦调度
s = 0.008
steps = np.arange(T + 1) / T
f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cosine = f[1:] / f[0]
betas_cosine = 1 - alpha_bar_cosine[1:] / alpha_bar_cosine[:-1]
betas_cosine = np.clip(betas_cosine, 0, 0.999)
```

---

### 三、反向过程（Reverse Process）

从 $x_T$（纯噪声）一步步走回 $x_0$：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1};\; \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t))$$

$\mu_\theta$ 和 $\Sigma_\theta$ 由神经网络（通常是 U-Net）参数化。

**数学保证**：如果每步加的噪声足够小，反向过程每步也近似高斯分布。

---

### 四、训练目标：变分下界 VLB（Problem 2.3）

通过 KL 散度非负性推导：

$$-\log p_\theta(x_0) \leq \underbrace{D_{\text{KL}}(q(x_T|x_0) \| p_\theta(x_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} + \underbrace{(-\log p_\theta(x_0|x_1))}_{L_0}$$

| 项 | 含义 |
|----|------|
| $L_T$ | 前向终点 vs 先验的差距（基本是常数） |
| $L_{t-1}$ | **核心训练损失**：让模型逼近真实后验 |
| $L_0$ | 重建项 |

#### 深入理解：对 $q(x_{0:T})$ 取期望为什么就变成了 KL 散度？

这是 Problem 2.3 推导中最容易让人困惑的一步。让我们彻底搞清楚。

##### 问题回顾

我们已经把 log 比值拆成了三部分：

$$\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})} = \underbrace{\log \frac{q(x_T|x_0)}{p_\theta(x_T)}}_{\text{(A)}} + \sum_{t=2}^{T} \underbrace{\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)}}_{\text{(B}_t\text{)}} + \underbrace{(-\log p_\theta(x_0|x_1))}_{\text{(C)}}$$

现在要对 $q(x_{0:T}) = q(x_0, x_1, x_2, \ldots, x_T)$ 取期望。这个联合分布涉及 $T+1$ 个变量，但每一项只涉及其中少数几个。困惑就在这里：

> **那些不出现在表达式中的变量怎么办？出现在条件 $|$ 左边的和右边的变量，处理方式一样吗？**

##### 关键原理：边缘化（Marginalization）

对联合分布 $q(x_{0:T})$ 取期望，就是对所有变量做积分：

$$\mathbb{E}_{q(x_{0:T})}[f(\cdot)] = \int \cdots \int q(x_0, x_1, \ldots, x_T) \cdot f(\cdot) \, dx_0 \, dx_1 \cdots dx_T$$

如果 $f$ 只依赖于其中部分变量（比如 $x_0, x_T$），那些不出现的变量可以**先积分掉**，因为：

$$\int q(x_0, x_1, \ldots, x_T) \, dx_1 \cdots dx_{T-1} = q(x_0, x_T)$$

这就是概率论中的**边缘化**：对不相关的变量积分，联合分布退化为边缘分布。积分结果恒为 1 的部分自动消失。

##### 以 (A) 项为例：完整推导

(A) 项是 $\log \frac{q(x_T|x_0)}{p_\theta(x_T)}$，只涉及 $x_0$ 和 $x_T$。

$$\mathbb{E}_{q(x_{0:T})} \left[ \log \frac{q(x_T|x_0)}{p_\theta(x_T)} \right]$$

**第 1 步：展开联合分布，积掉不相关的变量**

$$= \underbrace{\int \cdots \int}_{\text{对 } x_0 \sim x_T} q(x_0, x_1, \ldots, x_T) \cdot \log \frac{q(x_T|x_0)}{p_\theta(x_T)} \, dx_0 \cdots dx_T$$

由于 $\log$ 里面只有 $x_0$ 和 $x_T$，我们可以先对 $x_1, x_2, \ldots, x_{T-1}$ 积分：

$$= \int \int \underbrace{\left( \int \cdots \int q(x_0, x_1, \ldots, x_T) \, dx_1 \cdots dx_{T-1} \right)}_{= q(x_0, x_T)} \cdot \log \frac{q(x_T|x_0)}{p_\theta(x_T)} \, dx_0 \, dx_T$$

$$= \int \int q(x_0, x_T) \cdot \log \frac{q(x_T|x_0)}{p_\theta(x_T)} \, dx_0 \, dx_T$$

**第 2 步：拆分联合分布 $q(x_0, x_T) = q(x_0) \cdot q(x_T | x_0)$**

$$= \int q(x_0) \left[ \int q(x_T|x_0) \cdot \log \frac{q(x_T|x_0)}{p_\theta(x_T)} \, dx_T \right] dx_0$$

**第 3 步：认出内层积分就是 KL 散度的定义**

$$= \int q(x_0) \cdot D_{\text{KL}}\big(q(x_T|x_0) \,\|\, p_\theta(x_T)\big) \, dx_0$$

$$= \mathbb{E}_{q(x_0)} \left[ D_{\text{KL}}\big(q(x_T|x_0) \,\|\, p_\theta(x_T)\big) \right] = L_T$$

> 回忆 KL 散度的定义：$D_{\text{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx$。内层积分恰好是这个形式。

##### 以 (B$_t$) 项为例：条件变量更多的情况

(B$_t$) 项是 $\log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)}$，涉及三个变量 $x_0, x_{t-1}, x_t$。

$$\mathbb{E}_{q(x_{0:T})} \left[ \log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \right]$$

**先积掉不出现的变量**（除 $x_0, x_{t-1}, x_t$ 以外的所有变量）：

$$= \int \int \int q(x_0, x_{t-1}, x_t) \cdot \log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \, dx_0 \, dx_{t-1} \, dx_t$$

**拆分联合分布** $q(x_0, x_{t-1}, x_t) = q(x_0, x_t) \cdot q(x_{t-1} | x_t, x_0)$：

$$= \int \int q(x_0, x_t) \left[ \int q(x_{t-1}|x_t, x_0) \cdot \log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} \, dx_{t-1} \right] dx_0 \, dx_t$$

**认出内层积分 = KL 散度**（对 $x_{t-1}$ 积分，$x_0, x_t$ 是固定的条件）：

$$= \mathbb{E}_{q(x_0, x_t)} \left[ D_{\text{KL}}\big(q(x_{t-1}|x_t, x_0) \,\|\, p_\theta(x_{t-1}|x_t)\big) \right] = L_{t-1}$$

##### 总结规律

对于任何形如 $\log \frac{q(\text{某些变量}|\text{条件变量})}{p_\theta(\text{某些变量}|\text{条件变量})}$ 的项，在 $\mathbb{E}_{q(x_{0:T})}$ 下的处理方式都一样：

| 变量类型 | 处理方式 |
|----------|----------|
| **完全不出现在该项中的变量**（如 $x_1, \ldots, x_{T-1}$） | 直接积分掉，$\int q(\cdot) d\cdot = 1$，自动消失 |
| **出现在条件 $\|$ 右边的变量**（如 $x_0, x_t$） | 保留为外层期望 $\mathbb{E}_{q(x_0, x_t)}[\cdot]$ |
| **出现在条件 $\|$ 左边的变量**（如 $x_{t-1}$） | 作为 KL 散度内部的积分变量 |

这就是为什么不管 $q(x_{0:T})$ 有多少个变量，最终每一项都能精确地变成对应的 KL 散度——本质上就是**边缘化 + 条件分解 + KL 散度定义**这三步。

##### 追问：题干中写一个总的 $\mathbb{E}_q$ 是不是没写清楚？

**不是。两种写法都是完全正确的，而且数学上等价。** 下面解释为什么。

题干的写法是把所有项放在一个大期望下：

$$\mathbb{E}_{q(x_{0:T})} \left[ \log \frac{q(x_T|x_0)}{p_\theta(x_T)} + \sum_{t=2}^T \log \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)} - \log p_\theta(x_0|x_1) \right]$$

而我们拆开后，每一项的外层期望确实不一样：

$$\underbrace{\mathbb{E}_{q(x_0)}}_{L_T \text{ 的外层}} \left[ D_{\text{KL}}(\cdots) \right] + \sum_{t=2}^{T} \underbrace{\mathbb{E}_{q(x_0, x_t)}}_{L_{t-1} \text{ 的外层}} \left[ D_{\text{KL}}(\cdots) \right] + \underbrace{\mathbb{E}_{q(x_0, x_1)}}_{L_0 \text{ 的外层}} \left[ -\log p_\theta(x_0|x_1) \right]$$

**这两种写法为什么等价？** 靠的是两条性质：

**性质 1：期望的线性性**

$$\mathbb{E}[A + B + C] = \mathbb{E}[A] + \mathbb{E}[B] + \mathbb{E}[C]$$

所以一个大期望下的求和可以拆成每项单独取期望。

**性质 2：对不出现的变量积分 = 边缘化 = 自动降维**

$$\mathbb{E}_{q(x_{0:T})}[f(x_0, x_t)] = \mathbb{E}_{q(x_0, x_t)}[f(x_0, x_t)]$$

这是因为：

$$\int \cdots \int q(x_{0:T}) \cdot f(x_0, x_t) \, dx_0 \cdots dx_T = \int \int \underbrace{\left(\int \cdots \int q(x_{0:T}) \prod_{j \neq 0,t} dx_j \right)}_{= q(x_0, x_t)} f(x_0, x_t) \, dx_0 \, dx_t$$

也就是说，**用 $q(x_{0:T})$ 对只含 $(x_0, x_t)$ 的函数取期望，和用 $q(x_0, x_t)$ 取期望的结果完全一样**。多余的变量会自动被积掉。

**所以结论是：**

| 写法 | 含义 | 是否正确 |
|------|------|----------|
| 一个大 $\mathbb{E}_{q(x_{0:T})}[\text{整个求和}]$ | 对全部变量的联合分布取期望 | ✅ 完全正确 |
| 每项单独写 $\mathbb{E}_{q(\text{相关变量})}[D_{\text{KL}}(\cdots)]$ | 只对该项涉及的变量取期望 | ✅ 也完全正确 |

两者等价。题干用一个总的 $\mathbb{E}_q$ 是标准且合法的简写——因为数学上它会**自动退化**为每一项所需要的更小的期望。论文和教科书中通常偏爱这种简洁写法，而把"边缘化自动发生"视为读者应当理解的隐含步骤。

---

### 五、模型到底输入输出什么？

#### 理论上

模型应输出反向过程的均值 $\mu_\theta(x_t, t)$ 和方差 $\Sigma_\theta(x_t, t)$。

#### 实践上（DDPM 原论文）

**简化 1：方差固定，不学习**

$$\Sigma_\theta = \sigma_t^2 I, \quad \sigma_t^2 = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$$

方差是写死的超参数，不需要网络预测。

**简化 2：网络预测噪声 $\epsilon$，间接得到均值**

真实后验均值（Problem 2.2 推导结果）：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)$$

模型参数化的均值：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

#### 实际的网络结构

```
输入：x_t (加噪后的图片), t (时间步)
  │
  ▼
┌──────────────┐
│    U-Net     │
│ ε_θ(x_t, t) │
└──────────────┘
  │
  ▼
输出：ε̂ (预测的噪声，和 x_t 形状完全一样)
```

- **输入**：$(x_t, t)$ — 加噪图片 + 时间步
- **输出**：$\hat{\epsilon}$ — 预测的噪声
- **损失**：$\|\epsilon - \hat{\epsilon}\|^2$（MSE）
- **均值 / 方差**：通过公式事后计算

---

### 六、为什么预测噪声比直接预测均值效果好？

Ho et al. 做了实验对比，发现预测噪声 $\epsilon$ 比直接预测均值 $\mu$ 效果更好。原因有以下几点：

**1. 预测目标的尺度一致性**

- 如果预测均值 $\mu$：不同时间步 $t$ 的均值尺度差异很大（$t$ 小时接近原图，$t$ 大时接近零），网络需要在不同尺度之间切换
- 如果预测噪声 $\epsilon$：**噪声始终是标准正态分布 $\mathcal{N}(0, I)$**，无论 $t$ 是多少，预测目标的统计特性不变，网络学起来更稳定

**2. 与去噪分数匹配的等价性**

预测噪声 $\epsilon$ 本质上等价于估计数据分布的**分数函数（Score Function）**：

$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

所以 $\epsilon_\theta(x_t, t)$ 实际上是在估计 $-\sqrt{1-\bar{\alpha}_t} \cdot \nabla_{x_t} \log q_t(x_t)$，这正好和 Score-Based 生成模型联系了起来（也就是 Problem 5 要讨论的内容）。

**3. 简化后的损失函数**

最终的训练损失（Problem 2.4 推导结果）：

$$L_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,\; t) \|^2 \right]$$

一句话：**随机选时间步 $t$，算出 $x_t$，让网络预测噪声，和真实噪声算 MSE。**

---

### 七、采样（生成）过程

```
1. 从标准高斯分布采样 x_T ~ N(0, I)
2. for t = T, T-1, ..., 1:
      用网络预测噪声：ε̂ = ε_θ(x_t, t)
      计算均值：μ = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) · ε̂)
      采样：x_{t-1} = μ + σ_t · z,  z ~ N(0,I)  (t>1 时)
3. 输出 x_0
```

---

### 八、总结

```
训练阶段：
  真实图片 x₀ ──加噪──> 噪声图 xₜ ──网络──> 预测噪声 ε̂
                                              ↕ 比较 (MSE)
                          真实噪声 ε ─────────────┘

生成阶段：
  纯噪声 x_T ──去噪──> x_{T-1} ──去噪──> ... ──去噪──> x₀ (新图片)
```

| 概念 | 作用 |
|------|------|
| 前向过程 $q$ | 加噪声，不需要学习 |
| 反向过程 $p_\theta$ | 去噪声，需要训练 |
| $\epsilon_\theta$ | U-Net 网络，预测噪声 |
| $\bar{\alpha}_t$ | 噪声调度，控制信噪比 |
| 重参数化 | 可以一步从 $x_0$ 跳到任意 $x_t$ |

---

---

## Problem 3：Fisher Divergence 与 Score Matching

### 1. Fisher Divergence 想比较什么？

Fisher Divergence 不直接比较两个概率密度 $p_{\mathrm{data}}(x)$ 和 $p_\theta(x)$，而是比较它们的 **score function**：

$$
\nabla_x \log p_{\mathrm{data}}(x), \qquad \nabla_x \log p_\theta(x)
$$

目标是：

$$
F(p_{\mathrm{data}}\|p_\theta)
=
\frac{1}{2}
\mathbb{E}_{x\sim p_{\mathrm{data}}}
\left[
\left\|
\nabla_x \log p_{\mathrm{data}}(x)
-
\nabla_x \log p_\theta(x)
\right\|_2^2
\right]
$$

score 可以理解为“当前点附近概率密度上升最快的方向”。因为：

$$
\nabla_x \log p(x)=\frac{1}{p(x)}\nabla_x p(x)
$$

只要 $p(x)>0$，它和 $\nabla_x p(x)$ 的方向相同，只是长度被 $1/p(x)$ 缩放。因此 score 确实指向概率密度上升的方向。

### 2. 为什么不用普通梯度 $\nabla_x p(x)$？

使用 $\nabla_x \log p(x)$ 有两个好处：

1. $\log$ 更数值稳定，概率密度很小时不会直接处理极小数。
2. 如果模型是未归一化密度，例如

$$
p_\theta(x)=\frac{1}{Z_\theta}\exp(-E_\theta(x))
$$

那么：

$$
\nabla_x \log p_\theta(x)
=
-\nabla_x E_\theta(x)
$$

归一化常数 $Z_\theta$ 会消失，这对能量模型和 score matching 很重要。

### 3. Problem 3 的推导目标

Fisher Divergence 里面有一项：

$$
\nabla_x \log p_{\mathrm{data}}(x)
$$

但真实数据分布 $p_{\mathrm{data}}$ 不知道，所以这个 score 也不知道。Problem 3 的目标就是通过分部积分，把真实数据 score 消掉。

展开平方：

$$
\frac{1}{2}\|s_{\mathrm{data}}-s_\theta\|^2
=
\frac{1}{2}\|s_{\mathrm{data}}\|^2
+
\frac{1}{2}\|s_\theta\|^2
-
s_{\mathrm{data}}^T s_\theta
$$

其中：

$$
s_{\mathrm{data}}=\nabla_x\log p_{\mathrm{data}}(x),\qquad
s_\theta=\nabla_x\log p_\theta(x)
$$

第一项不依赖 $\theta$，可以并入常数项。关键是交叉项：

$$
-
\mathbb{E}_{p_{\mathrm{data}}}
\left[
\nabla_x\log p_{\mathrm{data}}(x)^T
\nabla_x\log p_\theta(x)
\right]
$$

利用：

$$
\nabla_x\log p_{\mathrm{data}}(x)
=
\frac{\nabla_x p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)}
$$

得到：

$$
-
\int
\nabla_x p_{\mathrm{data}}(x)^T
\nabla_x\log p_\theta(x)
dx
$$

对每个维度做分部积分，并假设边界项消失：

$$
-
\int
\nabla_x p_{\mathrm{data}}(x)^T
\nabla_x\log p_\theta(x)
dx
=
\mathbb{E}_{p_{\mathrm{data}}}
\left[
\mathrm{tr}\left(\nabla_x^2\log p_\theta(x)\right)
\right]
$$

最后得到：

$$
F(p_{\mathrm{data}}\|p_\theta)
=
\mathbb{E}_{p_{\mathrm{data}}}
\left[
\frac{1}{2}\|\nabla_x\log p_\theta(x)\|_2^2
+
\mathrm{tr}\left(\nabla_x^2\log p_\theta(x)\right)
\right]
+ Const.
$$

### 4. 这一步的实际意义

Problem 3 说明：我们可以不显式知道真实数据 score，也能训练模型 score。

但它还有实际困难：目标中有 Hessian trace：

$$
\mathrm{tr}\left(\nabla_x^2\log p_\theta(x)\right)
$$

这对神经网络训练很麻烦，而且真实数据分布常常集中在低维流形附近，直接学习原始分布的 score 也可能不稳定。因此 Problem 4 会引入 Denoising Score Matching。

---

## Problem 4：Denoising Score Matching 是怎么来的？

### 1. 从原始数据 score 到带噪数据 score

Problem 4 的思想是：不直接学 $p_{\mathrm{data}}(x)$ 的 score，而是先给样本加高斯噪声：

$$
\tilde{x}=x+\sigma\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
$$

于是得到带噪分布：

$$
q_\sigma(\tilde{x})
=
\int p_{\mathrm{data}}(x)q_\sigma(\tilde{x}|x)dx
$$

现在学习：

$$
s_\theta(\tilde{x},\sigma)
\approx
\nabla_{\tilde{x}}\log q_\sigma(\tilde{x})
$$

这仍然是 score matching，只是目标分布从原始数据分布变成了带噪数据分布。

### 2. 为什么目标里出现这个交叉项？

如果对带噪分布 $q_\sigma$ 做 Fisher divergence，目标是：

$$
\frac{1}{2}
\mathbb{E}_{\tilde{x}\sim q_\sigma}
\left[
\left\|
s_\theta(\tilde{x})
-
\nabla_{\tilde{x}}\log q_\sigma(\tilde{x})
\right\|^2
\right]
$$

展开后有交叉项：

$$
-
\mathbb{E}_{q_\sigma}
\left[
\nabla_{\tilde{x}}\log q_\sigma(\tilde{x})^T
s_\theta(\tilde{x})
\right]
$$

Problem 4 证明的核心就是：这个含有未知边缘 score 的项，可以改写成含有已知条件 score 的项：

$$
\mathbb{E}_{\tilde{x}\sim q_\sigma}
\left[
\nabla_{\tilde{x}}\log q_\sigma(\tilde{x})^T
s_\theta(\tilde{x})
\right]
=
\mathbb{E}_{x\sim p_{\mathrm{data}},\tilde{x}\sim q_\sigma(\tilde{x}|x)}
\left[
\nabla_{\tilde{x}}\log q_\sigma(\tilde{x}|x)^T
s_\theta(\tilde{x})
\right]
$$

推导的关键步骤是 log derivative trick：

$$
q_\sigma(\tilde{x})\nabla_{\tilde{x}}\log q_\sigma(\tilde{x})
=
\nabla_{\tilde{x}}q_\sigma(\tilde{x})
$$

再利用：

$$
q_\sigma(\tilde{x})
=
\int p_{\mathrm{data}}(x)q_\sigma(\tilde{x}|x)dx
$$

交换积分顺序后，再对条件分布使用：

$$
\nabla_{\tilde{x}}q_\sigma(\tilde{x}|x)
=
q_\sigma(\tilde{x}|x)
\nabla_{\tilde{x}}\log q_\sigma(\tilde{x}|x)
$$

于是未知的边缘 score 被换成了已知的条件 score。

### 3. 高斯加噪时，训练标签是什么？

如果：

$$
q_\sigma(\tilde{x}|x)=\mathcal{N}(\tilde{x};x,\sigma^2I)
$$

那么：

$$
\nabla_{\tilde{x}}\log q_\sigma(\tilde{x}|x)
=
-
\frac{\tilde{x}-x}{\sigma^2}
=
\frac{x-\tilde{x}}{\sigma^2}
$$

所以训练时可以让网络预测：

$$
s_\theta(\tilde{x},\sigma)
\approx
\frac{x-\tilde{x}}{\sigma^2}
$$

这就是“denoising”的来源：模型看到带噪样本 $\tilde{x}$，学习指向干净样本 $x$ 的方向。

### 4. 只学带噪分布，怎么得到真实数据？

如果只学一个固定 $\sigma$，确实只能得到被高斯平滑后的数据分布：

$$
q_\sigma=p_{\mathrm{data}}*\mathcal{N}(0,\sigma^2I)
$$

真正的 score-based generative modeling 会学习一整组噪声水平：

$$
\sigma_1>\sigma_2>\cdots>\sigma_T\approx 0
$$

训练：

$$
s_\theta(x,\sigma)
\approx
\nabla_x\log q_\sigma(x)
$$

生成时从大噪声开始，再逐渐降低噪声：

$$
\text{pure noise}
\rightarrow
\sigma_1
\rightarrow
\sigma_2
\rightarrow
\cdots
\rightarrow
\sigma_T\approx 0
$$

因为当 $\sigma_T\to 0$ 时：

$$
q_{\sigma_T}(x)\approx p_{\mathrm{data}}(x)
$$

所以最后的样本会逼近真实数据分布。

### 5. 实际采样不是直接反解

训练时标签是：

$$
\frac{x-\tilde{x}}{\sigma^2}
$$

但生成时没有干净样本 $x$，所以不能直接反解。生成时使用网络预测的 score：

$$
s_\theta(x,\sigma)
$$

一种典型采样方法是 annealed Langevin dynamics：

$$
x_{k+1}
=
x_k
+
\eta s_\theta(x_k,\sigma)
+
\sqrt{2\eta}z_k,
\qquad
z_k\sim\mathcal{N}(0,I)
$$

其中：

- $\eta s_\theta(x_k,\sigma)$ 让样本往高概率区域移动。
- $\sqrt{2\eta}z_k$ 保留随机性，避免只做确定性爬坡。
- $\sigma$ 逐步减小，让样本从粗到细靠近真实数据。

可以把生成过程理解成：

$$
\text{纯噪声}
\rightarrow
\text{粗略结构}
\rightarrow
\text{清晰形状}
\rightarrow
\text{细节纹理}
\rightarrow
\text{真实样本}
$$

---

## Problem 5：DDPM 与 NCSN 的关系

### 1. DDPM 的噪声预测其实是 score 预测

DDPM 的前向过程为：

$$
q(x_t|x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
$$

等价写成：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
$$

对条件分布求 score：

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-
\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}
{1-\bar{\alpha}_t}
$$

又因为：

$$
x_t-\sqrt{\bar{\alpha}_t}x_0
=
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

所以：

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-
\frac{\epsilon}
{\sqrt{1-\bar{\alpha}_t}}
$$

因此，预测噪声 $\epsilon$ 等价于预测条件 score，只差一个已知缩放因子。

### 2. 条件 score 和边缘 score 的关系

DDPM 训练目标通常是：

$$
\mathbb{E}_{x_0,\epsilon,t}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right]
$$

对固定的 $x_t,t$，MSE 最优预测是：

$$
\epsilon_\theta(x_t,t)
=
\mathbb{E}[\epsilon|x_t]
$$

由 denoising score matching/Tweedie identity，可得带噪边缘分布 $q_t(x_t)$ 的 score：

$$
\nabla_{x_t}\log q_t(x_t)
=
-
\frac{\mathbb{E}[\epsilon|x_t]}
{\sqrt{1-\bar{\alpha}_t}}
$$

所以 DDPM 的噪声预测网络可以转换成 score 网络：

$$
s_\theta(x_t,t)
\approx
\nabla_{x_t}\log q_t(x_t)
\approx
-
\frac{\epsilon_\theta(x_t,t)}
{\sqrt{1-\bar{\alpha}_t}}
$$

这一步解释了为什么 DDPM 虽然训练目标写成预测噪声，本质上却是在学习不同噪声水平下的数据 score。

### 3. 为什么 $x_0$ 会从条件 score 里消失？

这里最容易混淆的是两个不同的 score：

$$
\nabla_{x_t}\log q(x_t|x_0)
$$

这是 **条件 score**：已经知道原始样本 $x_0$ 时，$x_t$ 的概率密度上升方向。

另一个是：

$$
\nabla_{x_t}\log q_t(x_t)
$$

这是 **边缘 score**：只看到 $x_t$，不知道它来自哪个 $x_0$ 时，整体带噪分布的概率密度上升方向。

边缘分布定义为：

$$
q_t(x_t)
=
\int q_{\mathrm{data}}(x_0)q(x_t|x_0)dx_0
$$

也就是：从真实数据分布采样 $x_0$，再加噪得到 $x_t$，最后把所有可能的 $x_0$ 积分掉。

对边缘分布求 score：

$$
\nabla_{x_t}\log q_t(x_t)
=
\frac{\nabla_{x_t}q_t(x_t)}{q_t(x_t)}
$$

代入 $q_t(x_t)$：

$$
\nabla_{x_t}q_t(x_t)
=
\nabla_{x_t}
\int q_{\mathrm{data}}(x_0)q(x_t|x_0)dx_0
$$

在常规光滑性条件下，求导和积分可以交换：

$$
\nabla_{x_t}q_t(x_t)
=
\int q_{\mathrm{data}}(x_0)
\nabla_{x_t}q(x_t|x_0)dx_0
$$

使用 log derivative trick：

$$
\nabla_{x_t}q(x_t|x_0)
=
q(x_t|x_0)
\nabla_{x_t}\log q(x_t|x_0)
$$

所以：

$$
\nabla_{x_t}\log q_t(x_t)
=
\int
\frac{
q_{\mathrm{data}}(x_0)q(x_t|x_0)
}{
q_t(x_t)
}
\nabla_{x_t}\log q(x_t|x_0)
dx_0
$$

根据 Bayes 公式：

$$
q(x_0|x_t)
=
\frac{
q_{\mathrm{data}}(x_0)q(x_t|x_0)
}{
q_t(x_t)
}
$$

于是得到 Fisher identity：

$$
\nabla_{x_t}\log q_t(x_t)
=
\mathbb{E}_{q(x_0|x_t)}
\left[
\nabla_{x_t}\log q(x_t|x_0)
\right]
$$

这说明：边缘 score 等于条件 score 对所有可能原图 $x_0$ 的后验平均。$x_0$ 不是凭空消失了，而是被积分掉了。

### 4. 为什么可以写成 $\mathbb{E}[\epsilon|x_t]$？

DDPM 前向过程为：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

如果同时知道 $x_t$ 和 $x_0$，噪声可以直接反解：

$$
\epsilon
=
\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}
{\sqrt{1-\bar{\alpha}_t}}
$$

所以条件 score 可以写成：

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-
\frac{\epsilon}
{\sqrt{1-\bar{\alpha}_t}}
$$

把它代入 Fisher identity：

$$
\nabla_{x_t}\log q_t(x_t)
=
\mathbb{E}_{q(x_0|x_t)}
\left[
-
\frac{\epsilon}
{\sqrt{1-\bar{\alpha}_t}}
\right]
$$

由于 $\sqrt{1-\bar{\alpha}_t}$ 对给定时间步 $t$ 是常数：

$$
\nabla_{x_t}\log q_t(x_t)
=
-
\frac{
\mathbb{E}_{q(x_0|x_t)}[\epsilon]
}
{\sqrt{1-\bar{\alpha}_t}}
$$

这里的 $\epsilon$ 实际上是 $x_0$ 和 $x_t$ 的函数：

$$
\epsilon(x_0,x_t)
=
\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}
{\sqrt{1-\bar{\alpha}_t}}
$$

因此：

$$
\mathbb{E}_{q(x_0|x_t)}[\epsilon]
=
\mathbb{E}[\epsilon|x_t]
$$

展开写就是：

$$
\mathbb{E}[\epsilon|x_t]
=
\mathbb{E}
\left[
\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}
{\sqrt{1-\bar{\alpha}_t}}
\middle|x_t
\right]
$$

因为 $x_t$ 已经给定，是常量：

$$
\mathbb{E}[\epsilon|x_t]
=
\frac{
x_t-\sqrt{\bar{\alpha}_t}\mathbb{E}[x_0|x_t]
}
{\sqrt{1-\bar{\alpha}_t}}
$$

所以 $\epsilon$ 并不是和 $x_0$ 没关系。更准确地说：

- 前向采样时，$\epsilon$ 和 $x_0$ 是独立采样的。
- 但给定 $x_t$ 之后，$\epsilon$ 和 $x_0$ 会通过方程 $x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$ 产生后验相关性。
- $\mathbb{E}[\epsilon|x_t]$ 表示只看到 $x_t$ 时，对所有可能 $x_0$ 对应噪声的平均估计。

这就是：

$$
\nabla_{x_t}\log q_t(x_t)
=
-
\frac{\mathbb{E}[\epsilon|x_t]}
{\sqrt{1-\bar{\alpha}_t}}
$$

的来源。

DDPM 训练时知道真实 $x_0$，所以可以计算真实噪声 $\epsilon$ 来监督网络。但网络输入只有 $(x_t,t)$，因此 MSE 最优预测器满足：

$$
\epsilon_\theta(x_t,t)
\approx
\mathbb{E}[\epsilon|x_t,t]
$$

于是：

$$
-
\frac{\epsilon_\theta(x_t,t)}
{\sqrt{1-\bar{\alpha}_t}}
\approx
\nabla_{x_t}\log q_t(x_t)
$$

这就是 DDPM 噪声预测和边缘 score 估计之间的完整数学关系。

### 5. NCSN 做什么？

NCSN 显式训练：

$$
s_\theta(x,\sigma)
\approx
\nabla_x\log q_\sigma(x)
$$

其中：

$$
q_\sigma(x)
=
p_{\mathrm{data}}*\mathcal{N}(0,\sigma^2I)
$$

并使用一组噪声水平：

$$
\sigma_1>\sigma_2>\cdots>\sigma_T
$$

采样时，NCSN 通常用 annealed Langevin dynamics 从大噪声逐步走向小噪声。

### 6. DDPM 和 NCSN 的统一视角

二者都在学习一串噪声分布的 score：

| 方法 | 学什么 | 噪声水平 | 采样方式 |
|------|--------|----------|----------|
| NCSN | 直接预测 $s_\theta(x,\sigma)$ | $\sigma_1,\ldots,\sigma_T$ | Annealed Langevin dynamics |
| DDPM | 预测 $\epsilon_\theta(x_t,t)$，再换算成 score | $1-\bar{\alpha}_t$ 控制噪声强度 | 反向马尔可夫链 $p_\theta(x_{t-1}|x_t)$ |

核心等价关系是：

$$
s_\theta(x_t,t)
\approx
-
\frac{\epsilon_\theta(x_t,t)}
{\sqrt{1-\bar{\alpha}_t}}
$$

一句话总结：

> DDPM 的噪声预测目标和 NCSN 的 score matching 目标本质上是同一件事：它们都学习不同噪声水平下的 score field，然后从高噪声样本逐步去噪生成真实数据样本。
