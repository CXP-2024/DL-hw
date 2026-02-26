## Deep Learning (Spring 2026) Coding Project 1

### 1. Project Overview

In this project, you will derive and implement the backward pass (gradient computation) of several common layers used in deep learning. Backpropagation is the cornerstone of training neural networks, and understanding how gradients flow through each layer is essential for building, debugging, and optimizing deep learning models. By implementing these gradients from scratch, you will develop a concrete, mathematical understanding of what automatic differentiation frameworks do under the hood.

The `forward` methods for each layer are already provided -- your job is to implement the corresponding `backward` methods using **NumPy only**. You are **NOT** allowed to use any autograd framework (e.g., PyTorch, TensorFlow, JAX) in your implementation code. PyTorch is used only in the provided test suite to validate your results.

Let $L$ denote the loss function, $z$ the input to a layer, and $y$ the output. Assume the upstream gradient $\frac{\partial L}{\partial y}$ (denoted `grad` in the code) is given. Your goal is to compute:

1. $\frac{\partial L}{\partial z}$ -- the gradient w.r.t. the layer input.
2. (For layers with trainable parameters) $\frac{\partial L}{\partial \text{weight}}$ and $\frac{\partial L}{\partial \text{bias}}$.

Every module inherits from `Module` (defined in `modules/module.py`) and its `backward` method receives:
- `grad` -- the upstream gradient $\frac{\partial L}{\partial y}$
- `x` -- the original input to the layer

### 2. Learning Objectives

By the end of this project, you will be able to:

- Derive the backward pass (gradient computation) for common neural network layers using the chain rule.
- Implement gradient computations from scratch using NumPy, without relying on autograd frameworks.
- Understand the relationship between forward and backward passes for reshape, activation, linear, pooling, and convolution layers.
- Verify the correctness of your gradient implementations by comparing against PyTorch autograd.
- Gain hands-on experience with tensor shapes, broadcasting, and batched operations in NumPy.

### 3. Prerequisites and Environment Setup

**Prerequisites:**

- [uv](https://docs.astral.sh/uv/) package manager

**Setup:**

1. Install uv:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   If you encounter network issues, you can try:

   ```bash
   curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Verify the environment by running the test suite. Before you start, all tests should fail with `NotImplementedError`:

   ```bash
   uv run python -m unittest discover -s tests -v
   ```

### 4. Codebase Structure

Familiarize yourself with the starter code. Do not modify files marked as read-only.

```
.
├── modules/
│   ├── module.py          # Abstract base class (DO NOT EDIT)
│   ├── flatten.py         # Unflatten layer
│   ├── activation.py      # ReLU and Tanh
│   ├── linear.py          # Linear (fully-connected) layer
│   ├── pooling.py         # 2D max-pooling layer
│   └── conv.py            # 2D convolutional layer
├── tests/                 # Unit tests (DO NOT EDIT)
│   ├── test_flatten.py
│   ├── test_activation.py
│   ├── test_linear.py
│   ├── test_pooling.py
│   └── test_conv.py
├── pyproject.toml
└── README.md
```

Each module under `modules/` contains a `backward` method stub marked with:

```python
# YOUR CODE BEGIN.

raise NotImplementedError

# YOUR CODE END.
```

Replace `raise NotImplementedError` with your implementation.

### 5. Implementation Tasks

Tasks are ordered from easiest to hardest. We recommend completing them in order.

---

**Task 1: Unflatten Layer (5 pts)**

*File:* `modules/flatten.py`

The `Unflatten` layer reshapes a single dimension of the input tensor into a given multi-dimensional shape (analogous to `torch.nn.Unflatten`).

Implement `backward(self, grad, x)` which must return:

- `dx`: gradient w.r.t. input, same shape as `x`

---

**Task 2: Activation Functions (10 pts)**

*File:* `modules/activation.py`

Implement the backward pass for two activation functions.

**ReLU** is defined as:

$$y = \max(0, z)$$

**Tanh** is defined as:

$$y = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

For each, implement `backward(self, grad, x)` which must return:

- `dx`: gradient w.r.t. input, same shape as `x`

---

**Task 3: Linear Layer (15 pts)**

*File:* `modules/linear.py`

The forward pass computes:

$$y = zW + b$$

where weight $W$ has shape `(in_features, out_features)`, bias $b$ has shape `(out_features,)`, and input $z$ has shape `(batch_size, in_features)`.

> **Note:** The weight convention here is transposed relative to `torch.nn.Linear`. Our weight shape is `(in_features, out_features)`, while PyTorch uses `(out_features, in_features)`.

Implement `backward(self, grad, x)` which must return a tuple of:

- `dx`: gradient w.r.t. input, shape `(batch_size, in_features)`
- `Gradients(weight, bias)`: a named tuple containing:
  - `weight`: gradient w.r.t. `self.weight`, shape `(in_features, out_features)`
  - `bias`: gradient w.r.t. `self.bias`, shape `(out_features,)`

---

**Task 4: 2D Max-Pooling Layer (15 pts)**

*File:* `modules/pooling.py`

Given input shape `(batch_size, C, H, W)`, kernel size $(k_H, k_W)$, and stride $(s_H, s_W)$:

$$y(b, c, h, w) = \max_{m=0,\ldots,k_H-1} \max_{n=0,\ldots,k_W-1} z(b, c,\; s_H \cdot h + m,\; s_W \cdot w + n)$$

Implement `backward(self, grad, x)` which must return:

- `dx`: gradient w.r.t. input, shape `(batch_size, C, H, W)`. The gradient passes through only to the position of the maximum element in each pooling window.

---

**Task 5: 2D Convolutional Layer (25 pts)**

*File:* `modules/conv.py`

The convolutional layer has weight shape `(out_channels, in_channels, kernel_h, kernel_w)` and bias shape `(out_channels,)`. Input shape is `(batch_size, in_channels, H, W)`. No padding is applied.

$$y(b, j, h, w) = \text{bias}(j) + \sum_{k=0}^{C_{in}-1} \sum_{m=0}^{k_H-1} \sum_{n=0}^{k_W-1} \text{weight}(j, k, m, n) \cdot z(b, k,\; h \cdot s_H + m,\; w \cdot s_W + n)$$

Implement `backward(self, grad, x)` which must return a tuple of:

- `dx`: gradient w.r.t. input, shape `(batch_size, in_channels, H, W)`
- `Gradients(weight, bias)`: a named tuple containing:
  - `weight`: gradient w.r.t. `self.weight`, shape `(out_channels, in_channels, kernel_h, kernel_w)`
  - `bias`: gradient w.r.t. `self.bias`, shape `(out_channels,)`

---

### 6. Testing and Debugging

To test a specific task, run its corresponding test:

```bash
uv run python -m unittest tests.test_flatten -v      # Task 1
uv run python -m unittest tests.test_activation -v   # Task 2
uv run python -m unittest tests.test_linear -v       # Task 3
uv run python -m unittest tests.test_pooling -v      # Task 4
uv run python -m unittest tests.test_conv -v         # Task 5
```

To run all tests at once:

```bash
uv run python -m unittest discover -s tests -v
```

All tests compare your NumPy implementations against PyTorch autograd. Passing all tests is necessary but may not be sufficient -- additional tests may be used during grading.

### 7. Submission Guidelines

Follow these exact steps to ensure your project is successfully received and graded.

Your submission must be a **single ZIP archive** named `submission.zip` containing your source code and a report.

**Archive structure:**

```
submission.zip
├── modules/
│   ├── activation.py
│   ├── conv.py
│   ├── flatten.py
│   ├── linear.py
│   └── pooling.py
└── report.md
```

To create the archive:

```bash
zip -r submission.zip modules/ report.md
```

**Report requirements:**

Your `report.md` should include the following sections:

**1. Cover information**

Your name and student ID.

**2. Derivations and implementations**

For each of the five tasks:

- The mathematical derivation of the backward pass, showing how you arrived at the gradient formulas from the chain rule. Clearly state the input/output shapes and the final expressions you implemented.
- A brief description of your implementation approach (e.g., how you handled batched inputs, any NumPy operations you used, etc.).

**3. Generative AI usage disclosure**

Please honestly disclose whether you used generative AI tools while completing this assignment.

- If you did **not** use any AI, write "None."
- If you **did** use AI, please describe:
  - Which AI tool(s) you used (e.g., ChatGPT, GitHub Copilot, Claude, etc.).
  - How you used it (e.g., for brainstorming, for editing, for generating code, for debugging, for understanding concepts, etc.).

### 8. Grading Rubric

Your project will be evaluated based on the following criteria:

| Criteria | Weight | Description |
| --- | --- | --- |
| Unflatten (reshape) | 5 pts | Correct backward pass for the Unflatten layer |
| Activation functions (ReLU, Tanh) | 10 pts | Correct backward pass for ReLU and Tanh activations |
| Linear (fully-connected) | 15 pts | Correct backward pass for the Linear layer, including weight and bias gradients |
| 2D Max-Pooling | 15 pts | Correct backward pass for 2D max-pooling, routing gradients through max positions |
| 2D Convolution | 25 pts | Correct backward pass for 2D convolution, including weight and bias gradients |
| Report | 30 pts | Quality of mathematical derivations, implementation descriptions, and AI usage disclosure |
| **Total** | **100 pts** | |

### 9. Academic Integrity and AI Policy

- Only modify code between the `# YOUR CODE BEGIN.` and `# YOUR CODE END.` markers.
- Do **NOT** edit `modules/module.py` or any files under `tests/`.
- Do **NOT** add or remove any files.
- Do **NOT** use any autograd framework (PyTorch, TensorFlow, JAX, etc.) in your implementations. Only NumPy is allowed.
- Make sure all tests pass before submission.
- You must disclose any use of generative AI tools in your report (see Submission Guidelines).
- **Avoid plagiarism. Any student who violates academic integrity will be seriously dealt with and receive an F for the course.**
