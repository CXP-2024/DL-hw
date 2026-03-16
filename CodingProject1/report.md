# Report for CodingProject 1

Name: ChangxunPan

Id: 2024011323

## Task 1: Flatten Layer

Since in this layer, we only do shape changing, we simply reshape the gradient back to the original shape:

```python
input_grad = np.reshape(grad, x.shape)
# shape: x.shape
```

## Task 2: Activation Functions

- For ReLU, only when input $x[i] > 0$, $\frac{\partial L}{\partial x[i]} = \frac{\partial L}{\partial y[i]}$, else $\frac{\partial L}{\partial x[i]} = 0$.

  Thus, we only do:

  ```python
  relu_grad = np.where(x > 0, 1.0, 0.0)
  return grad * relu_grad
  # shape: (batch, channel, h, w) or (batch, features)
  ```

- For Tanh,

  $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (1 - \tanh^2 x)$$

  Thus, we do:

  ```python
  tanh_x = np.tanh(x)
  tanh_grad = 1.0 - tanh_x ** 2
  return grad * tanh_grad
  # shape: (batch, out_channel, out_h, out_w) or (batch, features)
  ```

## Task 3: Linear Layer

$$y = xW + b$$

- For gradient w.r.t. input $x$:

  $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot W^T$$

  The code will be:

  ```python
  grad_input = grad @ weight.T
  # weight shape: (in_features, out_features)
  # grad shape: (batch, out_features)
  # grad_input shape: (batch, in_features)
  ```

- For gradient w.r.t. weight:

  $$\frac{\partial L}{\partial W} = x^T \cdot \frac{\partial L}{\partial y}$$

  The code is:

  ```python
  grad_weight = x.T @ grad
  # grad_weight shape: (in_features, out_features)
  ```

- For gradient w.r.t. bias:

  $$\frac{\partial L}{\partial b_o} = \sum_b \frac{\partial L}{\partial y_{(b,\, o)}} \cdot 1$$

  We need to sum over all batch dimensions:

  ```python
  grad_bias = np.sum(grad, axis=0)
  # grad_bias shape: (out_features,)
  ```

## Task 4: 2D Max-Pooling Layer

Given input shape `(batch_size, C, H, W)`, kernel size $(k_H, k_W)$, and stride $(s_H, s_W)$:

$$y(b, c, h, w) = \max_{m=0,\ldots,k_H-1}\ \max_{n=0,\ldots,k_W-1}\ x(b, c,\ s_H \cdot h + m,\ s_W \cdot w + n)$$

The gradient is propagated by routing it back to the position that achieved the max value in each pooling window.

We use the same index construction as in the forward pass to build 6-dimensional windows:

```python
idx_h = np.arange(out_h)[:, None] * stride_h + np.arange(kernel_h)
idx_w = np.arange(out_w)[:, None] * stride_w + np.arange(kernel_w)

windows = x[:, :, idx_h[:, None, :, None], idx_w[None, :, None, :]]
# windows shape: (batch_size, channels, out_h, out_w, kernel_h, kernel_w)
```

Then `windows[:, :, i, j, :, :]` gives us the input window for output pixel $(i, j)$. We find the argmax position in each window and build a mask:

```python
for i in range(out_h):
    for j in range(out_w):
        window = windows[:, :, i, j, :, :]
        # flatten spatial dims, find argmax
        flat = window.reshape(b, c, -1)
        max_pos = np.argmax(flat, axis=-1)  # shape: (batch, channels)
        # build mask: 1 at max position, 0 elsewhere
        mask = np.zeros_like(flat)
        mask[np.arange(b)[:, None], np.arange(c)[None, :], max_pos] = 1
        mask = mask.reshape(b, c, kernel_h, kernel_w)
        # route gradient to max positions
        grad_input[:, :, idx_h[i][:, None], idx_w[j][None, :]] += mask * grad[:, :, i:i+1, j:j+1]
```

## Task 5: 2D Convolutional Layer

The convolutional layer has weight shape `(out_channels, in_channels, kernel_h, kernel_w)` and bias shape `(out_channels,)`. Input shape is `(batch_size, in_channels, H, W)`. No padding is applied.

$$y(b, o, h, w) = \text{bias}(o) + \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{k_H-1} \sum_{n=0}^{k_W-1} \text{weight}(o, c, m, n) \cdot x(b, c,\ h \cdot s_H + m,\ w \cdot s_W + n)$$

- **Gradient w.r.t. weight:**

  $$\frac{\partial L}{\partial w_{(o,c,m,n)}} = \sum_{b,h,w} \frac{\partial L}{\partial y_{(b,o,h,w)}} \cdot x(b, c,\ h \cdot s_H + m,\ w \cdot s_W + n)$$

  Construct 6-dimensional windows as before,  $ window(b,c,h,w,m,n) = x(b, c,\ h \cdot s_H + m,\ w \cdot s_W + n)$

  ```python
  windows = x[:, :, idx_h[:, None, :, None], idx_w[None, :, None, :]]
  # windows shape: (batch_size, in_channels, out_h, out_w, kernel_h, kernel_w)
  ```

  Then compute via einsum:

  ```python
  grad_weight = np.einsum("bohw,bchwmn->ocmn", grad, windows)
  # grad_weight shape: (out_channels, in_channels, kernel_h, kernel_w)
  ```

- **Gradient w.r.t. bias:**

  $$\frac{\partial L}{\partial \text{bias}_{(o)}} = \sum_{b,h,w} \frac{\partial L}{\partial y_{(b,o,h,w)}}$$

  ```python
  grad_bias = np.sum(grad, axis=(0, 2, 3))  # sum over batch, out_h, out_w
  # grad_bias shape: (out_channels,)
  ```

- **Gradient w.r.t. input $x$:**

  $$\frac{\partial L}{\partial x_{(b,c,ih,iw)}} = \sum_{o,h,w} \frac{\partial L}{\partial y_{(b,o,h,w)}} \cdot w_{(o,c,m,n)}$$

  where $ih = h \cdot s_H + m$ and $iw = w \cdot s_W + n$.

  For each output position $(h, w)$, we accumulate gradients into the corresponding input window by first summing over the `out_channels` dimension via einsum:

  ```python
  for i in range(out_h):
      for j in range(out_w):
          hs, ws = i * stride_h, j * stride_w
          g = grad[:, :, i:i+1, j:j+1]
          # g shape: (batch_size, out_channels, 1, 1)
          grad_input[:, :, hs:hs+kernel_h, ws:ws+kernel_w] += np.einsum("boxy,ocmn->bcmn", g, self.weight)
          # weight shape: (out_channels, in_channels, kernel_h, kernel_w)
          # grad_input shape: (batch_size, in_channels, in_h, in_w)
  ```

## Generative AI Usage Disclosure

I used Generative AI (GitHub Copilot / Gemini) during this project. Specifically, I used it to:
- Learn about and understand NumPy usage examples and methods (like `np.einsum`, `np.where`, etc.).
- Get explanations for related concepts in deep learning and backpropagation.
- Assist in understanding the mathematical derivations of gradients for the backward passes.