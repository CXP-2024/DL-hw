# Deep Learning Coding Project 3: MNIST Generative Modeling

## 1. Project Overview

In this project, you will implement and train generative models on the MNIST handwritten digit dataset using PyTorch.

You must complete:

- an **Energy-Based Model (EBM)** for image inpainting (**mandatory**)
- **one** class-conditional generative model: **Conditional GAN** or **Conditional VAE**

The goal is to implement the core training components, run experiments, and evaluate model quality with the provided scripts.

## 2. Learning Objectives

By the end of this project, you will be able to:

- implement Langevin dynamics and contrastive divergence for energy-based learning
- build and train class-conditional generative models on image data
- implement the objective functions behind VAEs and GANs
- evaluate inpainting with MSE and conditional generation with FID

## 3. Prerequisites and Environment Setup

**Prerequisites:** Familiarity with Python, PyTorch, and deep learning

**Environment Setup**

1. Install [uv](https://docs.astral.sh/uv/):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   If you encounter network issues, use the mirror:

   ```bash
   curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

**Data Preparation**

- MNIST is downloaded automatically to `data/` the first time you run training or evaluation.
- Download the FID evaluation checkpoint before running conditional-model evaluation:

```bash
bash download_checkpoints.sh
```

### Dataset Details

| Property | Value |
| --- | --- |
| Training images | 60,000 |
| Validation images | 10,000 |
| Image shape | `(1, 28, 28)` |
| Number of classes | 10 |
| Data directory | `data/` |

## 4. Codebase Structure

Unless explicitly allowed below, do not modify files marked as read-only.

```
deep-learning-coding-project-3/
├── modules/
│   ├── ebm.py              # [TODO] EBM model implementation
│   ├── gan.py              # [TODO] GAN generator and discriminator
│   └── vae.py              # [TODO] Conditional VAE model
├── evaluate_ebm.py         # [Read-only] EBM evaluation script (inpainting MSE)
├── evaluate_fid.py         # [Read-only] GAN/VAE evaluation script (per-class FID)
├── download_checkpoints.sh # [Read-only] Download FID evaluation checkpoint
├── train_ebm.py            # [TODO] EBM training logic
├── train_gan.py            # [TODO] GAN training logic
├── train_vae.py            # [TODO] VAE training logic
├── pyproject.toml          # Project configuration and dependencies
└── uv.lock                 # Dependency lock file
```

## 5. Implementation Tasks

You must complete **EBM** and exactly **one** of **GAN** or **VAE**.

Global requirements:

- Do not add new source files.
- Keep the public interfaces expected by the evaluation scripts unchanged.
- For `model.inpaint()` and `model.generate()`, images pixel values should be in `[0, 1]`.
- The sum of all submitted checkpoint files, before compression, must be under **200 MB**.

Global notes:

- Save the checkpoint you consider your best final model for each required component.
- Periodically inspect generated or reconstructed samples during training.
- Small adjustments outside TODO blocks are allowed when necessary to make your implementation work correctly.

**Task 1: Energy-Based Model (`modules/ebm.py`, `train_ebm.py`)**

Complete:

- `CustomEBMModel` in `modules/ebm.py`
- `train()` in `train_ebm.py`

Requirements:

- Implement the EBM as an MLP.
- Train it on MNIST for image inpainting, where alternating rows are corrupted with noise.

Notes:

- Naive contrastive divergence often diverges quickly; add an L2 regularization term `alpha(E_theta(x+)^2 + E_theta(x-)^2)` to stabilize training.
- Inspect generated or reconstructed samples during training to monitor behavior.
- You may consult [Implicit Generation and Generalization in Energy Based Models](https://arxiv.org/pdf/1903.08689.pdf) for useful training tricks.

**Task 2: Conditional GAN (`modules/gan.py`, `train_gan.py`)**

If you choose GAN, complete:

- `CustomGANGenerator` in `modules/gan.py`
- `CustomGANDiscriminator` in `modules/gan.py`
- the training loop in `train_gan.py`

Requirements:

- Implement a class-conditional DCGAN-style model; use fully convolutional networks for both generator and discriminator, except for linear projection heads if needed.
- Generate recognizable class-conditional MNIST digits with reasonable within-class diversity.
- Avoid severe mode collapse.

Notes:

- Monitor generated images during training.
- See [this overview](https://developers.google.com/machine-learning/gan/problems) for common mode-collapse mitigation strategies.

**Task 3: Conditional VAE (`modules/vae.py`, `train_vae.py`)**

If you choose VAE, complete:

- `CustomVAEModel` in `modules/vae.py`
- `train()` in `train_vae.py`

Requirements:

- Implement a class-conditional VAE with an MLP encoder and an MLP decoder.
- Generate recognizable class-conditional MNIST digits with reasonable within-class diversity.

Notes:

- Assume the prior is `p(z)=N(0, I)`.
- Assume both `q(z|x, y)` and `p(x|z, y)` are Gaussian distributions.
- Since `p(x|z, y)` is modeled as a real-valued Gaussian while images lie in `[0, 1]`, you may need to transform or scale `x` when computing the reconstruction term.
- Using pre-trained models for initialization is permitted, but you must disclose any external resources used in your report.

**Task 4: Report (`report.md` or `report.pdf`)**

Create `report.md` or `report.pdf` in the repository root with the following sections:

1. **Cover Information** - Your name and student ID.
2. **Generative AI Usage Disclosure** - State `None` if you did not use AI. Otherwise, describe which tool(s) you used and how.
3. **EBM Implementation** - Describe your energy model and training choices.
4. **Conditional Model Implementation** - Describe your GAN or VAE architecture and training choices.
5. **Hyperparameters** - Document batch size, learning rate, optimizer, epochs, and other key settings.
6. **Results** - Include EBM inpainting examples and MSE. For your chosen conditional model, include generated samples, FID, and per-class sample standard deviation.

## 6. Testing and Debugging

**Train the EBM**

```bash
uv run python train_ebm.py checkpoints/ebm_best.pth
```

**Evaluate the EBM**

```bash
uv run python evaluate_ebm.py checkpoints/ebm_best.pth
```

**Train the GAN**

```bash
uv run python train_gan.py checkpoints/gan_best.pth
```

**Evaluate the GAN**

```bash
uv run python evaluate_fid.py checkpoints/gan_best.pth --arch gan
```

To also save generated images:

```bash
uv run python evaluate_fid.py checkpoints/gan_best.pth --arch gan --generate
```

Generated images are saved under `generated/gan/<digit>/<digit>_<idx>.png`.

**Train the VAE**

```bash
uv run python train_vae.py checkpoints/vae_best.pth
```

**Evaluate the VAE**

```bash
uv run python evaluate_fid.py checkpoints/vae_best.pth --arch vae
```

To also save generated images:

```bash
uv run python evaluate_fid.py checkpoints/vae_best.pth --arch vae --generate
```

Generated images are saved under `generated/vae/<digit>/<digit>_<idx>.png`.

Evaluation outputs:

- `evaluate_ebm.py` reports inpainting MSE on the MNIST test split.
- `evaluate_fid.py` reports per-class FID over digits `0` through `9` on the MNIST test split; it also reports per-class sample standard deviation.

## 7. Submission Guidelines

Follow these steps to prepare your submission.

1. Finalize the required code files and your report file (`report.md` or `report.pdf`).
2. Create a ZIP archive named `submission.zip` include the following files:

   - `modules/ebm.py`
   - `train_ebm.py`
   - `report.md` or `report.pdf`
   - **either** `modules/gan.py` and `train_gan.py` **or** `modules/vae.py` and `train_vae.py`
   - your best checkpoint files
   - generated images for your chosen conditional model, produced by `evaluate_fid.py --generate`

   Requirements:

   - The sum of all submitted checkpoint files, before compression, must be under **200 MB**. Submissions that violate the checkpoint-size limit may receive no credit.
   - Submit only the checkpoint files needed for grading.
   - Do not include additional source files beyond the required ones listed here.

   You may run the following command to package everything in one shot:

   ```bash
   zip -r submission.zip report.* modules/*.py train_*.py checkpoints/*.pth generated/*
   ```

3. One valid archive layout is:

   ```
   submission.zip
   ├── report.md or report.pdf
   ├── generated/
   │   └── gan/ or vae/
   │       └── 0/ ... 9/
   ├── modules/
   │   ├── ebm.py
   │   ├── gan.py or vae.py
   │   └── __init__.py
   ├── train_ebm.py
   ├── train_gan.py or train_vae.py
   └── checkpoints/
       ├── ebm_best.pth
       └── gan_best.pth or vae_best.pth
   ```

## 8. Grading Rubric

Your project will be evaluated as follows:

| Criteria | Weight | Description |
| --- | --- | --- |
| EBM Performance | 40% | Inpainting quality, including the reconstruction metric and whether the recovered images are visually reasonable. |
| Conditional Model Performance | 40% | Generation quality for your chosen GAN or VAE, including FID, visual quality, and diversity. |
| Report | 20% | Completeness and clarity of implementation details, hyperparameters, results, and AI usage disclosure. |

For conditional generation, evaluation uses per-class FID over digits `0` through `9`, along with a check that samples are recognizable and reasonably diverse.

For GAN, the following per-digit diversity thresholds are provided as reference:

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Minimum std | 0.17 | 0.08 | 0.17 | 0.15 | 0.14 | 0.16 | 0.15 | 0.13 | 0.15 | 0.13 |

Major penalties or zero credit may apply if:

- a GAN submission does not use the required FCN generator and FCN discriminator
- the conditional model shows severe mode collapse or fails basic diversity requirements
- an EBM submission does not use the required MLP-style design, or a VAE submission does not use the required MLP encoder/decoder
- The sum of all submitted checkpoint files, before compression, exceeds **200MB**.

Examples of penalized issues include:

- EBM outputs that do not meaningfully recover corrupted rows
- conditional samples that are not visually recognizable as digits
- conditional outputs with obvious artifacts, incorrect intensity range, or near-identical samples within a class

**Grading Environment**

Your submission will be executed on a grading platform with at least the following specifications:

| Resource | Specification |
| --- | --- |
| GPU VRAM | 32 GB |
| System RAM | 64 GB |
| Running Time | 30 minutes (in total) |

**TA Reference Baseline (Not a Scoring Criterion)**

One TA implementation (completed in tens of minutes) reported the following final metrics for quick sanity check only:

- EBM inpainting: MSE `0.02110`
- Conditional GAN: mean FID `3.8279`
- Conditional VAE: mean FID `3.5479`

## 9. Academic Integrity and AI Policy

**Plagiarism in any form will result in an F for the course.**

You must disclose any use of generative AI tools in the **Generative AI Usage Disclosure** section of your report. If you did not use AI, state `None`. If you did, describe which tool(s) you used and how. Undisclosed AI use is a violation of academic integrity.

All submitted code must be your own work. You may discuss high-level ideas with classmates, but sharing or copying code is strictly prohibited.
