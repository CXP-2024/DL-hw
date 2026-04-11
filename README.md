# Deep Learning (Spring 2026) - IIIS, Tsinghua University

Homework repository for the Deep Learning course, containing both theoretical assignments (LaTeX) and coding projects (Python/PyTorch).

**Student:** Pan Changxun | **ID:** 2024011323

## Repository Structure

```
.
├── Assignment0/            # Written: Prerequisite review
├── Assignment1/            # Written: Deep learning foundations
├── Assignment2/            # Written: EBMs, Hopfield networks, RBMs
├── Assignment3/            # Written: GANs, JSD, Wasserstein distance
├── Assignment4/            # Written: VAEs, beta-VAE, EM algorithm
│   ├── cfg_slides.tex      #   Bonus: Classifier-Free Guidance slides
│   ├── vae_cvae_notes.tex  #   Bonus: VAE to CVAE introduction
│   └── vqvae_vs_vae_slides.tex  # Bonus: VQ-VAE vs VAE slides
├── Assignment5/            # Written: Autoregressive models, normalizing flows
│   └── pixelcnn_slides.tex #   Bonus: PixelCNN slides
├── CodingProject1/         # Backpropagation from scratch (NumPy)
│   ├── modules/            #   Flatten, Activation, Linear, Pooling, Conv2D
│   └── tests/              #   Unit tests (PyTorch autograd reference)
├── CodingProject2/         # Tiny ImageNet classification (SE-ResNet, PyTorch)
│   ├── modules.py          #   Custom CNN architecture
│   ├── train.py            #   Training loop
│   └── evaluate.py         #   Validation evaluation
├── CodingProject3/         # MNIST generative modeling (EBM + Conditional GAN/VAE)
│   ├── modules/            #   ebm.py, gan.py, vae.py
│   ├── train_ebm.py        #   EBM training (inpainting)
│   ├── train_gan.py        #   Conditional GAN training
│   └── gmvae.py            #   Gaussian Mixture VAE
├── CodingProject3_2024/    # Generative modeling (2024 notebook version)
│   ├── ebm.ipynb           #   Energy-Based Model
│   ├── gan.ipynb           #   GAN
│   ├── vae.ipynb           #   VAE
│   └── flow.ipynb          #   Normalizing Flow
├── CodingProject4/         # VLM fine-tuning (Qwen2.5-VL + IconQA)
│   ├── train.py            #   SFT fine-tuning with LoRA
│   ├── evaluate.py         #   Evaluation pipeline
│   └── processors.py       #   Data processing
├── slides/                 # Course lecture slides (lec1–lec13)
└── data/                   # Shared datasets (MNIST)
```

## Assignments

| # | Type | Topic |
|---|------|-------|
| Assignment 0 | Written | Prerequisite review |
| Assignment 1 | Written | Foundations: ReLU, BatchNorm/Dropout, GroupNorm, gradient descent convergence |
| Assignment 2 | Written | Energy-based models, Hopfield networks, RBMs, generative classification |
| Assignment 3 | Written | GANs: FID, discriminator training, JSD properties, Wasserstein distance |
| Assignment 4 | Written | VAEs: KL divergence, reparameterization trick, beta-VAE, EM algorithm |
| Assignment 5 | Written | Autoregressive models: dequantization, autoregressive normalizing flows |
| Coding Project 1 | Code | Implementing backward pass for common layers using NumPy |
| Coding Project 2 | Code | Training SE-ResNet on Tiny ImageNet (200-class, 64x64) |
| Coding Project 3 | Code | MNIST generative modeling: EBM inpainting + conditional GAN/VAE |
| Coding Project 4 | Code | Vision-Language Model fine-tuning on IconQA (Qwen2.5-VL + LoRA) |

## Viewing Original Starter Code

Each assignment/project has an initial commit containing the original problem set or starter code. Use `git show <commit>:<path>` to view a specific file, or `git diff <commit> -- <dir>` to see what was changed.

| Directory | Initial Commit | Command to View Starter |
|---|---|---|
| Assignment0 | `40b2e9b` | `git show 40b2e9b:Assignment0/main.tex` |
| Assignment1 | `23c46f8` | `git show 23c46f8:Assignment1/main.tex` |
| Assignment2 | `b294af6` | `git show b294af6:Assignment2/main.tex` |
| Assignment3 | `b294af6` | `git show b294af6:Assignment3/main.tex` |
| Assignment4 | `e9109a9` | `git diff e9109a9 -- Assignment4/` |
| Assignment5 | `c3e7fa9` | `git show c3e7fa9:Assignment5/main.tex` |
| CodingProject1 | `40b2e9b` | `git diff 40b2e9b -- CodingProject1/` |
| CodingProject2 | `a8c9a4a` | `git diff a8c9a4a -- CodingProject2/` |
| CodingProject3 | `fc97823` | `git diff fc97823 -- CodingProject3/` |
| CodingProject3_2024 | `cde873a` | `git diff cde873a -- CodingProject3_2024/` |
| CodingProject4 | `9582a0b` | `git diff 9582a0b -- CodingProject4/` |

To restore a directory to its original state (in a detached state for viewing only):

```bash
# Example: view the original CodingProject3 starter code
git stash          # save current work if needed
git checkout fc97823 -- CodingProject3/
# ... browse the original files ...
git checkout HEAD -- CodingProject3/   # restore back to current version
git stash pop      # restore stashed work
```

## Environment

All coding projects use [uv](https://docs.astral.sh/uv/) for dependency management. To set up:

```bash
cd CodingProject1  # or any CodingProjectN
uv sync
```

See each project's own `README.md` for detailed instructions.