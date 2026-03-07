# Deep Learning (Spring 2026) - IIIS, Tsinghua University

Homework repository for the Deep Learning course, containing both theoretical assignments (LaTeX) and coding projects (Python/PyTorch).

**Student:** Pan Changxun | **ID:** 2024011323

## Repository Structure

```
.
├── Assignment0/          # Written assignment 0
│   └── main.tex
├── Assignment1/          # Written assignment 1
│   └── main.tex
├── CodingProject1/       # Backpropagation from scratch (NumPy)
│   ├── modules/          #   Flatten, Activation, Linear, Pooling, Conv2D
│   ├── tests/            #   Unit tests (PyTorch autograd reference)
│   └── report.md
├── CodingProject2/       # Tiny ImageNet classification (PyTorch)
│   ├── modules.py        #   Custom CNN architecture
│   ├── train.py          #   Training loop
│   ├── evaluate.py       #   Validation evaluation
│   └── datasets.py       #   TinyImageNetDataset
└── README.md
```

## Assignments

| # | Type | Topic |
|---|------|-------|
| Assignment 0 | Written | Prerequisite review |
| Assignment 1 | Written | Theoretical foundations of deep learning |
| Coding Project 1 | Code | Implementing backward pass for common layers using NumPy |
| Coding Project 2 | Code | Training a CNN on Tiny ImageNet (200-class, 64x64) |

## Environment

All coding projects use [uv](https://docs.astral.sh/uv/) for dependency management. To set up:

```bash
cd CodingProject1  # or CodingProject2
uv sync
```

See each project's own `README.md` for detailed instructions.