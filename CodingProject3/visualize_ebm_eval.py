"""Visualize EBM inpainting: corrupted (left) vs repaired (right) side by side.

Uses the exact same corruption as evaluate_ebm.py / train_ebm.py.
Shows a 5x2 grid of digits (0-9), each with corrupted | repaired pair.
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from modules.ebm import CustomEBMModel
from utils import DEVICE, ensure_reproducibility
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ensure_reproducibility()

TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

# Load model
model = CustomEBMModel().to(DEVICE)
model.load_state_dict(torch.load("checkpoints/ebm_best.pth", map_location=DEVICE, weights_only=True))
model.eval()

# Load test data
dataset = MNIST(root="data", train=False, download=True, transform=TRANSFORM)

# Pick first example of each digit 0-9
indices = []
for digit in range(10):
    for i in range(len(dataset)):
        if dataset.targets[i] == digit:
            indices.append(i)
            break

originals = torch.stack([dataset[i][0] for i in indices]).to(DEVICE)  # (10,1,28,28)

# Corrupt using EXACT same logic as evaluate_ebm.py
def corrupt_images(x):
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., ::2, :] = True  # even rows = known
    noise = 0.3 * torch.randn_like(x)
    corrupted = torch.where(mask, x, noise).clamp(0, 1)
    return corrupted, mask

corrupted, mask = corrupt_images(originals)

# Inpaint
repaired = model.inpaint(corrupted, mask)

# MSE
corrupted_region = ~mask
total_se = (repaired - originals).square()[corrupted_region].sum().item()
total_px = corrupted_region.sum().item()
overall_mse = total_se / total_px

# Plot: 3 rows x 10 cols — original / corrupted / repaired
fig, axes = plt.subplots(3, 10, figsize=(20, 7))

for col in range(10):
    for row, (imgs, label) in enumerate([
        (originals, "Original"),
        (corrupted, "Corrupted"),
        (repaired, "Repaired"),
    ]):
        axes[row, col].imshow(imgs[col, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, col].axis("off")
    axes[0, col].set_title(f"Digit {col}", fontsize=12, fontweight="bold")

axes[0, 0].set_ylabel("Original", fontsize=14, rotation=0, labelpad=70, va="center")
axes[1, 0].set_ylabel("Corrupted", fontsize=14, rotation=0, labelpad=70, va="center")
axes[2, 0].set_ylabel("Repaired", fontsize=14, rotation=0, labelpad=70, va="center")

plt.suptitle(f"EBM Inpainting  |  MSE = {overall_mse:.4f}  (TA baseline: 0.0211)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("ebm_eval_inpainting.png", dpi=150, bbox_inches="tight")
print(f"Saved to ebm_eval_inpainting.png")
print(f"MSE = {overall_mse:.4f}")
