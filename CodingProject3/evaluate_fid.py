import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import models
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torchvision.utils import save_image

from modules.gan import CustomGANGenerator
from modules.vae import CustomVAEModel
from utils import DEVICE, ensure_reproducibility

logger = logging.getLogger(__name__)

NUM_TEST_SAMPLES = 100


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("--arch", type=str, required=True, choices=("gan", "vae"))
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    ensure_reproducibility()

    # Load the model.
    match args.arch:
        case "gan":
            model = CustomGANGenerator()

        case "vae":
            model = CustomVAEModel()

        case _:
            raise ValueError(f"Unsupported architecture: {args.arch}")

    model.to(DEVICE)

    model.load_state_dict(
        torch.load(args.ckpt_path, map_location=DEVICE, weights_only=True)
    )

    # Load the feature extractor.
    feature_extractor = _load_feature_extractor()

    dataset = MNIST(root="data", train=False, download=True)

    mean_fid, per_digit_fid, per_digit_std = evaluate(
        model,
        feature_extractor,
        dataset,
        args.arch,
        args.generate,
    )

    logger.info(f"Mean FID score: {mean_fid}")
    logger.info(f"Per-digit FID scores: {per_digit_fid}")
    logger.info(f"Per-digit std scores: {per_digit_std}")


@torch.inference_mode()
def evaluate(
    model: CustomGANGenerator | CustomVAEModel,
    feature_extractor: nn.Module,
    dataset: MNIST,
    arch: str,
    generate: bool,
) -> tuple[float, list[float], list[float]]:
    model.eval()

    fid = FrechetInceptionDistance(feature=feature_extractor, normalize=True).to(DEVICE)

    fid_scores: list[float] = []
    std_scores: list[float] = []
    output_root = Path("generated") / arch

    for digit in range(10):
        fid.reset()

        real = (
            dataset.data[dataset.targets == digit][:NUM_TEST_SAMPLES]
            .float()
            .div(255.0)
            .unsqueeze(1)
            .expand(-1, 3, -1, -1)
        )

        labels = torch.full((NUM_TEST_SAMPLES,), digit, dtype=torch.long, device=DEVICE)

        fake = model.generate(labels)
        std_scores.append(fake.std(dim=0).mean().item())

        if generate:
            digit_dir = output_root / str(digit)
            digit_dir.mkdir(parents=True, exist_ok=True)
            for idx, image in enumerate(fake):
                save_image(image.cpu(), digit_dir / f"{digit}_{idx:04d}.png")

        fake = fake.expand(-1, 3, -1, -1)

        fid.update(real.to(DEVICE), real=True)
        fid.update(fake.to(DEVICE), real=False)

        fid_scores.append(fid.compute().item())

    return sum(fid_scores) / len(fid_scores), fid_scores, std_scores


def _load_feature_extractor() -> nn.Module:
    class MnistInceptionV3(nn.Module):
        def __init__(self, in_channels=3):
            super(MnistInceptionV3, self).__init__()

            self.model = models.inception_v3()

            self.model.fc = nn.Linear(self.model.fc.in_features, 10)

        def forward(self, x):
            x = v2.functional.resize(x, [299, 299])

            return self.model(x)

    model = MnistInceptionV3()

    model.load_state_dict(
        torch.load(
            "checkpoints/MnistInceptionV3.pth", map_location="cpu", weights_only=True
        )
    )

    setattr(model.model, "fc", nn.Identity())

    model.eval()

    return model


if __name__ == "__main__":
    main()
