import random
from typing import Final

import numpy as np
import torch

DEVICE: Final = torch.accelerator.current_accelerator(
    check_available=True
) or torch.device("cpu")


def ensure_reproducibility() -> None:
    random.seed(33550336)
    np.random.seed(33550336)
    torch.manual_seed(33550336)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
