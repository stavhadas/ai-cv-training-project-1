import os
import random
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    """
    Seed everything for reproducibility.

    Args:
        seed (int): The seed value to use.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # cuBLAS reads this when it starts (the first GPU matmul), so set it before anything else.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: AdaptiveAvgPool2d's CUDA backward (ResNet's global pool) has no deterministic
        # version and would otherwise raise; it warns and stays non-deterministic instead.
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    """
    Create a torch.Generator with the given seed.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g