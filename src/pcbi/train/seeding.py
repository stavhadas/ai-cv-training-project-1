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

def make_generator(seed: int) -> torch.Generator:
    """
    Create a torch.Generator with the given seed.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g