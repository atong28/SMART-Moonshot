import torch
import pytorch_lightning as pl
import numpy as np


def set_global_seed(seed: int):
    """
    Set deterministic seeds for all hardware-accelerated backends.

    This function synchronizes random seeds across PyTorch, CUDA, 
    NumPy, and PyTorch Lightning to ensure reproducible results 
    across training runs. It enables Lightning's `seed_everything` 
    with worker seeding and also sets manual seeds for CPU and all 
    available CUDA devices.

    Args:
        seed (int): The seed value to apply across all libraries.

    Returns:
        None
    """
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
