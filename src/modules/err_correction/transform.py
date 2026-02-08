import torch

def normalize_hsqc(hsqc: torch.Tensor) -> torch.Tensor:
    """Normalize the intensity values (3rd column) of HSQC peaks."""
    hsqc = hsqc.clone()
    I = hsqc[:, 2]
    hsqc[:, 2] = I / (I.abs().max() + 1e-6)
    return hsqc
