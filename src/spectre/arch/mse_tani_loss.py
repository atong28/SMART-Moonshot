# mse_tanimoto_hybrid_loss.py
import torch
import torch.nn as nn

def tanimoto(a, b, eps=1e-12):
    # expects nonnegative
    ab = torch.sum(a * b, dim=1)
    aa = torch.sum(a * a, dim=1)
    bb = torch.sum(b * b, dim=1)
    return ab / (aa + bb - ab + eps)

class MSETanimotoHybridLoss(nn.Module):
    def __init__(self, lambda_mse=0.5, eps=1e-12):
        super().__init__()
        self.lambda_mse = float(lambda_mse)
        self.mse = nn.MSELoss(reduction="mean")
        self.eps = eps

    def forward(self, pred, target):
        # ensure nonnegativity for tanimoto term
        p = torch.clamp(pred, min=0)
        t = torch.clamp(target, min=0)
        mse_term = self.mse(pred, target)
        tan = tanimoto(p, t, eps=self.eps).mean()
        # maximize tanimoto => subtract from loss
        return self.lambda_mse * mse_term + (1.0 - self.lambda_mse) * (1.0 - tan)
