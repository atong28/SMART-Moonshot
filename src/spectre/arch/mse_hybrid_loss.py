import torch
import torch.nn as nn
import torch.nn.functional as F

class MSECosineHybridLoss(nn.Module):
    """
    Blend of cosine distance and MSE for real-valued FP regression.
    Targets/preds are compared as-is (logits); no sigmoid.
    """
    def __init__(self, lambda_mse=0.6, eps=1e-12):
        super().__init__()
        self.lambda_mse, self.eps = lambda_mse, eps
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        p = F.normalize(pred, dim=1)
        t = F.normalize(target, dim=1)
        cos = 1.0 - torch.sum(p * t, dim=1)
        return (1 - self.lambda_mse) * cos.mean() + self.lambda_mse * self.mse(pred, target)
